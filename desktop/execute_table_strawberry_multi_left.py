#!/usr/bin/env python3
"""Execute ordered table-strawberry pick/place cycles from one synchronized image."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import can
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT_ROOT.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "table_strawberry_multi_grasp_left.json"
DEFAULT_REPORT = PROJECT_ROOT / "config" / "last_table_strawberry_multi_left.json"
G20_SDK = WORKSPACE / "linkerhand" / "linkerhand-python-sdk-main"

sys.path.insert(0, str(SCRIPT_DIR))
import execute_table_strawberry_vision_left as vision

sys.path.insert(0, str(G20_SDK))
import g20_pose_control_left as g20


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def image_size_wh(path: Path) -> list[int] | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return [int(image.shape[1]), int(image.shape[0])]


def read_pose(arm) -> list[float]:
    code, pose = arm.get_position(is_radian=False)
    if code != 0:
        raise RuntimeError(f"get_position failed: {code}")
    return [float(value) for value in pose]


def check_arm(arm, label: str) -> None:
    if arm.state == 4 or arm.error_code != 0 or arm.warn_code != 0:
        raise RuntimeError(
            f"{label}: state={arm.state}, error={arm.error_code}, warn={arm.warn_code}"
        )


def move(arm, pose: list[float], speed: float, acc: float, label: str) -> list[float]:
    code = arm.set_position(
        *pose,
        speed=speed,
        mvacc=acc,
        wait=True,
        radius=-1,
    )
    if code != 0:
        raise RuntimeError(f"{label}: set_position failed with code {code}")
    actual = read_pose(arm)
    check_arm(arm, label)
    print(f"{label}: {[round(value, 3) for value in actual]}")
    return actual


def open_bus(channel: str, attempts: int = 3):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return can.interface.Bus(
                channel=channel,
                interface="pcan",
                bitrate=1_000_000,
            )
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(1.0)
    raise RuntimeError(f"cannot open PCAN channel {channel}: {last_error}")


def release_thumb_connected(
    bus,
    can_id: int,
    target_raw: int,
    speed: int,
    torque: int,
    timeout: float,
    poll_interval: float,
    tolerance: int = 8,
) -> dict:
    current = g20.read_pose(bus, can_id)
    if len(current) != len(g20.FINGERS):
        raise RuntimeError("G20 readback incomplete before release")
    target = {finger: list(values) for finger, values in current.items()}
    target["thumb"][g20.SLOTS["base"]] = int(target_raw)
    g20.send(bus, can_id, g20.FINGERS["thumb"]["speed"], [speed] * 6)
    g20.send(bus, can_id, g20.FINGERS["thumb"]["torque"], [torque] * 6)
    g20.send(bus, can_id, g20.FINGERS["thumb"]["pos"], target["thumb"])

    deadline = time.monotonic() + timeout
    last = current
    while time.monotonic() < deadline:
        last = g20.read_pose(bus, can_id)
        if len(last) == len(g20.FINGERS):
            actual = int(last["thumb"][g20.SLOTS["base"]])
            if abs(actual - target_raw) <= tolerance:
                print(
                    "release thumb.base -> "
                    f"{target_raw}, readback={actual}, tolerance={tolerance}"
                )
                return last
        time.sleep(poll_interval)
    raise RuntimeError(
        "thumb release timeout: "
        f"target={target_raw}, tolerance={tolerance}, readback={last}"
    )


def release_thumb(
    channel: str,
    can_id: int,
    target_raw: int,
    speed: int,
    torque: int,
    timeout: float,
    poll_interval: float,
    tolerance: int = 8,
) -> dict:
    bus = open_bus(channel)
    try:
        return release_thumb_connected(
            bus,
            can_id,
            target_raw,
            speed,
            torque,
            timeout,
            poll_interval,
            tolerance,
        )
    finally:
        bus.shutdown()


def return_to_capture(
    arm,
    capture_pose: list[float],
    safe_z: float,
    horizontal_speed: float,
    vertical_speed: float,
    acc: float,
    preserve_rpy: bool = False,
) -> list[dict]:
    actual = read_pose(arm)
    records = []
    high_z = max(float(safe_z), float(capture_pose[2]), float(actual[2]))

    high_current = list(actual)
    high_current[2] = high_z
    if abs(actual[2] - high_z) > 1.0:
        actual = move(
            arm, high_current, vertical_speed, acc, "return_raise_to_safe_z"
        )
        records.append({"name": "return_raise_to_safe_z", "pose": actual})

    high_capture = list(capture_pose)
    high_capture[2] = high_z
    if preserve_rpy:
        high_capture[3:] = actual[3:]
    actual = move(
        arm,
        high_capture,
        horizontal_speed,
        acc,
        (
            "return_high_to_capture_xy_keep_rpy"
            if preserve_rpy
            else "return_high_to_capture_xy_rpy"
        ),
    )
    records.append(
        {
            "name": (
                "return_high_to_capture_xy_keep_rpy"
                if preserve_rpy
                else "return_high_to_capture_xy_rpy"
            ),
            "pose": actual,
        }
    )

    if abs(high_z - capture_pose[2]) > 1.0:
        capture_z_pose = list(capture_pose)
        if preserve_rpy:
            capture_z_pose[3:] = actual[3:]
        actual = move(
            arm, capture_z_pose, vertical_speed, acc, "return_capture_z"
        )
        records.append({"name": "return_capture_z", "pose": actual})
    return records


def placement_slot_pose(
    placement: dict,
    reference_pose: list[float],
    placement_index: int,
    safe_z: float,
    release_z: float,
    place_rpy_override: list[float] | None = None,
    orientation_info: dict | None = None,
) -> tuple[list[float], list[float], dict]:
    place_high = list(reference_pose)
    mode = placement.get("mode", "")
    if mode == "manual_first_slot_y_rows_then_x_rows":
        slots_per_row = max(1, int(placement.get("slots_per_row", 5)))
        row = int(placement_index) // slots_per_row
        col = int(placement_index) % slots_per_row
        x_spacing = float(placement.get("row_x_spacing_mm", -50.0))
        y_spacing = float(placement.get("slot_y_spacing_mm", -40.0))
        place_high[0] = float(reference_pose[0]) + x_spacing * row
        place_high[1] = float(reference_pose[1]) + y_spacing * col
        slot_info = {
            "mode": mode,
            "row": row,
            "col": col,
            "slots_per_row": slots_per_row,
            "row_x_spacing_mm": x_spacing,
            "slot_y_spacing_mm": y_spacing,
        }
    else:
        spacing = float(placement.get("slot_spacing_mm", 0.0))
        place_high[0] = float(reference_pose[0]) + spacing * int(placement_index)
        slot_info = {
            "mode": mode or "legacy_x_decrement",
            "slot_spacing_mm": spacing,
            "legacy_axis": placement.get("slot_axis", "base_x"),
        }
    place_high[2] = safe_z
    if place_rpy_override is not None:
        place_high[3:] = [float(value) for value in place_rpy_override]
        slot_info["rpy_source"] = "direction_aligned_override"
    else:
        slot_info["rpy_source"] = "placement_reference_pose"
    if orientation_info is not None:
        slot_info["orientation_alignment"] = orientation_info
    place_release = list(place_high)
    place_release[2] = release_z
    return place_high, place_release, slot_info


def direction_line_image_angle_deg(
    grasp_payload: dict,
    image_size: list[int] | None = None,
    normalized: bool = False,
) -> tuple[float, dict]:
    line = grasp_payload.get("direction_line_2d")
    if not isinstance(line, dict):
        raise RuntimeError("missing direction_line_2d for placement orientation")
    start = line.get("from_ignored_tip_vertex")
    end = line.get("to_support_midpoint")
    if start is None or end is None:
        raise RuntimeError("incomplete direction_line_2d for placement orientation")
    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    vector = [ex - sx, ey - sy]
    angle_vector = list(vector)
    if normalized:
        if not image_size or float(image_size[0]) <= 0 or float(image_size[1]) <= 0:
            raise RuntimeError("normalized direction angle requires image_size")
        angle_vector = [vector[0] / float(image_size[0]), vector[1] / float(image_size[1])]
    angle = float(np.degrees(np.arctan2(angle_vector[1], angle_vector[0])))
    return angle, {
        "from_ignored_tip_vertex": [sx, sy],
        "to_support_midpoint": [ex, ey],
        "vector_image_px": vector,
        "angle_vector": angle_vector,
        "image_size_wh": image_size,
        "normalized": normalized,
        "angle_image_deg": angle,
    }


def placement_orientation_for_target(
    target_grasp: dict,
    server_dir: Path,
    placement: dict,
    reference_pose: list[float],
) -> tuple[list[float] | None, dict]:
    alignment = placement.get("orientation_alignment", {})
    if not bool(alignment.get("enabled", False)):
        return None, {"enabled": False}
    if alignment.get("mode") != "match_direction_line_2d_to_reference":
        return None, {
            "enabled": True,
            "applied": False,
            "reason": f"unsupported mode {alignment.get('mode')!r}",
        }

    raw_image, _grasp_json_path, grasp_payload = vision.load_server_grasp(server_dir)
    angle_mode = alignment.get("angle_mode", "pixel")
    normalized = angle_mode == "normalized_image"
    target_image_size = image_size_wh(raw_image)
    target_angle, target_line = direction_line_image_angle_deg(
        grasp_payload,
        target_image_size,
        normalized,
    )
    if normalized:
        reference_angle = float(alignment["reference_line_angle_normalized_image_deg"])
    else:
        reference_angle = float(alignment["reference_line_angle_image_deg"])
    angle_delta = vision.normalize_angle_deg(reference_angle - target_angle)

    place_rpy = [float(value) for value in reference_pose[3:]]
    place_rpy[2] = vision.normalize_angle_deg(float(reference_pose[5]) + angle_delta)
    return place_rpy, {
        "enabled": True,
        "applied": True,
        "mode": "match_direction_line_2d_to_reference",
        "angle_mode": angle_mode,
        "reference_line_angle_image_deg": reference_angle,
        "target_line_angle_image_deg": target_angle,
        "angle_delta_deg": angle_delta,
        "target_line": target_line,
        "reference_line": alignment.get("reference_direction_line_2d"),
        "placement_reference_rpy_deg": [float(value) for value in reference_pose[3:]],
        "placement_reference_rz_deg": float(reference_pose[5]),
        "place_rpy_deg": place_rpy,
        "rz_rule": "place_rz = placement_reference_rz + normalize(reference_line_angle - target_line_angle)",
        "red_tip_policy": alignment.get("red_tip_policy", "directed_line_preserves_red_tip_side"),
        "reference_source": alignment.get("reference_server_dir"),
    }


def apply_red_tip_base_proximity_constraint(
    grasp_payload: dict,
    capture_pose: list[float],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    camera_to_tcp: np.ndarray,
    table_z: float,
    target_rpy: list[float],
    orientation_following: dict,
) -> dict:
    constraint = orientation_following.get("red_tip_base_proximity_constraint", {})
    if not constraint or not bool(constraint.get("enabled", False)):
        return {"enabled": False}
    line = grasp_payload.get("direction_line_2d")
    if not isinstance(line, dict):
        return {
            "enabled": True,
            "applied": False,
            "reason": "missing direction_line_2d",
        }
    red_pixel = line.get("from_ignored_tip_vertex")
    support_pixel = line.get("to_support_midpoint")
    if red_pixel is None or support_pixel is None:
        return {
            "enabled": True,
            "applied": False,
            "reason": "missing red-tip or support-midpoint pixel",
        }
    red_point, red_distance = vision.pixel_to_table_point(
        red_pixel,
        capture_pose,
        camera_matrix,
        distortion,
        camera_to_tcp,
        table_z,
    )
    support_point, support_distance = vision.pixel_to_table_point(
        support_pixel,
        capture_pose,
        camera_matrix,
        distortion,
        camera_to_tcp,
        table_z,
    )
    red_base_distance = float(np.linalg.norm(red_point[:2]))
    support_base_distance = float(np.linalg.norm(support_point[:2]))
    would_flip = red_base_distance > support_base_distance
    control_rz = bool(constraint.get("control_rz", False))
    applied = bool(would_flip and control_rz)
    original_rz = float(target_rpy[2])
    if applied:
        target_rpy[2] = vision.normalize_angle_deg(original_rz + 180.0)
    return {
        "enabled": True,
        "applied": applied,
        "would_flip_rz": would_flip,
        "control_rz": control_rz,
        "rule": (
            "diagnose whether the red-tip is closer to the xArm base origin "
            "than the support midpoint; Rz is changed only when control_rz=true"
        ),
        "red_tip_pixel": [float(v) for v in red_pixel],
        "support_midpoint_pixel": [float(v) for v in support_pixel],
        "red_tip_table_base_mm": red_point.tolist(),
        "support_midpoint_table_base_mm": support_point.tolist(),
        "red_tip_base_xy_distance_mm": red_base_distance,
        "support_midpoint_base_xy_distance_mm": support_base_distance,
        "red_ray_distance_mm": float(red_distance),
        "support_ray_distance_mm": float(support_distance),
        "rz_before_constraint_deg": original_rz,
        "rz_after_constraint_deg": float(target_rpy[2]),
    }


def check_target_workspace(
    target_xy: list[float],
    target_z: float,
    target_rpy: list[float],
    motion: dict,
) -> dict:
    limits = motion.get("target_workspace_limits", {})
    if not bool(limits.get("enabled", False)):
        return {"enabled": False, "ok": True}
    x = float(target_xy[0])
    y = float(target_xy[1])
    radius = float(np.linalg.norm([x, y]))
    checks = {
        "x_min_mm": x >= float(limits.get("x_min_mm", -float("inf"))),
        "x_max_mm": x <= float(limits.get("x_max_mm", float("inf"))),
        "y_min_mm": y >= float(limits.get("y_min_mm", -float("inf"))),
        "y_max_mm": y <= float(limits.get("y_max_mm", float("inf"))),
        "radius_min_mm": radius >= float(limits.get("radius_min_mm", 0.0)),
        "radius_max_mm": radius <= float(limits.get("radius_max_mm", float("inf"))),
        "z_min_mm": float(target_z) >= float(limits.get("z_min_mm", -float("inf"))),
        "z_max_mm": float(target_z) <= float(limits.get("z_max_mm", float("inf"))),
    }
    ok = all(checks.values())
    return {
        "enabled": True,
        "ok": ok,
        "target_tcp_xy_mm": [x, y],
        "target_tcp_z_mm": float(target_z),
        "target_rpy_deg": [float(v) for v in target_rpy],
        "base_xy_radius_mm": radius,
        "limits": limits,
        "checks": checks,
    }


def compute_target_rpy(
    target_id: str,
    capture_json: Path,
    server_dir: Path,
    config: dict,
    height: dict,
    alignment_reference: dict,
    calibration_bundle: tuple,
    capture_pose: list[float],
) -> dict:
    camera_matrix, distortion, camera_to_tcp, _calibration = calibration_bundle
    orientation_following = config.get("orientation_following", {})

    target_pose = [float(v) for v in height["xarm"]["tcp_pose_base_mm_deg"]]
    target_rpy = target_pose[3:]
    reference_rpy = [
        float(v)
        for v in alignment_reference.get("target_rpy_deg", target_pose[3:])
    ]

    capture_pose_loaded, _capture_image, _capture_payload = vision.load_capture_pose(
        capture_json
    )
    if any(abs(a - b) > 1e-6 for a, b in zip(capture_pose_loaded, capture_pose)):
        raise RuntimeError(
            f"{target_id}: capture pose mismatch between manifest preload and file"
        )

    _server_raw_image, _grasp_json_path, grasp_payload = vision.load_server_grasp(
        server_dir
    )
    table_z = float(height["table_z_base_mm"])
    orientation_adjustment = {
        "enabled": bool(orientation_following.get("enabled", False)),
        "reference_rpy_deg": reference_rpy,
        "target_rpy_before_auto_rz_deg": [float(v) for v in target_rpy],
    }
    if orientation_following.get("enabled", False):
        mode = orientation_following.get("mode")
        if mode == "pair_to_thumb":
            reference_info = alignment_reference.get("pair_to_thumb_reference")
            if not reference_info or "angle_base_deg" not in reference_info:
                raise RuntimeError(
                    "missing pair-to-thumb reference angle in alignment reference"
                )
            reference_angle = float(reference_info["angle_base_deg"])
            pair_thumb_angle, pair_thumb_info = vision.pair_to_thumb_angle_base_deg(
                grasp_payload,
                capture_pose,
                camera_matrix,
                distortion,
                camera_to_tcp,
                table_z,
            )
            rz_delta = vision.normalize_angle_deg(pair_thumb_angle - reference_angle)
            target_rpy[2] = vision.normalize_angle_deg(
                float(reference_rpy[2]) + rz_delta
            )
            orientation_adjustment.update(
                {
                    "source": "pair_to_thumb",
                    "pair_to_thumb": pair_thumb_info,
                    "pair_to_thumb_reference": reference_info,
                    "pair_to_thumb_reference_angle_deg": reference_angle,
                    "pair_to_thumb_angle_delta_deg": rz_delta,
                    "rz_delta_from_reference_deg": rz_delta,
                    "target_rpy_after_auto_rz_deg": [float(v) for v in target_rpy],
                }
            )
        else:
            line_angle, line_info = vision.direction_line_angle_deg(grasp_payload)
            reference_angle = float(
                orientation_following.get("direction_line_reference_angle_deg", 0.0)
            )
            sign = float(orientation_following.get("direction_line_rz_sign", 1.0))
            angle_delta = vision.normalize_angle_deg(line_angle - reference_angle)
            rz_delta = sign * angle_delta
            target_rpy[2] = vision.normalize_angle_deg(
                float(reference_rpy[2]) + rz_delta
            )
            orientation_adjustment.update(
                {
                    "source": "direction_line",
                    "direction_line": line_info,
                    "direction_line_reference_angle_deg": reference_angle,
                    "direction_line_angle_delta_deg": angle_delta,
                    "direction_line_rz_sign": sign,
                    "rz_delta_from_reference_deg": rz_delta,
                    "target_rpy_after_auto_rz_deg": [float(v) for v in target_rpy],
                }
            )
    red_tip_constraint = apply_red_tip_base_proximity_constraint(
        grasp_payload,
        capture_pose,
        camera_matrix,
        distortion,
        camera_to_tcp,
        table_z,
        target_rpy,
        orientation_following,
    )
    orientation_adjustment["red_tip_base_proximity_constraint"] = red_tip_constraint
    if red_tip_constraint.get("applied"):
        orientation_adjustment["target_rpy_after_red_tip_constraint_deg"] = [
            float(v) for v in target_rpy
        ]

    return {
        "target_rpy_deg": [float(v) for v in target_rpy],
        "orientation_adjustment": orientation_adjustment,
    }


def execute_target_grasp_connected(
    arm,
    bus,
    target_id: str,
    capture_json: Path,
    server_dir: Path,
    target_report: Path,
    config: dict,
    preset: dict,
    height: dict,
    mount: dict,
    alignment_reference: dict,
    thumb_mapping: dict,
    calibration_bundle: tuple,
    capture_pose: list[float],
    cycle_index: int,
    skip_initial_return: bool,
    args: argparse.Namespace,
) -> dict:
    camera_matrix, distortion, camera_to_tcp, calibration = calibration_bundle
    motion = config["motion"]
    orientation_following = config.get("orientation_following", {})
    grasp_override = config.get("grasp_override", {})

    target_pose = [float(v) for v in height["xarm"]["tcp_pose_base_mm_deg"]]
    if "tcp_z_mm" in grasp_override:
        target_pose[2] = float(grasp_override["tcp_z_mm"])
    target_z = float(target_pose[2])
    target_rpy = target_pose[3:]
    reference_rpy = [
        float(v)
        for v in alignment_reference.get("target_rpy_deg", target_pose[3:])
    ]
    reference_tcp_xy_offset_mm = np.asarray(
        alignment_reference.get("tcp_xy_offset_mm", [0.0, 0.0]),
        dtype=np.float64,
    )
    tcp_xy_base_correction_mm = np.asarray(
        alignment_reference.get("tcp_xy_base_correction_mm", [0.0, 0.0]),
        dtype=np.float64,
    )

    capture_pose_loaded, capture_image, capture_payload = vision.load_capture_pose(
        capture_json
    )
    if any(abs(a - b) > 1e-6 for a, b in zip(capture_pose_loaded, capture_pose)):
        raise RuntimeError(
            f"{target_id}: capture pose mismatch between manifest preload and file"
        )
    server_raw_image, grasp_json_path, grasp_payload = vision.load_server_grasp(
        server_dir
    )
    if not capture_image.exists():
        raise RuntimeError(f"capture image missing: {capture_image}")
    image_check = vision.validate_server_image(capture_image, server_raw_image, 12.0)

    center_2d = grasp_payload.get("grasp_pair_center_2d")
    if not center_2d or len(center_2d) != 2:
        raise RuntimeError("grasp_points_triangle.json missing grasp_pair_center_2d")

    table_z = float(height["table_z_base_mm"])
    orientation_adjustment = {
        "enabled": bool(orientation_following.get("enabled", False)),
        "offset_rotation_enabled": bool(
            orientation_following.get("rotate_tcp_xy_offset_with_rz", True)
        ),
        "reference_rpy_deg": reference_rpy,
        "target_rpy_before_auto_rz_deg": [float(v) for v in target_rpy],
    }
    if orientation_following.get("enabled", False):
        mode = orientation_following.get("mode")
        if mode == "pair_to_thumb":
            reference_info = alignment_reference.get("pair_to_thumb_reference")
            if not reference_info or "angle_base_deg" not in reference_info:
                raise RuntimeError(
                    "missing pair-to-thumb reference angle in alignment reference"
                )
            reference_angle = float(reference_info["angle_base_deg"])
            pair_thumb_angle, pair_thumb_info = vision.pair_to_thumb_angle_base_deg(
                grasp_payload,
                capture_pose,
                camera_matrix,
                distortion,
                camera_to_tcp,
                table_z,
            )
            rz_delta = vision.normalize_angle_deg(pair_thumb_angle - reference_angle)
            target_rpy[2] = vision.normalize_angle_deg(
                float(reference_rpy[2]) + rz_delta
            )
            orientation_adjustment.update(
                {
                    "source": "pair_to_thumb",
                    "pair_to_thumb": pair_thumb_info,
                    "pair_to_thumb_reference": reference_info,
                    "pair_to_thumb_reference_angle_deg": reference_angle,
                    "pair_to_thumb_angle_delta_deg": rz_delta,
                    "rz_delta_from_reference_deg": rz_delta,
                    "target_rpy_after_auto_rz_deg": [float(v) for v in target_rpy],
                }
            )
        else:
            line_angle, line_info = vision.direction_line_angle_deg(grasp_payload)
            reference_angle = float(
                orientation_following.get("direction_line_reference_angle_deg", 0.0)
            )
            sign = float(orientation_following.get("direction_line_rz_sign", 1.0))
            angle_delta = vision.normalize_angle_deg(line_angle - reference_angle)
            rz_delta = sign * angle_delta
            target_rpy[2] = vision.normalize_angle_deg(
                float(reference_rpy[2]) + rz_delta
            )
            orientation_adjustment.update(
                {
                    "source": "direction_line",
                    "direction_line": line_info,
                    "direction_line_reference_angle_deg": reference_angle,
                    "direction_line_angle_delta_deg": angle_delta,
                    "direction_line_rz_sign": sign,
                    "rz_delta_from_reference_deg": rz_delta,
                    "target_rpy_after_auto_rz_deg": [float(v) for v in target_rpy],
                }
            )
    red_tip_constraint = apply_red_tip_base_proximity_constraint(
        grasp_payload,
        capture_pose,
        camera_matrix,
        distortion,
        camera_to_tcp,
        table_z,
        target_rpy,
        orientation_following,
    )
    orientation_adjustment["red_tip_base_proximity_constraint"] = red_tip_constraint
    if red_tip_constraint.get("applied"):
        orientation_adjustment["target_rpy_after_red_tip_constraint_deg"] = [
            float(v) for v in target_rpy
        ]

    geometry = vision.triangle_geometry(grasp_payload)
    thumb_mapping_relative_tolerance = 0.2
    recommended_thumb_steps, thumb_mapping_check = vision.mapped_thumb_steps(
        geometry,
        thumb_mapping,
        thumb_mapping_relative_tolerance,
    )
    table_point, ray_distance = vision.pixel_to_table_point(
        center_2d,
        capture_pose,
        camera_matrix,
        distortion,
        camera_to_tcp,
        table_z,
    )
    rz_delta_for_offset = vision.normalize_angle_deg(
        float(target_rpy[2]) - float(reference_rpy[2])
    )
    tcp_xy_offset_mm = np.asarray(reference_tcp_xy_offset_mm, dtype=np.float64)
    if orientation_following.get("rotate_tcp_xy_offset_with_rz", True):
        tcp_xy_offset_mm = vision.rotate_xy(tcp_xy_offset_mm, rz_delta_for_offset)
    tcp_xy_offset_mm = tcp_xy_offset_mm + tcp_xy_base_correction_mm
    orientation_adjustment.update(
        {
            "reference_tcp_xy_offset_mm": reference_tcp_xy_offset_mm.tolist(),
            "rz_delta_for_offset_rotation_deg": rz_delta_for_offset,
            "tcp_xy_base_correction_mm": tcp_xy_base_correction_mm.tolist(),
            "effective_tcp_xy_offset_mm": tcp_xy_offset_mm.tolist(),
        }
    )
    target_xy = (table_point[:2] + tcp_xy_offset_mm).tolist()

    current = read_pose(arm)
    check_arm(arm, f"{target_id} before motion")
    horizontal = list(current)
    horizontal[0] = float(target_xy[0])
    horizontal[1] = float(target_xy[1])
    oriented = list(horizontal)
    oriented[3:] = [float(v) for v in target_rpy]
    fast_approach_mm = float(motion.get("fast_approach_mm", 10.0))
    fast_approach = list(oriented)
    fast_approach[2] = target_z + fast_approach_mm
    final = list(oriented)
    final[2] = target_z
    rise = list(final)
    rise[2] = target_z + float(motion.get("post_grasp_rise_mm", 50.0))

    horizontal_distance = float(
        np.linalg.norm(np.asarray(current[:2]) - np.asarray(target_xy))
    )
    max_horizontal_mm = float(motion.get("max_horizontal_mm", 500.0))
    if horizontal_distance > max_horizontal_mm:
        raise RuntimeError(
            f"{target_id}: horizontal move {horizontal_distance:.3f} mm exceeds "
            f"limit {max_horizontal_mm:.3f} mm"
        )
    workspace_check = check_target_workspace(
        target_xy,
        target_z,
        target_rpy,
        motion,
    )
    if not workspace_check["ok"]:
        raise RuntimeError(
            f"{target_id}: target TCP pose rejected by workspace precheck: "
            f"{workspace_check}"
        )

    thumb_base_steps = [245, 140]
    thumb_speed = int(grasp_override.get("thumb_speed", 8))
    thumb_torque = int(grasp_override.get("thumb_torque", 4))
    thumb_preclose_wait_for_feedback = bool(
        grasp_override.get("thumb_preclose_wait_for_feedback", False)
    )
    g20_poll_interval = float(config.get("timing", {}).get("g20_poll_interval_s", 0.5))
    g20_feedback_timeout = 8.0
    thumb_contact_timeout = 20.0
    thumb_stall_window = float(grasp_override.get("thumb_stall_window_s", 1.0))
    thumb_stall_delta = 2
    thumb_contact_max_raw = int(grasp_override.get("thumb_contact_max_raw", 220))

    report = {
        "capture_json": str(capture_json.resolve()),
        "capture_image": str(capture_image.resolve()),
        "capture_metadata": capture_payload,
        "server_dir": str(server_dir.resolve()),
        "grasp_json": str(grasp_json_path.resolve()),
        "calibration": calibration,
        "mount": mount,
        "capture_pose": capture_pose,
        "target_point_table_base_mm": table_point.tolist(),
        "tcp_xy_offset_mm": tcp_xy_offset_mm.tolist(),
        "reference_tcp_xy_offset_mm": reference_tcp_xy_offset_mm.tolist(),
        "tcp_xy_base_correction_mm": tcp_xy_base_correction_mm.tolist(),
        "orientation_adjustment": orientation_adjustment,
        "alignment_reference": alignment_reference,
        "target_tcp_xy_mm": target_xy,
        "target_tcp_z_mm": target_z,
        "target_rpy_deg": [float(v) for v in target_rpy],
        "workspace_check": workspace_check,
        "image_check": image_check,
        "ray_distance_mm": float(ray_distance),
        "waypoints": [
            {
                "name": "horizontal",
                "pose": horizontal,
                "speed": motion.get("speed_horizontal_mm_s", 80.0),
                "acc": motion.get("acc_mm_s2", 150.0),
            },
            {
                "name": "orient",
                "pose": oriented,
                "speed": motion.get("speed_orient_mm_s", 80.0),
                "acc": motion.get("acc_mm_s2", 150.0),
            },
            {
                "name": "fast_approach",
                "pose": fast_approach,
                "speed": motion.get("speed_fast_descent_mm_s", 80.0),
                "acc": motion.get("acc_mm_s2", 150.0),
            },
            {
                "name": "final",
                "pose": final,
                "speed": motion.get("speed_final_descent_mm_s", 3.0),
                "acc": motion.get("acc_final_mm_s2", 35.0),
            },
        ],
        "speed_profile": {
            "g20_initial": int(config.get("timing", {}).get("g20_initial_speed", 100)),
            "horizontal": motion.get("speed_horizontal_mm_s", 80.0),
            "orient": motion.get("speed_orient_mm_s", 80.0),
            "fast_descent": motion.get("speed_fast_descent_mm_s", 80.0),
            "slow_final": motion.get("speed_final_descent_mm_s", 3.0),
            "rise": motion.get("speed_post_grasp_rise_mm_s", 100.0),
            "thumb": thumb_speed,
            "fast_approach_mm": fast_approach_mm,
            "fast_acc": motion.get("acc_mm_s2", 150.0),
            "final_acc": motion.get("acc_final_mm_s2", 35.0),
            "g20_poll_interval_s": g20_poll_interval,
        },
        "thumb_base_steps": thumb_base_steps,
        "thumb_contact_policy": {
            "mode": "low_torque_stall_detection",
            "validated_geometry_anchor_raw": int(thumb_mapping["contact_can_raw"]),
            "command_target_raw": 140,
            "preclose_raw": 245,
            "preclose_wait_for_feedback": thumb_preclose_wait_for_feedback,
            "contact_timeout_s": thumb_contact_timeout,
            "stall_window_s": thumb_stall_window,
            "stall_delta_raw": thumb_stall_delta,
            "contact_max_raw": thumb_contact_max_raw,
            "require_stalled_contact_before_rise": True,
        },
        "thumb_geometry": geometry,
        "thumb_mapping": thumb_mapping,
        "thumb_mapping_check": thumb_mapping_check,
        "recommended_thumb_base_steps": recommended_thumb_steps,
        "post_grasp_rise_mm": float(motion.get("post_grasp_rise_mm", 50.0)),
        "applied": bool(args.apply),
        "connected_mode": "single_process_xarm_and_g20_persistent",
        "xarm_actual": [],
        "g20_actual": [],
    }

    print(f"{target_id}: server_raw={server_raw_image.resolve()}")
    print(f"{target_id}: image_check={image_check}")
    print(f"{target_id}: table_point={np.round(table_point, 3).tolist()} mm")
    print(f"{target_id}: target_tcp_xy={np.round(target_xy, 3).tolist()} mm")
    print(f"{target_id}: target_rpy={np.round(target_rpy, 6).tolist()} deg")
    print(f"{target_id}: thumb_mapping_check={thumb_mapping_check}")

    capture_translation_error = float(
        np.linalg.norm(
            np.asarray(current[:3], dtype=np.float64)
            - np.asarray(capture_pose[:3], dtype=np.float64)
        )
    )
    capture_angle_errors = [
        vision.angular_difference_deg(actual, expected)
        for actual, expected in zip(current[3:], capture_pose[3:])
    ]
    target_angle_errors = [
        vision.angular_difference_deg(actual, expected)
        for actual, expected in zip(current[3:], target_rpy)
    ]
    first_target_preoriented = bool(
        config.get("orientation_optimization", {}).get(
            "preorient_first_target_before_g20_initialization", False
        )
    )
    if cycle_index > 0 or skip_initial_return:
        report["capture_pose_preflight"] = {
            "actual_tcp_pose": current,
            "translation_error_mm": capture_translation_error,
            "angle_errors_deg": capture_angle_errors,
            "skipped": True,
            "reason": "controlled_multi_target_cycle",
        }
        print(
            f"{target_id}: capture pose preflight skipped, "
            f"translation_error={capture_translation_error:.3f} mm"
        )
    elif (
        first_target_preoriented
        and capture_translation_error <= 5.0
        and max(target_angle_errors) <= 2.0
    ):
        report["capture_pose_preflight"] = {
            "actual_tcp_pose": current,
            "translation_error_mm": capture_translation_error,
            "capture_angle_errors_deg": capture_angle_errors,
            "target_angle_errors_deg": target_angle_errors,
            "skipped": True,
            "reason": "first_target_preoriented_at_capture_xy",
        }
        print(
            f"{target_id}: capture pose preflight allowed after first-target "
            f"preorientation, translation_error={capture_translation_error:.3f} mm"
        )
    elif capture_translation_error > 5.0 or max(capture_angle_errors) > 2.0:
        raise RuntimeError(
            f"{target_id}: current TCP no longer matches synchronized capture pose: "
            f"translation_error={capture_translation_error:.3f} mm, "
            f"angle_errors={np.round(capture_angle_errors, 3).tolist()} deg"
        )
    else:
        report["capture_pose_preflight"] = {
            "actual_tcp_pose": current,
            "translation_error_mm": capture_translation_error,
            "angle_errors_deg": capture_angle_errors,
            "skipped": False,
        }
        print(
            f"{target_id}: capture pose preflight passed, "
            f"translation_error={capture_translation_error:.3f} mm"
        )

    initial_g20 = vision.verify_g20_initial(
        bus,
        args.can_id,
        preset["g20_initial_pose"],
        tolerance=int(config.get("timing", {}).get("g20_initial_tolerance_raw", 8)),
    )
    report["g20_actual"].append(
        {"name": "initial_check_before_motion", "pose": initial_g20}
    )

    motion_waypoints = list(report["waypoints"])
    if max(target_angle_errors) <= 2.0:
        motion_waypoints = [
            waypoint for waypoint in motion_waypoints if waypoint["name"] != "orient"
        ]
        print(f"{target_id}: orient skipped; current RPY already matches grasp RPY")
    if current[2] <= fast_approach[2] + 2.0:
        motion_waypoints = [
            waypoint
            for waypoint in motion_waypoints
            if waypoint["name"] != "fast_approach"
        ]

    previous_z = current[2]
    for waypoint in motion_waypoints:
        name = waypoint["name"]
        pose = waypoint["pose"]
        code = arm.set_position(
            *pose,
            speed=waypoint["speed"],
            mvacc=waypoint["acc"],
            wait=True,
            radius=-1,
        )
        if code != 0:
            raise RuntimeError(f"{target_id} {name}: set_position failed with {code}")
        actual = read_pose(arm)
        check_arm(arm, f"{target_id} {name}")
        if name in ("horizontal", "orient") and actual[2] < current[2] - 1.0:
            raise RuntimeError(
                f"{target_id} {name}: unexpected Z drop to {actual[2]:.3f} mm"
            )
        if name in ("fast_approach", "final") and actual[2] > previous_z + 2.0:
            raise RuntimeError(
                f"{target_id} {name}: unexpected upward Z jump to {actual[2]:.3f} mm"
            )
        previous_z = actual[2]
        report["xarm_actual"].append({"name": name, "pose": actual, "code": code})
        print(f"{target_id} {name} actual={np.round(actual, 3).tolist()}")

    current_g20 = vision.verify_g20_initial(
        bus,
        args.can_id,
        preset["g20_initial_pose"],
        tolerance=int(config.get("timing", {}).get("g20_initial_tolerance_raw", 8)),
    )
    report["g20_actual"].append(
        {"name": "initial_check_after_motion", "pose": current_g20}
    )
    for index, step in enumerate(thumb_base_steps):
        is_contact_step = index == len(thumb_base_steps) - 1
        if is_contact_step:
            current_g20, contact_result = vision.close_thumb_until_contact(
                bus,
                args.can_id,
                current_g20,
                int(step),
                thumb_speed,
                thumb_torque,
                thumb_contact_timeout,
                g20_poll_interval,
                thumb_stall_window,
                thumb_stall_delta,
                thumb_contact_max_raw,
            )
            report["thumb_contact_result"] = contact_result
            if not contact_result["contact_detected"]:
                raise RuntimeError(
                    f"{target_id}: thumb reached target without stalled-contact signal"
                )
            print(f"{target_id}: thumb stalled contact detected")
        else:
            target = {finger: list(values) for finger, values in current_g20.items()}
            target["thumb"][g20.SLOTS["base"]] = int(step)
            if thumb_preclose_wait_for_feedback:
                current_g20 = vision.send_pose(
                    bus,
                    args.can_id,
                    target,
                    thumb_speed,
                    thumb_torque,
                    g20_feedback_timeout,
                    g20_poll_interval,
                )
            else:
                g20.send(
                    bus,
                    args.can_id,
                    g20.FINGERS["thumb"]["speed"],
                    [thumb_speed] * 6,
                )
                g20.send(
                    bus,
                    args.can_id,
                    g20.FINGERS["thumb"]["torque"],
                    [thumb_torque] * 6,
                )
                g20.send(
                    bus,
                    args.can_id,
                    g20.FINGERS["thumb"]["pos"],
                    target["thumb"],
                )
                current_g20 = target
        report["g20_actual"].append(
            {
                "name": f"thumb_base_{step}",
                "pose": current_g20,
                "feedback_waited": bool(is_contact_step or thumb_preclose_wait_for_feedback),
            }
        )
        print(f"{target_id}: thumb.base -> {step}, readback={current_g20['thumb']}")

    rise_code = arm.set_position(
        rise[0],
        rise[1],
        rise[2],
        rise[3],
        rise[4],
        rise[5],
        speed=motion.get("speed_post_grasp_rise_mm_s", 100.0),
        mvacc=motion.get("acc_mm_s2", 150.0),
        wait=True,
        radius=-1,
    )
    if rise_code != 0:
        raise RuntimeError(f"{target_id}: post-grasp rise failed with {rise_code}")
    rise_actual = read_pose(arm)
    check_arm(arm, f"{target_id} post-grasp-rise")
    report["xarm_actual"].append(
        {"name": "post_grasp_rise", "pose": rise_actual, "code": rise_code}
    )
    print(f"{target_id}: post_grasp_rise actual={np.round(rise_actual, 3).tolist()}")

    target_report.parent.mkdir(parents=True, exist_ok=True)
    target_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--robot-ip", default="192.168.1.200")
    parser.add_argument("--channel", default="PCAN_USBBUS1")
    parser.add_argument("--can-id", type=lambda value: int(value, 0), default=0x28)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--skip-initial-return",
        action="store_true",
        help=(
            "start the first target directly from the current TCP pose; useful "
            "when resuming after a completed placement"
        ),
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    config = read_json(args.config)
    manifest = read_json(args.manifest)
    capture_json = Path(manifest["capture_json"])
    capture_payload = read_json(capture_json)
    capture_pose = [
        float(value)
        for value in capture_payload["xarm"]["tcp_pose_base_mm_deg"]
    ]
    targets = sorted(
        manifest["targets"],
        key=lambda target: (int(target.get("order", 0)), str(target["id"])),
    )
    if not targets:
        raise RuntimeError("manifest contains no targets")

    place_center_xy = [
        float(value) for value in config["place_grasp_center_xy_base_mm"]
    ]
    tcp_xy_offset = [float(value) for value in config["tcp_xy_offset_mm"]]
    configured_place_pose = [
        float(value)
        for value in config["computed_place_tcp_pose_base_mm_deg"]
    ]
    place_pose = list(configured_place_pose)
    place_pose[0] = place_center_xy[0] + tcp_xy_offset[0]
    place_pose[1] = place_center_xy[1] + tcp_xy_offset[1]
    placement = config.get("placement", {})
    motion = config["motion"]
    timing = config.get("timing", {})
    process_switch_delay_s = max(
        0.0, float(timing.get("process_switch_delay_s", 0.5))
    )
    xarm_enable_settle_s = max(
        0.0, float(timing.get("xarm_enable_settle_s", 0.5))
    )
    release_poll_interval_s = max(
        0.05, float(timing.get("release_poll_interval_s", 0.5))
    )
    g20_poll_interval_s = max(
        0.05, float(timing.get("g20_poll_interval_s", 0.5))
    )
    g20_initial_speed = int(timing.get("g20_initial_speed", 100))
    g20_initial_torque = int(timing.get("g20_initial_torque", 12))
    g20_initial_feedback_timeout_s = float(
        timing.get("g20_initial_feedback_timeout_s", 8.0)
    )
    g20_initial_tolerance_raw = int(timing.get("g20_initial_tolerance_raw", 8))
    skip_g20_initialize_after_first_target = bool(
        timing.get("skip_g20_initialize_after_first_target", False)
    )
    release = config["release"]
    release_tolerance_raw = int(release.get("tolerance_raw", 8))
    orientation_following = config.get("orientation_following", {})
    grasp_tcp_z = float(config.get("grasp_override", {}).get("tcp_z_mm", 215.723053))
    placement_reference_pose = [
        float(value)
        for value in placement.get("reference_tcp_pose_base_mm_deg", place_pose)
    ]
    placement_slot_spacing_mm = float(placement.get("slot_spacing_mm", 0.0))
    placement_slots_per_row = int(placement.get("slots_per_row", 1))
    placement_slot_y_spacing_mm = float(placement.get("slot_y_spacing_mm", 0.0))
    placement_row_x_spacing_mm = float(placement.get("row_x_spacing_mm", 0.0))
    placement_release_z = grasp_tcp_z + float(
        placement.get("release_clearance_above_grasp_mm", 0.0)
    )
    if placement.get("safe_z_source") == "reference_tcp_pose_z":
        placement_safe_z = float(placement_reference_pose[2])
    elif placement.get("safe_z_source") == "grasp_tcp_z_plus_post_grasp_rise":
        placement_safe_z = grasp_tcp_z + float(motion.get("post_grasp_rise_mm", 50.0))
    else:
        placement_safe_z = float(motion["safe_z_mm"])
    report = {
        "format": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(args.manifest.resolve()),
        "capture_json": str(capture_json.resolve()),
        "capture_pose": capture_pose,
        "place_grasp_center_xy_mm": place_center_xy,
        "tcp_xy_offset_mm": tcp_xy_offset,
        "place_pose": place_pose,
        "placement": placement,
        "placement_reference_pose": placement_reference_pose,
        "placement_release_z": placement_release_z,
        "placement_safe_z": placement_safe_z,
        "timing": {
            "process_switch_delay_s": process_switch_delay_s,
            "process_switch_delay_used": False,
            "xarm_enable_settle_s": xarm_enable_settle_s,
            "release_poll_interval_s": release_poll_interval_s,
            "g20_poll_interval_s": g20_poll_interval_s,
            "g20_initial_speed": g20_initial_speed,
            "g20_initial_torque": g20_initial_torque,
            "g20_initial_feedback_timeout_s": g20_initial_feedback_timeout_s,
            "g20_initial_tolerance_raw": g20_initial_tolerance_raw,
            "skip_g20_initialize_after_first_target": (
                skip_g20_initialize_after_first_target
            ),
            "connected_mode": "single_process_xarm_and_g20_persistent",
        },
        "release_policy": {
            "target_raw": int(release["target_raw"]),
            "speed": int(release["speed"]),
            "torque": int(release["torque"]),
            "feedback_timeout_s": float(release["feedback_timeout_s"]),
            "tolerance_raw": release_tolerance_raw,
        },
        "target_order": [target["id"] for target in targets],
        "orientation_following": orientation_following,
        "applied": bool(args.apply),
        "timing_measurements": {},
        "cycles": [],
    }

    print(f"targets={report['target_order']}")
    print(f"capture_pose={[round(value, 3) for value in capture_pose]}")
    print(
        "place_grasp_center_xy="
        f"{[round(value, 3) for value in place_center_xy]}"
    )
    print(f"place_pose={[round(value, 3) for value in place_pose]}")
    if placement:
        print(
            "placement_reference_pose="
            f"{[round(value, 3) for value in placement_reference_pose]}"
        )
        print(
            f"placement_release_z={placement_release_z:.3f}, "
            f"slot_spacing_x={placement_slot_spacing_mm:.3f}, "
            f"slots_per_row={placement_slots_per_row}, "
            f"slot_y_spacing={placement_slot_y_spacing_mm:.3f}, "
            f"row_x_spacing={placement_row_x_spacing_mm:.3f}"
        )
    print(
        "timing="
        f"process_switch={process_switch_delay_s:.2f}s(ignored in connected mode), "
        f"xarm_enable_settle={xarm_enable_settle_s:.2f}s, "
        f"release_poll={release_poll_interval_s:.2f}s, "
        f"g20_poll={g20_poll_interval_s:.2f}s, "
        f"g20_initial_speed={g20_initial_speed}, "
        f"g20_initial_timeout={g20_initial_feedback_timeout_s:.1f}s, "
        f"g20_initial_tol={g20_initial_tolerance_raw}, "
        "skip_g20_init_after_first="
        f"{skip_g20_initialize_after_first_target}"
    )
    if not args.apply:
        for target in targets:
            print(f"dry-run target {target['id']}: {target['server_dir']}")
        print("Dry run only. Add --apply to execute all cycles.")
        return 0

    preset = read_json(vision.DEFAULT_PRESET)
    height = read_json(vision.DEFAULT_HEIGHT)
    mount = read_json(vision.DEFAULT_MOUNT)
    alignment_reference = read_json(vision.DEFAULT_ALIGNMENT_REFERENCE)
    thumb_mapping = read_json(vision.DEFAULT_THUMB_MAPPING)
    calibration_bundle = vision.perception.load_calibration(vision.DEFAULT_CALIB_DIR)

    from xarm.wrapper import XArmAPI

    arm = None
    bus = None
    try:
        arm = XArmAPI(args.robot_ip)
        if not arm.connected:
            raise RuntimeError(f"cannot connect to xArm {args.robot_ip}")
        bus = open_bus(args.channel)
        check_arm(arm, "multi-cycle preflight")
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(xarm_enable_settle_s)
        check_arm(arm, "multi-cycle enabled")

        if args.skip_initial_return:
            print("initial return to capture pose skipped")
        else:
            return_to_capture(
                arm,
                capture_pose,
                motion["safe_z_mm"],
                motion["speed_return_mm_s"],
                motion["speed_vertical_mm_s"],
                motion["acc_mm_s2"],
            )

        orientation_optimization = config.get("orientation_optimization", {})
        preorientation_start = time.monotonic()
        if bool(
            orientation_optimization.get(
                "preorient_first_target_before_g20_initialization", False
            )
        ) and not args.skip_initial_return:
            first_target = targets[0]
            first_rpy = compute_target_rpy(
                str(first_target["id"]),
                capture_json,
                Path(first_target["server_dir"]),
                config,
                height,
                alignment_reference,
                calibration_bundle,
                capture_pose,
            )
            current_pose = read_pose(arm)
            angle_errors = [
                vision.angular_difference_deg(actual, expected)
                for actual, expected in zip(current_pose[3:], first_rpy["target_rpy_deg"])
            ]
            if max(angle_errors) > 2.0:
                preorient_pose = list(current_pose)
                preorient_pose[3:] = first_rpy["target_rpy_deg"]
                actual_preorient = move(
                    arm,
                    preorient_pose,
                    motion["speed_orient_mm_s"],
                    motion["acc_mm_s2"],
                    "preorient_first_target_rpy",
                )
                report["first_target_preorientation"] = {
                    "enabled": True,
                    "target_id": str(first_target["id"]),
                    "target_rpy_deg": first_rpy["target_rpy_deg"],
                    "orientation_adjustment": first_rpy["orientation_adjustment"],
                    "actual_pose": actual_preorient,
                }
            else:
                report["first_target_preorientation"] = {
                    "enabled": True,
                    "target_id": str(first_target["id"]),
                    "target_rpy_deg": first_rpy["target_rpy_deg"],
                    "skipped": True,
                    "reason": "current_rpy_already_matches_first_target",
                }
        preorientation_elapsed_s = time.monotonic() - preorientation_start
        report["timing_measurements"]["first_target_preorientation_s"] = (
            preorientation_elapsed_s
        )
        print(
            "first_target_preorientation_elapsed="
            f"{preorientation_elapsed_s:.3f}s"
        )

        g20_initial_start = time.monotonic()
        initialized = vision.initialize_g20(
            bus,
            args.can_id,
            preset["g20_initial_pose"],
            speed=g20_initial_speed,
            torque=g20_initial_torque,
            feedback_timeout=g20_initial_feedback_timeout_s,
            poll_interval=g20_poll_interval_s,
            tolerance=g20_initial_tolerance_raw,
        )
        g20_initial_elapsed_s = time.monotonic() - g20_initial_start
        report["timing_measurements"]["g20_initialization_s"] = (
            g20_initial_elapsed_s
        )
        report["timing_measurements"]["pregrasp_preparation_total_s"] = (
            preorientation_elapsed_s + g20_initial_elapsed_s
        )
        report["g20_initialization"] = initialized
        print(
            "G20 initial pose ready once for connected multi-target cycle, "
            f"elapsed={g20_initial_elapsed_s:.3f}s"
        )

        for cycle_index, target in enumerate(targets):
            target_id = str(target["id"])
            placement_index = int(target.get("order", cycle_index))
            server_dir = Path(target["server_dir"])
            target_report = (
                args.report.parent
                / f"{args.report.stem}_{cycle_index:02d}_{target_id}.json"
            )
            print(f"cycle {cycle_index + 1}/{len(targets)}: {target_id}")
            grasp_error = None
            try:
                target_grasp = execute_target_grasp_connected(
                    arm,
                    bus,
                    target_id,
                    capture_json,
                    server_dir,
                    target_report,
                    config,
                    preset,
                    height,
                    mount,
                    alignment_reference,
                    thumb_mapping,
                    calibration_bundle,
                    capture_pose,
                    cycle_index,
                    args.skip_initial_return,
                    args,
                )
            except Exception as exc:
                grasp_error = exc
            if grasp_error is not None:
                if arm.error_code != 0 or arm.warn_code != 0 or arm.state == 4:
                    raise RuntimeError(
                        f"{target_id} failed with controller fault: "
                        f"state={arm.state}, error={arm.error_code}, "
                        f"warn={arm.warn_code}"
                    ) from grasp_error
                print(
                    f"{target_id}: target failed without controller fault; "
                    "opening the thumb, raising to safe Z, and skipping to next target"
                )
                recovery_motion = []
                actual_before_recovery = read_pose(arm)
                release_pose = release_thumb_connected(
                    bus,
                    args.can_id,
                    int(release["target_raw"]),
                    int(release["speed"]),
                    int(release["torque"]),
                    float(release["feedback_timeout_s"]),
                    release_poll_interval_s,
                    release_tolerance_raw,
                )
                arm.motion_enable(enable=True)
                arm.set_mode(0)
                arm.set_state(0)
                time.sleep(xarm_enable_settle_s)
                check_arm(arm, f"{target_id} recovery enabled")
                skip_safe = list(read_pose(arm))
                skip_safe[2] = max(placement_safe_z, skip_safe[2])
                if abs(skip_safe[2] - actual_before_recovery[2]) > 1.0:
                    actual_safe = move(
                        arm,
                        skip_safe,
                        motion["speed_vertical_mm_s"],
                        motion["acc_mm_s2"],
                        f"{target_id}_skip_raise_to_safe_z",
                    )
                    recovery_motion.append(
                        {"name": "skip_raise_to_safe_z", "pose": actual_safe}
                    )
                report["cycles"].append(
                    {
                        "id": target_id,
                        "server_dir": str(server_dir.resolve()),
                        "status": "target_skipped_after_non_controller_error",
                        "error": repr(grasp_error),
                        "controller_state": {
                            "state": int(arm.state),
                            "error_code": int(arm.error_code),
                            "warn_code": int(arm.warn_code),
                        },
                        "release_pose": release_pose,
                        "motion": recovery_motion,
                    }
                )
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                continue

            actual = read_pose(arm)
            high = list(actual)
            high[2] = max(placement_safe_z, actual[2])
            cycle_record = {
                "id": target_id,
                "status": "grasp_confirmed_and_placed",
                "server_dir": str(server_dir.resolve()),
                "single_grasp_report": str(target_report.resolve()),
                "target_tcp_xy_mm": target_grasp["target_tcp_xy_mm"],
                "target_rpy_deg": target_grasp["target_rpy_deg"],
                "thumb_contact_result": target_grasp.get("thumb_contact_result"),
                "placement_index": placement_index,
                "motion": [],
            }
            if abs(high[2] - actual[2]) > 1.0:
                actual = move(
                    arm,
                    high,
                    motion["speed_vertical_mm_s"],
                    motion["acc_mm_s2"],
                    f"{target_id}_raise_to_place_safe_z",
                )
                cycle_record["motion"].append(
                    {"name": "raise_to_place_safe_z", "pose": actual}
            )

            if placement:
                place_rpy_override, placement_orientation_info = (
                    placement_orientation_for_target(
                        target_grasp,
                        server_dir,
                        placement,
                        placement_reference_pose,
                    )
                )
                place_high, place_release, placement_slot_info = placement_slot_pose(
                    placement,
                    placement_reference_pose,
                    placement_index,
                    placement_safe_z,
                    placement_release_z,
                    place_rpy_override,
                    placement_orientation_info,
                )
                cycle_record["placement_slot"] = placement_slot_info
            else:
                place_high = list(place_pose)
                place_high[2] = max(float(motion["safe_z_mm"]), place_pose[2])
                place_release = list(place_high)

            place_xy_high = list(place_high)
            place_xy_high[3:] = read_pose(arm)[3:]
            actual = move(
                arm,
                place_xy_high,
                motion["speed_horizontal_mm_s"],
                motion["acc_mm_s2"],
                f"{target_id}_move_to_place_xy_high",
            )
            cycle_record["motion"].append(
                {"name": "move_to_place_xy_high", "pose": actual}
            )
            actual_rpy_error = max(
                vision.angular_difference_deg(actual_value, target_value)
                for actual_value, target_value in zip(actual[3:], place_high[3:])
            )
            if actual_rpy_error > 2.0:
                actual = move(
                    arm,
                    place_high,
                    motion["speed_orient_mm_s"],
                    motion["acc_mm_s2"],
                    f"{target_id}_orient_to_place_rpy",
                )
                cycle_record["motion"].append(
                    {"name": "orient_to_place_rpy", "pose": actual}
                )
            if placement:
                actual = move(
                    arm,
                    place_release,
                    motion["speed_vertical_mm_s"],
                    motion["acc_mm_s2"],
                    f"{target_id}_lower_to_place_release_z",
                )
                cycle_record["motion"].append(
                    {"name": "lower_to_place_release_z", "pose": actual}
                )
            cycle_record["release_pose"] = release_thumb_connected(
                bus,
                args.can_id,
                int(release["target_raw"]),
                int(release["speed"]),
                int(release["torque"]),
                float(release["feedback_timeout_s"]),
                release_poll_interval_s,
                release_tolerance_raw,
            )
            if placement:
                actual = move(
                    arm,
                    place_high,
                    motion["speed_vertical_mm_s"],
                    motion["acc_mm_s2"],
                    f"{target_id}_raise_after_release",
                )
                cycle_record["motion"].append(
                    {"name": "raise_after_release", "pose": actual}
                )
            if cycle_index < len(targets) - 1:
                cycle_record["motion"].append(
                    {
                        "name": "skip_return_to_capture_between_targets",
                        "pose": read_pose(arm),
                        "reason": "direct_next_target_optimization",
                    }
                )
            else:
                cycle_record["motion"].extend(
                    return_to_capture(
                        arm,
                        capture_pose,
                        placement_safe_z,
                        motion["speed_return_mm_s"],
                        motion["speed_vertical_mm_s"],
                        motion["acc_mm_s2"],
                        preserve_rpy=False,
                    )
                )
            report["cycles"].append(cycle_record)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        print("all targets completed; robot returned to capture pose")
        return 0
    finally:
        if bus is not None:
            bus.shutdown()
        if arm is not None:
            arm.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
