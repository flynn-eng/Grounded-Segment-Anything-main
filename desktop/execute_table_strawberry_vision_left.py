#!/usr/bin/env python3
"""Execute the left-hand table-strawberry grasp from a captured image and server grasp points."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import can
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT_ROOT.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(SCRIPT_DIR))

import table_perception as perception

sys.path.insert(0, str(WORKSPACE / "linkerhand" / "linkerhand-python-sdk-main"))
import g20_pose_control_left as g20


DEFAULT_CALIB_DIR = (
    WORKSPACE / "calib_20260611_153756" / "calib_20260611_153756"
)
DEFAULT_PRESET = PROJECT_ROOT / "config" / "table_strawberry_grasp_preset.json"
DEFAULT_HEIGHT = (
    PROJECT_ROOT
    / "config"
    / "table_strawberry_grasp_height_optimized_20260615.json"
)
DEFAULT_ALIGNMENT_REFERENCE = (
    PROJECT_ROOT
    / "config"
    / "table_strawberry_vision_alignment_reference_left.json"
)
DEFAULT_THUMB_MAPPING = (
    PROJECT_ROOT
    / "config"
    / "table_strawberry_thumb_base_contact_mapping_left_170.json"
)
DEFAULT_MOUNT = PROJECT_ROOT / "config" / "g20_mount_reference.json"
DEFAULT_REPORT = PROJECT_ROOT / "config" / "last_table_strawberry_vision_left.json"
DEFAULT_SERVER_DIR = WORKSPACE / "testdata" / "outputs_strawberry_03"
G20_POSE_TOLERANCE = 8
G20_INITIAL_SWAY_SLOT = g20.SLOTS["sway"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_image(path: Path) -> np.ndarray:
    buffer = np.fromfile(str(path), dtype=np.uint8)
    if buffer.size == 0:
        raise RuntimeError(f"cannot read image bytes: {path}")
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode image: {path}")
    return image


def image_shape_and_rmse(path_a: Path, path_b: Path) -> tuple[tuple[int, int, int], float]:
    image_a = load_image(path_a)
    image_b = load_image(path_b)
    if image_a.shape != image_b.shape:
        raise RuntimeError(
            f"image shape mismatch: {path_a.name}={image_a.shape}, {path_b.name}={image_b.shape}"
        )
    delta = image_a.astype(np.float32) - image_b.astype(np.float32)
    rmse = float(np.sqrt(np.mean(np.square(delta))))
    return image_a.shape, rmse


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_pose(arm) -> list[float]:
    code, pose = arm.get_position(is_radian=False)
    if code != 0:
        raise RuntimeError(f"get_position failed: {code}")
    return [float(value) for value in pose]


def check_arm_clean(arm, label: str) -> None:
    if arm.state == 4:
        raise RuntimeError(f"{label}: xArm state=4")
    if arm.error_code != 0:
        raise RuntimeError(f"{label}: xArm error={arm.error_code}")
    if arm.warn_code != 0:
        raise RuntimeError(f"{label}: xArm warn={arm.warn_code}")


def load_capture_pose(capture_json: Path) -> tuple[list[float], Path, dict]:
    payload = read_json(capture_json)
    image_path = Path(payload["image"])
    pose = [float(value) for value in payload["xarm"]["tcp_pose_base_mm_deg"]]
    return pose, image_path, payload


def load_server_grasp(server_dir: Path) -> tuple[Path, Path, dict]:
    raw_image = server_dir / "raw_image.jpg"
    grasp_json = server_dir / "grasp_points_triangle.json"
    mask_json = server_dir / "mask.json"
    mask_image = server_dir / "mask_0.png"
    vis_image = server_dir / "grasp_points_triangle_visualization.jpg"
    for path in (raw_image, grasp_json, mask_json, mask_image, vis_image):
        if not path.exists():
            raise RuntimeError(f"missing server output: {path}")
    return raw_image, grasp_json, read_json(grasp_json)


def validate_server_image(
    capture_image: Path,
    server_image: Path,
    max_rmse: float,
) -> dict:
    capture_hash = sha256(capture_image)
    server_hash = sha256(server_image)
    same_hash = capture_hash == server_hash
    shape, rmse = image_shape_and_rmse(capture_image, server_image)
    if not same_hash and rmse > max_rmse:
        raise RuntimeError(
            f"server image does not match capture: hash differs and rmse={rmse:.3f} > {max_rmse:.3f}"
        )
    return {
        "capture_sha256": capture_hash,
        "server_sha256": server_hash,
        "same_sha256": same_hash,
        "shape": list(shape),
        "rmse": rmse,
    }


def parse_xy_offset(text: str) -> np.ndarray:
    values = [float(value.strip()) for value in text.split(",")]
    if len(values) != 2:
        raise argparse.ArgumentTypeError("expected X,Y")
    return np.asarray(values, dtype=np.float64)


def parse_rpy(text: str) -> np.ndarray:
    values = [float(value.strip()) for value in text.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected roll,pitch,yaw")
    return np.asarray(values, dtype=np.float64)


def normalize_angle_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def angular_difference_deg(actual: float, expected: float) -> float:
    return abs((actual - expected + 180.0) % 360.0 - 180.0)


def rotate_xy(vector: np.ndarray, angle_deg: float) -> np.ndarray:
    angle = np.radians(float(angle_deg))
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    return rotation @ np.asarray(vector, dtype=np.float64).reshape(2)


def direction_line_angle_deg(grasp_payload: dict) -> tuple[float, dict]:
    line = grasp_payload.get("direction_line_2d")
    if line is not None:
        vector = line.get("vector")
        if vector is None or len(vector) != 2:
            start = np.asarray(line["from_ignored_tip_vertex"], dtype=np.float64)
            end = np.asarray(line["to_support_midpoint"], dtype=np.float64)
            vector = (end - start).tolist()
        source = "direction_line_2d"
    else:
        points = grasp_payload["grasp_points_2d"]
        start = np.asarray(points["P_thumb"], dtype=np.float64)
        end = np.asarray(grasp_payload["grasp_pair_center_2d"], dtype=np.float64)
        vector = (end - start).tolist()
        source = "P_thumb_to_grasp_pair_center_2d"
    vector_xy = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector_xy))
    if norm < 1e-9:
        raise RuntimeError("direction line is too short to compute Rz")
    angle = float(np.degrees(np.arctan2(vector_xy[1], vector_xy[0])))
    return angle, {
        "source": source,
        "vector_pixels": vector_xy.tolist(),
        "image_angle_deg": angle,
    }


def pixel_to_table_point(
    pixel: list[float],
    pose: list[float],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    camera_to_tcp: np.ndarray,
    table_z: float,
) -> tuple[np.ndarray, float]:
    camera_origin, ray = perception.camera_ray_in_base(
        pixel, pose, camera_matrix, distortion, camera_to_tcp
    )
    return perception.intersect_ray_plane(
        camera_origin,
        ray,
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        -table_z,
    )


def pair_to_thumb_angle_base_deg(
    grasp_payload: dict,
    pose: list[float],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    camera_to_tcp: np.ndarray,
    table_z: float,
) -> tuple[float, dict]:
    points = grasp_payload["grasp_points_2d"]
    pair_pixel = grasp_payload["grasp_pair_center_2d"]
    thumb_pixel = points["P_thumb"]
    pair_point, pair_distance = pixel_to_table_point(
        pair_pixel,
        pose,
        camera_matrix,
        distortion,
        camera_to_tcp,
        table_z,
    )
    thumb_point, thumb_distance = pixel_to_table_point(
        thumb_pixel,
        pose,
        camera_matrix,
        distortion,
        camera_to_tcp,
        table_z,
    )
    vector = thumb_point[:2] - pair_point[:2]
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise RuntimeError("pair-to-thumb vector is too short to compute Rz")
    angle = float(np.degrees(np.arctan2(vector[1], vector[0])))
    return angle, {
        "source": "grasp_pair_center_2d_to_P_thumb_projected_to_table",
        "pair_pixel": [float(v) for v in pair_pixel],
        "thumb_pixel": [float(v) for v in thumb_pixel],
        "pair_point_table_base_mm": pair_point.tolist(),
        "thumb_point_table_base_mm": thumb_point.tolist(),
        "vector_xy_base_mm": vector.tolist(),
        "angle_base_deg": angle,
        "pair_ray_distance_mm": float(pair_distance),
        "thumb_ray_distance_mm": float(thumb_distance),
    }


def triangle_geometry(grasp_payload: dict) -> dict:
    points = grasp_payload["grasp_points_2d"]
    index = np.asarray(points["P_index"], dtype=np.float64)
    middle = np.asarray(points["P_middle"], dtype=np.float64)
    thumb = np.asarray(points["P_thumb"], dtype=np.float64)
    pair = np.asarray(
        grasp_payload["grasp_pair_center_2d"],
        dtype=np.float64,
    )
    pair_spacing = float(np.linalg.norm(index - middle))
    thumb_to_pair = float(np.linalg.norm(thumb - pair))
    bbox = grasp_payload.get("diagnostics", {}).get("bbox_xywh")
    if not bbox or len(bbox) != 4:
        raise RuntimeError(
            "grasp_points_triangle.json missing diagnostics.bbox_xywh required "
            "for scale-invariant thumb mapping"
        )
    bbox_width = float(bbox[2])
    bbox_height = float(bbox[3])
    if bbox_width <= 0.0 or bbox_height <= 0.0:
        raise RuntimeError(f"invalid grasp bbox dimensions: {bbox}")
    return {
        "index_middle_distance": pair_spacing,
        "index_thumb_distance": float(np.linalg.norm(index - thumb)),
        "middle_thumb_distance": float(np.linalg.norm(middle - thumb)),
        "thumb_to_pair_center_distance": thumb_to_pair,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "pair_spacing_over_bbox_width": pair_spacing / bbox_width,
        "thumb_to_pair_over_bbox_height": thumb_to_pair / bbox_height,
        "thumb_to_pair_over_pair_spacing": (
            thumb_to_pair / pair_spacing if pair_spacing > 1e-9 else None
        ),
    }


def mapped_thumb_steps(
    geometry: dict,
    mapping: dict,
    relative_tolerance: float,
) -> tuple[list[int], dict]:
    reference = mapping["reference_geometry_pixels"]
    keys = (
        "pair_spacing_over_bbox_width",
        "thumb_to_pair_over_bbox_height",
        "thumb_to_pair_over_pair_spacing",
    )
    relative_errors = {
        key: abs(float(geometry[key]) - float(reference[key]))
        / max(abs(float(reference[key])), 1e-9)
        for key in keys
    }
    similar = all(error <= relative_tolerance for error in relative_errors.values())
    contact = int(mapping["contact_can_raw"])
    steps = []
    if similar:
        steps.append(245)
        if contact < 200:
            steps.append(200)
        steps.append(contact)
        steps = list(dict.fromkeys(steps))
    return steps, {
        "similar": similar,
        "relative_tolerance": relative_tolerance,
        "relative_errors": relative_errors,
        "max_relative_error": max(relative_errors.values()),
        "contact_can_raw": contact,
    }


def verify_g20_initial(
    bus,
    can_id: int,
    expected: dict,
    tolerance: int = G20_POSE_TOLERANCE,
) -> dict:
    pose = g20.read_pose(bus, can_id)
    if len(pose) != len(g20.FINGERS):
        raise RuntimeError("G20 readback incomplete")
    for finger, expected_values in expected.items():
        actual = pose.get(finger)
        if actual is None:
            raise RuntimeError(f"G20 missing finger readback: {finger}")
        diffs = [abs(int(a) - int(b)) for a, b in zip(actual, expected_values)]
        if any(delta > tolerance for delta in diffs):
            raise RuntimeError(
                f"G20 initial pose mismatch for {finger}: got {actual}, expected {expected_values}, diffs={diffs}"
            )
    return pose


def pose_matches(actual: dict, target: dict, tolerance: int) -> bool:
    if len(actual) != len(g20.FINGERS):
        return False
    for finger, expected_values in target.items():
        actual_values = actual.get(finger)
        if actual_values is None:
            return False
        if any(
            abs(int(a) - int(b)) > tolerance
            for a, b in zip(actual_values, expected_values)
        ):
            return False
    return True


def wait_for_g20_pose(
    bus,
    can_id: int,
    target: dict[str, list[int]],
    timeout: float,
    poll_interval: float,
    tolerance: int = G20_POSE_TOLERANCE,
) -> dict:
    deadline = time.monotonic() + timeout
    last_pose = {}
    while time.monotonic() < deadline:
        last_pose = g20.read_pose(bus, can_id)
        if pose_matches(last_pose, target, tolerance):
            return last_pose
        time.sleep(poll_interval)
    raise RuntimeError(
        f"G20 pose timeout after {timeout:.1f}s: got {last_pose}, expected {target}"
    )


def open_can_bus(channel: str, attempts: int = 3):
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


def send_pose(
    bus,
    can_id: int,
    target: dict[str, list[int]],
    speed: int,
    torque: int,
    feedback_timeout: float,
    poll_interval: float,
    tolerance: int = G20_POSE_TOLERANCE,
) -> dict:
    current = g20.read_pose(bus, can_id)
    if len(current) != len(g20.FINGERS):
        raise RuntimeError("G20 readback incomplete before send")
    changed = g20.changed_fingers(current, target)
    for finger in changed:
        g20.send(bus, can_id, g20.FINGERS[finger]["speed"], [speed] * 6)
        g20.send(bus, can_id, g20.FINGERS[finger]["torque"], [torque] * 6)
        g20.send(bus, can_id, g20.FINGERS[finger]["pos"], target[finger])
    return wait_for_g20_pose(
        bus,
        can_id,
        target,
        timeout=feedback_timeout,
        poll_interval=poll_interval,
        tolerance=tolerance,
    )


def close_thumb_until_contact(
    bus,
    can_id: int,
    current: dict[str, list[int]],
    target_raw: int,
    speed: int,
    torque: int,
    feedback_timeout: float,
    poll_interval: float,
    stall_window: float,
    stall_delta: int,
    contact_max_raw: int,
) -> tuple[dict, dict]:
    target = {finger: list(values) for finger, values in current.items()}
    target["thumb"][g20.SLOTS["base"]] = int(target_raw)
    g20.send(bus, can_id, g20.FINGERS["thumb"]["speed"], [speed] * 6)
    g20.send(bus, can_id, g20.FINGERS["thumb"]["torque"], [torque] * 6)
    g20.send(bus, can_id, g20.FINGERS["thumb"]["pos"], target["thumb"])

    started = time.monotonic()
    stable_started = None
    stable_min = None
    stable_max = None
    last_pose = current
    while time.monotonic() - started < feedback_timeout:
        last_pose = g20.read_pose(bus, can_id)
        if len(last_pose) != len(g20.FINGERS):
            time.sleep(poll_interval)
            continue
        now = time.monotonic()
        actual_raw = int(last_pose["thumb"][g20.SLOTS["base"]])

        if abs(actual_raw - target_raw) <= G20_POSE_TOLERANCE:
            return last_pose, {
                "result": "target_reached_without_stall",
                "command_raw": int(target_raw),
                "actual_raw": actual_raw,
                "contact_detected": False,
            }

        if stable_started is None:
            stable_started = now
            stable_min = actual_raw
            stable_max = actual_raw
        else:
            candidate_min = min(stable_min, actual_raw)
            candidate_max = max(stable_max, actual_raw)
            if candidate_max - candidate_min <= stall_delta:
                stable_min = candidate_min
                stable_max = candidate_max
            else:
                stable_started = now
                stable_min = actual_raw
                stable_max = actual_raw

        if (
            stable_started is not None
            and now - stable_started >= stall_window
            and actual_raw <= contact_max_raw
            and actual_raw > target_raw + G20_POSE_TOLERANCE
        ):
            return last_pose, {
                "result": "stalled_contact",
                "command_raw": int(target_raw),
                "actual_raw": actual_raw,
                "contact_detected": True,
                "stall_window_s": stall_window,
                "stall_range_raw": stable_max - stable_min,
            }
        time.sleep(poll_interval)

    raise RuntimeError(
        f"thumb contact timeout after {feedback_timeout:.1f}s: "
        f"got {last_pose}, command_raw={target_raw}"
    )


def initialize_g20(
    bus,
    can_id: int,
    expected: dict[str, list[int]],
    speed: int,
    torque: int,
    feedback_timeout: float,
    poll_interval: float,
    tolerance: int = G20_POSE_TOLERANCE,
) -> list[dict]:
    current = g20.read_pose(bus, can_id)
    if len(current) != len(g20.FINGERS):
        raise RuntimeError("G20 readback incomplete before initialization")

    stage1 = {finger: list(values) for finger, values in current.items()}
    for finger, expected_values in expected.items():
        for index, value in enumerate(expected_values):
            if index != G20_INITIAL_SWAY_SLOT:
                stage1[finger][index] = int(value)

    actual_stage1 = send_pose(
        bus,
        can_id,
        stage1,
        speed,
        torque,
        feedback_timeout,
        poll_interval,
        tolerance=tolerance,
    )
    final_target = {
        finger: [int(value) for value in values]
        for finger, values in expected.items()
    }
    actual_final = send_pose(
        bus,
        can_id,
        final_target,
        speed,
        torque,
        feedback_timeout,
        poll_interval,
        tolerance=tolerance,
    )
    return [
        {"name": "g20_initial_stage1", "pose": actual_stage1},
        {"name": "g20_initial_final", "pose": actual_final},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="192.168.1.200")
    parser.add_argument("--channel", default="PCAN_USBBUS1")
    parser.add_argument("--can-id", type=lambda value: int(value, 0), default=0x28)
    parser.add_argument("--capture-json", type=Path, required=True)
    parser.add_argument("--server-dir", type=Path, default=DEFAULT_SERVER_DIR)
    parser.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR)
    parser.add_argument("--preset", type=Path, default=DEFAULT_PRESET)
    parser.add_argument("--height", type=Path, default=DEFAULT_HEIGHT)
    parser.add_argument("--alignment-reference", type=Path, default=DEFAULT_ALIGNMENT_REFERENCE)
    parser.add_argument("--thumb-mapping", type=Path, default=DEFAULT_THUMB_MAPPING)
    parser.add_argument("--mount", type=Path, default=DEFAULT_MOUNT)
    parser.add_argument("--tcp-xy-offset-mm", type=parse_xy_offset, default=None)
    parser.add_argument("--target-rpy", type=parse_rpy, default=None)
    parser.add_argument(
        "--rotate-tcp-xy-offset-with-rz",
        action="store_true",
        help=(
            "rotate the empirical TCP-to-fingertip-center XY offset by the "
            "delta between reference Rz and target Rz"
        ),
    )
    parser.add_argument(
        "--auto-rz-from-direction-line",
        action="store_true",
        help=(
            "compute target Rz from grasp_points_triangle.json direction_line_2d "
            "so the blue-green line becomes horizontal in the reference view"
        ),
    )
    parser.add_argument(
        "--auto-rz-from-pair-thumb",
        action="store_true",
        help=(
            "compute target Rz from the table-projected vector from "
            "grasp_pair_center_2d to P_thumb, using the validated reference "
            "sample in the alignment-reference file"
        ),
    )
    parser.add_argument(
        "--direction-line-reference-angle-deg",
        type=float,
        default=0.0,
        help="image angle that corresponds to the reference Rz",
    )
    parser.add_argument(
        "--direction-line-rz-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help="sign from image direction-line angle delta to xArm Rz delta",
    )
    parser.add_argument(
        "--pair-thumb-reference-angle-deg",
        type=float,
        default=None,
        help=(
            "reference base-frame angle for grasp_pair_center_2d -> P_thumb; "
            "defaults to alignment-reference.pair_to_thumb_reference.angle_base_deg"
        ),
    )
    parser.add_argument(
        "--fast-approach-mm",
        "--clearance-mm",
        dest="fast_approach_mm",
        type=float,
        default=10.0,
        help="switch from fast descent to slow final approach this far above grasp Z",
    )
    parser.add_argument("--speed-horizontal", type=float, default=50.0)
    parser.add_argument("--speed-orient", type=float, default=45.0)
    parser.add_argument("--speed-vertical", type=float, default=40.0)
    parser.add_argument("--speed-final", type=float, default=3.0)
    parser.add_argument("--speed-rise", type=float, default=80.0)
    parser.add_argument("--acc", type=float, default=100.0)
    parser.add_argument("--acc-final", type=float, default=35.0)
    parser.add_argument("--xarm-enable-settle", type=float, default=0.5)
    parser.add_argument("--max-rmse", type=float, default=12.0)
    parser.add_argument("--max-horizontal-mm", type=float, default=220.0)
    parser.add_argument("--max-capture-pose-translation-mm", type=float, default=5.0)
    parser.add_argument("--max-capture-pose-angle-deg", type=float, default=2.0)
    parser.add_argument(
        "--capture-position-only",
        action="store_true",
        help="for multi-target cycles: validate capture XYZ but allow retained grasp RPY",
    )
    parser.add_argument(
        "--skip-capture-pose-preflight",
        action="store_true",
        help=(
            "for controlled multi-target cycles: allow starting from the prior "
            "place pose while still using the synchronized capture pose for "
            "vision projection"
        ),
    )
    parser.add_argument("--thumb-speed", type=int, default=8)
    parser.add_argument("--thumb-torque", type=int, default=4)
    parser.add_argument("--thumb-contact-target", type=int, default=140)
    parser.add_argument("--thumb-contact-timeout", type=float, default=20.0)
    parser.add_argument("--thumb-stall-window", type=float, default=1.0)
    parser.add_argument("--thumb-stall-delta", type=int, default=2)
    parser.add_argument("--thumb-contact-max-raw", type=int, default=220)
    parser.add_argument("--g20-initial-speed", type=int, default=100)
    parser.add_argument("--g20-initial-torque", type=int, default=12)
    parser.add_argument("--g20-feedback-timeout", type=float, default=8.0)
    parser.add_argument("--g20-poll-interval", type=float, default=0.5)
    parser.add_argument(
        "--thumb-base-steps",
        default="",
        help="explicit comma-separated thumb.base CAN steps; empty means no close",
    )
    parser.add_argument(
        "--use-mapped-thumb-contact",
        action="store_true",
        help="use the real-contact thumb.base anchor only when triangle geometry is similar",
    )
    parser.add_argument("--thumb-mapping-relative-tolerance", type=float, default=0.2)
    parser.add_argument("--post-grasp-rise-mm", type=float, default=100.0)
    parser.add_argument(
        "--stop-before-final",
        action="store_true",
        help=(
            "apply only XY/orientation and fast approach, then stop before the "
            "slow final descent and all G20 thumb grasp actions"
        ),
    )
    parser.add_argument("--initialize-g20", action="store_true", default=True)
    parser.add_argument(
        "--skip-g20-initialize",
        dest="initialize_g20",
        action="store_false",
    )
    parser.add_argument("--check-g20-initial", action="store_true", default=True)
    parser.add_argument("--skip-g20-initial-check", dest="check_g20_initial", action="store_false")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    preset = read_json(args.preset)
    height = read_json(args.height)
    mount = read_json(args.mount)
    alignment_reference = None
    if args.alignment_reference.exists():
        alignment_reference = read_json(args.alignment_reference)
    thumb_mapping = read_json(args.thumb_mapping)
    thumb_mappings = [thumb_mapping]

    table_z = float(height["table_z_base_mm"])
    target_pose = [float(v) for v in height["xarm"]["tcp_pose_base_mm_deg"]]
    target_z = float(target_pose[2])
    target_rpy = (
        [float(v) for v in args.target_rpy]
        if args.target_rpy is not None
        else target_pose[3:]
    )
    reference_rpy = [
        float(v)
        for v in (alignment_reference or {}).get("target_rpy_deg", target_pose[3:])
    ]
    tcp_xy_offset_mm = (
        np.asarray(args.tcp_xy_offset_mm, dtype=np.float64)
        if args.tcp_xy_offset_mm is not None
        else np.asarray(
            (alignment_reference or {}).get("tcp_xy_offset_mm", [0.0, 0.0]),
            dtype=np.float64,
        )
    )
    tcp_xy_base_correction_mm = (
        np.zeros(2, dtype=np.float64)
        if args.tcp_xy_offset_mm is not None
        else np.asarray(
            (alignment_reference or {}).get(
                "tcp_xy_base_correction_mm",
                [0.0, 0.0],
            ),
            dtype=np.float64,
        )
    )

    capture_pose, capture_image, capture_payload = load_capture_pose(args.capture_json)
    server_raw_image, grasp_json_path, grasp_payload = load_server_grasp(args.server_dir)

    if not capture_image.exists():
        raise RuntimeError(f"capture image missing: {capture_image}")
    image_check = validate_server_image(capture_image, server_raw_image, args.max_rmse)

    center_2d = grasp_payload.get("grasp_pair_center_2d")
    if not center_2d or len(center_2d) != 2:
        raise RuntimeError("grasp_points_triangle.json missing grasp_pair_center_2d")
    orientation_adjustment = {
        "enabled": bool(args.auto_rz_from_direction_line),
        "offset_rotation_enabled": bool(args.rotate_tcp_xy_offset_with_rz),
        "reference_rpy_deg": reference_rpy,
        "target_rpy_before_auto_rz_deg": [float(v) for v in target_rpy],
    }
    if args.auto_rz_from_direction_line and args.auto_rz_from_pair_thumb:
        raise RuntimeError(
            "choose only one Rz source: --auto-rz-from-direction-line or --auto-rz-from-pair-thumb"
        )

    camera_matrix, distortion, camera_to_tcp, calibration = perception.load_calibration(
        args.calib_dir
    )

    if args.auto_rz_from_pair_thumb:
        reference_angle = args.pair_thumb_reference_angle_deg
        reference_info = (alignment_reference or {}).get("pair_to_thumb_reference")
        if reference_angle is None:
            if not reference_info or "angle_base_deg" not in reference_info:
                raise RuntimeError(
                    "missing pair-to-thumb reference angle; provide "
                    "--pair-thumb-reference-angle-deg or add "
                    "alignment-reference.pair_to_thumb_reference.angle_base_deg"
                )
            reference_angle = float(reference_info["angle_base_deg"])
        pair_thumb_angle, pair_thumb_info = pair_to_thumb_angle_base_deg(
            grasp_payload,
            capture_pose,
            camera_matrix,
            distortion,
            camera_to_tcp,
            table_z,
        )
        rz_delta = normalize_angle_deg(pair_thumb_angle - float(reference_angle))
        target_rpy[2] = normalize_angle_deg(float(reference_rpy[2]) + rz_delta)
        orientation_adjustment.update(
            {
                "source": "pair_to_thumb",
                "pair_to_thumb": pair_thumb_info,
                "pair_to_thumb_reference": reference_info,
                "pair_to_thumb_reference_angle_deg": float(reference_angle),
                "pair_to_thumb_angle_delta_deg": rz_delta,
                "rz_delta_from_reference_deg": rz_delta,
                "target_rpy_after_auto_rz_deg": [float(v) for v in target_rpy],
            }
        )
    elif args.auto_rz_from_direction_line:
        line_angle, line_info = direction_line_angle_deg(grasp_payload)
        angle_delta = normalize_angle_deg(
            line_angle - float(args.direction_line_reference_angle_deg)
        )
        rz_delta = float(args.direction_line_rz_sign) * angle_delta
        target_rpy[2] = normalize_angle_deg(float(reference_rpy[2]) + rz_delta)
        orientation_adjustment.update(
            {
                "direction_line": line_info,
                "direction_line_reference_angle_deg": float(
                    args.direction_line_reference_angle_deg
                ),
                "direction_line_angle_delta_deg": angle_delta,
                "direction_line_rz_sign": float(args.direction_line_rz_sign),
                "rz_delta_from_reference_deg": rz_delta,
                "target_rpy_after_auto_rz_deg": [float(v) for v in target_rpy],
            }
        )
    geometry = triangle_geometry(grasp_payload)
    mapping_candidates = []
    for mapping in thumb_mappings:
        steps, check = mapped_thumb_steps(
            geometry,
            mapping,
            args.thumb_mapping_relative_tolerance,
        )
        mapping_candidates.append(
            {
                "name": mapping.get("name", str(mapping.get("contact_can_raw"))),
                "mapping": mapping,
                "steps": steps,
                "check": check,
            }
        )
    selected_mapping = min(
        mapping_candidates,
        key=lambda candidate: candidate["check"]["max_relative_error"],
    )
    recommended_thumb_steps = selected_mapping["steps"]
    thumb_mapping_check = selected_mapping["check"]
    thumb_mapping = selected_mapping["mapping"]

    table_point, ray_distance = pixel_to_table_point(
        center_2d,
        capture_pose,
        camera_matrix,
        distortion,
        camera_to_tcp,
        table_z,
    )
    reference_tcp_xy_offset_mm = np.asarray(tcp_xy_offset_mm, dtype=np.float64)
    rz_delta_for_offset = normalize_angle_deg(
        float(target_rpy[2]) - float(reference_rpy[2])
    )
    if args.rotate_tcp_xy_offset_with_rz:
        tcp_xy_offset_mm = rotate_xy(reference_tcp_xy_offset_mm, rz_delta_for_offset)
    tcp_xy_offset_mm = tcp_xy_offset_mm + tcp_xy_base_correction_mm
    orientation_adjustment.update(
        {
            "reference_tcp_xy_offset_mm": reference_tcp_xy_offset_mm.tolist(),
            "rz_delta_for_offset_rotation_deg": rz_delta_for_offset,
            "tcp_xy_base_correction_mm": tcp_xy_base_correction_mm.tolist(),
            "effective_tcp_xy_offset_mm": tcp_xy_offset_mm.tolist(),
        }
    )
    tcp_xy = table_point[:2] + tcp_xy_offset_mm
    target_xy = tcp_xy.tolist()

    from xarm.wrapper import XArmAPI

    arm = XArmAPI(args.robot_ip)
    try:
        if not arm.connected:
            raise RuntimeError(f"cannot connect to xArm {args.robot_ip}")
        current = read_pose(arm)
        check_arm_clean(arm, "before motion")

        horizontal = list(current)
        horizontal[0] = float(target_xy[0])
        horizontal[1] = float(target_xy[1])

        oriented = list(horizontal)
        oriented[3:] = [float(v) for v in target_rpy]

        fast_approach = list(oriented)
        fast_approach[2] = target_z + float(args.fast_approach_mm)
        final = list(oriented)
        final[2] = target_z

        rise = list(final)
        rise[2] = target_z + float(args.post_grasp_rise_mm)

        horizontal_distance = float(
            np.linalg.norm(np.asarray(current[:2]) - np.asarray(target_xy))
        )
        if horizontal_distance > args.max_horizontal_mm:
            raise RuntimeError(
                f"horizontal move {horizontal_distance:.3f} mm exceeds limit {args.max_horizontal_mm:.3f} mm"
            )

        thumb_base_steps = [
            int(value.strip())
            for value in args.thumb_base_steps.split(",")
            if value.strip()
        ]
        if not thumb_base_steps:
            if (
                not thumb_mapping_check["similar"]
                and not args.use_mapped_thumb_contact
            ):
                raise RuntimeError(
                    "new grasp triangle does not match any validated thumb mapping"
                )
            thumb_base_steps = recommended_thumb_steps
        if args.use_mapped_thumb_contact:
            if not thumb_mapping_check["similar"]:
                print(
                    "warning: grasp triangle differs from the validated 170 "
                    "anchor; continuing because TCP-to-fixed-fingertip alignment "
                    "is the primary constraint and thumb contact uses low-torque "
                    "stall detection"
                )
            thumb_base_steps = [245, int(args.thumb_contact_target)]

        report = {
            "capture_json": str(args.capture_json.resolve()),
            "capture_image": str(capture_image.resolve()),
            "capture_metadata": capture_payload,
            "server_dir": str(args.server_dir.resolve()),
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
            "image_check": image_check,
            "ray_distance_mm": float(ray_distance),
            "waypoints": [
                {
                    "name": "horizontal",
                    "pose": horizontal,
                    "speed": args.speed_horizontal,
                    "acc": args.acc,
                },
                {
                    "name": "orient",
                    "pose": oriented,
                    "speed": args.speed_orient,
                    "acc": args.acc,
                },
                {
                    "name": "fast_approach",
                    "pose": fast_approach,
                    "speed": args.speed_vertical,
                    "acc": args.acc,
                },
                {
                    "name": "final",
                    "pose": final,
                    "speed": args.speed_final,
                    "acc": args.acc_final,
                },
            ],
            "speed_profile": {
                "g20_initial": args.g20_initial_speed,
                "horizontal": args.speed_horizontal,
                "orient": args.speed_orient,
                "fast_descent": args.speed_vertical,
                "slow_final": args.speed_final,
                "rise": args.speed_rise,
                "thumb": args.thumb_speed,
                "fast_approach_mm": args.fast_approach_mm,
                "fast_acc": args.acc,
                "final_acc": args.acc_final,
            },
            "thumb_base_steps": thumb_base_steps,
            "thumb_contact_policy": {
                "mode": "low_torque_stall_detection",
                "validated_geometry_anchor_raw": int(
                    thumb_mapping["contact_can_raw"]
                ),
                "command_target_raw": int(args.thumb_contact_target),
                "contact_timeout_s": float(args.thumb_contact_timeout),
                "stall_window_s": float(args.thumb_stall_window),
                "stall_delta_raw": int(args.thumb_stall_delta),
                "contact_max_raw": int(args.thumb_contact_max_raw),
                "require_stalled_contact_before_rise": True,
            },
            "thumb_geometry": geometry,
            "thumb_mapping": thumb_mapping,
            "thumb_mapping_check": thumb_mapping_check,
            "thumb_geometry_policy": {
                "role": "advisory_only_in_low_torque_stall_mode",
                "tcp_fingertip_alignment_is_primary": True,
                "blocked_execution": False,
            },
            "thumb_mapping_candidates": [
                {
                    "name": candidate["name"],
                    "steps": candidate["steps"],
                    "check": candidate["check"],
                }
                for candidate in mapping_candidates
            ],
            "recommended_thumb_base_steps": recommended_thumb_steps,
            "post_grasp_rise_mm": float(args.post_grasp_rise_mm),
            "stop_before_final": bool(args.stop_before_final),
            "applied": bool(args.apply),
            "xarm_actual": [],
            "g20_actual": [],
        }

        print(f"capture={args.capture_json.resolve()}")
        print(f"capture_image={capture_image.resolve()}")
        print(f"server_raw={server_raw_image.resolve()}")
        print(f"image_check={image_check}")
        print(f"grasp_pair_center_2d={center_2d}")
        if args.auto_rz_from_direction_line:
            print(f"orientation_adjustment={orientation_adjustment}")
        print(f"table_point={np.round(table_point, 3).tolist()} mm")
        if args.rotate_tcp_xy_offset_with_rz:
            print(
                "rotated_tcp_xy_offset="
                f"{np.round(tcp_xy_offset_mm, 3).tolist()} mm "
                f"(reference={np.round(reference_tcp_xy_offset_mm, 3).tolist()} mm, "
                f"rz_delta={rz_delta_for_offset:.3f} deg)"
            )
        print(f"target_tcp_xy={np.round(target_xy, 3).tolist()} mm")
        print(f"target_tcp_z={target_z:.3f} mm")
        print(f"target_rpy={np.round(target_rpy, 6).tolist()} deg")
        print(f"thumb_geometry={geometry}")
        print(f"thumb_mapping_check={thumb_mapping_check}")
        print(f"selected_thumb_mapping={selected_mapping['name']}")
        print(f"recommended_thumb_base_steps={recommended_thumb_steps}")
        for waypoint in report["waypoints"]:
            name = waypoint["name"]
            pose = waypoint["pose"]
            speed = waypoint["speed"]
            acc = waypoint["acc"]
            print(
                f"{name}: {np.round(pose, 3).tolist()}, "
                f"speed={speed}, acc={acc}"
            )

        if not args.apply:
            print("Dry run only. Add --apply to move.")
        else:
            capture_translation_error = float(
                np.linalg.norm(
                    np.asarray(current[:3], dtype=np.float64)
                    - np.asarray(capture_pose[:3], dtype=np.float64)
                )
            )
            capture_angle_errors = [
                angular_difference_deg(actual, expected)
                for actual, expected in zip(current[3:], capture_pose[3:])
            ]
            if args.skip_capture_pose_preflight:
                report["capture_pose_preflight"] = {
                    "actual_tcp_pose": current,
                    "translation_error_mm": capture_translation_error,
                    "angle_errors_deg": capture_angle_errors,
                    "skipped": True,
                    "reason": "controlled_multi_target_cycle",
                }
                print(
                    "capture pose preflight skipped for controlled multi-target "
                    "cycle: "
                    f"translation_error={capture_translation_error:.3f} mm, "
                    f"angle_errors={np.round(capture_angle_errors, 3).tolist()} deg"
                )
            elif (
                capture_translation_error > args.max_capture_pose_translation_mm
                or (
                    not args.capture_position_only
                    and max(capture_angle_errors) > args.max_capture_pose_angle_deg
                )
            ):
                raise RuntimeError(
                    "current TCP no longer matches the synchronized capture pose: "
                    f"translation_error={capture_translation_error:.3f} mm, "
                    f"angle_errors={np.round(capture_angle_errors, 3).tolist()} deg"
                )
            else:
                report["capture_pose_preflight"] = {
                    "actual_tcp_pose": current,
                    "translation_error_mm": capture_translation_error,
                    "angle_errors_deg": capture_angle_errors,
                    "position_only": bool(args.capture_position_only),
                    "skipped": False,
                }
                print(
                    "capture pose preflight passed: "
                    f"translation_error={capture_translation_error:.3f} mm, "
                    f"angle_errors={np.round(capture_angle_errors, 3).tolist()} deg"
                )

            if args.initialize_g20 or args.check_g20_initial:
                bus = open_can_bus(args.channel)
                try:
                    if args.initialize_g20:
                        initialized = initialize_g20(
                            bus,
                            args.can_id,
                            preset["g20_initial_pose"],
                            args.g20_initial_speed,
                            args.g20_initial_torque,
                            args.g20_feedback_timeout,
                            args.g20_poll_interval,
                        )
                        report["g20_actual"].extend(initialized)
                        print(
                            "G20 initial pose ready, "
                            f"speed={args.g20_initial_speed}"
                        )
                    if args.check_g20_initial:
                        initial_g20 = verify_g20_initial(
                            bus, args.can_id, preset["g20_initial_pose"]
                        )
                        report["g20_actual"].append(
                            {"name": "initial_check_before_motion", "pose": initial_g20}
                        )
                finally:
                    bus.shutdown()

            if arm.error_code != 0 or arm.warn_code != 0:
                raise RuntimeError(
                    f"xArm not clean: state={arm.state}, error={arm.error_code}, warn={arm.warn_code}"
                )
            arm.motion_enable(enable=True)
            arm.set_mode(0)
            arm.set_state(0)
            time.sleep(max(0.0, float(args.xarm_enable_settle)))
            check_arm_clean(arm, "after enable")

            motion_waypoints = report["waypoints"]
            if args.stop_before_final:
                motion_waypoints = [
                    waypoint
                    for waypoint in motion_waypoints
                    if waypoint["name"] != "final"
                ]
                print(
                    "verification stop enabled: final descent, thumb grasp, "
                    "and post-grasp rise will be skipped"
                )
            current_target_angle_errors = [
                angular_difference_deg(actual, expected)
                for actual, expected in zip(current[3:], target_rpy)
            ]
            if max(current_target_angle_errors) <= args.max_capture_pose_angle_deg:
                motion_waypoints = [
                    waypoint
                    for waypoint in motion_waypoints
                    if waypoint["name"] != "orient"
                ]
                print("orient skipped: current RPY already matches grasp RPY")
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
                speed = waypoint["speed"]
                acc = waypoint["acc"]
                code = arm.set_position(
                    *pose,
                    speed=speed,
                    mvacc=acc,
                    wait=True,
                    radius=-1,
                )
                if code != 0:
                    raise RuntimeError(f"{name}: set_position failed with code {code}")
                actual = read_pose(arm)
                check_arm_clean(arm, name)
                if name in ("horizontal", "orient") and actual[2] < current[2] - 1.0:
                    raise RuntimeError(
                        f"{name}: unexpected Z drop to {actual[2]:.3f} mm from {current[2]:.3f} mm"
                    )
                if name in ("fast_approach", "final") and actual[2] > previous_z + 2.0:
                    raise RuntimeError(
                        f"{name}: unexpected upward Z jump to {actual[2]:.3f} mm"
                    )
                previous_z = actual[2]
                report["xarm_actual"].append({"name": name, "pose": actual, "code": code})
                print(f"{name} actual={np.round(actual, 3).tolist()}")

            if args.stop_before_final:
                print("stopped before final descent; no thumb grasp executed")
            elif args.check_g20_initial:
                bus = open_can_bus(args.channel)
                try:
                    current_g20 = verify_g20_initial(
                        bus, args.can_id, preset["g20_initial_pose"]
                    )
                    report["g20_actual"].append(
                        {"name": "initial_check_after_motion", "pose": current_g20}
                    )
                    for index, step in enumerate(thumb_base_steps):
                        target = {finger: list(values) for finger, values in current_g20.items()}
                        target["thumb"][g20.SLOTS["base"]] = int(step)
                        is_contact_step = (
                            args.use_mapped_thumb_contact
                            and index == len(thumb_base_steps) - 1
                        )
                        if is_contact_step:
                            current_g20, contact_result = close_thumb_until_contact(
                                bus,
                                args.can_id,
                                current_g20,
                                int(step),
                                args.thumb_speed,
                                args.thumb_torque,
                                args.thumb_contact_timeout,
                                args.g20_poll_interval,
                                args.thumb_stall_window,
                                args.thumb_stall_delta,
                                args.thumb_contact_max_raw,
                            )
                            report["thumb_contact_result"] = contact_result
                            if not contact_result["contact_detected"]:
                                raise RuntimeError(
                                    "thumb reached the low target without a "
                                    "stalled-contact signal; post-grasp rise blocked"
                                )
                            print(
                                "thumb stalled contact detected; "
                                "continuing directly to automatic TCP rise"
                            )
                        else:
                            current_g20 = send_pose(
                                bus,
                                args.can_id,
                                target,
                                args.thumb_speed,
                                args.thumb_torque,
                                args.g20_feedback_timeout,
                                args.g20_poll_interval,
                            )
                        report["g20_actual"].append(
                            {"name": f"thumb_base_{step}", "pose": current_g20}
                        )
                        print(f"thumb.base -> {step}, readback={current_g20['thumb']}")
                        if is_contact_step:
                            print(f"thumb contact={contact_result}")
                finally:
                    bus.shutdown()

            if (not args.stop_before_final) and args.post_grasp_rise_mm > 0.0:
                if args.use_mapped_thumb_contact:
                    print(
                        "automatic post-contact rise: "
                        f"{args.post_grasp_rise_mm:.1f} mm"
                    )
                rise_code = arm.set_position(
                    rise[0],
                    rise[1],
                    rise[2],
                    rise[3],
                    rise[4],
                    rise[5],
                    speed=args.speed_rise,
                    mvacc=args.acc,
                    wait=True,
                    radius=-1,
                )
                if rise_code != 0:
                    raise RuntimeError(f"post-grasp rise failed with code {rise_code}")
                rise_actual = read_pose(arm)
                check_arm_clean(arm, "post-grasp-rise")
                report["xarm_actual"].append(
                    {"name": "post_grasp_rise", "pose": rise_actual, "code": rise_code}
                )
                print(f"post_grasp_rise actual={np.round(rise_actual, 3).tolist()}")

        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"report={args.report.resolve()}")
        return 0
    finally:
        arm.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
