#!/usr/bin/env python3
"""Use existing eye-in-hand results for xArm 7 table localization.

This tool does not solve or overwrite camera/hand-eye calibration. It never
enables or moves the robot. It records fingertip contact poses, estimates the
table plane, and converts image pixels to xArm base coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_CALIB_DIR = (
    PROJECT_ROOT.parent.parent
    / "calib_20260611_153756"
    / "calib_20260611_153756"
)
DEFAULT_PLANE = PROJECT_ROOT / "config" / "table_plane.json"
DEFAULT_CONTACTS = PROJECT_ROOT / "config" / "table_contacts.json"


def make_homogeneous(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def pose_to_homogeneous(
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    is_radian: bool = False,
) -> np.ndarray:
    if not is_radian:
        roll, pitch, yaw = np.radians([roll, pitch, yaw])
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    return make_homogeneous(rotation, np.array([x, y, z]))


def parse_csv_floats(text: str, count: int, label: str) -> list[float]:
    try:
        values = [float(value.strip()) for value in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} 必须是逗号分隔的数字") from exc
    if len(values) != count:
        raise argparse.ArgumentTypeError(f"{label} 需要 {count} 个数值")
    return values


def parse_pose(text: str) -> list[float]:
    return parse_csv_floats(text, 6, "pose")


def parse_xyz(text: str) -> list[float]:
    return parse_csv_floats(text, 3, "XYZ")


def parse_pixel(text: str) -> list[float]:
    return parse_csv_floats(text, 2, "pixel")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_calibration(
    calib_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    intrinsics_path = calib_dir / "camera_intrinsics.npz"
    hand_eye_path = calib_dir / "hand_eye_result.npz"
    if not intrinsics_path.exists() or not hand_eye_path.exists():
        raise RuntimeError(f"标定目录缺少 npz 文件: {calib_dir}")

    intrinsics = np.load(intrinsics_path)
    hand_eye = np.load(hand_eye_path)
    camera_matrix = np.asarray(intrinsics["mtx"], dtype=np.float64)
    distortion = np.asarray(intrinsics["dist"], dtype=np.float64)
    camera_to_tcp = make_homogeneous(
        np.asarray(hand_eye["R_cam2gripper"], dtype=np.float64),
        np.asarray(hand_eye["t_cam2gripper"], dtype=np.float64),
    )
    metadata = {
        "calib_dir": str(calib_dir),
        "camera_intrinsics_sha256": file_sha256(intrinsics_path),
        "hand_eye_result_sha256": file_sha256(hand_eye_path),
    }
    return camera_matrix, distortion, camera_to_tcp, metadata


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 3:
        raise RuntimeError("拟合平面至少需要 3 个点")
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
    offset = -float(normal @ center)
    residuals = points @ normal + offset
    return normal, offset, residuals


def save_plane(
    path: Path,
    normal: np.ndarray,
    offset: float,
    residuals: np.ndarray,
    source: str,
    point_count: int,
) -> None:
    payload = {
        "format": 1,
        "frame": "xarm_base",
        "units": "mm",
        "equation": "normal dot point + offset = 0",
        "normal": [float(value) for value in normal],
        "offset": float(offset),
        "source": source,
        "point_count": int(point_count),
        "rms_residual_mm": float(np.sqrt(np.mean(np.square(residuals)))),
        "max_residual_mm": float(np.max(np.abs(residuals))),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_plane(path: Path) -> tuple[np.ndarray, float, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    normal = np.asarray(payload["normal"], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    return normal, float(payload["offset"]), payload


def read_robot_pose(robot_ip: str) -> list[float]:
    try:
        from xarm.wrapper import XArmAPI
    except ImportError as exc:
        raise RuntimeError("未安装 xarm-python-sdk，不能读取机械臂位姿") from exc

    arm = XArmAPI(robot_ip)
    try:
        if not arm.connected:
            raise RuntimeError(f"无法连接机械臂 {robot_ip}")
        code, pose = arm.get_position(is_radian=False)
        if code != 0:
            raise RuntimeError(f"get_position 返回错误码 {code}")
        return [float(value) for value in pose]
    finally:
        arm.disconnect()


def resolve_pose(args: argparse.Namespace) -> list[float]:
    if args.pose is not None:
        return args.pose
    return read_robot_pose(args.robot_ip)


def transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    point_h = np.append(np.asarray(point, dtype=np.float64), 1.0)
    return (transform @ point_h)[:3]


def record_contact(args: argparse.Namespace) -> int:
    pose = resolve_pose(args)
    tcp_to_base = pose_to_homogeneous(*pose, is_radian=False)
    contact_base = transform_point(tcp_to_base, np.asarray(args.tip_in_tcp))

    contacts_path = args.contacts
    payload = {"format": 1, "frame": "xarm_base", "units": "mm", "contacts": []}
    if contacts_path.exists() and not args.reset:
        payload = json.loads(contacts_path.read_text(encoding="utf-8"))
    payload.setdefault("contacts", []).append(
        {
            "point": [float(value) for value in contact_base],
            "tcp_pose": pose,
            "tip_in_tcp": args.tip_in_tcp,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    contacts_path.parent.mkdir(parents=True, exist_ok=True)
    contacts_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    points = np.asarray([entry["point"] for entry in payload["contacts"]])
    print(f"接触点 base XYZ: {contact_base.round(3).tolist()} mm")
    print(f"已记录接触点: {len(points)}")
    print(f"记录文件: {contacts_path.resolve()}")

    if len(points) >= 3:
        normal, offset, residuals = fit_plane(points)
        save_plane(
            args.output,
            normal,
            offset,
            residuals,
            source=f"fingertip_contacts:{contacts_path}",
            point_count=len(points),
        )
        print(f"已拟合桌面平面: {args.output.resolve()}")
    elif args.horizontal:
        normal = np.array([0.0, 0.0, 1.0])
        offset = -float(np.mean(points[:, 2]))
        residuals = points[:, 2] + offset
        save_plane(
            args.output,
            normal,
            offset,
            residuals,
            source=f"horizontal_fingertip_contacts:{contacts_path}",
            point_count=len(points),
        )
        print(f"按水平桌面生成平面: {args.output.resolve()}")
    else:
        print("再记录分散的接触点，累计 3 点后自动拟合桌面平面。")
    return 0


def record_touch_pose(args: argparse.Namespace) -> int:
    pose = resolve_pose(args)
    touches_path = args.touches
    payload = {
        "format": 1,
        "frame": "xarm_base",
        "units": "mm_and_degree",
        "tcp_definition": "must_match_hand_eye_calibration",
        "poses": [],
    }
    if touches_path.exists() and not args.reset:
        payload = json.loads(touches_path.read_text(encoding="utf-8"))
    payload.setdefault("poses", []).append(
        {
            "tcp_pose": pose,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    touches_path.parent.mkdir(parents=True, exist_ok=True)
    touches_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已记录接触 TCP 位姿: {[round(value, 3) for value in pose]}")
    print(f"当前数量: {len(payload['poses'])}，建议至少 6 个不同姿态")
    print(f"记录文件: {touches_path.resolve()}")
    return 0


def solve_horizontal_touches(args: argparse.Namespace) -> int:
    payload = json.loads(args.touches.read_text(encoding="utf-8"))
    poses = [entry["tcp_pose"] for entry in payload.get("poses", [])]
    if len(poses) < 4:
        raise RuntimeError("联合求指尖偏移和桌面高度至少需要 4 个接触姿态")

    rows = []
    rhs = []
    for pose in poses:
        tcp_to_base = pose_to_homogeneous(*pose, is_radian=False)
        rows.append([*tcp_to_base[2, :3], -1.0])
        rhs.append(-tcp_to_base[2, 3])
    matrix = np.asarray(rows, dtype=np.float64)
    vector = np.asarray(rhs, dtype=np.float64)
    solution, _, rank, singular_values = np.linalg.lstsq(matrix, vector, rcond=None)
    if rank < 4:
        raise RuntimeError("接触姿态变化不足，无法区分指尖偏移和桌面高度")

    tip_in_tcp = solution[:3]
    table_z = float(solution[3])
    residuals = matrix @ solution - vector
    condition = float(singular_values[0] / singular_values[-1])
    if condition > 1000:
        raise RuntimeError(
            f"接触姿态条件数过大 ({condition:.1f})，请增加末端姿态变化"
        )

    normal = np.array([0.0, 0.0, 1.0])
    save_plane(
        args.output,
        normal,
        -table_z,
        residuals,
        source=f"horizontal_touch_poses:{args.touches}",
        point_count=len(poses),
    )
    result = {
        "format": 1,
        "tip_in_calibrated_tcp_mm": [float(value) for value in tip_in_tcp],
        "table_z_base_mm": table_z,
        "rms_residual_mm": float(np.sqrt(np.mean(residuals**2))),
        "condition_number": condition,
        "touch_count": len(poses),
        "tcp_definition": "must_match_hand_eye_calibration",
    }
    args.tip_output.parent.mkdir(parents=True, exist_ok=True)
    args.tip_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"指尖相对标定 TCP: {tip_in_tcp.round(3).tolist()} mm")
    print(f"桌面 base Z: {table_z:.3f} mm")
    print(f"拟合 RMS: {result['rms_residual_mm']:.3f} mm")
    print(f"条件数: {condition:.1f}")
    print(f"桌面平面: {args.output.resolve()}")
    print(f"指尖结果: {args.tip_output.resolve()}")
    return 0


def camera_ray_in_base(
    pixel: list[float],
    pose: list[float],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    camera_to_tcp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    distorted_x = (float(pixel[0]) - cx) / fx
    distorted_y = (float(pixel[1]) - cy) / fy

    coefficients = np.zeros(5, dtype=np.float64)
    flat_distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)
    coefficients[: min(5, len(flat_distortion))] = flat_distortion[:5]
    k1, k2, p1, p2, k3 = coefficients

    x, y = distorted_x, distorted_y
    for _ in range(10):
        radius2 = x * x + y * y
        radial = 1.0 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
        delta_x = 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x * x)
        delta_y = p1 * (radius2 + 2.0 * y * y) + 2.0 * p2 * x * y
        if abs(radial) < 1e-12:
            raise RuntimeError("相机畸变反算失败")
        x = (distorted_x - delta_x) / radial
        y = (distorted_y - delta_y) / radial

    ray_camera = np.array([x, y, 1.0])
    ray_camera /= np.linalg.norm(ray_camera)

    tcp_to_base = pose_to_homogeneous(*pose, is_radian=False)
    camera_to_base = tcp_to_base @ camera_to_tcp
    origin = camera_to_base[:3, 3]
    direction = camera_to_base[:3, :3] @ ray_camera
    direction /= np.linalg.norm(direction)
    return origin, direction


def intersect_ray_plane(
    origin: np.ndarray,
    direction: np.ndarray,
    normal: np.ndarray,
    offset: float,
) -> tuple[np.ndarray, float]:
    denominator = float(normal @ direction)
    if abs(denominator) < 1e-8:
        raise RuntimeError("像素射线与目标平面近似平行")
    distance = -float(normal @ origin + offset) / denominator
    if distance <= 0:
        raise RuntimeError("目标平面位于相机射线后方，请检查位姿和标定方向")
    return origin + distance * direction, distance


def locate_pixel(args: argparse.Namespace) -> int:
    pose = resolve_pose(args)
    camera_matrix, distortion, camera_to_tcp, calibration_metadata = load_calibration(
        args.calib_dir.resolve()
    )
    normal, offset, plane_payload = load_plane(args.plane)
    elevated_offset = offset - float(args.height_mm)
    origin, direction = camera_ray_in_base(
        args.pixel, pose, camera_matrix, distortion, camera_to_tcp
    )
    point, ray_distance = intersect_ray_plane(
        origin, direction, normal, elevated_offset
    )

    result = {
        "pixel": args.pixel,
        "height_above_table_mm": float(args.height_mm),
        "point_base_mm": [float(value) for value in point],
        "camera_origin_base_mm": [float(value) for value in origin],
        "ray_distance_mm": float(ray_distance),
        "tcp_pose": pose,
        "plane": plane_payload,
        "calibration": calibration_metadata,
    }
    print(f"像素: {args.pixel}")
    print(f"目标高度: 桌面上方 {args.height_mm:.2f} mm")
    print(f"目标 base XYZ: {point.round(3).tolist()} mm")
    print(f"相机到目标的射线距离: {ray_distance:.2f} mm")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"结果: {args.output.resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    contact = subparsers.add_parser(
        "record-contact", help="已知指尖相对标定 TCP 时记录接触点"
    )
    contact.add_argument("--pose", type=parse_pose)
    contact.add_argument("--robot-ip", default="192.168.1.200")
    contact.add_argument(
        "--tip-in-tcp",
        type=parse_xyz,
        required=True,
        help="接触指尖在当前 xArm TCP 坐标系中的 XYZ(mm)",
    )
    contact.add_argument("--contacts", type=Path, default=DEFAULT_CONTACTS)
    contact.add_argument("--output", type=Path, default=DEFAULT_PLANE)
    contact.add_argument("--horizontal", action="store_true")
    contact.add_argument("--reset", action="store_true")
    contact.set_defaults(handler=record_contact)

    touch = subparsers.add_parser(
        "record-touch", help="未知指尖偏移时记录一个水平桌面接触姿态"
    )
    touch.add_argument("--pose", type=parse_pose)
    touch.add_argument("--robot-ip", default="192.168.1.200")
    touch.add_argument(
        "--touches",
        type=Path,
        default=PROJECT_ROOT / "config" / "horizontal_touch_poses.json",
    )
    touch.add_argument("--reset", action="store_true")
    touch.set_defaults(handler=record_touch_pose)

    solve_touch = subparsers.add_parser(
        "solve-touches", help="由多个水平桌面接触姿态求指尖偏移和桌面高度"
    )
    solve_touch.add_argument(
        "--touches",
        type=Path,
        default=PROJECT_ROOT / "config" / "horizontal_touch_poses.json",
    )
    solve_touch.add_argument("--output", type=Path, default=DEFAULT_PLANE)
    solve_touch.add_argument(
        "--tip-output",
        type=Path,
        default=PROJECT_ROOT / "config" / "contact_fingertip.json",
    )
    solve_touch.set_defaults(handler=solve_horizontal_touches)

    locate = subparsers.add_parser(
        "locate", help="将目标像素投影到桌面或桌面上方的平面"
    )
    locate.add_argument("--pixel", type=parse_pixel, required=True)
    locate.add_argument("--pose", type=parse_pose)
    locate.add_argument("--robot-ip", default="192.168.1.200")
    locate.add_argument("--height-mm", type=float, default=0.0)
    locate.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR)
    locate.add_argument("--plane", type=Path, default=DEFAULT_PLANE)
    locate.add_argument("--output", type=Path)
    locate.set_defaults(handler=locate_pixel)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.handler(args)
    except (RuntimeError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
