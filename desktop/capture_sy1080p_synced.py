#!/usr/bin/env python3
"""Capture one SY1080P frame and record the matching xArm pose."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
from xarm.wrapper import XArmAPI


WORKSPACE = Path(__file__).resolve().parents[3]
OUTPUT_DIR = WORKSPACE / "testdata"
CALIB_DIR = (
    WORKSPACE / "calib_20260611_153756" / "calib_20260611_153756"
)
ROBOT_IP = "192.168.1.200"
CAMERA_INDEX = 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        help="output base name without extension; defaults to a timestamp",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=CAMERA_INDEX,
        help="OpenCV camera index",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = args.name or f"sy1080p_{stamp}"
    image_path = OUTPUT_DIR / f"{base_name}.jpg"
    metadata_path = OUTPUT_DIR / f"{base_name}.json"

    arm = XArmAPI(ROBOT_IP)
    camera_index = args.camera_index
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    try:
        if not arm.connected:
            raise RuntimeError(f"cannot connect to xArm {ROBOT_IP}")
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera index {camera_index}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        frame = None
        for _ in range(20):
            ok, candidate = cap.read()
            if ok:
                frame = candidate
            time.sleep(0.03)
        if frame is None:
            raise RuntimeError("SY1080P did not return a frame")
        if frame.shape[1] != 640 or frame.shape[0] != 480:
            raise RuntimeError(
                f"unexpected resolution {frame.shape[1]}x{frame.shape[0]}"
            )

        pose_code, pose = arm.get_position(is_radian=False)
        angle_code, angles = arm.get_servo_angle(is_radian=False)
        if pose_code != 0 or angle_code != 0:
            raise RuntimeError(
                f"xArm read failed: pose={pose_code}, angles={angle_code}"
            )
        encoded_ok, encoded = cv2.imencode(".jpg", frame)
        if not encoded_ok:
            raise RuntimeError("failed to encode JPEG frame")
        image_path.write_bytes(encoded.tobytes())

        intrinsics = CALIB_DIR / "camera_intrinsics.npz"
        hand_eye = CALIB_DIR / "hand_eye_result.npz"
        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "image": str(image_path),
            "camera": {
                "name": "SY 1080P camera",
                "index": camera_index,
                "resolution": [640, 480],
            },
            "xarm": {
                "robot_ip": ROBOT_IP,
                "tcp_pose_base_mm_deg": [float(value) for value in pose],
                "servo_angles_deg": [float(value) for value in angles],
                "state": int(arm.state),
                "error_code": int(arm.error_code),
                "warn_code": int(arm.warn_code),
            },
            "calibration": {
                "directory": str(CALIB_DIR),
                "camera_intrinsics_sha256": sha256(intrinsics),
                "hand_eye_result_sha256": sha256(hand_eye),
            },
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"image={image_path}")
        print(f"metadata={metadata_path}")
        print(f"pose={metadata['xarm']['tcp_pose_base_mm_deg']}")
        print(
            f"state={arm.state} error={arm.error_code} warn={arm.warn_code}"
        )
        return 0
    finally:
        cap.release()
        arm.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
