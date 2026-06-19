#!/usr/bin/env python3
"""Record the current xArm TCP pose as the first strawberry placement slot."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from xarm.wrapper import XArmAPI


WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    WORKSPACE / "robot_try" / "robot_try" / "config" / "table_strawberry_multi_grasp_left.json"
)
ROBOT_IP = "192.168.1.200"


def read_pose(robot_ip: str) -> list[float]:
    arm = XArmAPI(robot_ip)
    try:
        if not arm.connected:
            raise RuntimeError(f"cannot connect to xArm {robot_ip}")
        code, pose = arm.get_position(is_radian=False)
        if code != 0:
            raise RuntimeError(f"get_position failed: {code}")
        if arm.state == 4 or arm.error_code != 0 or arm.warn_code != 0:
            raise RuntimeError(
                "xArm is not ready: "
                f"state={arm.state}, error={arm.error_code}, warn={arm.warn_code}"
            )
        return [float(value) for value in pose]
    finally:
        arm.disconnect()


def update_config(
    config_path: Path,
    pose: list[float],
    slots_per_row: int,
    slot_y_spacing_mm: float,
    row_x_spacing_mm: float,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    placement = dict(config.get("placement", {}))
    placement.update(
        {
            "mode": "manual_first_slot_y_rows_then_x_rows",
            "source": (
                "user-confirmed first strawberry box placement TCP on "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}; "
                "direct TCP pose reference including manually adjusted Rz"
            ),
            "reference_tcp_pose_base_mm_deg": pose,
            "layout": "row_major",
            "slots_per_row": int(slots_per_row),
            "slot_y_spacing_mm": float(slot_y_spacing_mm),
            "row_x_spacing_mm": float(row_x_spacing_mm),
            "release_z_source": "grasp_tcp_z_plus_release_clearance",
            "release_clearance_above_grasp_mm": float(
                placement.get("release_clearance_above_grasp_mm", 10.0)
            ),
            "release_pose_rule": (
                "direct TCP slot pose: slot_index i -> row=floor(i/slots_per_row), "
                "col=i%slots_per_row; X=reference_X+row_x_spacing*row, "
                "Y=reference_Y+slot_y_spacing*col, RPY=manual reference RPY, "
                "release Z=grasp_tcp_z+release_clearance"
            ),
            "tcp_to_fingertip_mapping": (
                "not_used_for_placement; the manually recorded TCP pose already "
                "places the fingertips over the first box slot"
            ),
            "safe_z_source": placement.get(
                "safe_z_source", "grasp_tcp_z_plus_post_grasp_rise"
            ),
        }
    )
    config["placement"] = placement
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--robot-ip", default=ROBOT_IP)
    parser.add_argument("--slots-per-row", type=int, default=5)
    parser.add_argument("--slot-y-spacing-mm", type=float, default=-40.0)
    parser.add_argument("--row-x-spacing-mm", type=float, default=-50.0)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the updated placement reference into the config file",
    )
    args = parser.parse_args()

    pose = read_pose(args.robot_ip)
    config = update_config(
        args.config,
        pose,
        args.slots_per_row,
        args.slot_y_spacing_mm,
        args.row_x_spacing_mm,
    )

    placement = config["placement"]
    print(f"current_tcp_pose_base_mm_deg={pose}")
    print(
        "placement_grid="
        f"slots_per_row={placement['slots_per_row']}, "
        f"slot_y_spacing_mm={placement['slot_y_spacing_mm']}, "
        f"row_x_spacing_mm={placement['row_x_spacing_mm']}"
    )
    if args.apply:
        args.config.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"updated_config={args.config}")
    else:
        print("dry_run=true; add --apply to write the config")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
