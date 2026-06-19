#!/usr/bin/env python3
"""Quick-start wrapper for table-strawberry multi-target pick/place."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT_ROOT.parent.parent
DEFAULT_OUTPUT_ROOT = WORKSPACE / "testdata"
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "config"
MULTI_SCRIPT = Path(__file__).resolve().parent / "execute_table_strawberry_multi_left.py"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_run_id(server_dir: Path) -> str:
    name = server_dir.name
    if name.startswith("outputs_multi_"):
        return name.removeprefix("outputs_multi_")
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_manifest(capture_json: Path, server_dir: Path, manifest_path: Path) -> dict:
    summary_path = server_dir / "all_grasp_points.json"
    if not summary_path.exists():
        raise RuntimeError(f"missing multi-target summary: {summary_path}")
    summary = read_json(summary_path)
    targets = []
    for index, target in enumerate(summary.get("targets", [])):
        if target.get("status") != "ok":
            continue
        target_id = str(target["target"])
        target_dir = server_dir / target_id
        required = [
            target_dir / "raw_image.jpg",
            target_dir / "mask_0.png",
            target_dir / "mask.json",
            target_dir / "grasp_points_triangle.json",
            target_dir / "grasp_points_triangle_visualization.jpg",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(
                f"{target_id} is incomplete; missing: {', '.join(missing)}"
            )
        targets.append(
            {
                "id": target_id,
                "order": int(index),
                "server_dir": str(target_dir.resolve()),
            }
        )
    if not targets:
        raise RuntimeError(f"no usable targets in {summary_path}")

    manifest = {
        "format": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(summary_path.resolve()),
        "capture_json": str(capture_json.resolve()),
        "order_policy": summary.get(
            "sort_order",
            "server_order_from_all_grasp_points_json",
        ),
        "targets": targets,
    }
    write_json(manifest_path, manifest)
    return manifest


def run(command: list[str], label: str) -> None:
    print(f"\n[{label}]")
    print(" ".join(command))
    subprocess.run(command, check=True, cwd=str(WORKSPACE))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture",
        required=True,
        help="capture base name like test_13, or full path to a capture json",
    )
    parser.add_argument(
        "--server-dir",
        type=Path,
        required=True,
        help="outputs_multi_XX directory",
    )
    parser.add_argument(
        "--run-id",
        help="id used for generated manifest/report; defaults to outputs_multi suffix",
    )
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--skip-dry-run", action="store_true")
    parser.add_argument("--skip-initial-return", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    capture_arg = Path(args.capture)
    if capture_arg.suffix.lower() == ".json":
        capture_json = capture_arg
    else:
        capture_json = DEFAULT_OUTPUT_ROOT / f"{args.capture}.json"
    server_dir = args.server_dir.resolve()
    if not capture_json.exists():
        raise RuntimeError(f"missing capture json: {capture_json}")
    if not server_dir.exists():
        raise RuntimeError(f"missing server dir: {server_dir}")

    run_id = args.run_id or parse_run_id(server_dir)
    manifest_path = (
        args.manifest_dir
        / f"table_strawberry_multi_targets_{run_id}.json"
    )
    report_path = args.report_dir / f"last_table_strawberry_multi_left_{run_id}.json"

    manifest = build_manifest(capture_json, server_dir, manifest_path)
    print(f"manifest={manifest_path.resolve()}")
    print(f"report={report_path.resolve()}")
    print(f"targets={[target['id'] for target in manifest['targets']]}")

    base_command = [
        sys.executable,
        str(MULTI_SCRIPT),
        "--manifest",
        str(manifest_path),
        "--report",
        str(report_path),
    ]
    if args.skip_initial_return:
        base_command.append("--skip-initial-return")

    if not args.skip_dry_run:
        run(base_command, "dry-run")

    if args.apply:
        run([*base_command, "--apply"], "apply")
    else:
        print("\nDry run complete. Add --apply to execute hardware.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
