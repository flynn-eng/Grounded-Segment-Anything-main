#!/usr/bin/env python3
"""Local one-command pipeline for synchronized capture, server upload, and grasp."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT_ROOT.parent.parent
TESTDATA = WORKSPACE / "testdata"
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CONFIG = CONFIG_DIR / "table_strawberry_auto_pipeline_local.json"
CAPTURE_SCRIPT = Path(__file__).resolve().parent / "capture_sy1080p_synced.py"
MULTI_QUICK_SCRIPT = Path(__file__).resolve().parent / "run_table_strawberry_multi_quick.py"

PLACEHOLDER_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c6360000002000100ffff03000006000557bfab5d000000"
    "0049454e44ae426082"
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def next_run_name(prefix: str, run_root: Path, output_root: Path) -> str:
    day = datetime.now().strftime("%Y%m%d")
    pattern = re.compile(rf"^{re.escape(prefix)}_{day}_(\d+)$")
    used = []
    for root in (run_root, output_root):
        if not root.exists():
            continue
        for path in root.iterdir():
            match = pattern.match(path.name)
            if match:
                used.append(int(match.group(1)))
    return f"{prefix}_{day}_{(max(used) + 1) if used else 1}"


def output_dir_name(output_prefix: str, run_name: str, capture_prefix: str) -> str:
    stripped = run_name
    prefix_marker = f"{capture_prefix}_"
    if run_name.startswith(prefix_marker):
        stripped = run_name[len(prefix_marker):]
    return f"{output_prefix}_{stripped}"


def run_command(command: list[str], label: str) -> float:
    started = time.monotonic()
    print(f"\n[{label}]")
    print(" ".join(command))
    subprocess.run(command, check=True, cwd=str(WORKSPACE))
    return time.monotonic() - started


def normalize_server_url(server_url: str, endpoint: str) -> str:
    server_url = server_url.rstrip("/")
    endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    return f"{server_url}{endpoint}"


def multipart_post(url: str, fields: dict[str, str], files: dict[str, Path], timeout: float) -> tuple[bytes, str]:
    boundary = f"----codexstrawberry{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("ascii"))
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("ascii")
        )
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")
    for name, path in files.items():
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        chunks.append(f"--{boundary}\r\n".encode("ascii"))
        chunks.append(
            (
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{path.name}"\r\n'
            ).encode("utf-8")
        )
        chunks.append(f"Content-Type: {mime}\r\n\r\n".encode("ascii"))
        chunks.append(path.read_bytes())
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("ascii"))
    body = b"".join(chunks)
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(), response.headers.get("Content-Type", "")


def http_get(url: str, timeout: float) -> tuple[bytes, str]:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(), response.headers.get("Content-Type", "")


def extract_result_zip(body: bytes, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir.with_suffix(".zip")
    zip_path.write_bytes(body)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)

    summary_path = output_dir / "all_grasp_points.json"
    if summary_path.exists():
        return read_json(summary_path)

    candidates = list(output_dir.rglob("all_grasp_points.json"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"server zip missing all_grasp_points.json at root and found "
            f"{len(candidates)} nested candidates under {output_dir}"
        )
    nested_root = candidates[0].parent
    for child in nested_root.iterdir():
        destination = output_dir / child.name
        if destination.exists():
            continue
        shutil.move(str(child), str(destination))
    return read_json(output_dir / "all_grasp_points.json")


def ensure_target_payload(target: dict, target_id: str) -> dict:
    payload = dict(target)
    if "grasp_points_2d" not in payload:
        raise RuntimeError(f"{target_id}: missing grasp_points_2d in server JSON")
    if "grasp_pair_center_2d" not in payload:
        raise RuntimeError(f"{target_id}: missing grasp_pair_center_2d in server JSON")
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, dict) or "bbox_xywh" not in diagnostics:
        raise RuntimeError(
            f"{target_id}: server JSON must include diagnostics.bbox_xywh; "
            "the local grasp script uses it for thumb geometry scaling"
        )
    return payload


def materialize_json_response(response: dict, capture_image: Path, output_dir: Path) -> dict:
    targets = response.get("targets")
    if not isinstance(targets, list) or not targets:
        raise RuntimeError("server JSON response contains no targets")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(capture_image, output_dir / "raw_image.jpg")
    summary_targets = []
    for index, target in enumerate(targets):
        target_id = str(target.get("id") or target.get("target") or f"target_{index:02d}")
        if not target_id.startswith("target_"):
            target_id = f"target_{index:02d}"
        target_dir = output_dir / target_id
        target_dir.mkdir(parents=True, exist_ok=True)
        payload = ensure_target_payload(target, target_id)
        shutil.copy2(capture_image, target_dir / "raw_image.jpg")
        write_json(target_dir / "grasp_points_triangle.json", payload)
        write_json(
            target_dir / "mask.json",
            {
                "source": "server_json_response",
                "note": "mask image omitted by sync JSON server response; placeholder exists only for compatibility",
            },
        )
        (target_dir / "mask_0.png").write_bytes(PLACEHOLDER_PNG)
        shutil.copy2(capture_image, target_dir / "grasp_points_triangle_visualization.jpg")
        summary_targets.append(
            {
                "target": target_id,
                "status": "ok",
                "order": int(target.get("order", index)),
                "source": "sync_json_response",
            }
        )
    summary_targets.sort(key=lambda item: (int(item["order"]), str(item["target"])))
    summary = {
        "format": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sort_order": response.get("sort_order", "server_order"),
        "targets": summary_targets,
        "server_response": {
            key: value
            for key, value in response.items()
            if key not in {"targets"}
        },
    }
    write_json(output_dir / "all_grasp_points.json", summary)
    return summary


def materialize_server_response(
    body: bytes,
    content_type: str,
    capture_image: Path,
    output_dir: Path,
    server_url: str,
    timeout: float,
) -> dict:
    content_type_lower = content_type.lower()
    if "zip" in content_type_lower or body[:4] == b"PK\x03\x04":
        return extract_result_zip(body, output_dir)

    response = json.loads(body.decode("utf-8"))
    if response.get("status") not in (None, "done"):
        raise RuntimeError(f"server returned non-done status: {response}")
    if "result_zip_url" in response:
        result_zip_url = str(response["result_zip_url"])
        zip_url = urllib.parse.urljoin(server_url.rstrip("/") + "/", result_zip_url)
        zip_body, _zip_content_type = http_get(zip_url, timeout)
        return extract_result_zip(zip_body, output_dir)
    if "result_zip_base64" in response:
        raise RuntimeError(
            "result_zip_base64 is not implemented in the local pipeline yet; "
            "return application/zip or JSON targets"
        )
    return materialize_json_response(response, capture_image, output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--name", help="run/capture name; defaults to auto_YYYYMMDD_N")
    parser.add_argument("--server-url", help="server base URL, e.g. http://host:8765")
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--skip-dry-run", action="store_true")
    parser.add_argument(
        "--no-grasp",
        action="store_true",
        help="capture/upload/download/materialize results, then stop before robot grasp",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    config = read_json(args.config)
    run_root = WORKSPACE / config.get("run_root", "testdata/auto_pipeline/runs")
    output_root = TESTDATA
    run_name = args.name or next_run_name(
        config.get("capture_prefix", "auto"),
        run_root,
        output_root,
    )
    capture_prefix = config.get("capture_prefix", "auto")
    run_dir = run_root / run_name
    output_dir = output_root / output_dir_name(
        config.get("output_prefix", "outputs_multi_auto"),
        run_name,
        capture_prefix,
    )
    server_url = args.server_url or config["default_server_url"]
    endpoint = normalize_server_url(server_url, config["server_api"]["sync_endpoint"])
    camera_index = int(args.camera_index if args.camera_index is not None else config["camera_index"])
    timeout = float(args.timeout if args.timeout is not None else config["server_api"]["timeout_s"])
    capture_image = TESTDATA / f"{run_name}.jpg"
    capture_json = TESTDATA / f"{run_name}.json"
    report_path = CONFIG_DIR / f"last_table_strawberry_auto_pipeline_{run_name}.json"

    report = {
        "format": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scheme": config["scheme"],
        "run_name": run_name,
        "run_dir": str(run_dir.resolve()),
        "capture_image": str(capture_image.resolve()),
        "capture_json": str(capture_json.resolve()),
        "server_endpoint": endpoint,
        "output_dir": str(output_dir.resolve()),
        "applied": bool(args.apply),
        "no_grasp": bool(args.no_grasp),
        "timing": {},
        "steps": [],
    }
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"run_name={run_name}")
    print(f"server_endpoint={endpoint}")
    print(f"output_dir={output_dir}")
    print(f"report={report_path}")

    if not args.apply:
        report["steps"].append(
            {
                "name": "dry_run_only",
                "note": "Add --apply to capture, upload, and execute grasp.",
            }
        )
        write_json(report_path, report)
        print("Dry run only. No camera, server upload, or robot motion executed.")
        return 0

    report["timing"]["capture_s"] = run_command(
        [
            sys.executable,
            str(CAPTURE_SCRIPT),
            "--name",
            run_name,
            "--camera-index",
            str(camera_index),
        ],
        "capture_sy1080p_synced",
    )
    if not capture_image.exists() or not capture_json.exists():
        raise RuntimeError(f"capture did not create expected files for {run_name}")

    upload_started = time.monotonic()
    try:
        body, content_type = multipart_post(
            endpoint,
            fields={"run_name": run_name, "return_format": "json_or_zip"},
            files={"image": capture_image, "metadata": capture_json},
            timeout=timeout,
        )
    except urllib.error.URLError as exc:
        raise RuntimeError(f"server upload failed: {exc}") from exc
    report["timing"]["server_upload_and_process_s"] = time.monotonic() - upload_started
    report["server_content_type"] = content_type

    materialize_started = time.monotonic()
    summary = materialize_server_response(
        body,
        content_type,
        capture_image,
        output_dir,
        server_url,
        timeout,
    )
    report["timing"]["materialize_result_s"] = time.monotonic() - materialize_started
    report["target_count"] = len([t for t in summary.get("targets", []) if t.get("status") == "ok"])
    write_json(run_dir / "server_summary.json", summary)

    if args.no_grasp:
        report["steps"].append(
            {
                "name": "stopped_before_grasp",
                "reason": "--no-grasp",
                "output_dir": str(output_dir.resolve()),
            }
        )
        report["timing"]["total_s"] = sum(report["timing"].values())
        write_json(report_path, report)
        print(f"stopped_before_grasp output_dir={output_dir.resolve()}")
        print(f"auto_pipeline_report={report_path.resolve()}")
        return 0

    quick_command = [
        sys.executable,
        str(MULTI_QUICK_SCRIPT),
        "--capture",
        run_name,
        "--server-dir",
        str(output_dir),
        "--run-id",
        run_name,
    ]
    if args.skip_dry_run:
        quick_command.append("--skip-dry-run")
    if args.apply:
        quick_command.append("--apply")
    report["timing"]["grasp_total_s"] = run_command(quick_command, "multi_grasp")
    report["multi_report"] = str(
        (CONFIG_DIR / f"last_table_strawberry_multi_left_{run_name}.json").resolve()
    )
    report["timing"]["total_s"] = sum(report["timing"].values())
    write_json(report_path, report)
    print(f"auto_pipeline_report={report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
