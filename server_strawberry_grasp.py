#!/usr/bin/env python3
"""HTTP API for server-side multi-strawberry grasp generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_JOBS_DIR = PROJECT_ROOT / "strawberry_server_jobs"
DEFAULT_CONFIG = PROJECT_ROOT / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DEFAULT_GROUNDED_CHECKPOINT = PROJECT_ROOT / "groundingdino_swint_ogc.pth"
DEFAULT_SAM_CHECKPOINT = PROJECT_ROOT / "sam_vit_h_4b8939.pth"
DEFAULT_DEVICE = "cuda"
EXPECTED_WIDTH = 640
EXPECTED_HEIGHT = 480


class PipelineError(RuntimeError):
    """Expected request or inference failure returned as JSON."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve strawberry multi-grasp API.")
    parser.add_argument("--jobs-dir", default=str(DEFAULT_JOBS_DIR))
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--text-prompt", default="strawberry")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--grounded-checkpoint", default=str(DEFAULT_GROUNDED_CHECKPOINT))
    parser.add_argument("--sam-checkpoint", default=str(DEFAULT_SAM_CHECKPOINT))
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=True)
        file.write("\n")


def make_job_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def validate_image_bytes(image_bytes: bytes) -> Dict[str, Any]:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise PipelineError("uploaded image is not a readable image")
    height, width = image.shape[:2]
    if (width, height) != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
        raise PipelineError(
            f"image must be {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}, got {width}x{height}"
        )
    return {"width": width, "height": height}


def run_command(command: List[str], timeout_s: int) -> Dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gsa-server")
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise PipelineError(f"model inference timed out after {timeout_s}s", 504) from exc

    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        tail = (completed.stdout or "").strip().splitlines()[-30:]
        raise PipelineError(
            "command failed: "
            + " ".join(command)
            + "\n"
            + "\n".join(tail),
            500,
        )
    return {"elapsed_s": elapsed, "output": completed.stdout}


def copy_original_raw_image(raw_image_path: Path, output_dir: Path) -> None:
    shutil.copy2(raw_image_path, output_dir / "raw_image.jpg")
    for target_dir in output_dir.glob("target_*"):
        if target_dir.is_dir():
            shutil.copy2(raw_image_path, target_dir / "raw_image.jpg")


def create_result_zip(job_dir: Path, output_dir: Path) -> Path:
    zip_path = job_dir / "result.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        root_name = output_dir.name
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, Path(root_name) / path.relative_to(output_dir))
    return zip_path


def compact_targets(all_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for order, target in enumerate(all_data.get("targets", [])):
        score = None
        target_dir = target.get("target_dir")
        if target_dir:
            mask_json = Path(target_dir) / "mask.json"
            if mask_json.exists():
                for item in read_json(mask_json):
                    if item.get("value") == 1:
                        score = item.get("logit")
                        break
        item: Dict[str, Any] = {
            "id": target.get("target", f"target_{order:02d}"),
            "order": order,
            "status": target.get("status"),
            "source_mask_index": target.get("source_mask_index"),
            "sort_center": target.get("sort_center"),
            "mask_bbox_xywh": target.get("mask_bbox_xywh"),
            "grasp_points_2d": target.get("grasp_points_2d"),
            "grasp_pair_center_2d": target.get("grasp_pair_center_2d"),
            "direction_line_2d": target.get("direction_line_2d"),
            "score": score,
        }
        compact.append(item)
    return compact


def build_app(args: Optional[argparse.Namespace] = None) -> FastAPI:
    settings = args or parse_args()
    jobs_dir = Path(settings.jobs_dir).resolve()
    jobs_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Strawberry Multi-Grasp Server", version="1.0")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "project_root": str(PROJECT_ROOT),
            "jobs_dir": str(jobs_dir),
            "device": settings.device,
        }

    @app.post("/api/strawberry/multi_grasp")
    async def multi_grasp(
        image: UploadFile = File(...),
        metadata: UploadFile = File(...),
        return_format: str = Form("json"),
    ):
        job_id = make_job_id()
        job_dir = jobs_dir / job_id
        input_dir = job_dir / "input"
        raw_output_dir = job_dir / "output_raw"
        output_dir = job_dir / "output" / f"outputs_multi_auto_{job_id}"
        input_dir.mkdir(parents=True, exist_ok=False)

        started = time.monotonic()
        try:
            image_bytes = await image.read()
            metadata_bytes = await metadata.read()
            image_info = validate_image_bytes(image_bytes)

            raw_image_path = input_dir / "raw_image.jpg"
            metadata_path = input_dir / "metadata.json"
            raw_image_path.write_bytes(image_bytes)
            metadata_path.write_bytes(metadata_bytes)

            try:
                metadata_json = json.loads(metadata_bytes.decode("utf-8"))
            except Exception as exc:
                raise PipelineError("metadata is not valid UTF-8 JSON") from exc

            raw_sha256 = sha256_bytes(image_bytes)
            write_json(
                job_dir / "manifest.json",
                {
                    "job_id": job_id,
                    "status": "running",
                    "raw_image_sha256": raw_sha256,
                    "metadata": metadata_json,
                    **image_info,
                },
            )

            ground_cmd = [
                sys.executable,
                "grounded_sam_demo.py",
                "--config",
                str(settings.config),
                "--grounded_checkpoint",
                str(settings.grounded_checkpoint),
                "--sam_checkpoint",
                str(settings.sam_checkpoint),
                "--input_image",
                str(raw_image_path),
                "--output_dir",
                str(raw_output_dir),
                "--box_threshold",
                str(settings.box_threshold),
                "--text_threshold",
                str(settings.text_threshold),
                "--text_prompt",
                str(settings.text_prompt),
                "--device",
                str(settings.device),
            ]
            ground_result = run_command(ground_cmd, timeout_s=180)

            masks = sorted(raw_output_dir.glob("mask_*.png"))
            if not masks:
                raise PipelineError("no strawberry detected", 422)

            grasp_cmd = [
                sys.executable,
                "tools/generate_multi_strawberry_grasps.py",
                "--sam-output-dir",
                str(raw_output_dir),
                "--output-dir",
                str(output_dir),
                "--layout",
                "triangle-paired",
                "--thumb-side",
                "top",
                "--overwrite",
            ]
            grasp_result = run_command(grasp_cmd, timeout_s=60)

            copy_original_raw_image(raw_image_path, raw_output_dir)
            copy_original_raw_image(raw_image_path, output_dir)
            all_data = read_json(output_dir / "all_grasp_points.json")
            targets = compact_targets(all_data)
            if not targets:
                raise PipelineError("mask count is zero after grasp generation", 422)
            if all(target.get("status") != "ok" for target in targets):
                raise PipelineError("grasp point generation failed for all targets", 500)

            zip_path = create_result_zip(job_dir, output_dir)
            total_time_s = time.monotonic() - started
            response = {
                "job_id": job_id,
                "status": "done",
                "raw_image_sha256": raw_sha256,
                "width": image_info["width"],
                "height": image_info["height"],
                "target_count": len(targets),
                "failed_targets": all_data.get("failed_targets", 0),
                "targets": targets,
                "server_output_dir": str(output_dir),
                "all_grasp_points_json": str(output_dir / "all_grasp_points.json"),
                "result_zip_url": f"/api/strawberry/jobs/{job_id}/result.zip",
                "timing": {
                    "grounded_sam_s": ground_result["elapsed_s"],
                    "grasp_generation_s": grasp_result["elapsed_s"],
                    "total_s": total_time_s,
                },
            }
            write_json(
                job_dir / "manifest.json",
                {
                    **response,
                    "status": "done",
                    "input_image": str(raw_image_path),
                    "input_metadata": str(metadata_path),
                    "raw_output_dir": str(raw_output_dir),
                    "result_zip": str(zip_path),
                },
            )

            if return_format.lower() == "zip":
                return FileResponse(
                    zip_path,
                    media_type="application/zip",
                    filename=f"{output_dir.name}.zip",
                )
            return response
        except PipelineError as exc:
            error = {"job_id": job_id, "status": "failed", "error": str(exc)}
            write_json(job_dir / "manifest.json", error)
            return JSONResponse(status_code=exc.status_code, content=error)
        except Exception as exc:
            error = {"job_id": job_id, "status": "failed", "error": str(exc)}
            write_json(job_dir / "manifest.json", error)
            return JSONResponse(status_code=500, content=error)

    @app.get("/api/strawberry/jobs/{job_id}")
    def job_status(job_id: str):
        manifest = jobs_dir / job_id / "manifest.json"
        if not manifest.exists():
            return JSONResponse(
                status_code=404,
                content={"job_id": job_id, "status": "failed", "error": "job not found"},
            )
        return read_json(manifest)

    @app.get("/api/strawberry/jobs/{job_id}/result.zip")
    def job_result_zip(job_id: str):
        zip_path = jobs_dir / job_id / "result.zip"
        if not zip_path.exists():
            return JSONResponse(
                status_code=404,
                content={"job_id": job_id, "status": "failed", "error": "result.zip not found"},
            )
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=f"{job_id}_result.zip",
        )

    return app


app = build_app(argparse.Namespace(
    jobs_dir=str(DEFAULT_JOBS_DIR),
    device=DEFAULT_DEVICE,
    box_threshold=0.3,
    text_threshold=0.25,
    text_prompt="strawberry",
    config=str(DEFAULT_CONFIG),
    grounded_checkpoint=str(DEFAULT_GROUNDED_CHECKPOINT),
    sam_checkpoint=str(DEFAULT_SAM_CHECKPOINT),
))


if __name__ == "__main__":
    import uvicorn

    cli_args = parse_args()
    uvicorn.run(build_app(cli_args), host="0.0.0.0", port=8000)
