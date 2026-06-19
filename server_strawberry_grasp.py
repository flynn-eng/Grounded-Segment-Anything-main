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
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gsa-server")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "GroundingDINO"))
sys.path.insert(0, str(PROJECT_ROOT / "segment_anything"))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from grounded_sam_demo import (
    SamPredictor,
    get_grounding_output,
    load_image,
    load_model,
    sam_model_registry,
    save_mask_data,
    show_box,
    show_mask,
)

DEFAULT_JOBS_DIR = PROJECT_ROOT / "strawberry_server_jobs"
DEFAULT_CONFIG = PROJECT_ROOT / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DEFAULT_GROUNDED_CHECKPOINT = PROJECT_ROOT / "groundingdino_swint_ogc.pth"
DEFAULT_SAM_CHECKPOINT = PROJECT_ROOT / "sam_vit_h_4b8939.pth"
DEFAULT_BERT_BASE_UNCASED_PATH = PROJECT_ROOT / "bert-base-uncased"
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
    parser.add_argument("--bert-base-uncased-path", default=str(DEFAULT_BERT_BASE_UNCASED_PATH))
    return parser.parse_args()


def log_event(job_id: str, message: str) -> None:
    stamp = datetime.now().isoformat(timespec="seconds")
    print(f"[strawberry-server] {stamp} job={job_id} {message}", flush=True)



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


def update_manifest(job_dir: Path, updates: Dict[str, Any]) -> None:
    manifest = job_dir / "manifest.json"
    current: Dict[str, Any] = {}
    if manifest.exists():
        current = read_json(manifest)
    write_json(manifest, {**current, **updates})


class GroundedSamEngine:
    def __init__(self, settings: argparse.Namespace):
        self.settings = settings
        self.device = str(settings.device)
        self.lock = threading.Lock()
        bert_path = Path(settings.bert_base_uncased_path)
        if not bert_path.is_dir():
            raise PipelineError(
                f"bert-base-uncased local path not found: {bert_path}. "
                "Download it and symlink it to the project root before running Grounded-SAM.",
                500,
            )

        started = time.monotonic()
        print("[strawberry-server] loading GroundingDINO/SAM models into GPU", flush=True)
        self.grounding_model = load_model(
            str(settings.config),
            str(settings.grounded_checkpoint),
            str(settings.bert_base_uncased_path),
            device=self.device,
        ).to(self.device)
        self.grounding_model.eval()
        self.predictor = SamPredictor(
            sam_model_registry["vit_h"](checkpoint=str(settings.sam_checkpoint)).to(self.device)
        )
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        print(f"[strawberry-server] models ready elapsed_s={elapsed:.2f}", flush=True)

    def run(
        self,
        input_image: Path,
        output_dir: Path,
        text_prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> Dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with self.lock, torch.inference_mode():
            image_pil, image_tensor = load_image(str(input_image))
            image_pil.save(output_dir / "raw_image.jpg")

            boxes_filt, pred_phrases = get_grounding_output(
                self.grounding_model,
                image_tensor,
                text_prompt,
                box_threshold,
                text_threshold,
                device=self.device,
            )

            image = cv2.imread(str(input_image))
            if image is None:
                raise PipelineError(f"failed to read image for inference: {input_image}", 500)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(image)

            width, height = image_pil.size
            boxes_xyxy = boxes_filt.clone()
            for index in range(boxes_xyxy.size(0)):
                boxes_xyxy[index] = boxes_xyxy[index] * torch.Tensor([width, height, width, height])
                boxes_xyxy[index][:2] -= boxes_xyxy[index][2:] / 2
                boxes_xyxy[index][2:] += boxes_xyxy[index][:2]

            boxes_xyxy = boxes_xyxy.cpu()
            transformed_boxes = self.predictor.transform.apply_boxes_torch(
                boxes_xyxy, image.shape[:2]
            ).to(self.device)

            masks, _, _ = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(self.device),
                multimask_output=False,
            )

            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            for mask in masks:
                show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            for box, label in zip(boxes_xyxy, pred_phrases):
                show_box(box.numpy(), plt.gca(), label)
            plt.axis("off")
            plt.savefig(
                output_dir / "grounded_sam_output.jpg",
                bbox_inches="tight",
                dpi=300,
                pad_inches=0.0,
            )
            plt.close()

            save_mask_data(str(output_dir), masks, boxes_xyxy, pred_phrases)
            plt.close("all")
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()

        return {"elapsed_s": time.monotonic() - started, "output": "in-process"}


def run_pipeline_job(
    settings: argparse.Namespace,
    engine: GroundedSamEngine,
    job_dir: Path,
    raw_image_path: Path,
    metadata_path: Path,
    raw_sha256: str,
    image_info: Dict[str, Any],
    metadata_json: Dict[str, Any],
    started: float,
) -> Dict[str, Any]:
    raw_output_dir = job_dir / "output_raw"
    output_dir = job_dir / "output" / f"outputs_multi_auto_{job_dir.name}"

    log_event(job_dir.name, "stage=grounded_sam starting Grounded-SAM")
    update_manifest(
        job_dir,
        {
            "status": "running",
            "stage": "grounded_sam",
            "stage_message": "running Grounded-SAM detection and segmentation",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    ground_result = engine.run(
        raw_image_path,
        raw_output_dir,
        str(settings.text_prompt),
        float(settings.box_threshold),
        float(settings.text_threshold),
    )

    masks = sorted(raw_output_dir.glob("mask_*.png"))
    if not masks:
        raise PipelineError("no strawberry detected", 422)

    log_event(job_dir.name, f"stage=grasp_generation masks={len(masks)}")
    update_manifest(
        job_dir,
        {
            "stage": "grasp_generation",
            "stage_message": f"generating grasp points for {len(masks)} masks",
            "mask_count": len(masks),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

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
    log_event(job_dir.name, f"stage=done targets={len(targets)} zip={zip_path}")
    total_time_s = time.monotonic() - started
    response = {
        "job_id": job_dir.name,
        "status": "done",
        "stage": "done",
        "raw_image_sha256": raw_sha256,
        "width": image_info["width"],
        "height": image_info["height"],
        "target_count": len(targets),
        "failed_targets": all_data.get("failed_targets", 0),
        "targets": targets,
        "server_output_dir": str(output_dir),
        "all_grasp_points_json": str(output_dir / "all_grasp_points.json"),
        "result_zip_url": f"/api/strawberry/jobs/{job_dir.name}/result.zip",
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
            "metadata": metadata_json,
            "raw_output_dir": str(raw_output_dir),
            "result_zip": str(zip_path),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return response


def run_pipeline_job_background(
    settings: argparse.Namespace,
    engine: GroundedSamEngine,
    job_dir: Path,
    raw_image_path: Path,
    metadata_path: Path,
    raw_sha256: str,
    image_info: Dict[str, Any],
    metadata_json: Dict[str, Any],
    started: float,
) -> None:
    try:
        run_pipeline_job(
            settings,
            engine,
            job_dir,
            raw_image_path,
            metadata_path,
            raw_sha256,
            image_info,
            metadata_json,
            started,
        )
    except PipelineError as exc:
        log_event(job_dir.name, f"stage=failed error={exc}")
        update_manifest(
            job_dir,
            {
                "status": "failed",
                "stage": "failed",
                "error": str(exc),
                "status_code": exc.status_code,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
    except Exception as exc:
        log_event(job_dir.name, f"stage=failed error={exc}")
        update_manifest(
            job_dir,
            {
                "status": "failed",
                "stage": "failed",
                "error": str(exc),
                "status_code": 500,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )


async def prepare_job(
    jobs_dir: Path,
    image: UploadFile,
    metadata: UploadFile,
) -> Dict[str, Any]:
    job_id = make_job_id()
    job_dir = jobs_dir / job_id
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=False)

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
    log_event(job_id, "stage=accepted request received and input saved")
    write_json(
        job_dir / "manifest.json",
        {
            "job_id": job_id,
            "status": "accepted",
            "stage": "accepted",
            "stage_message": "request received and input files saved",
            "raw_image_sha256": raw_sha256,
            "metadata": metadata_json,
            "input_image": str(raw_image_path),
            "input_metadata": str(metadata_path),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            **image_info,
        },
    )
    return {
        "job_id": job_id,
        "job_dir": job_dir,
        "raw_image_path": raw_image_path,
        "metadata_path": metadata_path,
        "raw_sha256": raw_sha256,
        "image_info": image_info,
        "metadata_json": metadata_json,
        "started": time.monotonic(),
    }


def build_app(args: Optional[argparse.Namespace] = None) -> FastAPI:
    settings = args or parse_args()
    jobs_dir = Path(settings.jobs_dir).resolve()
    jobs_dir.mkdir(parents=True, exist_ok=True)
    engine = GroundedSamEngine(settings)

    app = FastAPI(title="Strawberry Multi-Grasp Server", version="1.0")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "project_root": str(PROJECT_ROOT),
            "jobs_dir": str(jobs_dir),
            "device": settings.device,
            "model_loaded": True,
        }

    @app.post("/api/strawberry/multi_grasp")
    async def multi_grasp(
        image: UploadFile = File(...),
        metadata: UploadFile = File(...),
        return_format: str = Form("json"),
    ):
        try:
            job = await prepare_job(jobs_dir, image, metadata)
            response = run_pipeline_job(
                settings,
                engine,
                job["job_dir"],
                job["raw_image_path"],
                job["metadata_path"],
                job["raw_sha256"],
                job["image_info"],
                job["metadata_json"],
                job["started"],
            )
            if return_format.lower() == "zip":
                zip_path = job["job_dir"] / "result.zip"
                output_dir = Path(response["server_output_dir"])
                return FileResponse(
                    zip_path,
                    media_type="application/zip",
                    filename=f"{output_dir.name}.zip",
                )
            return response
        except PipelineError as exc:
            job_id = locals().get("job", {}).get("job_id", make_job_id())
            job_dir = locals().get("job", {}).get("job_dir", jobs_dir / job_id)
            log_event(job_id, f"stage=failed error={exc}")
            error = {"job_id": job_id, "status": "failed", "stage": "failed", "error": str(exc)}
            write_json(job_dir / "manifest.json", error)
            return JSONResponse(status_code=exc.status_code, content=error)
        except Exception as exc:
            job_id = locals().get("job", {}).get("job_id", make_job_id())
            job_dir = locals().get("job", {}).get("job_dir", jobs_dir / job_id)
            log_event(job_id, f"stage=failed error={exc}")
            error = {"job_id": job_id, "status": "failed", "stage": "failed", "error": str(exc)}
            write_json(job_dir / "manifest.json", error)
            return JSONResponse(status_code=500, content=error)

    @app.post("/api/strawberry/multi_grasp_async")
    async def multi_grasp_async(
        background_tasks: BackgroundTasks,
        image: UploadFile = File(...),
        metadata: UploadFile = File(...),
        return_format: str = Form("json"),
    ):
        try:
            job = await prepare_job(jobs_dir, image, metadata)
            background_tasks.add_task(
                run_pipeline_job_background,
                settings,
                engine,
                job["job_dir"],
                job["raw_image_path"],
                job["metadata_path"],
                job["raw_sha256"],
                job["image_info"],
                job["metadata_json"],
                job["started"],
            )
            return {
                "job_id": job["job_id"],
                "status": "accepted",
                "stage": "accepted",
                "message": "request received; poll job_status_url for progress",
                "job_status_url": f"/api/strawberry/jobs/{job['job_id']}",
                "result_zip_url": f"/api/strawberry/jobs/{job['job_id']}/result.zip",
            }
        except PipelineError as exc:
            job_id = locals().get("job", {}).get("job_id", make_job_id())
            job_dir = locals().get("job", {}).get("job_dir", jobs_dir / job_id)
            error = {"job_id": job_id, "status": "failed", "stage": "failed", "error": str(exc)}
            write_json(job_dir / "manifest.json", error)
            return JSONResponse(status_code=exc.status_code, content=error)
        except Exception as exc:
            job_id = locals().get("job", {}).get("job_id", make_job_id())
            job_dir = locals().get("job", {}).get("job_dir", jobs_dir / job_id)
            error = {"job_id": job_id, "status": "failed", "stage": "failed", "error": str(exc)}
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



if __name__ == "__main__":
    import uvicorn

    cli_args = parse_args()
    uvicorn.run(build_app(cli_args), host="0.0.0.0", port=8000)
