#!/usr/bin/env python3
"""Split multi-strawberry masks into ordered targets and generate grasps."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


Point2D = Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create target_XX folders from a Grounded-SAM output directory and "
            "generate one grasp for each strawberry mask. Targets are ordered by "
            "lower image position first, then rightmost position."
        )
    )
    parser.add_argument(
        "--sam-output-dir",
        required=True,
        help="Grounded-SAM output containing raw_image.jpg, mask.json, mask_*.png.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory, e.g. outputs_multi_01.",
    )
    parser.add_argument(
        "--grasp-script",
        default=str(Path(__file__).with_name("generate_strawberry_grasp.py")),
        help="Path to generate_strawberry_grasp.py.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run the grasp script.",
    )
    parser.add_argument(
        "--layout",
        choices=("triangle-paired", "paired-opposed", "tripod"),
        default="triangle-paired",
        help="Grasp layout passed to the single-target grasp script.",
    )
    parser.add_argument(
        "--thumb-side",
        choices=("top", "bottom", "left", "right"),
        default="top",
        help="Thumb side passed to the single-target grasp script.",
    )
    parser.add_argument(
        "--paired-side",
        choices=("left", "right", "top", "bottom"),
        default="top",
        help="Paired side passed to the single-target grasp script.",
    )
    parser.add_argument(
        "--mask-transform",
        choices=("auto", "none", "transpose", "rotate-cw", "rotate-ccw"),
        default="auto",
        help="Mask transform passed to the single-target grasp script.",
    )
    parser.add_argument("--depth", help="Optional depth image for 3D points.")
    parser.add_argument("--intrinsics", help="Optional camera intrinsics.")
    parser.add_argument("--T-hand-cam", dest="t_hand_cam", help="Optional camera-to-hand transform.")
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--depth-radius", type=int, default=5)
    parser.add_argument("--mm-per-pixel", type=float)
    parser.add_argument("--pair-spacing-ratio", type=float, default=0.22)
    parser.add_argument("--pair-spacing-mm", type=float)
    parser.add_argument("--contact-inset-ratio", type=float, default=0.015)
    parser.add_argument("--contact-inset-mm", type=float)
    parser.add_argument("--morph-kernel", type=int, default=0)
    parser.add_argument("--safe-margin-ratio", type=float, default=0.012)
    parser.add_argument("--min-spacing-ratio", type=float, default=0.12)
    parser.add_argument("--min-area-ratio", type=float, default=0.02)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output directory before writing.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop at the first target whose grasp generation fails.",
    )
    return parser.parse_args()


def mask_index(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.split("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Unexpected mask filename: {path.name}") from exc


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=True)
        file.write("\n")


def bbox_from_mask(mask_path: Path) -> Tuple[int, int, int, int]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        raise ValueError(f"Mask is empty: {mask_path}")
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    )


def bbox_center(bbox: Sequence[float]) -> Point2D:
    x0, y0, x1, y1 = bbox
    return (float(x0 + x1) / 2.0, float(y0 + y1) / 2.0)


def xywh_center(bbox: Sequence[int]) -> Point2D:
    x, y, w, h = bbox
    return (float(x) + float(w - 1) / 2.0, float(y) + float(h - 1) / 2.0)


def metadata_by_mask_index(mask_json: Path) -> Dict[int, Dict[str, Any]]:
    if not mask_json.exists():
        return {}
    data = read_json(mask_json)
    result: Dict[int, Dict[str, Any]] = {}
    for item in data:
        value = item.get("value")
        if isinstance(value, int) and value > 0:
            result[value - 1] = item
    return result


def sorted_targets(sam_output_dir: Path) -> List[Dict[str, Any]]:
    metadata = metadata_by_mask_index(sam_output_dir / "mask.json")
    targets: List[Dict[str, Any]] = []
    for mask_path in sorted(sam_output_dir.glob("mask_*.png"), key=mask_index):
        index = mask_index(mask_path)
        mask_bbox_xywh = bbox_from_mask(mask_path)
        item = metadata.get(index, {})
        if "box" in item:
            cx, cy = bbox_center(item["box"])
        else:
            cx, cy = xywh_center(mask_bbox_xywh)
        targets.append(
            {
                "source_mask_index": index,
                "source_mask": mask_path,
                "source_metadata": item,
                "sort_center": [cx, cy],
                "mask_bbox_xywh": list(mask_bbox_xywh),
            }
        )
    return sorted(
        targets,
        key=lambda target: (-target["sort_center"][1], -target["sort_center"][0]),
    )


def target_mask_json(target: Dict[str, Any]) -> List[Dict[str, Any]]:
    item = target.get("source_metadata") or {}
    if item:
        target_item = dict(item)
        target_item["value"] = 1
    else:
        x, y, w, h = target["mask_bbox_xywh"]
        target_item = {
            "value": 1,
            "label": "strawberry",
            "box": [x, y, x + w - 1, y + h - 1],
        }
    target_item["source_mask_index"] = target["source_mask_index"]
    target_item["sort_center"] = target["sort_center"]
    target_item["mask_bbox_xywh"] = target["mask_bbox_xywh"]
    return [{"value": 0, "label": "background"}, target_item]


def optional_flag(command: List[str], name: str, value: Optional[Any]) -> None:
    if value is not None:
        command.extend([name, str(value)])


def run_grasp(args: argparse.Namespace, target_dir: Path) -> Dict[str, Any]:
    output_json = target_dir / "grasp_points_triangle.json"
    output_vis = target_dir / "grasp_points_triangle_visualization.jpg"
    command = [
        args.python,
        args.grasp_script,
        "--mask",
        str(target_dir / "mask_0.png"),
        "--image",
        str(target_dir / "raw_image.jpg"),
        "--layout",
        args.layout,
        "--thumb-side",
        args.thumb_side,
        "--paired-side",
        args.paired_side,
        "--mask-transform",
        args.mask_transform,
        "--depth-scale",
        str(args.depth_scale),
        "--depth-radius",
        str(args.depth_radius),
        "--pair-spacing-ratio",
        str(args.pair_spacing_ratio),
        "--contact-inset-ratio",
        str(args.contact_inset_ratio),
        "--morph-kernel",
        str(args.morph_kernel),
        "--safe-margin-ratio",
        str(args.safe_margin_ratio),
        "--min-spacing-ratio",
        str(args.min_spacing_ratio),
        "--min-area-ratio",
        str(args.min_area_ratio),
        "--output-json",
        str(output_json),
        "--output-vis",
        str(output_vis),
    ]
    optional_flag(command, "--depth", args.depth)
    optional_flag(command, "--intrinsics", args.intrinsics)
    optional_flag(command, "--T-hand-cam", args.t_hand_cam)
    optional_flag(command, "--mm-per-pixel", args.mm_per_pixel)
    optional_flag(command, "--pair-spacing-mm", args.pair_spacing_mm)
    optional_flag(command, "--contact-inset-mm", args.contact_inset_mm)
    subprocess.run(command, check=True)
    return read_json(output_json)


def direction_line_from_grasp(grasp: Dict[str, Any]) -> Optional[Dict[str, List[float]]]:
    diagnostics = grasp.get("diagnostics", {})
    ignored = diagnostics.get("ignored_tip_vertex")
    thumb = diagnostics.get("thumb_support_vertex")
    pair = diagnostics.get("pair_support_vertex")
    if ignored is None or thumb is None or pair is None:
        return None

    ignored_pt = np.asarray(ignored, dtype=np.float64)
    thumb_pt = np.asarray(thumb, dtype=np.float64)
    pair_pt = np.asarray(pair, dtype=np.float64)
    midpoint = (thumb_pt + pair_pt) / 2.0
    vector = midpoint - ignored_pt
    return {
        "from_ignored_tip_vertex": ignored_pt.tolist(),
        "to_support_midpoint": midpoint.tolist(),
        "vector": vector.tolist(),
    }


def draw_direction_summary(
    raw_image_path: Path,
    output_path: Path,
    target_items: Sequence[Dict[str, Any]],
) -> None:
    image = cv2.imread(str(raw_image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read raw image: {raw_image_path}")
    canvas = image.copy()
    overlay = canvas.copy()
    palette = [
        (0, 255, 255),
        (0, 165, 255),
        (255, 0, 255),
        (255, 80, 0),
        (0, 255, 0),
        (255, 255, 0),
    ]
    scale = max(0.55, min(canvas.shape[:2]) / 1000.0)
    thickness = max(2, round(3 * scale))
    radius = max(5, round(8 * scale))

    for index, item in enumerate(target_items):
        if item.get("status") != "ok":
            continue
        target_dir = Path(item["target_dir"])
        mask = cv2.imread(str(target_dir / "mask_0.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        color = palette[index % len(palette)]
        overlay[mask > 0] = color

    canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)

    for index, item in enumerate(target_items):
        if item.get("status") != "ok":
            continue
        target_dir = Path(item["target_dir"])
        mask = cv2.imread(str(target_dir / "mask_0.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(
                canvas,
                contours,
                -1,
                (0, 255, 0),
                thickness,
                cv2.LINE_AA,
            )

        direction = item.get("direction_line_2d")
        if not direction:
            continue
        start = tuple(np.rint(direction["from_ignored_tip_vertex"]).astype(int))
        end = tuple(np.rint(direction["to_support_midpoint"]).astype(int))
        color = palette[index % len(palette)]

        cv2.arrowedLine(canvas, start, end, color, thickness, cv2.LINE_AA, tipLength=0.25)
        cv2.circle(canvas, start, radius, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, end, radius, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, end, radius, color, thickness, cv2.LINE_AA)
        cv2.putText(
            canvas,
            item["target"],
            (end[0] + radius + 4, end[1] - radius - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"Could not write direction visualization: {output_path}")


def copy_target_files(sam_output_dir: Path, output_dir: Path, target: Dict[str, Any], order: int) -> Path:
    target_dir = output_dir / f"target_{order:02d}"
    target_dir.mkdir(parents=True, exist_ok=True)
    raw_image = sam_output_dir / "raw_image.jpg"
    if not raw_image.exists():
        raise FileNotFoundError(f"Missing raw image: {raw_image}")
    shutil.copy2(raw_image, target_dir / "raw_image.jpg")
    shutil.copy2(target["source_mask"], target_dir / "mask_0.png")
    write_json(target_dir / "mask.json", target_mask_json(target))
    return target_dir


def main() -> None:
    args = parse_args()
    sam_output_dir = Path(args.sam_output_dir)
    output_dir = Path(args.output_dir)
    if not sam_output_dir.is_dir():
        raise NotADirectoryError(f"Grounded-SAM output dir not found: {sam_output_dir}")
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dir already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ordered_targets = sorted_targets(sam_output_dir)
    if not ordered_targets:
        raise ValueError(f"No mask_*.png files found in {sam_output_dir}")

    summary: List[Dict[str, Any]] = []
    failures = 0
    for order, target in enumerate(ordered_targets):
        target_dir = copy_target_files(sam_output_dir, output_dir, target, order)
        item = {
            "target": f"target_{order:02d}",
            "source_mask_index": target["source_mask_index"],
            "sort_center": target["sort_center"],
            "mask_bbox_xywh": target["mask_bbox_xywh"],
            "grasp_json": str(target_dir / "grasp_points_triangle.json"),
            "visualization": str(target_dir / "grasp_points_triangle_visualization.jpg"),
        }
        try:
            grasp = run_grasp(args, target_dir)
        except subprocess.CalledProcessError as exc:
            failures += 1
            error = {
                "status": "failed",
                "returncode": exc.returncode,
                "command": exc.cmd,
            }
            write_json(target_dir / "grasp_error.json", error)
            item.update(
                {
                    "status": "failed",
                    "error_json": str(target_dir / "grasp_error.json"),
                }
            )
            if args.strict:
                raise
        else:
            direction_line = direction_line_from_grasp(grasp)
            if direction_line is not None:
                grasp["direction_line_2d"] = direction_line
                write_json(target_dir / "grasp_points_triangle.json", grasp)
            item.update(
                {
                    "status": "ok",
                    "grasp_points_2d": grasp["grasp_points_2d"],
                    "grasp_pair_center_2d": grasp.get("grasp_pair_center_2d"),
                    "target_dir": str(target_dir),
                }
            )
            if direction_line is not None:
                item["direction_line_2d"] = direction_line
        summary.append(item)

    direction_vis = output_dir / "strawberry_direction_visualization.jpg"
    draw_direction_summary(sam_output_dir / "raw_image.jpg", direction_vis, summary)

    all_data = {
        "sam_output_dir": str(sam_output_dir),
        "output_dir": str(output_dir),
        "sort_order": "descending center_y, then descending center_x",
        "direction_standard": (
            "line from ignored_tip_vertex to the midpoint of "
            "thumb_support_vertex and pair_support_vertex"
        ),
        "direction_visualization": str(direction_vis),
        "failed_targets": failures,
        "targets": summary,
    }
    write_json(output_dir / "all_grasp_points.json", all_data)
    print(json.dumps(all_data, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
