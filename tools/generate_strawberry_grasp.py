#!/usr/bin/env python3
"""Generate a three-finger grasp from a strawberry binary mask."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


Point2D = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paired-finger/opposed-thumb strawberry grasp contacts."
    )
    parser.add_argument("--mask", required=True, help="Binary mask image.")
    parser.add_argument("--image", help="RGB image used for alignment and visualization.")
    parser.add_argument("--depth", help="Optional depth image (.npy or image file).")
    parser.add_argument(
        "--intrinsics",
        help="Optional camera intrinsics: JSON, TXT, or NPY containing a 3x3 K matrix.",
    )
    parser.add_argument(
        "--T-hand-cam",
        dest="t_hand_cam",
        help="Optional JSON, TXT, or NPY 4x4 transform from camera to hand_base.",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="Multiplier converting raw depth to output units (for mm to m use 0.001).",
    )
    parser.add_argument(
        "--depth-radius",
        type=int,
        default=5,
        help="Radius for valid-depth median sampling around each grasp point.",
    )
    parser.add_argument(
        "--mask-transform",
        choices=("auto", "none", "transpose", "rotate-cw", "rotate-ccw"),
        default="auto",
        help="Transform mask into image coordinates. Auto handles EXIF/cv2 mismatch.",
    )
    parser.add_argument(
        "--layout",
        choices=("triangle-paired", "paired-opposed", "tripod"),
        default="triangle-paired",
        help="Triangle-based paired grasp, image-side paired grasp, or original tripod.",
    )
    parser.add_argument(
        "--thumb-side",
        choices=("top", "bottom", "left", "right"),
        default="top",
        help="Select which non-tip triangle vertex is used by the thumb.",
    )
    parser.add_argument(
        "--paired-side",
        choices=("left", "right", "top", "bottom"),
        default="top",
        help="Image side occupied by the paired index/middle fingers.",
    )
    parser.add_argument(
        "--mm-per-pixel",
        type=float,
        help="Optional planar scale at the strawberry plane.",
    )
    parser.add_argument(
        "--pair-spacing-ratio",
        type=float,
        default=0.22,
        help="Index/middle spacing relative to the bbox dimension along the side.",
    )
    parser.add_argument(
        "--pair-spacing-mm",
        type=float,
        help="Physical index/middle contact spacing; requires --mm-per-pixel.",
    )
    parser.add_argument(
        "--contact-inset-ratio",
        type=float,
        default=0.015,
        help="Contact inset from the mask boundary relative to min bbox dimension.",
    )
    parser.add_argument(
        "--contact-inset-mm",
        type=float,
        help="Physical contact inset from the boundary; requires --mm-per-pixel.",
    )
    parser.add_argument(
        "--morph-kernel",
        type=int,
        default=0,
        help="Odd morphology kernel size; 0 chooses a size from the object bbox.",
    )
    parser.add_argument(
        "--safe-margin-ratio",
        type=float,
        default=0.012,
        help="Safe erosion margin relative to min(bbox width, bbox height).",
    )
    parser.add_argument(
        "--min-spacing-ratio",
        type=float,
        default=0.12,
        help="Required minimum point distance relative to min bbox dimension.",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.02,
        help="Required triangle area relative to bbox area.",
    )
    parser.add_argument(
        "--output-json",
        default="grasp_points.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--output-vis",
        default="grasp_points_visualization.jpg",
        help="Output visualization path.",
    )
    return parser.parse_args()


def read_image(path: str, flags: int, name: str) -> np.ndarray:
    image = cv2.imread(path, flags)
    if image is None:
        raise FileNotFoundError(f"Could not read {name}: {path}")
    return image


def odd_kernel_size(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def alignment_score(mask: np.ndarray, image: np.ndarray) -> float:
    """Score mask/image alignment using contour contrast and image gradients."""
    mask_u8 = (mask > 0).astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)
    inner = cv2.erode(mask_u8, kernel)
    ring = cv2.dilate(mask_u8, kernel) - mask_u8
    boundary = cv2.dilate(mask_u8, kernel) - inner
    if not np.any(inner) or not np.any(ring):
        return -np.inf

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    color_contrast = float(
        np.linalg.norm(lab[inner > 0].mean(axis=0) - lab[ring > 0].mean(axis=0))
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    boundary_gradient = float(gradient[boundary > 0].mean())
    return color_contrast + 0.25 * boundary_gradient


def align_mask(
    mask: np.ndarray, image: Optional[np.ndarray], transform: str
) -> Tuple[np.ndarray, str, Dict[str, float]]:
    operations = {
        "none": lambda x: x,
        "transpose": lambda x: x.T,
        "rotate-cw": lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
        "rotate-ccw": lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }
    if transform != "auto":
        aligned = operations[transform](mask)
        if image is not None and aligned.shape != image.shape[:2]:
            raise ValueError(
                f"Transformed mask shape {aligned.shape} does not match image "
                f"shape {image.shape[:2]}."
            )
        return aligned, transform, {}

    if image is None or mask.shape == image.shape[:2]:
        return mask, "none", {}

    candidates = {}
    scores = {}
    for name, operation in operations.items():
        candidate = operation(mask)
        if candidate.shape == image.shape[:2]:
            candidates[name] = candidate
            scores[name] = alignment_score(candidate, image)
    if not candidates:
        raise ValueError(
            f"Mask shape {mask.shape} cannot be aligned to image shape {image.shape[:2]} "
            "by transpose or 90-degree rotation."
        )
    selected = max(scores, key=scores.get)
    return candidates[selected], selected, scores


def preprocess_mask(mask: np.ndarray, morph_kernel: int) -> Tuple[np.ndarray, np.ndarray]:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    rough = cv2.findNonZero(binary)
    if rough is None:
        raise ValueError("The input mask is empty.")
    _, _, rough_w, rough_h = cv2.boundingRect(rough)
    if morph_kernel <= 0:
        morph_kernel = odd_kernel_size(round(0.008 * min(rough_w, rough_h)))
    else:
        morph_kernel = odd_kernel_size(morph_kernel)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel)
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        raise ValueError("No foreground component remains after preprocessing.")
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        largest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour = max(contours, key=cv2.contourArea)
    return largest, contour


def nearest_safe_point(point: Sequence[float], safe_mask: np.ndarray) -> Point2D:
    x = int(round(point[0]))
    y = int(round(point[1]))
    h, w = safe_mask.shape
    if 0 <= x < w and 0 <= y < h and safe_mask[y, x] > 0:
        return x, y
    ys, xs = np.nonzero(safe_mask)
    if len(xs) == 0:
        raise ValueError("Safe mask is empty.")
    distances = (xs - point[0]) ** 2 + (ys - point[1]) ** 2
    index = int(np.argmin(distances))
    return int(xs[index]), int(ys[index])


def nearest_contour_index(point: Sequence[float], contour: np.ndarray) -> int:
    contour_points = contour.reshape(-1, 2).astype(np.float64)
    distances = np.sum((contour_points - np.asarray(point)) ** 2, axis=1)
    return int(np.argmin(distances))


def walk_contour(
    contour: np.ndarray, start_index: int, distance: float, direction: int
) -> Tuple[Point2D, int]:
    points = contour.reshape(-1, 2)
    count = len(points)
    current = start_index
    travelled = 0.0
    for _ in range(count):
        following = (current + direction) % count
        travelled += float(np.linalg.norm(points[following] - points[current]))
        current = following
        if travelled >= distance:
            break
    return (int(points[current, 0]), int(points[current, 1])), current


def row_boundaries(mask: np.ndarray, target_y: float, y_min: int, y_max: int):
    target = int(round(target_y))
    max_delta = max(target - y_min, y_max - target)
    for delta in range(max_delta + 1):
        rows = (target,) if delta == 0 else (target - delta, target + delta)
        for y in rows:
            if y_min <= y <= y_max:
                xs = np.flatnonzero(mask[y] > 0)
                if len(xs) >= 2:
                    return y, int(xs[0]), int(xs[-1])
    raise ValueError("Could not find mask boundaries near the support row.")


def column_boundaries(mask: np.ndarray, target_x: float, x_min: int, x_max: int):
    target = int(round(target_x))
    max_delta = max(target - x_min, x_max - target)
    for delta in range(max_delta + 1):
        columns = (target,) if delta == 0 else (target - delta, target + delta)
        for x in columns:
            if x_min <= x <= x_max:
                ys = np.flatnonzero(mask[:, x] > 0)
                if len(ys) >= 2:
                    return x, int(ys[0]), int(ys[-1])
    raise ValueError("Could not find mask boundaries near the paired-finger column.")


def build_safe_mask(mask: np.ndarray, margin: int) -> Tuple[np.ndarray, int]:
    for current_margin in range(max(1, margin), 0, -1):
        size = 2 * current_margin + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        safe = cv2.erode(mask, kernel)
        if np.any(safe):
            return safe, current_margin
    return mask.copy(), 0


def triangle_metrics(points: List[Point2D]) -> Tuple[List[float], float]:
    p = np.asarray(points, dtype=np.float64)
    distances = [
        float(np.linalg.norm(p[0] - p[1])),
        float(np.linalg.norm(p[0] - p[2])),
        float(np.linalg.norm(p[1] - p[2])),
    ]
    first = p[1] - p[0]
    second = p[2] - p[0]
    area = float(abs(first[0] * second[1] - first[1] * second[0]) * 0.5)
    return distances, area


def validate_triangle(
    points: Dict[str, Point2D],
    order: Sequence[str],
    w: int,
    h: int,
    min_spacing_ratio: float,
    min_area_ratio: float,
) -> Dict[str, object]:
    ordered = [points[name] for name in order]
    distances, area = triangle_metrics(ordered)
    min_distance = min_spacing_ratio * min(w, h)
    min_area = min_area_ratio * w * h
    stable = min(distances) >= min_distance and area >= min_area
    if not stable:
        raise ValueError(
            "Generated points do not form a stable triangle: "
            f"min distance={min(distances):.2f} (required {min_distance:.2f}), "
            f"area={area:.2f} (required {min_area:.2f})."
        )
    return {
        "pairwise_distances_pixels": distances,
        "triangle_area_pixels2": area,
        "stable_triangle": stable,
    }


def fit_strawberry_triangle(
    contour: np.ndarray,
) -> Tuple[np.ndarray, int, str, float]:
    """Return three on-contour fruit vertices and the index of the sharp tip."""
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    quadrilateral = None
    for epsilon_ratio in np.linspace(0.01, 0.18, 69):
        approximation = cv2.approxPolyDP(
            hull, float(epsilon_ratio * perimeter), True
        ).reshape(-1, 2)
        if len(approximation) == 4:
            quadrilateral = approximation.astype(np.float64)
            break

    if quadrilateral is not None:
        angles = []
        for index in range(4):
            previous_vector = quadrilateral[index - 1] - quadrilateral[index]
            next_vector = quadrilateral[(index + 1) % 4] - quadrilateral[index]
            cosine = np.dot(previous_vector, next_vector) / (
                np.linalg.norm(previous_vector) * np.linalg.norm(next_vector)
            )
            angles.append(float(np.arccos(np.clip(cosine, -1.0, 1.0))))
        tip_quad_index = int(np.argmin(angles))
        base_quad_index = (tip_quad_index + 2) % 4
        retained_indices = [
            index for index in range(4) if index != base_quad_index
        ]
        triangle = quadrilateral[retained_indices]
        tip_index = retained_indices.index(tip_quad_index)
        area = abs(float(cv2.contourArea(triangle.astype(np.float32))))
        return triangle, tip_index, "convex-hull-shoulders", area

    area, triangle = cv2.minEnclosingTriangle(contour)
    if triangle is None:
        raise ValueError("Could not fit a triangle to the mask contour.")
    triangle = triangle.reshape(3, 2).astype(np.float64)
    edges = [
        float(np.linalg.norm(triangle[0] - triangle[1])),
        float(np.linalg.norm(triangle[1] - triangle[2])),
        float(np.linalg.norm(triangle[2] - triangle[0])),
    ]
    tip_index = (int(np.argmin(edges)) + 2) % 3
    return triangle, tip_index, "minimum-enclosing-triangle", float(area)


def generate_tripod_grasp(
    mask: np.ndarray,
    contour: np.ndarray,
    safe_margin_ratio: float,
    min_spacing_ratio: float,
    min_area_ratio: float,
) -> Tuple[Dict[str, Point2D], Dict[str, object], np.ndarray]:
    x, y, w, h = cv2.boundingRect(contour)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        raise ValueError("Mask contour has zero area.")
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    support_y, left_x, right_x = row_boundaries(
        mask, y + 0.65 * (h - 1), y, y + h - 1
    )
    inset = 0.05 * w
    candidates = {
        "P_top": (cx, y + 0.25 * (h - 1)),
        "P_left": (left_x + inset, support_y),
        "P_right": (right_x - inset, support_y),
    }

    requested_margin = max(1, round(safe_margin_ratio * min(w, h)))
    safe_mask, actual_margin = build_safe_mask(mask, requested_margin)
    points = {
        name: nearest_safe_point(point, safe_mask)
        for name, point in candidates.items()
    }

    diagnostics = {
        "bbox_xywh": [int(x), int(y), int(w), int(h)],
        "centroid": [float(cx), float(cy)],
        "safe_margin_pixels": int(actual_margin),
        "layout": "tripod",
        **validate_triangle(
            points,
            ("P_top", "P_left", "P_right"),
            w,
            h,
            min_spacing_ratio,
            min_area_ratio,
        ),
    }
    return points, diagnostics, safe_mask


def generate_paired_grasp(
    mask: np.ndarray,
    contour: np.ndarray,
    paired_side: str,
    pair_spacing_pixels: float,
    contact_inset_pixels: float,
    safe_margin_ratio: float,
    min_spacing_ratio: float,
    min_area_ratio: float,
) -> Tuple[Dict[str, Point2D], Dict[str, object], np.ndarray]:
    x, y, w, h = cv2.boundingRect(contour)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        raise ValueError("Mask contour has zero area.")
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    inset = max(1.0, contact_inset_pixels)
    half_spacing = pair_spacing_pixels * 0.5

    if paired_side in ("left", "right"):
        upper_y, upper_left, upper_right = row_boundaries(
            mask, cy - half_spacing, y, y + h - 1
        )
        lower_y, lower_left, lower_right = row_boundaries(
            mask, cy + half_spacing, y, y + h - 1
        )
        thumb_y, thumb_left, thumb_right = row_boundaries(
            mask, cy, y, y + h - 1
        )
        if paired_side == "right":
            candidates = {
                "P_index": (upper_right - inset, upper_y),
                "P_middle": (lower_right - inset, lower_y),
                "P_thumb": (thumb_left + inset, thumb_y),
            }
        else:
            candidates = {
                "P_index": (upper_left + inset, upper_y),
                "P_middle": (lower_left + inset, lower_y),
                "P_thumb": (thumb_right - inset, thumb_y),
            }
    else:
        left_x, left_top, left_bottom = column_boundaries(
            mask, cx - half_spacing, x, x + w - 1
        )
        right_x, right_top, right_bottom = column_boundaries(
            mask, cx + half_spacing, x, x + w - 1
        )
        thumb_x, thumb_top, thumb_bottom = column_boundaries(
            mask, cx, x, x + w - 1
        )
        if paired_side == "bottom":
            candidates = {
                "P_index": (left_x, left_bottom - inset),
                "P_middle": (right_x, right_bottom - inset),
                "P_thumb": (thumb_x, thumb_top + inset),
            }
        else:
            candidates = {
                "P_index": (left_x, left_top + inset),
                "P_middle": (right_x, right_top + inset),
                "P_thumb": (thumb_x, thumb_bottom - inset),
            }

    requested_margin = max(1, round(safe_margin_ratio * min(w, h)))
    safe_mask, actual_margin = build_safe_mask(mask, requested_margin)
    points = {
        name: nearest_safe_point(point, safe_mask)
        for name, point in candidates.items()
    }
    diagnostics = {
        "bbox_xywh": [int(x), int(y), int(w), int(h)],
        "centroid": [float(cx), float(cy)],
        "safe_margin_pixels": int(actual_margin),
        "layout": "paired-opposed",
        "paired_side": paired_side,
        "pair_spacing_pixels": float(pair_spacing_pixels),
        "contact_inset_pixels": float(inset),
        **validate_triangle(
            points,
            ("P_index", "P_middle", "P_thumb"),
            w,
            h,
            min_spacing_ratio,
            min_area_ratio,
        ),
    }
    return points, diagnostics, safe_mask


def generate_triangle_paired_grasp(
    mask: np.ndarray,
    contour: np.ndarray,
    thumb_side: str,
    pair_spacing_pixels: float,
    contact_inset_pixels: float,
    safe_margin_ratio: float,
    min_spacing_ratio: float,
    min_area_ratio: float,
) -> Tuple[Dict[str, Point2D], Dict[str, object], np.ndarray]:
    x, y, w, h = cv2.boundingRect(contour)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        raise ValueError("Mask contour has zero area.")
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    triangle, tip_index, triangle_method, triangle_area = fit_strawberry_triangle(
        contour
    )
    support_indices = [index for index in range(3) if index != tip_index]

    contour_indices = [
        nearest_contour_index(vertex, contour) for vertex in triangle
    ]
    contour_points = contour.reshape(-1, 2)
    support_points = [
        contour_points[contour_indices[index]].astype(np.float64)
        for index in support_indices
    ]
    axis = 1 if thumb_side in ("top", "bottom") else 0
    choose_minimum = thumb_side in ("top", "left")
    values = [point[axis] for point in support_points]
    thumb_position = int(np.argmin(values) if choose_minimum else np.argmax(values))
    pair_position = 1 - thumb_position
    thumb_triangle_index = support_indices[thumb_position]
    pair_triangle_index = support_indices[pair_position]
    thumb_boundary = support_points[thumb_position]
    pair_center_boundary = support_points[pair_position]
    pair_center_index = contour_indices[pair_triangle_index]

    half_spacing = max(1.0, pair_spacing_pixels * 0.5)
    index_boundary, _ = walk_contour(
        contour, pair_center_index, half_spacing, direction=-1
    )
    middle_boundary, _ = walk_contour(
        contour, pair_center_index, half_spacing, direction=1
    )

    requested_margin = max(
        1,
        round(safe_margin_ratio * min(w, h)),
        round(contact_inset_pixels),
    )
    safe_mask, actual_margin = build_safe_mask(mask, requested_margin)
    candidates = {
        "P_index": index_boundary,
        "P_middle": middle_boundary,
        "P_thumb": thumb_boundary,
    }
    points = {
        name: nearest_safe_point(point, safe_mask)
        for name, point in candidates.items()
    }
    pair_center = (
        int(round((points["P_index"][0] + points["P_middle"][0]) * 0.5)),
        int(round((points["P_index"][1] + points["P_middle"][1]) * 0.5)),
    )

    diagnostics = {
        "bbox_xywh": [int(x), int(y), int(w), int(h)],
        "centroid": [float(cx), float(cy)],
        "safe_margin_pixels": int(actual_margin),
        "layout": "triangle-paired",
        "thumb_side": thumb_side,
        "pair_spacing_pixels": float(pair_spacing_pixels),
        "contact_inset_pixels": float(contact_inset_pixels),
        "triangle_fit_method": triangle_method,
        "strawberry_triangle_area_pixels2": float(triangle_area),
        "strawberry_triangle_vertices": triangle.astype(float).tolist(),
        "ignored_tip_vertex": triangle[tip_index].astype(float).tolist(),
        "thumb_support_vertex": triangle[thumb_triangle_index].astype(float).tolist(),
        "pair_support_vertex": triangle[pair_triangle_index].astype(float).tolist(),
        "pair_center_2d": [int(pair_center[0]), int(pair_center[1])],
        **validate_triangle(
            points,
            ("P_index", "P_middle", "P_thumb"),
            w,
            h,
            min_spacing_ratio,
            min_area_ratio,
        ),
    }
    return points, diagnostics, safe_mask


def load_matrix(path: str, expected_shape: Tuple[int, int], name: str) -> np.ndarray:
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        matrix = np.load(path)
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict):
            for key in ("matrix", "K", "intrinsics", "T_hand_cam", "transform"):
                if key in data:
                    data = data[key]
                    break
        matrix = np.asarray(data, dtype=np.float64)
    else:
        matrix = np.loadtxt(path, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {matrix.shape}.")
    return matrix


def load_depth(path: str, target_shape: Tuple[int, int]) -> np.ndarray:
    if Path(path).suffix.lower() == ".npy":
        depth = np.load(path)
    else:
        depth = read_image(path, cv2.IMREAD_UNCHANGED, "depth image")
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.shape != target_shape:
        raise ValueError(
            f"Depth shape {depth.shape} does not match mask shape {target_shape}."
        )
    return depth.astype(np.float64)


def sample_depth(depth: np.ndarray, point: Point2D, radius: int) -> float:
    x, y = point
    y0, y1 = max(0, y - radius), min(depth.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(depth.shape[1], x + radius + 1)
    values = depth[y0:y1, x0:x1]
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0:
        raise ValueError(f"No valid depth near grasp point ({x}, {y}).")
    return float(np.median(values))


def backproject_points(
    points: Dict[str, Point2D],
    depth: np.ndarray,
    intrinsics: np.ndarray,
    depth_scale: float,
    depth_radius: int,
) -> Dict[str, List[float]]:
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    if fx == 0 or fy == 0:
        raise ValueError("Camera focal lengths fx and fy must be non-zero.")
    output = {}
    for name, (u, v) in points.items():
        z = sample_depth(depth, (u, v), depth_radius) * depth_scale
        output[name] = [
            float((u - cx) * z / fx),
            float((v - cy) * z / fy),
            float(z),
        ]
    return output


def transform_points(
    points: Dict[str, List[float]], transform: np.ndarray
) -> Dict[str, List[float]]:
    output = {}
    for name, point in points.items():
        homogeneous = np.r_[point, 1.0]
        transformed = transform @ homogeneous
        if transformed[3] == 0:
            raise ValueError("T_hand_cam produced a homogeneous coordinate of zero.")
        output[name] = (transformed[:3] / transformed[3]).astype(float).tolist()
    return output


def draw_visualization(
    image: Optional[np.ndarray],
    mask: np.ndarray,
    contour: np.ndarray,
    points: Dict[str, Point2D],
    diagnostics: Dict[str, object],
    output_path: str,
) -> None:
    if image is None:
        canvas = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()
        overlay = canvas.copy()
        overlay[mask > 0] = (0, 180, 255)
        canvas = cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0)

    scale = max(0.7, min(canvas.shape[:2]) / 1200.0)
    thickness = max(2, round(3 * scale))
    radius = max(7, round(10 * scale))
    cv2.drawContours(canvas, [contour], -1, (0, 255, 0), thickness)
    if diagnostics.get("layout") == "triangle-paired":
        fitted_triangle = np.asarray(
            diagnostics["strawberry_triangle_vertices"], dtype=np.int32
        ).reshape((-1, 1, 2))
        cv2.polylines(
            canvas, [fitted_triangle], True, (0, 255, 255), thickness, cv2.LINE_AA
        )
        tip = tuple(np.rint(diagnostics["ignored_tip_vertex"]).astype(int))
        cross_size = 2 * radius
        cv2.line(
            canvas,
            (tip[0] - cross_size, tip[1] - cross_size),
            (tip[0] + cross_size, tip[1] + cross_size),
            (0, 0, 255),
            thickness,
            cv2.LINE_AA,
        )
        cv2.line(
            canvas,
            (tip[0] - cross_size, tip[1] + cross_size),
            (tip[0] + cross_size, tip[1] - cross_size),
            (0, 0, 255),
            thickness,
            cv2.LINE_AA,
        )
        pair_center = tuple(diagnostics["pair_center_2d"])
        cv2.drawMarker(
            canvas,
            pair_center,
            (255, 255, 0),
            cv2.MARKER_CROSS,
            3 * radius,
            thickness,
            cv2.LINE_AA,
        )
        thumb = np.asarray(diagnostics["thumb_support_vertex"], dtype=np.float64)
        pair = np.asarray(diagnostics["pair_support_vertex"], dtype=np.float64)
        direction_midpoint = tuple(np.rint((thumb + pair) / 2.0).astype(int))
        cv2.arrowedLine(
            canvas,
            tip,
            direction_midpoint,
            (255, 255, 0),
            thickness,
            cv2.LINE_AA,
            tipLength=0.25,
        )
        cv2.circle(canvas, direction_midpoint, radius, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, direction_midpoint, radius, (255, 255, 0), thickness, cv2.LINE_AA)
    colors = {
        "P_top": (0, 0, 255),
        "P_left": (255, 80, 0),
        "P_right": (255, 0, 255),
        "P_index": (0, 0, 255),
        "P_middle": (0, 165, 255),
        "P_thumb": (255, 0, 255),
    }
    for name, point in points.items():
        cv2.circle(canvas, point, radius, colors[name], -1, cv2.LINE_AA)
        cv2.circle(canvas, point, radius, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(
            canvas,
            name,
            (point[0] + radius + 5, point[1] - radius - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            colors[name],
            thickness,
            cv2.LINE_AA,
        )
    triangle = np.asarray(list(points.values()), np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [triangle], True, (255, 255, 255), thickness, cv2.LINE_AA)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise OSError(f"Could not write visualization: {output}")


def main() -> None:
    args = parse_args()
    raw_mask = read_image(args.mask, cv2.IMREAD_GRAYSCALE, "mask")
    image = (
        read_image(args.image, cv2.IMREAD_COLOR, "RGB image") if args.image else None
    )
    aligned_mask, selected_transform, alignment_scores = align_mask(
        raw_mask, image, args.mask_transform
    )
    mask, contour = preprocess_mask(aligned_mask, args.morph_kernel)
    if args.mm_per_pixel is not None and args.mm_per_pixel <= 0:
        raise ValueError("--mm-per-pixel must be positive.")
    if args.pair_spacing_mm is not None and args.mm_per_pixel is None:
        raise ValueError("--pair-spacing-mm requires --mm-per-pixel.")
    if args.contact_inset_mm is not None and args.mm_per_pixel is None:
        raise ValueError("--contact-inset-mm requires --mm-per-pixel.")

    _, _, bbox_w, bbox_h = cv2.boundingRect(contour)
    if args.layout in ("triangle-paired", "paired-opposed"):
        side_dimension = bbox_h if args.paired_side in ("left", "right") else bbox_w
        pair_spacing_pixels = (
            args.pair_spacing_mm / args.mm_per_pixel
            if args.pair_spacing_mm is not None
            else args.pair_spacing_ratio * side_dimension
        )
        contact_inset_pixels = (
            args.contact_inset_mm / args.mm_per_pixel
            if args.contact_inset_mm is not None
            else args.contact_inset_ratio * min(bbox_w, bbox_h)
        )
        if args.layout == "triangle-paired":
            points_2d, diagnostics, _ = generate_triangle_paired_grasp(
                mask,
                contour,
                args.thumb_side,
                pair_spacing_pixels,
                contact_inset_pixels,
                args.safe_margin_ratio,
                args.min_spacing_ratio,
                args.min_area_ratio,
            )
        else:
            points_2d, diagnostics, _ = generate_paired_grasp(
                mask,
                contour,
                args.paired_side,
                pair_spacing_pixels,
                contact_inset_pixels,
                args.safe_margin_ratio,
                args.min_spacing_ratio,
                args.min_area_ratio,
            )
    else:
        points_2d, diagnostics, _ = generate_tripod_grasp(
            mask,
            contour,
            args.safe_margin_ratio,
            args.min_spacing_ratio,
            args.min_area_ratio,
        )
    diagnostics["mm_per_pixel"] = args.mm_per_pixel

    points_cam = None
    points_hand = None
    pair_center_2d = diagnostics.get("pair_center_2d")
    pair_center_cam = None
    pair_center_hand = None
    if bool(args.depth) != bool(args.intrinsics):
        raise ValueError("--depth and --intrinsics must be provided together.")
    if args.depth:
        depth = load_depth(args.depth, mask.shape)
        intrinsics = load_matrix(args.intrinsics, (3, 3), "intrinsics")
        points_cam = backproject_points(
            points_2d, depth, intrinsics, args.depth_scale, args.depth_radius
        )
        if pair_center_2d is not None:
            pair_center_cam = backproject_points(
                {"P_pair_center": tuple(pair_center_2d)},
                depth,
                intrinsics,
                args.depth_scale,
                args.depth_radius,
            )["P_pair_center"]
    if args.t_hand_cam:
        if points_cam is None:
            raise ValueError("--T-hand-cam requires --depth and --intrinsics.")
        t_hand_cam = load_matrix(args.t_hand_cam, (4, 4), "T_hand_cam")
        points_hand = transform_points(points_cam, t_hand_cam)
        if pair_center_cam is not None:
            pair_center_hand = transform_points(
                {"P_pair_center": pair_center_cam}, t_hand_cam
            )["P_pair_center"]

    result = {
        "grasp_points_2d": {
            name: [int(point[0]), int(point[1])]
            for name, point in points_2d.items()
        },
        "grasp_pair_center_2d": pair_center_2d,
        "grasp_points_cam": points_cam,
        "grasp_pair_center_cam": pair_center_cam,
        "grasp_points_hand": points_hand,
        "grasp_pair_center_hand": pair_center_hand,
        "diagnostics": {
            **diagnostics,
            "mask_transform": selected_transform,
            "alignment_scores": alignment_scores,
        },
    }
    if args.layout == "triangle-paired":
        ignored = np.asarray(diagnostics["ignored_tip_vertex"], dtype=np.float64)
        thumb = np.asarray(diagnostics["thumb_support_vertex"], dtype=np.float64)
        pair = np.asarray(diagnostics["pair_support_vertex"], dtype=np.float64)
        support_midpoint = (thumb + pair) / 2.0
        result["direction_line_2d"] = {
            "from_ignored_tip_vertex": ignored.tolist(),
            "to_support_midpoint": support_midpoint.tolist(),
            "vector": (support_midpoint - ignored).tolist(),
        }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=True)
        file.write("\n")
    draw_visualization(image, mask, contour, points_2d, diagnostics, args.output_vis)

    print(json.dumps(result, indent=2, ensure_ascii=True))
    print(f"Saved JSON: {output_json}")
    print(f"Saved visualization: {args.output_vis}")


if __name__ == "__main__":
    main()
