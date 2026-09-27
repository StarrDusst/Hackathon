"""Load and use normalized fixed-camera scene calibration."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

CONFIG_PATH = Path(__file__).resolve().parent / "scene_calibration.json"


def load_calibration(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path else CONFIG_PATH
    if not config_path.is_file():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("version") == 1:
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def polygon_pixels(points: list, width: int, height: int) -> np.ndarray:
    return np.asarray([[round(float(x) * width), round(float(y) * height)] for x, y in points], dtype=np.int32)


def point_in_polygon(point: tuple[float, float], points: list, width: int, height: int) -> bool:
    if not points or len(points) < 3:
        return False
    poly = polygon_pixels(points, width, height)
    return cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0


def boxes_intersect_polygon(box: tuple, points: list, width: int, height: int) -> bool:
    if not points:
        return False
    foot = ((box[0] + box[2]) / 2.0, box[3])
    center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
    return point_in_polygon(foot, points, width, height) or point_in_polygon(center, points, width, height)


def segment_side(point: tuple[float, float], line: list[list[float]], width: int, height: int) -> float:
    if len(line) != 2:
        return 0.0
    (x1, y1), (x2, y2) = polygon_pixels(line, width, height).astype(float)
    return (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)


def line_crosses_segment(a: tuple[float, float], b: tuple[float, float], line: list,
                         width: int, height: int) -> bool:
    s1 = segment_side(a, line, width, height)
    s2 = segment_side(b, line, width, height)
    if s1 * s2 >= 0:
        return False
    (x1, y1), (x2, y2) = polygon_pixels(line, width, height).astype(float)
    dx, dy = b[0] - a[0], b[1] - a[1]
    lx, ly = x2 - x1, y2 - y1
    denom = dx * ly - dy * lx
    if abs(denom) < 1e-6:
        return False
    u = ((x1 - a[0]) * ly - (y1 - a[1]) * lx) / denom
    v = ((x1 - a[0]) * dy - (y1 - a[1]) * dx) / denom
    return 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0
