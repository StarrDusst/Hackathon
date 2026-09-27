"""Auto-calibrate a first-pass road polygon, traffic-flow corridors and signal ROIs.

Automatic output is a draft. Add crosswalks, stop lines and prohibited-manoeuvre rules
through the Streamlit calibration tab or by editing scene_calibration.json.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from solution import ANIMAL_IDS, MODEL_PATH, VEHICLE_IDS, _center, _iou, _predict, _update_tracks

ROOT = Path(__file__).resolve().parent


def _norm_box(box, width, height):
    x1, y1, x2, y2 = (float(v) for v in box)
    return [round(x1 / width, 5), round(y1 / height, 5), round(x2 / width, 5), round(y2 / height, 5)]


def _lane_candidates(tracks, width, height):
    buckets = defaultdict(list)
    for tr in tracks:
        if tr["class_id"] not in VEHICLE_IDS or len(tr["obs"]) < 4:
            continue
        obs = tr["obs"]
        p0 = _center(obs[0][1:5]); p1 = _center(obs[-1][1:5])
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        if math_hypot(dx, dy) < 70:
            continue
        if abs(dx) > 1.35 * abs(dy):
            axis, sign, coord = "horizontal", (1 if dx > 0 else -1), (p0[1] + p1[1]) / 2
            norm = height
        elif abs(dy) > 1.35 * abs(dx):
            axis, sign, coord = "vertical", (1 if dy > 0 else -1), (p0[0] + p1[0]) / 2
            norm = width
        else:
            continue
        band = round(coord / max(1.0, norm * 0.08))
        buckets[(axis, sign, band)].append((coord, tr))
    lanes = []
    for (axis, sign, _), rows in buckets.items():
        if len(rows) < 3:
            continue
        coords = [r[0] for r in rows]
        coord = float(np.median(coords))
        spread = max(float(np.percentile(coords, 90) - np.percentile(coords, 10)), (height if axis == "horizontal" else width) * 0.035)
        dim = height if axis == "horizontal" else width
        lo, hi = max(0.0, coord - spread / 2 - dim * 0.02), min(float(dim), coord + spread / 2 + dim * 0.02)
        if axis == "horizontal":
            polygon = [[0.0, round(lo / height, 5)], [1.0, round(lo / height, 5)],
                       [1.0, round(hi / height, 5)], [0.0, round(hi / height, 5)]]
            direction = [float(sign), 0.0]
        else:
            polygon = [[round(lo / width, 5), 0.0], [round(hi / width, 5), 0.0],
                       [round(hi / width, 5), 1.0], [round(lo / width, 5), 1.0]]
            direction = [0.0, float(sign)]
        lanes.append({"id": f"{axis}_{'positive' if sign > 0 else 'negative'}_{len(lanes)+1}",
                      "axis": axis, "sign": sign, "polygon_norm": polygon, "direction": direction,
                      "track_count": len(rows), "confidence": "draft-auto"})
    # Keep only the most frequently observed corridors per orientation/direction;
    # tiny bins are detector noise rather than useful lane calibration.
    compact = []
    for axis in ("horizontal", "vertical"):
        for sign in (-1, 1):
            selected = sorted([lane for lane in lanes if lane["axis"] == axis and lane["sign"] == sign],
                              key=lambda lane: lane["track_count"], reverse=True)[:3]
            compact.extend(selected)
    for i, lane in enumerate(compact, 1):
        lane["id"] = f"{lane['axis']}_{'positive' if lane['sign'] > 0 else 'negative'}_{i}"
    return compact


def math_hypot(x, y):
    return float(np.hypot(x, y))


def calibrate(video: Path, output: Path, sample_hz: float = 0.5) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, round(fps / sample_hz))
    tracks, completed, next_id, frame_i = [], [], 0, 0
    road_points, signal_tracks, stop_candidates = [], [], []
    first_frame = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if first_frame is None:
                first_frame = frame.copy()
            if frame_i % stride == 0:
                detections = _predict(frame, conf=0.15)
                next_id = _update_tracks(tracks, detections, frame_i / fps, next_id,
                                         max_gap=2.2, completed=completed)
                for det in detections:
                    x1, y1, x2, y2, conf, cls = det
                    if cls in VEHICLE_IDS:
                        road_points.append((int((x1 + x2) / 2), int(y2)))
            frame_i += 1
    finally:
        cap.release()
    duration = n_frames / fps if n_frames else frame_i / fps
    all_tracks = completed + tracks
    vehicle_tracks = [tr for tr in all_tracks if tr["class_id"] in VEHICLE_IDS]
    light_tracks = [tr for tr in all_tracks if tr["class_id"] == 9 and len(tr["obs"]) >= 2]
    stop_sign_tracks = [tr for tr in all_tracks if tr["class_id"] == 11 and len(tr["obs"]) >= 2]
    animal_tracks = [tr for tr in all_tracks if tr["class_id"] in ANIMAL_IDS and len(tr["obs"]) >= 2]

    road_polygon = []
    if len(road_points) >= 3:
        hull = cv2.convexHull(np.asarray(road_points, dtype=np.int32))
        eps = max(2.0, 0.004 * cv2.arcLength(hull, True))
        hull = cv2.approxPolyDP(hull, eps, True).reshape(-1, 2)
        road_polygon = [[round(float(x) / width, 5), round(float(y) / height, 5)] for x, y in hull]
    lanes = _lane_candidates(vehicle_tracks, width, height)

    traffic_lights = []
    for tr in light_tracks:
        boxes = np.asarray([o[1:5] for o in tr["obs"]], dtype=np.float32)
        box = np.median(boxes, axis=0)
        traffic_lights.append({"bbox_norm": _norm_box(box, width, height),
                               "detections": len(tr["obs"]), "confidence": "draft-auto", "state": "unknown"})
    stop_signs = []
    for tr in stop_sign_tracks:
        boxes = np.asarray([o[1:5] for o in tr["obs"]], dtype=np.float32)
        stop_signs.append({"bbox_norm": _norm_box(np.median(boxes, axis=0), width, height),
                           "detections": len(tr["obs"]), "confidence": "draft-auto"})
    def deduplicate(rows):
        kept = []
        for row in sorted(rows, key=lambda r: r["detections"], reverse=True):
            box = row["bbox_norm"]
            if not any(_iou(box, prior["bbox_norm"]) >= .45 for prior in kept):
                kept.append(row)
        return kept
    traffic_lights = deduplicate(traffic_lights)
    stop_signs = deduplicate(stop_signs)

    for tr in vehicle_tracks:
        obs = tr["obs"]
        if len(obs) >= 8:
            speeds = [math_hypot(_center(b[1:5])[0] - _center(a[1:5])[0],
                                 _center(b[1:5])[1] - _center(a[1:5])[1]) / max(.05, b[0] - a[0])
                      for a, b in zip(obs, obs[1:])]
            if np.median(speeds[-5:]) < 8 and obs[-1][0] - obs[max(0, len(obs)-5)][0] >= 1.5:
                cx, cy = _center(obs[-1][1:5])
                stop_candidates.append({"point_norm": [round(cx / width, 5), round(cy / height, 5)],
                                        "confidence": "review", "note": "recurrent/slow-vehicle sample; not a stop line"})
    # Deduplicate stopped-position proposals into a coarse grid.
    unique_stops = {}
    for candidate in stop_candidates:
        key = (round(candidate["point_norm"][0] / .03), round(candidate["point_norm"][1] / .03))
        unique_stops.setdefault(key, candidate)
    stop_candidates = list(unique_stops.values())[:50]

    config = {
        "version": 1, "status": "auto-draft-needs-visual-review",
        "reference_video": video.name, "reference_size": [width, height],
        "reference_fps": fps, "reference_duration_sec": round(duration, 3),
        "calibration_sample_hz": sample_hz,
        "road_polygon_norm": road_polygon,
        "lanes": lanes,
        "crosswalk_polygons_norm": [],
        "intersection_polygon_norm": [],
        "stop_lines": [],
        "solid_boundaries": [],
        "hazard_rois": [],
        "traffic_lights": traffic_lights,
        "stop_signs": stop_signs,
        "stopped_vehicle_zone_candidates": stop_candidates,
        "prohibited_turns": [],
        "u_turn_allowed": None,
        "manual_review_required": True,
        "notes": [
            "All geometry uses normalized [0,1] image coordinates.",
            "Road polygon and direction corridors are inferred from YOLO vehicle detections and require visual correction.",
            "Add crosswalk_polygons_norm, stop_lines and solid_boundaries manually before enabling those event rules.",
            "Traffic-light objects are not signal-state labels; red/green ROI colors also require review."
        ],
        "track_counts": {"vehicles": len(vehicle_tracks), "traffic_lights": len(light_tracks),
                         "stop_signs": len(stop_sign_tracks), "animals": len(animal_tracks)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2), encoding="utf-8")

    if first_frame is not None:
        overlay = cv2.resize(first_frame, (min(1280, width), round(height * min(1280, width) / width)))
        scale = overlay.shape[1] / width
        if road_polygon:
            pts = np.asarray([[int(x * width * scale), int(y * height * scale)] for x, y in road_polygon], np.int32)
            cv2.polylines(overlay, [pts], True, (0, 255, 255), 3)
            layer = overlay.copy(); cv2.fillPoly(layer, [pts], (0, 255, 255)); overlay = cv2.addWeighted(layer, .12, overlay, .88, 0)
        for lane in lanes:
            pts = np.asarray([[int(x * width * scale), int(y * height * scale)] for x, y in lane["polygon_norm"]], np.int32)
            cv2.polylines(overlay, [pts], True, (255, 70, 20), 2)
            cv2.putText(overlay, lane["id"], tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 70, 20), 2)
        for light in traffic_lights:
            x1, y1, x2, y2 = light["bbox_norm"]
            cv2.rectangle(overlay, (int(x1*width*scale), int(y1*height*scale)),
                          (int(x2*width*scale), int(y2*height*scale)), (0, 0, 255), 2)
        for candidate in stop_candidates:
            x, y = candidate["point_norm"]
            cv2.circle(overlay, (int(x*width*scale), int(y*height*scale)), 6, (0, 140, 255), -1)
        cv2.putText(overlay, "AUTO DRAFT: road yellow, flow blue, signals red, stop candidates orange", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2, cv2.LINE_AA)
        overlay_dir = ROOT / "reports"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = overlay_dir / "calibration_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay)
        config["overlay_image"] = str(overlay_path.name)
        output.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=str(ROOT.parent / "samples" / "C3896.MP4"))
    parser.add_argument("--out", default=str(ROOT / "scene_calibration.json"))
    parser.add_argument("--hz", type=float, default=0.5)
    args = parser.parse_args()
    if not MODEL_PATH.is_file():
        raise SystemExit("Missing weights; run python weights/download.py first")
    config = calibrate(Path(args.video), Path(args.out), max(.2, args.hz))
    print(f"Wrote calibration draft: {args.out}")
    print("road polygon points:", len(config["road_polygon_norm"]),
          "lane corridors:", len(config["lanes"]), "traffic lights:", len(config["traffic_lights"]),
          "stop zones:", len(config["stopped_vehicle_zone_candidates"]))
    print("Review calibration_overlay.jpg and correct scene_calibration.json before trusting rule events.")


if __name__ == "__main__":
    main()
