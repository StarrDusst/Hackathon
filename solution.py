"""Offline fixed-camera traffic-event baseline using shipped Ultralytics YOLOv8n weights.

Object detection is learned (COCO-pretrained YOLOv8n). Event boundaries and risk are
rule-based over causal, sampled detections/tracks. No hosted API or runtime download
is used: weights must already exist in ``weights/yolov8n.pt``.
"""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path

import numpy as np

from calibration import boxes_intersect_polygon, line_crosses_segment, load_calibration, point_in_polygon, segment_side

CLASSES: list[str] = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
]
RISK_HORIZON_SEC = 5.0
MODEL_PATH = Path(__file__).resolve().parent / "weights" / "yolov8n.pt"
COCO_TRAFFIC_IDS = (0, 1, 2, 3, 5, 7, 9, 10, 11, 14, 15, 16, 17, 18, 19)
VEHICLE_IDS = {2, 3, 5, 7}
ANIMAL_IDS = {14, 15, 16, 17, 18, 19}
_PERSON_ID = 0
_TRAFFIC_LIGHT_ID = 9
_STOP_SIGN_ID = 11
_MODEL = None
_DEVICE = None


def _get_model():
    """Load only the local shipped checkpoint; never download weights implicitly."""
    global _MODEL, _DEVICE
    if _MODEL is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Missing pretrained weights: {MODEL_PATH}. Run `python weights/download.py` first."
            )
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "YOLO inference dependencies are missing. Install requirements.txt (or run setup.ps1)."
            ) from exc
        _DEVICE = 0 if torch.cuda.is_available() else "cpu"
        _MODEL = YOLO(str(MODEL_PATH))
    return _MODEL


def _predict(frame: np.ndarray, conf: float = 0.20) -> list[tuple[float, float, float, float, float, int]]:
    """Run one local YOLO inference; boxes are xyxy in source-frame pixels."""
    model = _get_model()
    result = model.predict(
        source=frame, imgsz=640, conf=conf, iou=0.50, classes=list(COCO_TRAFFIC_IDS),
        max_det=100, device=_DEVICE, verbose=False,
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    labels = result.boxes.cls.detach().cpu().numpy().astype(int)
    return [(float(x1), float(y1), float(x2), float(y2), float(score), int(label))
            for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels)]


def _center(box: tuple) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _iou(a: tuple, b: tuple) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + ab - inter) if aa + ab > inter else 0.0


def _new_track(track_id: int, t: float, det: tuple) -> dict:
    return {"id": track_id, "class_id": det[5], "last_t": t,
            "obs": [(t, *det[:5])], "class_votes": Counter({det[5]: 1})}


def _update_tracks(tracks: list[dict], detections: list[tuple], t: float,
                   next_id: int, max_gap: float = 2.0,
                   completed: list[dict] | None = None) -> int:
    """Greedy IoU/constant-velocity matcher suitable for 1–6 Hz sampled frames."""
    eligible = [tr for tr in tracks if t - tr["last_t"] <= max_gap]
    pairs = []
    for ti, tr in enumerate(eligible):
        last = tr["obs"][-1]
        predicted = list(last[1:5])
        if len(tr["obs"]) >= 2:
            prev = tr["obs"][-2]
            dt = max(0.05, last[0] - prev[0])
            lead = min(max(0.0, t - last[0]), 1.0)
            vx = ((_center(last[1:5])[0] - _center(prev[1:5])[0]) / dt) * lead
            vy = ((_center(last[1:5])[1] - _center(prev[1:5])[1]) / dt) * lead
            predicted[0] += vx; predicted[2] += vx
            predicted[1] += vy; predicted[3] += vy
        pcx, pcy = _center(predicted)
        pw, ph = predicted[2] - predicted[0], predicted[3] - predicted[1]
        for di, det in enumerate(detections):
            # Keep identity within compatible COCO object groups.
            old_vehicle = tr["class_id"] in VEHICLE_IDS
            new_vehicle = det[5] in VEHICLE_IDS
            if old_vehicle != new_vehicle or (not old_vehicle and tr["class_id"] != det[5]):
                continue
            cx, cy = _center(det[:4])
            dist = math.hypot(cx - pcx, cy - pcy)
            radius = max(55.0, 0.8 * math.hypot(pw, ph))
            overlap = _iou(predicted, det[:4])
            if dist < radius or overlap > 0.01:
                pairs.append((dist / radius + 0.4 * (1.0 - overlap), ti, di))
    used_t, used_d = set(), set()
    for _, ti, di in sorted(pairs):
        if ti in used_t or di in used_d:
            continue
        tr, det = eligible[ti], detections[di]
        tr["obs"].append((t, *det[:5]))
        tr["obs"] = tr["obs"][-80:]
        tr["last_t"] = t
        tr["class_votes"][det[5]] += 1
        tr["class_id"] = tr["class_votes"].most_common(1)[0][0]
        used_t.add(ti); used_d.add(di)
    for di, det in enumerate(detections):
        if di not in used_d:
            tracks.append(_new_track(next_id, t, det))
            next_id += 1
    stale = [tr for tr in tracks if t - tr["last_t"] > max_gap]
    if completed is not None:
        completed.extend(stale)
    tracks[:] = [tr for tr in tracks if t - tr["last_t"] <= max_gap]
    return next_id


def _track_speed(track: dict, idx: int) -> float:
    obs = track["obs"]
    if idx <= 0 or idx >= len(obs):
        return 0.0
    a, b = obs[idx - 1], obs[idx]
    dt = max(0.05, b[0] - a[0])
    ax, ay = _center(a[1:5]); bx, by = _center(b[1:5])
    return math.hypot(bx - ax, by - ay) / dt


def _stopped_segments(track: dict) -> list[tuple[float, float]]:
    """Require a moving vehicle to become stationary for the official >=10 s rule."""
    obs = track["obs"]
    if len(obs) < 8:
        return []
    result, anchor = [], 0
    stationary_start = None
    for i in range(1, len(obs)):
        ax, ay = _center(obs[anchor][1:5])
        x, y = _center(obs[i][1:5])
        elapsed = obs[i][0] - obs[anchor][0]
        if math.hypot(x - ax, y - ay) > 28.0:
            if stationary_start is not None and obs[i][0] - stationary_start >= 10.0:
                result.append((stationary_start, obs[i][0]))
            anchor, stationary_start = i, None
        elif stationary_start is None and elapsed >= 1.5:
            # Avoid a static object that was never observed moving.
            moved = max(math.hypot(_center(o[1:5])[0] - _center(obs[0][1:5])[0],
                                   _center(o[1:5])[1] - _center(obs[0][1:5])[1]) for o in obs[1:anchor + 1]) if anchor else 0.0
            if moved >= 35.0:
                stationary_start = obs[anchor][0]
    end = obs[-1][0]
    if stationary_start is None and end - obs[anchor][0] >= 1.5:
        moved = max(math.hypot(_center(o[1:5])[0] - _center(obs[0][1:5])[0],
                               _center(o[1:5])[1] - _center(obs[0][1:5])[1]) for o in obs[1:anchor + 1]) if anchor else 0.0
        if moved >= 35.0:
            stationary_start = obs[anchor][0]
    if stationary_start is not None and end - stationary_start >= 10.0:
        result.append((stationary_start, end))
    return result


def _light_color(frame: np.ndarray, roi_norm: list, width: int, height: int) -> str:
    """Classify red/green pixels in a manually reviewed traffic-signal crop."""
    import cv2
    if not roi_norm or len(roi_norm) != 4:
        return "unknown"
    x, y, w, h = roi_norm
    x1, y1 = max(0, int(x * width)), max(0, int(y * height))
    x2, y2 = min(width, int((x + w) * width)), min(height, int((y + h) * height))
    if x2 <= x1 or y2 <= y1:
        return "unknown"
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "unknown"
    crop = cv2.resize(crop, (max(20, crop.shape[1] * 3), max(20, crop.shape[0] * 3)))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, np.array([0, 90, 100]), np.array([12, 255, 255]))
    red |= cv2.inRange(hsv, np.array([168, 90, 100]), np.array([180, 255, 255]))
    green = cv2.inRange(hsv, np.array([35, 70, 90]), np.array([95, 255, 255]))
    nr, ng = int(np.count_nonzero(red)), int(np.count_nonzero(green))
    if nr >= 3 and nr >= 1.4 * max(1, ng):
        return "red"
    if ng >= 3 and ng >= 1.4 * max(1, nr):
        return "green"
    return "unknown"


def _distance_to_line(point: tuple[float, float], line: list, width: int, height: int) -> float:
    if len(line) != 2:
        return float("inf")
    ax, ay = float(line[0][0]) * width, float(line[0][1]) * height
    bx, by = float(line[1][0]) * width, float(line[1][1]) * height
    dx, dy = bx - ax, by - ay
    denom = dx*dx + dy*dy
    if denom < 1e-6:
        return float("inf")
    t = min(1.0, max(0.0, ((point[0]-ax)*dx + (point[1]-ay)*dy) / denom))
    return math.hypot(point[0] - (ax + t*dx), point[1] - (ay + t*dy))


def _hazard_is_visible(frame: np.ndarray, roi_norm: list, width: int, height: int,
                       kind: str, threshold: float = 0.04) -> bool:
    """Low-confidence color cue, active only inside an operator-reviewed hazard ROI."""
    import cv2
    if len(roi_norm) != 4:
        return False
    x, y, rw, rh = roi_norm
    x1, y1 = max(0, int(x*width)), max(0, int(y*height))
    x2, y2 = min(width, int((x+rw)*width)), min(height, int((y+rh)*height))
    if x2 <= x1 or y2 <= y1:
        return False
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    if kind == "fire":
        mask = cv2.inRange(hsv, np.array([0, 100, 170]), np.array([38, 255, 255]))
        mask |= cv2.inRange(hsv, np.array([170, 100, 170]), np.array([180, 255, 255]))
        return float(np.mean(mask > 0)) >= threshold
    if kind == "smoke":
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        low_sat = cv2.inRange(hsv, np.array([0, 0, 75]), np.array([180, 75, 235]))
        fraction = float(np.mean(low_sat > 0))
        return fraction >= max(.72, threshold) and float(np.std(gray)) >= 9.0
    return False


def _signal_is_red(signal_samples: list[dict], roi: list | None, t: float) -> bool:
    if not roi or not signal_samples:
        return False
    cx, cy = roi[0] + roi[2] / 2, roi[1] + roi[3] / 2
    candidates = []
    for row in signal_samples:
        r = row["roi"]
        rcx, rcy = r[0] + r[2] / 2, r[1] + r[3] / 2
        dist = math.hypot(cx - rcx, cy - rcy)
        if abs(row["t"] - t) <= 1.0 and dist <= 0.08:
            candidates.append((abs(row["t"] - t) + dist, row["state"]))
    return bool(candidates and min(candidates)[1] == "red")


def _geometry_events(tracks: list[dict], duration: float, shape: tuple[int, ...],
                     config: dict, signal_samples: list[dict], hazard_samples: list[dict]) -> list[list]:
    """Use reviewed road polygons, lane directions, crossings and virtual lines."""
    h, w = shape[:2]
    events: list[list] = []
    vehicles = [tr for tr in tracks if tr["class_id"] in VEHICLE_IDS and len(tr["obs"]) >= 3]
    people = [tr for tr in tracks if tr["class_id"] == _PERSON_ID and len(tr["obs"]) >= 3]
    road = config.get("road_polygon_norm", [])
    crosswalks = config.get("crosswalk_polygons_norm", [])
    intersection = config.get("intersection_polygon_norm", [])

    # Lane-relative reverse travel requires the operator to clear the draft flag
    # after reviewing/correcting the estimated corridors; raw auto lanes are unsafe.
    for tr in vehicles:
        if config.get("manual_review_required", True):
            break
        obs = tr["obs"]
        p0, p1 = _center(obs[0][1:5]), _center(obs[-1][1:5])
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        if math.hypot(dx, dy) < 90 or obs[-1][0] - obs[0][0] < 3:
            continue
        covered = 0
        candidates = []
        for lane in config.get("lanes", []):
            if lane.get("track_count", 0) < 5 or not lane.get("polygon_norm") or not lane.get("direction"):
                continue
            inside = sum(point_in_polygon(_center(o[1:5]), lane["polygon_norm"], w, h) for o in obs)
            if inside / len(obs) >= 0.6:
                direction = lane["direction"]
                if (abs(direction[0]) > 0 and abs(dy) > .45 * abs(dx)) or (abs(direction[1]) > 0 and abs(dx) > .45 * abs(dy)):
                    continue
                projection = dx * direction[0] + dy * direction[1]
                if projection < -max(45.0, .55 * math.hypot(dx, dy)):
                    candidates.append(lane)
        if candidates:
            events.append([obs[0][0], min(duration, obs[-1][0] + 0.5), "wrong_way"])

    # Calibrated illegal-crossing segments require both the road footprint and at
    # least one reviewed crosswalk polygon; otherwise legality is unknown, so emit nothing.
    if road and crosswalks:
        for tr in people:
            obs = tr["obs"]
            run = []
            def finish_run(points):
                if len(points) < 3 or points[-1][0] - points[0][0] < 1.0 or points[-1][0] - points[0][0] > 14.0:
                    return
                centers = [_center(o[1:5]) for o in points]
                span_x = max(p[0] for p in centers) - min(p[0] for p in centers)
                span_y = max(p[1] for p in centers) - min(p[1] for p in centers)
                if max(span_x, span_y) < max(35.0, .035 * max(h, w)):
                    return
                in_crosswalk = 0
                for point in centers:
                    if any(point_in_polygon(point, poly, w, h) for poly in crosswalks):
                        in_crosswalk += 1
                if crosswalks and in_crosswalk / len(centers) >= .5:
                    return
                events.append([points[0][0], min(duration, points[-1][0] + .5), "jaywalking"])
            for o in obs:
                foot = ((o[1] + o[3]) / 2, o[4])
                if point_in_polygon(foot, road, w, h):
                    if run and o[0] - run[-1][0] > 1.25:
                        finish_run(run); run = []
                    run.append(o)
                elif run:
                    finish_run(run); run = []
            if run:
                finish_run(run)

    # A calibrated crossing polygon plus close vehicle passage without braking is a
    # failure-to-yield candidate; no prediction is made if no crosswalk was reviewed.
    for crosswalk in crosswalks:
        people_in = [(tr, o) for tr in people for o in tr["obs"]
                     if point_in_polygon(_center(o[1:5]), crosswalk, w, h)]
        for person_track, person_obs in people_in:
            pc = _center(person_obs[1:5])
            for vehicle in vehicles:
                close = [o for o in vehicle["obs"] if abs(o[0] - person_obs[0]) <= .6
                         and math.dist(_center(o[1:5]), pc) < max(45.0, .7 * math.hypot(o[3]-o[1], o[4]-o[2]))]
                if not close:
                    continue
                obs = vehicle["obs"]
                ix = min(range(len(obs)), key=lambda i: abs(obs[i][0] - close[0][0]))
                before = _track_speed(vehicle, max(1, ix))
                after = _track_speed(vehicle, min(len(obs)-1, ix+1))
                if before > 15 and after > .75 * before:
                    events.append([max(0.0, person_obs[0] - .5), min(duration, person_obs[0] + 1.5), "failure_to_yield"])

    # Scene-specific stop-line and red-signal rules. Format and setup are described
    # in scene_calibration.json; a red-light event is impossible without this geometry.
    for line_cfg in config.get("stop_lines", []):
        line = line_cfg.get("points_norm", [])
        roi = line_cfg.get("signal_roi_norm")
        approach_sign = float(line_cfg.get("approach_sign", 1.0))
        if len(line) != 2 or not roi:
            continue
        for tr in vehicles:
            obs = tr["obs"]
            for a, b in zip(obs, obs[1:]):
                ca, cb = _center(a[1:5]), _center(b[1:5])
                if not line_crosses_segment(ca, cb, line, w, h):
                    continue
                if segment_side(ca, line, w, h) * approach_sign <= 0:
                    continue
                if _signal_is_red(signal_samples, roi, b[0]):
                    events.append([max(0.0, b[0] - .5), min(duration, b[0] + 3.0), "red_light"])
            # Stationary beyond the stop line while its configured signal is red.
            for s, e in _stopped_segments(tr):
                for o in obs:
                    if not s <= o[0] <= e:
                        continue
                    center = _center(o[1:5])
                    if segment_side(center, line, w, h) * approach_sign < 0 and _signal_is_red(signal_samples, roi, o[0]):
                        events.append([s, min(duration, e + .5), "stop_line"])
                        break

    # Crossings of configured solid-boundary polylines are directly testable once
    # the road marking has been drawn in calibration.
    for boundary in config.get("solid_boundaries", []):
        points = boundary if isinstance(boundary, list) else boundary.get("points_norm", [])
        if len(points) < 2:
            continue
        for tr in vehicles:
            for a, b in zip(tr["obs"], tr["obs"][1:]):
                ca, cb = _center(a[1:5]), _center(b[1:5])
                if any(line_crosses_segment(ca, cb, [p0, p1], w, h) for p0, p1 in zip(points, points[1:])):
                    events.append([a[0], min(duration, b[0] + .5), "solid_line_crossing"])

    # Prohibited turns are only emitted when explicitly declared in calibration;
    # trajectory shape alone cannot determine legal vs illegal intent.
    for tr in vehicles:
        obs = tr["obs"]
        if len(obs) < 8:
            continue
        first, last = _center(obs[0][1:5]), _center(obs[-1][1:5])
        start_heading = _center(obs[2][1:5])
        end_heading = _center(obs[-3][1:5])
        va = (start_heading[0] - first[0], start_heading[1] - first[1])
        vb = (last[0] - end_heading[0], last[1] - end_heading[1])
        norm_a, norm_b = math.hypot(*va), math.hypot(*vb)
        if norm_a < 30 or norm_b < 30:
            continue
        angle = math.degrees(math.atan2(va[0]*vb[1] - va[1]*vb[0], va[0]*vb[0] + va[1]*vb[1]))
        angle_abs = abs(angle)
        turn = "u_turn" if angle_abs >= 140 else ("right" if angle > 35 else "left" if angle < -35 else "straight")
        path = [_center(o[1:5]) for o in obs]
        in_intersection = (not intersection or any(point_in_polygon(p, intersection, w, h) for p in path))
        if turn == "u_turn" and config.get("u_turn_allowed") is False and in_intersection:
            events.append([obs[0][0], min(duration, obs[-1][0] + .5), "illegal_u_turn"])
        for rule in config.get("prohibited_turns", []):
            poly = rule.get("polygon_norm", intersection)
            if rule.get("allowed") is False and rule.get("turn") == turn and poly and any(point_in_polygon(p, poly, w, h) for p in path):
                label = "illegal_u_turn" if turn == "u_turn" else "illegal_turn"
                events.append([obs[0][0], min(duration, obs[-1][0] + .5), label])

    # Visible COCO animals persistently on the calibrated carriageway can be an
    # obstacle; ordinary parked cars and unclassified debris are not guessed.
    if road:
        for tr in tracks:
            if tr["class_id"] not in ANIMAL_IDS or len(tr["obs"]) < 5:
                continue
            obs = tr["obs"]
            if obs[-1][0] - obs[0][0] < 2.0:
                continue
            speed = _track_speed(tr, len(obs)-1)
            if speed < 6 and any(boxes_intersect_polygon(o[1:5], road, w, h) for o in obs):
                events.append([obs[0][0], min(duration, obs[-1][0] + .5), "road_obstacle"])

    # Fire/smoke pixels are evaluated only in explicitly reviewed ROIs and must
    # persist across several samples; this remains an appearance heuristic.
    for hazard in config.get("hazard_rois", []):
        roi, kind = hazard.get("roi_norm"), hazard.get("kind")
        if not roi or kind not in {"fire", "smoke"}:
            continue
        active = sorted(row["t"] for row in hazard_samples
                        if row["kind"] == kind and tuple(row["roi"]) == tuple(roi) and row["active"])
        groups = []
        for t in active:
            if not groups or t - groups[-1][-1] > 1.25:
                groups.append([t])
            else:
                groups[-1].append(t)
        for group in groups:
            if len(group) >= 3 and group[-1] - group[0] >= 1.0:
                events.append([group[0], min(duration, group[-1] + .5), "fire_smoke"])
    return events


def _infer_events(tracks: list[dict], duration: float, frame_shape: tuple[int, ...],
                  config: dict | None = None, signal_samples: list[dict] | None = None,
                  hazard_samples: list[dict] | None = None) -> list[list]:
    """Causal-agnostic Part A rules over detector tracks, then clip/merge segments."""
    events = []
    config = config or load_calibration()
    signal_samples = signal_samples or []
    height, width = frame_shape[:2]
    vehicles = [tr for tr in tracks if tr["class_id"] in VEHICLE_IDS and len(tr["obs"]) >= 5]
    people = [tr for tr in tracks if tr["class_id"] == _PERSON_ID and len(tr["obs"]) >= 4]

    for tr in vehicles:
        for s, e in _stopped_segments(tr):
            queued_at_red = False
            for line_cfg in config.get("stop_lines", []):
                line, roi = line_cfg.get("points_norm", []), line_cfg.get("signal_roi_norm")
                if len(line) != 2 or not roi:
                    continue
                for o in tr["obs"]:
                    if s <= o[0] <= e and _distance_to_line(_center(o[1:5]), line, width, height) < max(45, .025*max(width,height)):
                        queued_at_red |= _signal_is_red(signal_samples, roi, o[0])
            if not queued_at_red:
                events.append([s, min(duration, e + 0.5), "stopped_vehicle"])

    # Congestion candidate: several distinct vehicles remain nearly stationary together.
    # A long isolated stop is left to stopped_vehicle; a queue is conservative and requires 6+.
    times = sorted({round(o[0], 2) for tr in vehicles for o in tr["obs"]})
    congested = []
    for t in times:
        active = 0
        for tr in vehicles:
            obs = [o for o in tr["obs"] if abs(o[0] - t) <= 0.6]
            if len(obs) >= 2 and _track_speed(tr, tr["obs"].index(obs[-1])) < 8.0:
                active += 1
        if active >= 6:
            congested.append(t)
    if congested:
        groups = [[congested[0]]]
        for t in congested[1:]:
            (groups[-1] if t - groups[-1][-1] <= 1.5 else groups.append([t]))
        for g in groups:
            if g[-1] - g[0] >= 8.0:
                events.append([g[0], min(duration, g[-1] + 0.5), "congestion"])

    # Interaction candidates at near-overlapping boxes; only label collision when
    # the detections overlap and both tracked vehicles abruptly slow afterwards.
    for i, a in enumerate(vehicles):
        for b in vehicles[i + 1:]:
            if (a["obs"][0][0] > b["obs"][-1][0] + 0.5
                    or b["obs"][0][0] > a["obs"][-1][0] + 0.5):
                continue
            matches = []
            j = 0
            for oa in a["obs"]:
                while j + 1 < len(b["obs"]) and b["obs"][j + 1][0] <= oa[0] + 0.35:
                    j += 1
                if j < len(b["obs"]) and abs(b["obs"][j][0] - oa[0]) <= 0.35:
                    ob = b["obs"][j]
                    ca, cb = _center(oa[1:5]), _center(ob[1:5])
                    scale = .5 * ((oa[3]-oa[1]) + (ob[3]-ob[1]))
                    matches.append((max(oa[0], ob[0]), _iou(oa[1:5], ob[1:5]),
                                    math.hypot(ca[0]-cb[0], ca[1]-cb[1]), scale))
            if not matches:
                continue
            # Near miss: clear approach and separation with no box contact.
            k = min(range(len(matches)), key=lambda idx: matches[idx][2])
            if 1 < k < len(matches) - 2:
                tm, overlap_m, dmin, scale = matches[k]
                approach = matches[k-2][2] - dmin
                separate = matches[k+2][2] - dmin
                ia = min(range(len(a["obs"])), key=lambda n: abs(a["obs"][n][0] - tm))
                ib = min(range(len(b["obs"])), key=lambda n: abs(b["obs"][n][0] - tm))
                va, vb = _track_speed(a, ia), _track_speed(b, ib)
                if (overlap_m < .01 and dmin < max(28.0, .38 * scale)
                        and approach > max(12.0, .18 * scale)
                        and separate > max(12.0, .18 * scale) and va > 18 and vb > 18):
                    events.append([max(0.0, tm - .75), min(duration, tm + 1.25), "near_miss"])
            t_contact, overlap, _, _ = max(matches, key=lambda item: item[1])
            if overlap < 0.12:
                continue
            ia = min(range(len(a["obs"])), key=lambda idx: abs(a["obs"][idx][0] - t_contact))
            ib = min(range(len(b["obs"])), key=lambda idx: abs(b["obs"][idx][0] - t_contact))
            after_a = _track_speed(a, min(ia + 1, len(a["obs"]) - 1))
            after_b = _track_speed(b, min(ib + 1, len(b["obs"]) - 1))
            before_a = _track_speed(a, max(1, ia))
            before_b = _track_speed(b, max(1, ib))
            if before_a > 15 and before_b > 15 and after_a < before_a * 0.55 and after_b < before_b * 0.55:
                events.append([max(0.0, t_contact - 0.5), min(duration, t_contact + 3.0), "accident"])

    events.extend(_geometry_events(tracks, duration, frame_shape, config, signal_samples, hazard_samples or []))
    clipped = [[max(0.0, float(s)), min(duration, float(e)), label]
               for s, e, label in events if 0 <= s < e and label in CLASSES]
    return _merge_events(clipped)


def _merge_events(events: list[list], gap: float = 0.75) -> list[list]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for s, e, label in events:
        if label in CLASSES and math.isfinite(s) and math.isfinite(e) and s < e:
            grouped.setdefault(label, []).append((float(s), float(e)))
    result = []
    for label, spans in grouped.items():
        spans.sort()
        merged = []
        for s, e in spans:
            if merged and s <= merged[-1][1] + gap:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        result.extend([[round(s, 3), round(e, 3), label] for s, e in merged if e > s])
    return sorted(result, key=lambda event: (event[0], event[1], event[2]))


def _signal_observations(frame: np.ndarray, detections: list[tuple], config: dict, t: float) -> list[dict]:
    h, w = frame.shape[:2]
    rois = []
    for item in config.get("traffic_lights", []):
        roi = item.get("signal_roi_norm")
        if roi:
            rois.append(roi)
        elif item.get("bbox_norm"):
            x1, y1, x2, y2 = item["bbox_norm"]
            cx, cy, bw, bh = (x1+x2)/2, (y1+y2)/2, max(.01, (x2-x1)*2.5), max(.02, (y2-y1)*2.5)
            rois.append([max(0,cx-bw/2), max(0,cy-bh/2), min(1,bw), min(1,bh)])
    for item in config.get("stop_lines", []):
        if item.get("signal_roi_norm"):
            rois.append(item["signal_roi_norm"])
    for det in detections:
        if det[5] == _TRAFFIC_LIGHT_ID:
            x1,y1,x2,y2 = det[:4]
            cx,cy = (x1+x2)/(2*w),(y1+y2)/(2*h)
            bw,bh = max(.01,2.5*(x2-x1)/w),max(.02,2.5*(y2-y1)/h)
            rois.append([max(0,cx-bw/2),max(0,cy-bh/2),min(1,bw),min(1,bh)])
    unique = {}
    for roi in rois:
        key = tuple(round(float(v),3) for v in roi)
        unique[key] = roi
    return [{"t":float(t),"roi":list(roi),"state":_light_color(frame,roi,w,h)} for roi in unique.values()]


def _hazard_observations(frame: np.ndarray, config: dict, t: float) -> list[dict]:
    h, w = frame.shape[:2]
    rows = []
    for hazard in config.get("hazard_rois", []):
        roi, kind = hazard.get("roi_norm"), hazard.get("kind")
        if roi and kind in {"fire", "smoke"}:
            rows.append({"t":float(t), "roi":list(roi), "kind":kind,
                         "active":_hazard_is_visible(frame,roi,w,h,kind,float(hazard.get("min_fraction",.04)))})
    return rows


def detect_events(video_path: str, progress_callback=None) -> list[list]:
    """Sample the video at 2 Hz, detect COCO road users, track and classify events."""
    import cv2
    config = load_calibration()
    signal_samples = []
    hazard_samples = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = n_frames / fps if n_frames else 0.0
    stride = max(1, round(fps / 2.0))
    tracks: list[dict] = []
    completed: list[dict] = []
    next_id = 0
    frame_i = 0
    shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3)
    try:
        while True:
            if frame_i % stride == 0:
                ok, frame = cap.read()
            else:
                ok, frame = cap.grab(), None
            if not ok:
                break
            if frame is not None:
                t = frame_i / fps
                dets = _predict(frame)
                signal_samples.extend(_signal_observations(frame, dets, config, t))
                hazard_samples.extend(_hazard_observations(frame, config, t))
                next_id = _update_tracks(tracks, dets, t, next_id, completed=completed)
                shape = frame.shape
            frame_i += 1
            if progress_callback is not None and frame_i % max(1, round(fps * 5)) == 0:
                progress_callback(min(.99, frame_i / max(1, n_frames)), frame_i / fps)
    finally:
        cap.release()
    if duration <= 0:
        duration = frame_i / fps if fps else 0.0
    return _infer_events(completed + tracks, duration, shape, config, signal_samples, hazard_samples)


class RiskEstimator:
    """Causal risk score using only prior YOLO detections and track velocities."""
    def reset(self, meta: dict) -> None:
        self.meta = dict(meta)
        self.fps = max(1.0, float(meta.get("fps", 25.0)))
        self.stride = max(1, round(self.fps / 3.0))
        self.index = 0
        self.tracks: list[dict] = []
        self.next_id = 0
        self.score = 0.01
        self.last_t = 0.0

    def step(self, frame: np.ndarray | None, t_sec: float) -> float:
        """Advance one source frame; only sampled frames need decoded pixels."""
        self.index += 1
        if self.index % self.stride or frame is None:
            return float(self.score)
        detections = _predict(frame, conf=0.18)
        self.next_id = _update_tracks(self.tracks, detections, float(t_sec), self.next_id, max_gap=1.5)
        risk = 0.01
        vehicles = [tr for tr in self.tracks if tr["class_id"] in VEHICLE_IDS and len(tr["obs"]) >= 3]
        speeds = []
        for tr in vehicles:
            obs = tr["obs"]
            v_now = _track_speed(tr, len(obs) - 1)
            v_old = _track_speed(tr, len(obs) - 2)
            x, y = _center(obs[-1][1:5])
            speeds.append((x, y, v_now, tr))
            if v_old > 18.0 and v_now < 0.45 * v_old:
                risk = max(risk, min(0.82, 0.30 + (v_old - v_now) / 100.0))
        for i, (x1, y1, v1, tr1) in enumerate(speeds):
            p1, q1 = tr1["obs"][-2], tr1["obs"][-1]
            dt1 = max(0.05, q1[0] - p1[0])
            c0 = _center(p1[1:5]); vx1, vy1 = (x1 - c0[0]) / dt1, (y1 - c0[1]) / dt1
            for x2, y2, v2, tr2 in speeds[i + 1:]:
                p2, q2 = tr2["obs"][-2], tr2["obs"][-1]
                dt2 = max(0.05, q2[0] - p2[0])
                c0b = _center(p2[1:5]); vx2, vy2 = (x2 - c0b[0]) / dt2, (y2 - c0b[1]) / dt2
                rx, ry, rvx, rvy = x2 - x1, y2 - y1, vx2 - vx1, vy2 - vy1
                vv = rvx * rvx + rvy * rvy
                if vv < 1.0:
                    continue
                ttc = -(rx * rvx + ry * rvy) / vv
                if not 0.0 < ttc <= RISK_HORIZON_SEC:
                    continue
                closest = math.hypot(rx + rvx * ttc, ry + rvy * ttc)
                safe_dist = max(20.0, 0.35 * math.hypot(
                    q1[3] - q1[1] + q2[3] - q2[1], q1[4] - q1[2] + q2[4] - q2[2]))
                if closest < safe_dist:
                    risk = max(risk, 0.35 + 0.5 * (1.0 - closest / safe_dist) * (1.0 - ttc / 6.0))
        dt = max(0.0, float(t_sec) - self.last_t)
        self.score = min(0.98, max(risk, self.score * math.exp(-dt / 1.5)))
        self.last_t = float(t_sec)
        return float(np.clip(self.score, 0.0, 1.0))
