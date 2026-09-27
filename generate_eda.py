"""Generate reproducible sample-video EDA charts and annotated low-rate MP4s."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from solution import VEHICLE_IDS, _center, _iou, _predict

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT.parent / "samples"
OUT = ROOT / "reports"
NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
         9: "traffic_light", 10: "fire_hydrant", 11: "stop_sign", 14: "bird", 15: "cat",
         16: "dog", 17: "horse", 18: "sheep", 19: "cow"}
COLORS = {"person": "#d95f02", "bicycle": "#7570b3", "car": "#1b9e77",
          "motorcycle": "#e7298a", "bus": "#66a61e", "truck": "#e6ab02",
          "traffic_light": "#d62728", "fire_hydrant": "#8c564b", "stop_sign": "#bcbd22",
          "bird": "#17becf", "cat": "#9467bd", "dog": "#ff7f0e", "horse": "#7f7f7f",
          "sheep": "#aec7e8", "cow": "#c5b0d5"}


def _event_candidate_pair(detections: list[tuple]) -> set[int]:
    """Choose the closest/most-overlapping vehicle pair for event-time review only."""
    indices = [i for i, d in enumerate(detections) if d[5] in VEHICLE_IDS]
    pairs = []
    for n, i in enumerate(indices):
        for j in indices[n+1:]:
            overlap = _iou(detections[i][:4], detections[j][:4])
            distance = float(np.hypot(_center(detections[i][:4])[0] - _center(detections[j][:4])[0],
                                      _center(detections[i][:4])[1] - _center(detections[j][:4])[1]))
            pairs.append((overlap, distance, i, j))
    if not pairs:
        return set()
    touching = [p for p in pairs if p[0] > 0.01]
    selected = max(touching, key=lambda p: p[0]) if touching else min(pairs, key=lambda p: p[1])
    if not touching:
        i, j = selected[2], selected[3]
        width_sum = (detections[i][2] - detections[i][0]) + (detections[j][2] - detections[j][0])
        if selected[1] > max(48.0, .55 * width_sum):
            return set()
    return {selected[2], selected[3]}


def _events_for_video(path: Path) -> list[list]:
    pred_file = ROOT / "predictions_samples.json"
    if not pred_file.is_file():
        return []
    try:
        return json.loads(pred_file.read_text(encoding="utf-8"))["videos"].get(path.name, {}).get("events", [])
    except Exception:
        return []


def process(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frames / fps if fps else 0.0
    events = _events_for_video(path)
    stride = max(1, round(fps / 2.0))
    sample_times, counts, motion, heat = [], [], [], None
    previous_gray = None
    annotated_path = OUT / f"{path.stem}_event_review.mp4"
    out_w = max(2, min(960, width) // 2 * 2)
    out_h = max(2, round(height * out_w / max(1, width)) // 2 * 2)
    writer = cv2.VideoWriter(str(annotated_path), cv2.VideoWriter_fourcc(*"avc1"), 2.0, (out_w, out_h))
    if not writer.isOpened():
        cap.release(); writer.release()
        raise RuntimeError("Could not create browser-playable H.264 event-review video; check OpenCV FFmpeg avc1 support.")
    last_frame_shape = (out_h, out_w)
    try:
        frame_i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_i % stride == 0:
                t = frame_i / fps
                detections = _predict(frame)
                sample_times.append(t)
                row = {name: 0 for name in NAMES.values()}
                view = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                scale = out_w / frame.shape[1]
                active = [ev for ev in events if ev[0] <= t <= ev[1]]
                active_labels = {ev[2] for ev in active}
                pair = _event_candidate_pair(detections) if active_labels.intersection({"accident", "near_miss"}) else set()
                for det_i, (x1, y1, x2, y2, conf, cls) in enumerate(detections):
                    name = NAMES.get(cls, str(cls)); row[name] = row.get(name, 0) + 1
                    p1 = (round(x1 * scale), round(y1 * scale)); p2 = (round(x2 * scale), round(y2 * scale))
                    if det_i in pair:
                        color, thickness, suffix = (0, 0, 255), 4, "  EVENT PAIR?"
                    elif "stopped_vehicle" in active_labels and cls in VEHICLE_IDS:
                        color, thickness, suffix = (0, 165, 255), 3, "  STOP CANDIDATE"
                    else:
                        color, thickness, suffix = (40, 210, 80), 2, ""
                    cv2.rectangle(view, p1, p2, color, thickness)
                    cv2.putText(view, f"{name} {conf:.2f}{suffix}", (p1[0], max(15, p1[1] - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
                counts.append(row)
                gray = cv2.cvtColor(cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                if previous_gray is None:
                    heat = np.zeros_like(gray, dtype=np.float32)
                    motion.append(0.0)
                else:
                    diff = cv2.absdiff(previous_gray, gray).astype(np.float32)
                    heat += diff
                    motion.append(float(np.mean(diff)))
                previous_gray = gray
                if active:
                    labels = ", ".join(sorted(active_labels))
                    cv2.rectangle(view, (0, 0), (out_w, 52), (0, 0, 150), -1)
                    cv2.putText(view, f"UNVERIFIED EVENT CANDIDATE: {labels}", (10, 21),
                                cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(view, f"t={t:.1f}s | red=closest/overlap pair | inspect before trusting", (10, 44),
                                cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.rectangle(view, (0, 0), (out_w, 28), (15, 20, 28), -1)
                    cv2.putText(view, f"{t:.1f}s | detections: {len(detections)}", (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
                writer.write(view)
                last_frame_shape = view.shape[:2]
            frame_i += 1
    finally:
        cap.release(); writer.release()
    if heat is None:
        heat = np.zeros((180, 320), dtype=np.float32)
    mean_counts = {name: float(np.mean([r.get(name, 0) for r in counts])) if counts else 0.0
                   for name in NAMES.values()}
    totals = {name: int(sum(r.get(name, 0) for r in counts)) for name in NAMES.values()}
    chart_path = OUT / f"{path.stem}_eda.png"
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    ax = axes[0, 0]
    for name in NAMES.values():
        ax.plot(sample_times, [r.get(name, 0) for r in counts], label=name, color=COLORS[name], linewidth=1)
    ax.set(title="COCO road-user detections per 0.5 s", xlabel="time (s)", ylabel="count"); ax.legend(ncol=3, fontsize=8)
    ax = axes[0, 1]
    ax.imshow(heat, cmap="inferno"); ax.set(title="Accumulated frame-difference motion heatmap", xticks=[], yticks=[])
    ax = axes[1, 0]
    ax.plot(sample_times, motion, color="#3366cc", linewidth=.8)
    ax.set(title="Mean frame-difference motion", xlabel="time (s)", ylabel="mean |Δgray|")
    ax = axes[1, 1]
    for y, ev in enumerate(events):
        ax.barh(y, ev[1] - ev[0], left=ev[0], color="#59a14f")
        ax.text(ev[0], y, f" {ev[2]}", va="center", fontsize=8)
    ax.set(title="Model output event timeline (not ground truth)", xlabel="time (s)", yticks=[])
    if not events: ax.text(.5, .5, "No event predictions", transform=ax.transAxes, ha="center")
    fig.suptitle(f"{path.name} — {width}×{height}, {fps:.2f} FPS, {duration:.1f}s", fontsize=14)
    fig.savefig(chart_path, dpi=150); plt.close(fig)
    return {"duration_sec": round(duration, 3), "fps": fps, "resolution": f"{width}×{height}",
            "n_frames": frames, "sample_rate_hz": round(fps / stride, 3),
            "summary": {"detections_per_class_total": totals, "mean_count_per_sample": mean_counts,
                        "mean_motion": float(np.mean(motion)) if motion else 0.0,
                        "sampled_frames": len(sample_times)},
            "chart": chart_path.name, "annotated_video": annotated_path.name}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    videos = sorted(p for p in SAMPLES.iterdir() if p.suffix.lower() == ".mp4")
    if not videos:
        raise SystemExit(f"No MP4 samples found in {SAMPLES}")
    data = {"method": "YOLOv8n COCO inference at 2 Hz; gray-frame difference motion; no sample ground truth",
            "videos": {p.name: process(p) for p in videos}}
    (OUT / "eda.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {OUT / 'eda.json'} and {len(videos)} chart/annotated-video pair(s)")
    for name, row in data["videos"].items():
        print(name, row["resolution"], f"{row['fps']:.3f}fps", f"{row['duration_sec']:.1f}s", row["summary"]["detections_per_class_total"])


if __name__ == "__main__":
    main()
