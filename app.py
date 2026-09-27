"""Local Streamlit website and upload demo for the WIUT CV-track submission."""
from __future__ import annotations

import inspect
import json
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from solution import MODEL_PATH, VEHICLE_IDS, RiskEstimator, _center, _iou, _predict, detect_events
from calibration import CONFIG_PATH, load_calibration

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT.parent / "samples"
REPORTS = ROOT / "reports"
st.set_page_config(page_title="WIUT Traffic Vision", page_icon="🚦", layout="wide")


MAX_UPLOAD_BYTES = 10 * 1024**3
UPLOAD_LIMIT_GIB = 10
UPLOAD_CHUNK_BYTES = 8 * 1024**2
TEMP_OUTPUT_RESERVE_BYTES = 512 * 1024**2
TEMP_SPACE_FACTOR = 2


def _upload_temp_dir() -> Path:
    """Use a writable, configurable volume rather than a possibly tiny OS temp drive."""
    configured = os.environ.get("WIUT_TEMP_DIR")
    temp_dir = Path(configured).expanduser() if configured else ROOT / ".upload_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def _required_temp_space_bytes(upload_size: int) -> int:
    """Reserve room for both the source and a conservative output allowance."""
    return max(TEMP_OUTPUT_RESERVE_BYTES, upload_size * TEMP_SPACE_FACTOR)


def _copy_upload_to_temp(upload) -> str:
    """Copy Streamlit's in-memory upload to disk in bounded chunks.

    Streamlit still buffers the upload in RAM; chunking avoids making another
    full-size bytes copy while staging it for OpenCV.
    """
    temp_dir = _upload_temp_dir()
    free_bytes = shutil.disk_usage(temp_dir).free
    required_bytes = _required_temp_space_bytes(int(upload.size))
    if free_bytes < required_bytes:
        raise OSError(
            f"Not enough free space on the temporary disk ({temp_dir}). "
            f"Need about {required_bytes / 1024**3:.1f} GiB free for input and output; "
            f"only {free_bytes / 1024**3:.1f} GiB is available."
        )

    path = None
    copied = 0
    try:
        upload.seek(0)
        with tempfile.NamedTemporaryFile(suffix=".mp4", dir=temp_dir, delete=False) as tmp:
            path = tmp.name
            while True:
                chunk = upload.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                tmp.write(chunk)
                copied += len(chunk)
        if copied != int(upload.size):
            raise OSError(f"Upload copy was incomplete ({copied} of {upload.size} bytes). Please retry.")
        return path
    except Exception:
        if path and os.path.exists(path):
            os.unlink(path)
        raise
    finally:
        try:
            upload.seek(0)
        except Exception:
            pass


def _candidate_vehicle_pair(detections: list[tuple]) -> set[int]:
    """Pick a visual pointer pair; this does not validate that an event occurred."""
    indices = [i for i, d in enumerate(detections) if d[5] in VEHICLE_IDS]
    pairs = []
    for n, i in enumerate(indices):
        for j in indices[n + 1:]:
            overlap = _iou(detections[i][:4], detections[j][:4])
            a, b = _center(detections[i][:4]), _center(detections[j][:4])
            distance = float(np.hypot(a[0] - b[0], a[1] - b[1]))
            pairs.append((overlap, distance, i, j))
    if not pairs:
        return set()
    touching = [p for p in pairs if p[0] > .01]
    chosen = max(touching, key=lambda p: p[0]) if touching else min(pairs, key=lambda p: p[1])
    if not touching:
        i, j = chosen[2], chosen[3]
        width_sum = (detections[i][2] - detections[i][0]) + (detections[j][2] - detections[j][0])
        if chosen[1] > max(48, .55 * width_sum):
            return set()
    return {chosen[2], chosen[3]}


def _metadata(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise ValueError("OpenCV cannot open this MP4. Try an H.264/AAC MP4 or re-encode it.")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return {"fps": fps, "width": width, "height": height,
                "n_frames": n_frames, "duration_sec": n_frames / fps if n_frames > 0 else 0.0,
                "video_id": Path(path).name}
    finally:
        cap.release()


def _calibration_preview(config: dict) -> np.ndarray:
    """Render the current editor JSON on its reference frame before saving."""
    video = SAMPLES / config.get("reference_video", "C3896.MP4")
    cap = cv2.VideoCapture(str(video))
    try:
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Cannot read calibration reference video: {video.name}")
    finally:
        cap.release()
    height, width = frame.shape[:2]
    out_w = min(1280, width); out_h = round(height * out_w / width)
    view = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
    sx, sy = out_w / width, out_h / height

    def polygon(points):
        return np.asarray([[round(float(x)*width*sx), round(float(y)*height*sy)] for x,y in points], np.int32)

    layers = [("road_polygon_norm", (0, 220, 255), .12),
              ("intersection_polygon_norm", (255, 90, 0), .16)]
    for key, color, alpha in layers:
        points = config.get(key, [])
        if len(points) >= 3:
            pts = polygon(points); layer = view.copy()
            cv2.fillPoly(layer, [pts], color); view = cv2.addWeighted(layer, alpha, view, 1-alpha, 0)
            cv2.polylines(view, [pts], True, color, 2)
    for idx, points in enumerate(config.get("crosswalk_polygons_norm", []), 1):
        if len(points) >= 3:
            pts = polygon(points); layer = view.copy()
            cv2.fillPoly(layer, [pts], (40, 220, 40)); view = cv2.addWeighted(layer, .20, view, .80, 0)
            cv2.polylines(view, [pts], True, (40, 220, 40), 2)
            cv2.putText(view, f"crosswalk {idx}", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, .55, (40,220,40), 2)
    for lane in config.get("lanes", []):
        points = lane.get("polygon_norm", [])
        if len(points) >= 3:
            pts = polygon(points); cv2.polylines(view, [pts], True, (255, 80, 20), 2)
    for line in config.get("stop_lines", []):
        points = line.get("points_norm", [])
        if len(points) == 2:
            pts = polygon(points); cv2.line(view, tuple(pts[0]), tuple(pts[1]), (0,0,255), 4)
        roi = line.get("signal_roi_norm")
        if roi and len(roi) == 4:
            x,y,rw,rh = roi; cv2.rectangle(view, (int(x*width*sx),int(y*height*sy)),
                                             (int((x+rw)*width*sx),int((y+rh)*height*sy)), (0,0,255), 2)
    for boundary in config.get("solid_boundaries", []):
        points = boundary if isinstance(boundary, list) else boundary.get("points_norm", [])
        if len(points) >= 2:
            pts = polygon(points); cv2.polylines(view, [pts], False, (255,255,255), 3)
    for item in config.get("traffic_lights", []):
        roi = item.get("signal_roi_norm")
        if roi and len(roi) == 4:
            x,y,rw,rh = roi
        elif item.get("bbox_norm"):
            x1,y1,x2,y2 = item["bbox_norm"]; x,y,rw,rh = x1,y1,x2-x1,y2-y1
        else:
            continue
        cv2.rectangle(view,(int(x*width*sx),int(y*height*sy)),
                      (int((x+rw)*width*sx),int((y+rh)*height*sy)),(0,0,255),2)
    for item in config.get("hazard_rois", []):
        roi=item.get("roi_norm")
        if roi and len(roi)==4:
            x,y,rw,rh=roi; cv2.rectangle(view,(int(x*width*sx),int(y*height*sy)),
                                          (int((x+rw)*width*sx),int((y+rh)*height*sy)),(0,180,255),2)
    for item in config.get("stopped_vehicle_zone_candidates", []):
        x,y=item.get("point_norm",[0,0]); cv2.circle(view,(int(x*width*sx),int(y*height*sy)),5,(0,140,255),-1)
    return cv2.cvtColor(view, cv2.COLOR_BGR2RGB)


def _risk_curve(path: str, meta: dict, progress=None) -> list[list[float]]:
    cap = cv2.VideoCapture(path)
    estimator = RiskEstimator()
    estimator.reset(meta)
    values, frame_i = [], 0
    fps = max(1.0, meta["fps"])
    try:
        while True:
            if (estimator.index + 1) % estimator.stride == 0:
                ok, frame = cap.read()
            else:
                ok, frame = cap.grab(), None
            if not ok:
                break
            t = frame_i / fps
            score = estimator.step(frame, t)
            if frame_i % max(1, round(fps)) == 0:
                values.append([round(t, 3), round(float(score), 4)])
            frame_i += 1
            if progress is not None and frame_i % max(1, round(fps)) == 0:
                progress.progress(min(.72, .42 + .30 * frame_i / max(1, meta["n_frames"])), text=f"Risk scan: {t:.0f}s")
    finally:
        cap.release()
    return values


def _annotate(path: str, events: list[list], output: str, progress=None) -> None:
    cap = cv2.VideoCapture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    stride = max(1, round(fps / 2.0))
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = max(2, min(960, source_width) // 2 * 2)
    height = max(2, round(source_height * width / max(1, source_width)) // 2 * 2)
    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"avc1"), 2.0, (width, height))
    if not writer.isOpened():
        cap.release()
        writer.release()
        raise RuntimeError("Could not create browser-playable H.264 MP4. Check that OpenCV FFmpeg supports avc1 encoding.")
    frame_i = 0
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
                detections = _predict(frame)
                active = [e for e in events if e[0] <= t <= e[1]]
                active_labels = {e[2] for e in active}
                pair = _candidate_vehicle_pair(detections) if active_labels.intersection({"accident", "near_miss"}) else set()
                view = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                scale_x = width / frame.shape[1]
                scale_y = height / frame.shape[0]
                for det_i, (x1, y1, x2, y2, conf, cls) in enumerate(detections):
                    label = _COCO.get(cls, str(cls))
                    a = (round(x1 * scale_x), round(y1 * scale_y)); b = (round(x2 * scale_x), round(y2 * scale_y))
                    if det_i in pair:
                        color, thickness, suffix = (0, 0, 255), 4, " CANDIDATE PAIR?"
                    elif "stopped_vehicle" in active_labels and cls in VEHICLE_IDS:
                        color, thickness, suffix = (0, 165, 255), 4, " STOP CANDIDATE"
                    else:
                        color, thickness, suffix = (30, 220, 90), 2, ""
                    cv2.rectangle(view, a, b, color, thickness)
                    cv2.putText(view, f"{label} {conf:.2f}{suffix}", (a[0], max(18, a[1] - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                banner = "  |  ".join(f"candidate: {e[2]} {e[0]:.1f}-{e[1]:.1f}s" for e in active[:3]) or "No event candidate"
                if active:
                    cv2.rectangle(view, (0, 0), (width, 48), (0, 0, 150), -1)
                    cv2.putText(view, f"UNVERIFIED EVENT CANDIDATE | t={t:.1f}s", (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(view, banner[:115], (10, 41), cv2.FONT_HERSHEY_SIMPLEX,
                                0.44, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.rectangle(view, (0, 0), (width, 31), (15, 20, 28), -1)
                    cv2.putText(view, f"{t:.1f}s | No event candidate", (10, 21), cv2.FONT_HERSHEY_SIMPLEX,
                                0.52, (255, 255, 255), 1, cv2.LINE_AA)
                writer.write(view)
            frame_i += 1
            if progress is not None and frame_i % max(1, round(fps * 5)) == 0:
                progress.progress(min(.99, .72 + .27 * frame_i / max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))),
                                  text=f"Rendering: {frame_i / fps:.0f}s")
    finally:
        cap.release()
        writer.release()


_COCO = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
         9: "traffic_light", 10: "fire_hydrant", 11: "stop_sign", 14: "bird", 15: "cat",
         16: "dog", 17: "horse", 18: "sheep", 19: "cow"}

st.title("🚦 WIUT Traffic Vision")
st.caption("Fixed-camera traffic-event detection · YOLOv8n + trajectory rules · runs locally/offline")

with st.sidebar:
    st.header("Navigation")
    st.markdown("**Overview · Calibration · EDA · Results · Live demo · Report · Team**")
    st.divider()
    st.write("Model weights:", "✅ found" if MODEL_PATH.is_file() else "❌ missing")
    try:
        import torch
        if torch.cuda.is_available():
            st.success(f"Inference device: {torch.cuda.get_device_name(0)} (CUDA)")
        else:
            st.warning("Inference device: CPU (CUDA wheel/GPU unavailable)")
    except Exception as exc:
        st.warning(f"Cannot detect inference device: {exc}")
    if not MODEL_PATH.is_file():
        st.code("python weights/download.py", language="powershell")

overview, calibration_tab, eda, results, demo, report, team = st.tabs(
    ["Overview", "Camera calibration", "Sample EDA", "Sample results", "Live demo", "Report", "Team"]
)

with overview:
    st.header("Problem & approach")
    st.markdown("""
    **Input:** a fixed road-camera MP4. **Output:** `[start_sec, end_sec, class]` segments and a
    causal accident-risk curve. Detection uses the shipped COCO-pretrained YOLOv8n checkpoint;
    event rules and TTC/deceleration risk estimates are local and deterministic. No hosted AI API
    is called and no weights are downloaded at inference time.
    """)
    st.code("MP4 → YOLOv8n (road users + signal/sign proposals) → tracks + auto camera draft → calibrated event rules → events + causal risk", language="text")
    st.info("Auto road/lane geometry is generated from the reference clip and requires visual review. Red-light/stop-line, crosswalk, solid-line and prohibited-turn rules need verified ROIs/polygons; an object box alone does not establish a violation.")
    st.markdown("**Reproduce:** `python calibrate_scene.py` · review `reports/calibration_overlay.jpg` and `scene_calibration.json` · `python run_submission.py --videos ../samples --out predictions_samples.json --team YOUR_TEAM` · `streamlit run app.py`.")

with calibration_tab:
    st.header("Fixed-camera geometry calibration")
    st.markdown("Run the automatic draft once: `python calibrate_scene.py --video ..\\samples\\C3896.MP4`. It estimates the drivable footprint, dominant flow corridors, traffic-light/sign ROIs and stationary-vehicle zones. **The draft must be visually reviewed**; geometry-based labels are disabled when the required ROIs are absent. The saved calibration applies to every processed video, so recalibrate for a different camera/viewpoint.")
    current_calibration = load_calibration()
    if current_calibration:
        m1, m2, m3 = st.columns(3)
        m1.metric("Calibration state", current_calibration.get("status", "manual"))
        m2.metric("Flow corridors", len(current_calibration.get("lanes", [])))
        m3.metric("Signal candidates", len(current_calibration.get("traffic_lights", [])))
        if current_calibration.get("manual_review_required"):
            st.warning("Auto geometry is an unverified draft; confirm the road, lanes, signal crops and stop-position markers before interpreting violations.")
    if CONFIG_PATH.is_file():
        config_text = CONFIG_PATH.read_text(encoding="utf-8")
        overlay = REPORTS / "calibration_overlay.jpg"
        if overlay.is_file(): st.image(str(overlay), caption="Auto-calibrated road/lane draft — edit polygons against the frame")
        st.caption("Polygons use normalized [x, y] values in [0,1]. Edit JSON → Preview edited geometry → verify it → set manual_review_required=false only after confirming the road/lane draft → Validate and save.")
        edited = st.text_area("scene_calibration.json", value=config_text, height=420, key="calibration_json")
        if st.button("Preview edited geometry"):
            try:
                st.image(_calibration_preview(json.loads(edited)), caption="Edited geometry preview — save only after visually checking it")
            except Exception as exc:
                st.error(f"Cannot render preview: {exc}")
        if st.button("Validate and save calibration", type="primary"):
            try:
                candidate = json.loads(edited)
                if candidate.get("version") != 1: raise ValueError("version must be 1")
                for key in ("road_polygon_norm", "intersection_polygon_norm", "crosswalk_polygons_norm"):
                    for poly in candidate.get(key, []):
                        if len(poly) < 3 or any(len(p) != 2 or any(not 0 <= float(v) <= 1 for v in p) for p in poly):
                            raise ValueError(f"{key} must contain polygons of normalized [x,y] points")
                def valid_roi(roi):
                    return (isinstance(roi, list) and len(roi) == 4
                            and all(0 <= float(v) <= 1 for v in roi)
                            and float(roi[2]) > 0 and float(roi[3]) > 0
                            and float(roi[0]) + float(roi[2]) <= 1
                            and float(roi[1]) + float(roi[3]) <= 1)
                for line in candidate.get("stop_lines", []):
                    points = line.get("points_norm", [])
                    if (len(points) != 2 or any(len(p) != 2 or any(not 0 <= float(v) <= 1 for v in p) for p in points)
                            or not valid_roi(line.get("signal_roi_norm"))):
                        raise ValueError("Each stop_line requires two normalized points and a normalized signal ROI rectangle")
                for hazard in candidate.get("hazard_rois", []):
                    if hazard.get("kind") not in {"fire", "smoke"} or not valid_roi(hazard.get("roi_norm")):
                        raise ValueError("Each hazard ROI requires kind fire/smoke and [x,y,width,height] in [0,1]")
                CONFIG_PATH.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
                st.success("Calibration saved. The next inference run will use it.")
            except Exception as exc:
                st.error(f"Calibration not saved: {exc}")
        st.download_button("Download current calibration JSON", config_text, file_name="scene_calibration.json", mime="application/json")
    else:
        st.warning("No calibration file yet. Run `python calibrate_scene.py` from the project folder.")
    with st.expander("Calibration fields and what they enable"):
        st.code('''
"road_polygon_norm": [[x,y], ...],
"crosswalk_polygons_norm": [[[x,y], ...]],
"intersection_polygon_norm": [[x,y], ...],
"lanes": [{"polygon_norm": [[x,y], ...], "direction": [dx,dy], "track_count": 4}],
"stop_lines": [{"points_norm": [[x1,y1],[x2,y2]], "approach_sign": 1,
                "signal_roi_norm": [left,top,width,height]}],
"solid_boundaries": [{"points_norm": [[x,y], ...]}],
"hazard_rois": [{"kind":"fire", "roi_norm":[left,top,width,height], "min_fraction":0.04}],
"u_turn_allowed": false,
"prohibited_turns": [{"turn":"left", "allowed":false, "polygon_norm":[[x,y], ...]}]
''', language="json")
        st.write("Draw stop-lines across the stop bar and make signal_roi_norm cover the actual lamp housing; verify red/green pixel states before enabling violations. Set a crosswalk polygon to distinguish jaywalking from a legal crossing. Put a solid boundary in solid_boundaries. Prohibited turns must be entered only when confirmed by signs/markings.")
        st.caption("Pixel-to-normalized conversion: x_norm = x_pixels / frame_width, y_norm = y_pixels / frame_height. ROI format is [left, top, width, height] in normalized coordinates. approach_sign=1 means vehicles approach from the positive side of the endpoint-ordered stop line; use -1 if direction is reversed.")

with eda:
    st.header("Exploratory data analysis")
    st.write("Sample clips are unlabelled; charts describe detected COCO objects and frame-difference motion, not ground truth.")
    eda_file = REPORTS / "eda.json"
    if eda_file.is_file():
        data = json.loads(eda_file.read_text(encoding="utf-8"))
        options = list(data.get("videos", {}))
        if options:
            selected = st.selectbox("Sample video", options)
            row = data["videos"][selected]
            c1, c2, c3 = st.columns(3)
            c1.metric("Duration", f"{row['duration_sec']:.1f}s")
            c2.metric("Resolution", row["resolution"])
            c3.metric("FPS", f"{row['fps']:.2f}")
            chart = REPORTS / row["chart"]
            annotated = REPORTS / row["annotated_video"]
            if chart.is_file(): st.image(str(chart), caption="Object detections, motion heatmap, and event timeline")
            if annotated.is_file(): st.video(str(annotated))
            st.json(row["summary"])
    else:
        st.warning("Generate sample EDA first: `python generate_eda.py`.")

with results:
    st.header("Sample predictions")
    pred_path = ROOT / "predictions_samples.json"
    if pred_path.is_file():
        pred = json.loads(pred_path.read_text(encoding="utf-8"))
        for name, entry in pred.get("videos", {}).items():
            with st.expander(f"{name}: {len(entry.get('events', []))} events"):
                st.dataframe([{"start": e[0], "end": e[1], "class": e[2]} for e in entry.get("events", [])],
                             width="stretch")
                eda_path = REPORTS / "eda.json"
                if eda_path.is_file():
                    media = json.loads(eda_path.read_text(encoding="utf-8")).get("videos", {}).get(name, {})
                    review_video = REPORTS / media.get("annotated_video", f"{Path(name).stem}_event_review.mp4")
                    chart = REPORTS / media.get("chart", "")
                    if review_video.is_file():
                        st.markdown("**Event-review video — red outlines = unverified accident/near-miss pair; amber = stop candidate**")
                        st.video(str(review_video))
                    if chart.is_file():
                        st.image(str(chart), caption="Sample detections, motion and unverified event timeline")
                curve = entry.get("risk", [])
                if curve:
                    fig, ax = plt.subplots(figsize=(10, 2.5)); ax.plot([x[0] for x in curve], [x[1] for x in curve])
                    ax.axhline(0.5, color="red", linestyle="--", linewidth=1); ax.set(xlabel="seconds", ylabel="risk", ylim=(0, 1))
                    st.pyplot(fig); plt.close(fig)
    else:
        st.info("Run the harness to create predictions_samples.json.")
    st.caption("No sample ground-truth labels are supplied; official metrics cannot be estimated on these clips.")

with demo:
    st.header("Live upload demo")
    st.write(f"Upload an MP4 up to {UPLOAD_LIMIT_GIB} GiB. 4K files are accepted; inference resizes sampled frames to YOLO's 640-pixel input and the review video is capped at 960 px wide. Processing time depends heavily on duration and hardware.")
    st.warning("Streamlit holds the complete upload in server RAM. A 5 GiB file therefore consumes at least 5 GiB of RAM before analysis; budget roughly 16 GiB available RAM and 10 GiB free in the upload workspace. The workspace defaults to .upload_tmp beside the project (set WIUT_TEMP_DIR to use another writable drive). A 32 GiB-RAM machine is safer, especially for 4K/long clips. These are practical estimates, not guarantees; very long videos may take hours. After processing, clear the selected file with the uploader's × control to release its server-side buffer.")
    upload = st.file_uploader(f"Choose a traffic video (max {UPLOAD_LIMIT_GIB} GiB)", type=["mp4", "MP4"])
    if upload is not None:
        if upload.size > MAX_UPLOAD_BYTES:
            st.error(f"Maximum upload size is {UPLOAD_LIMIT_GIB} GiB.")
        else:
            size_gib = upload.size / (1024**3)
            st.caption(f"{upload.name} · {size_gib:.2f} GiB · no fixed duration limit")
            if upload.size >= 4 * 1024**3:
                st.warning("Large upload: it is already buffered in server RAM by Streamlit. Close memory-heavy apps first; copying to disk is chunked, but does not make the network upload itself disk-streamed.")
            if st.button("Analyze video", type="primary"):
                temp_path = None
                annotated_path = None
                previous = st.session_state.pop("demo_result", None)
                if previous and previous[3] and os.path.isfile(previous[3]):
                    try:
                        os.unlink(previous[3])
                    except OSError:
                        pass
                try:
                    temp_path = _copy_upload_to_temp(upload)
                    meta = _metadata(temp_path)
                    if meta["width"] <= 0 or meta["height"] <= 0 or meta["n_frames"] <= 0:
                        raise ValueError("OpenCV could not read this video. Check that it is a valid MP4.")
                    duration_minutes = meta["duration_sec"] / 60.0
                    st.info(f"Source: {meta['width']}×{meta['height']} · {meta['fps']:.2f} FPS · about {duration_minutes:.1f} min. YOLO inference uses 640 px; output is downscaled for review.")
                    if meta["width"] * meta["height"] >= 8_000_000:
                        st.warning("4K/high-resolution source detected. Decoding still reads source frames, but unused frames are skipped without creating full-size image arrays. Expect longer processing time; the result is not 4K.")
                    progress = st.progress(0.02, text="YOLO event scan starting")
                    def event_scan_progress(fraction, seconds):
                        progress.progress(.02 + .40 * fraction, text=f"YOLO scan: {seconds:.0f}s")
                    if "progress_callback" in inspect.signature(detect_events).parameters:
                        events = detect_events(temp_path, progress_callback=event_scan_progress)
                    else:
                        events = detect_events(temp_path)
                    progress.progress(0.42, text="Event scan complete; computing causal risk")
                    risk = _risk_curve(temp_path, meta, progress)
                    progress.progress(0.72, text="Rendering unverified event-review video")
                    with tempfile.NamedTemporaryFile(suffix=".mp4", dir=_upload_temp_dir(), delete=False) as annotated:
                        annotated_path = annotated.name
                    _annotate(temp_path, events, annotated_path, progress)
                    progress.progress(1.0, text="Complete")
                    st.session_state["demo_result"] = (upload.name, events, risk, annotated_path)
                    annotated_path = None
                except OSError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.exception(exc)
                finally:
                    if temp_path and os.path.exists(temp_path): os.unlink(temp_path)
                    if annotated_path and os.path.exists(annotated_path): os.unlink(annotated_path)
    if "demo_result" in st.session_state:
        name, events, risk, video_out = st.session_state["demo_result"]
        st.subheader(f"Results: {name}")
        if events:
            markers = " · ".join(f"{e[2]} {e[0]:.1f}–{e[1]:.1f}s" for e in events)
            st.markdown(f"**Marked event intervals:** {markers}")
        else:
            st.info("No event candidates were returned; the annotated full video is still available below.")
        if os.path.isfile(video_out):
            st.markdown("### Annotated event video")
            st.video(str(video_out))
            st.caption("Full video, sampled at 2 FPS. Red outlines mark event-window road users and accident/near-miss candidate pairs; amber outlines mark stop candidates. These are visual review cues, not confirmed violations.")
            if events:
                st.markdown("**Highlighted event frames** — previews are taken from the annotated video at each interval midpoint.")
                preview_cap = cv2.VideoCapture(str(video_out))
                if preview_cap.isOpened():
                    for start in range(0, min(len(events), 6), 3):
                        preview_cols = st.columns(3)
                        for col, event in zip(preview_cols, events[start:start + 3]):
                            midpoint = max(0.0, (float(event[0]) + float(event[1])) / 2.0)
                            preview_cap.set(cv2.CAP_PROP_POS_MSEC, midpoint * 1000.0)
                            frame_ok, preview = preview_cap.read()
                            if frame_ok:
                                col.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                                          caption=f"{event[2]} · {event[0]:.1f}–{event[1]:.1f}s",
                                          width="stretch")
                            else:
                                col.caption(f"{event[2]} · {event[0]:.1f}–{event[1]:.1f}s")
                    preview_cap.release()
        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Detected events**")
            st.dataframe([{"start (s)": e[0], "end (s)": e[1], "class": e[2]} for e in events],
                         width="stretch")
            if events:
                fig, ax = plt.subplots(figsize=(9, max(2, .45 * len(events))))
                for y, ev in enumerate(events):
                    ax.barh(y, ev[1] - ev[0], left=ev[0], height=.55)
                    ax.text(ev[0], y, f"  {ev[2]}  {ev[0]:.1f}-{ev[1]:.1f}s", va="center", fontsize=8)
                ax.set_yticks([]); ax.set_xlabel("time (seconds)"); ax.invert_yaxis(); st.pyplot(fig); plt.close(fig)
            else:
                st.info("No events met the current rules.")
        with right:
            st.markdown("**Causal accident-risk curve**")
            if risk:
                fig, ax = plt.subplots(figsize=(9, 3)); ax.plot([x[0] for x in risk], [x[1] for x in risk], linewidth=.8)
                ax.axhline(.5, color="red", linestyle="--", label="alarm threshold 0.5")
                ax.set(xlabel="time (seconds)", ylabel="risk", ylim=(0, 1)); ax.legend(); st.pyplot(fig); plt.close(fig)

with report:
    st.header("Technical report")
    st.markdown("### What we built\nA fixed-camera baseline with local COCO YOLOv8n detections, greedy trajectory association, temporal stop/crowd/interaction rules, and a causal risk estimator based on track deceleration and TTC.")
    st.markdown("### What works / what does not\nObject categories are learned from COCO. Event labels are not trained: most are rules. No labels are provided for the sample clips, so no credible F1/AP claim is made. Stop-line, red-light, prohibited-turn and solid-line classes require scene calibration/extra training; close-approach rules can confuse near traffic with an incident.")
    st.markdown("### Next steps\nAnnotate the supplied clips, calibrate the fixed camera (lanes, stop line, crossing), add event-specific validation, tune the thresholds, and benchmark AP/F1 on a held-out dev set before claiming accuracy.")
    st.markdown("### Data and licensing\nSamples are organizer-provided. The detector uses Ultralytics YOLOv8n pretrained on COCO; see the repository license and model page linked in README. No extra traffic footage is used.")

with team:
    st.header("Team")
    st.info("Team name: Qwen 3.8. Roles below are proposed project assignments; verified public profile links can be added when supplied.")
    st.markdown("- **Team name:** Qwen 3.8\n- **Inomov Saidkamol — Team lead & system integration:** event pipeline, rule architecture, and final integration\n- **Karimov Alisher — Computer vision & GPU inference:** YOLO/CUDA setup and fixed-camera calibration\n- **Bekasil Ongarbaev — Demo, EDA & validation:** Streamlit UI, visual review outputs, and checklist/testing\n- **Standalone team site:** `team_site/index.html`")
    st.markdown("Public deployment is not configured. Run locally with `streamlit run app.py`.")

st.divider()
st.caption("WIUT Hackathon CV Track · Local demo · Uploads are processed on-device and temporary input files are deleted after inference.")
