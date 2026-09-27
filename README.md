# WIUT Hackathon 2026 — Computer Vision track

Offline fixed-camera traffic-event baseline with a local YOLOv8n object detector, track-based temporal rules, causal accident-risk estimation, sample EDA, and a local upload-demo website.

> **No sample labels are supplied.** `predictions_samples.json` is not ground truth, and a meaningful official F1/AP score cannot be computed until you annotate a development set. The model currently emits only classes supported by implemented rules; see **Known limitations**.

## Quick start — Windows

From this folder (`wiut_cv_scripts`):

```powershell
.\setup.ps1
python calibrate_scene.py --video ..\samples\C3896.MP4
# review reports/calibration_overlay.jpg and edit scene_calibration.json if needed
python run_submission.py --videos ..\samples --out predictions_samples.json --team YOUR_TEAM
python evaluate.py --pred predictions_samples.json --validate-only
python generate_eda.py
streamlit run app.py
```

`setup.ps1` creates `.venv`, installs Python dependencies, installs the CUDA 12.8 PyTorch/torchvision wheels for NVIDIA GPUs, and downloads the checkpoint once. The detector auto-selects CUDA when `torch.cuda.is_available()`; otherwise it visibly falls back to CPU. Activate the environment manually if you opened a new terminal:

```powershell
.\.venv\Scripts\Activate.ps1
```

If the current PowerShell execution policy blocks the setup script, run `powershell -ExecutionPolicy Bypass -File .\setup.ps1` for this invocation only.

For Linux/macOS:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
sh weights/download.sh
python calibrate_scene.py --video ../samples/C3896.MP4
# visually review reports/calibration_overlay.jpg and edit scene_calibration.json
python run_submission.py --videos ../samples --out predictions_samples.json --team YOUR_TEAM
python evaluate.py --pred predictions_samples.json --validate-only
python generate_eda.py
streamlit run app.py
```

The demo listens on `http://localhost:8501`. Streamlit is configured to accept MP4 files up to 10 GiB with no fixed duration cutoff. Uploads are processed locally and temporary input files are deleted after analysis. This is a configured maximum, not a hardware guarantee: Streamlit buffers the uploaded file, so a 10 GiB upload realistically needs about 32 GB RAM and at least 20 GB free disk for input/output; inference/rendering time grows with duration. Public hosting is not configured.

## Official evaluation commands

The organizer harness and metric are unchanged:

```bash
python run_submission.py --videos /data/test --out predictions.json --team YOUR_TEAM
python evaluate.py --pred predictions.json --gt ground_truth.json
```

The sample clips are one directory above this repository. For the official challenge package, use the organizers' test and ground-truth paths. The local sample validation command checks the JSON format only; it does not report model quality without labels.

## What the system does

1. **Learned detection:** official Ultralytics YOLOv8n weights, pretrained on COCO. It detects person, bicycle, car, motorcycle, bus, truck, traffic-light/stop-sign proposals and selected animal classes at about 2 Hz for Part A.
2. **Tracking:** deterministic greedy IoU / constant-velocity association over sampled boxes; short occlusions are tolerated.
3. **Rule-based events:** temporal stop/crowd rules, calibrated road/crosswalk entry (`jaywalking`), close approach (`near_miss`), contact plus braking (`accident`), animal obstacle candidates, and lane-relative wrong-way candidates. When the operator configures scene geometry, the engine can additionally test red-light/stop-line crossings, failure-to-yield, solid-boundary crossings and prohibited turns. Thresholds are heuristics, not fitted to organizer annotations.
4. **Causal anticipation:** every frame is accepted by `RiskEstimator.step`; YOLO is evaluated at a lower cadence and the score is held between evaluations. Only prior detections are used. Relative velocity / closest-point TTC and sharp deceleration increase the score.
5. **Post-processing:** clips intervals to video duration, merges same-class fragments, and sorts output.

### Known limitations (read before trusting results)

- COCO is an object dataset, not an accident/traffic-violation dataset. The supplied sample videos have no labels, so rule thresholds have not been calibrated.
- `calibrate_scene.py` estimates a road footprint, horizontal/vertical flow corridors, signal/sign ROIs and stop-position candidates from detections. It writes an **auto draft**, not verified ground truth. Open `reports/calibration_overlay.jpg`, then review/edit `scene_calibration.json` in the website calibration tab; set `manual_review_required` to `false` only after confirming/correcting the road/lane overlays to enable wrong-way labels.
- Jaywalking is withheld until at least one legal-crosswalk polygon is reviewed; merely stepping onto the auto road polygon is not enough to call a crossing illegal. Red-light/stop-line, failure-to-yield, solid-boundary and prohibited-turn rules only activate when their exact line/polygon and signal crop are entered in calibration. Auto geometry cannot determine whether a turn is legally prohibited. Signal-state classification is a color heuristic and requires checking.
- Wrong-way, near-miss, jaywalking, congestion and accident rules remain track heuristics and can be wrong; static vehicles may include queues. Animal obstacles are only proposed when COCO recognizes an animal on the calibrated road polygon. Fire/smoke currently use only a low-confidence color-persistence cue inside a manually specified ROI; generic debris and several sign/marking classes still need a dedicated trained detector.
- No sample ground truth exists. The current `predictions_samples.json` must not be treated as verified labels. Annotate sample clips and build a labeled dev set before tuning thresholds or claiming accuracy.
- The risk score is a heuristic probability proxy, not a calibrated probability. Tune/calibrate it with labeled accident timelines and validate Part B before making a performance claim.
- Accuracy was not measured on the provided samples because no sample ground truth exists.

## Model weights, offline run and licenses

`weights/yolov8n.pt` is shipped locally (~6.55 MB); inference does not access the network. To fetch/replace it before an offline evaluation:

```bash
python weights/download.py
# or: sh weights/download.sh
```

The downloader checks the SHA-256 digest and retrieves the official release:
`https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt`

The model implementation is Ultralytics (AGPL-3.0; review the license and its obligations before redistribution/commercial use):
`https://github.com/ultralytics/ultralytics/blob/main/LICENSE`
The checkpoint is pretrained on COCO; review COCO's dataset and image-specific terms at:
`https://cocodataset.org/#home`
No other datasets or external traffic footage are used. Organizer-provided clips are used only for the requested sample analysis.

For network-isolated evaluation, install `requirements.txt` and fetch/ship `weights/yolov8n.pt` **before** disabling internet. Total shipped weight is well below the 5 GB limit. The script deliberately raises a clear error rather than silently downloading a missing model during inference.

## EDA, visualizations, and local website

```bash
python generate_eda.py
streamlit run app.py
```

`generate_eda.py` produces per-sample COCO detection counts, frame-difference motion heatmaps, time-series plots, event timelines, `reports/eda.json`, and 2 Hz `*_event_review.mp4` videos under `reports/` with predicted event windows and red outlines on the closest/overlapping vehicle pair for accident/near-miss candidates; stop candidates use amber boxes. These overlays are unverified model outputs, not ground truth. `calibrate_scene.py` (0.5 Hz default) estimates a car-footprint polygon, lane-flow corridors, traffic-light/stop-sign ROIs, and candidate stopped-vehicle zones from a reference clip. Use the calibration tab in Streamlit to edit JSON and preview the overlays before saving; normalized coordinates are `[0,1]` in `[x,y]` order. Add hand-reviewed `crosswalk_polygons_norm`, `intersection_polygon_norm`, `stop_lines`, `solid_boundaries`, `prohibited_turns`, `hazard_rois` and `u_turn_allowed` when you can verify them in the image. Without those annotations the corresponding violation detectors remain disabled. The saved calibration is global and applies to every video; recalibrate for a different camera/viewpoint. The Streamlit page includes approach, calibration, EDA, sample outputs/risk curves, upload demo, report, and a team section with explicit placeholders. Replace those placeholders with real team member names, roles, GitHub, LinkedIn and portfolio links before publishing. No public deployment credentials or team details were supplied.

## Reproducibility

- No random augmentations or online inference are used. Tracking and temporal post-processing use fixed thresholds and ordered frame sampling.
- YOLO inference uses CUDA device 0 when the CUDA-enabled PyTorch wheel and supported NVIDIA GPU are available; video decoding, tracking, event rules and plotting remain CPU-side. Otherwise the app reports a CPU fallback. Small numerical differences between CPU/GPU backends are possible.
- `run_submission.py` and `evaluate.py` are the organizer starter files and should remain unchanged.

## Package layout

```text
solution.py                  YOLO, tracking, event rules, causal RiskEstimator
calibration.py                normalized scene geometry predicates
calibrate_scene.py            auto-draft fixed-camera calibration
scene_calibration.json        auto-generated road/lane/light draft; review required
run_submission.py             organizer harness (unchanged)
evaluate.py                   official format validator / metric (unchanged)
requirements.txt              runtime + local-demo dependencies
weights/yolov8n.pt            shipped pretrained checkpoint
weights/download.py           SHA-256 checked one-time downloader
weights/download.sh           shell entry point
setup.ps1                     Windows environment setup
app.py                        local upload/calibration website and demo
generate_eda.py               reproducible sample analysis and annotations
predictions_samples.json      harness output on supplied clips
reports/                      charts, EDA JSON, overlay and annotated clips
```

## Team

Project roles below are proposed assignments and can be changed by the team. Verified public profile links can be added when supplied.

- Team name: **Qwen 3.8**
- **Inomov Saidkamol — Team lead & system integration:** event pipeline, rule architecture, and final integration.
- **Karimov Alisher — Computer vision & GPU inference:** YOLO/CUDA setup and fixed-camera calibration.
- **Bekasil Ongarbaev — Demo, EDA & validation:** Streamlit UI, visual review outputs, and testing.
- Standalone team site source: [`team_site/index.html`](team_site/index.html).

## Technical report

We use an open-weight COCO detector to localize road users, then use fixed-camera track geometry for a limited set of temporal events and risk cues. Object detection is learned; event classification and risk calibration are rule-based. This makes the package small and offline-capable, but it is not a substitute for traffic-event training data. The most important next step is to review the automatic fixed-camera draft, add verified crosswalk/stop-line/turn restrictions, label sample clips, inspect per-class false positives and misses, and tune thresholds on a held-out dev split. Current failure modes include merged/fragmented detections, occlusion, perspective-dependent TTC, traffic queues mistaken for violations, and missing coverage for sign/marking/signal classes.
