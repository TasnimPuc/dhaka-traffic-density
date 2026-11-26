# Dhaka Traffic Density Analyzer

YOLO11 fine-tuned on **Dhaka-AI** (21 local vehicle classes, incl. **rickshaw**, **CNG**, **human hauler**, **auto rickshaw**).
It goes beyond detection: it **counts** vehicles, gives a **congestion score (0–100)** with a **heatmap**, and counts
unique vehicles in video with **ByteTrack**.

```
image/video ─► YOLO11 ─► boxes ─► counts + PCU load + occupancy (+ slowness) ─► score · heatmap
                                └─► ByteTrack ─► unique vehicles · line crossings · score timeline
```

## Pipeline (`notebooks/dhaka_traffic_density.ipynb`)
1. **Data:** DatasetNinja archive (Supervisely format), with the original Kaggle (Pascal VOC) version as fallback. The cleaned
   YOLO copy is cached in your GCS bucket.
2. **Clean:** drops the 956 unlabeled images, exact duplicates and unknown classes; fixes EXIF rotation; clips boxes; splits
   80/10/10 by image hash.
3. **Train:** YOLO11s at 960 px on 2×T4. `last.pt` / `best.pt` are mirrored to the bucket every 2 min, so *Run All*
   after a stop **resumes from the same epoch**.
4. **Evaluate:** held-out test split with mAP50, mAP50-95 and per-class AP next to box counts (rare classes are reported, not
   hidden).
5. **Analyze:** congestion score + heatmap on images, ByteTrack counting and score timeline on video.
6. **Ship:** `.pt` (Ultralytics), `.pth` (state dict), ONNX, and a Gradio demo in `app/`.

## Congestion score
A transparent heuristic, not a learned label (`app/traffic.py`, all constants editable per camera):
- **Image:** 60% occupancy (union of vehicle boxes / frame; 35% ≈ jam) + 40% load in PCU (bus/truck 3, minibus 2, car 1,
  rickshaw 1, CNG 0.8, motorbike/bicycle 0.5; 40 PCU ≈ jam).
- **Video:** 45% occupancy + 20% PCU + 35% slowness (median track motion vs free flow).
- **Levels:** Low < 30 ≤ Moderate < 55 ≤ Heavy < 75 ≤ Jam.

## Results
After a run, see `results/RESULTS.md`, `results/figures/` (training curves, confusion matrix, PR curve, image analysis, video
timeline).

## Run
**Kaggle:** import the notebook → GPU **T4 ×2** → Internet **On** → Add-ons → Google Cloud Services → Cloud Storage → *Run All*
(≈ 2.5–3.5 h). Add a Dhaka traffic video with *Add Input* or set `CFG["video"]`. Otherwise a synthetic pan is used only as a
pipeline check.

**Demo (local or Hugging Face Space):**
```bash
cd app && pip install -r requirements.txt
MODEL_PATH=dhaka_yolo11s.pt python app.py
```

## Repository
```
notebooks/dhaka_traffic_density.ipynb   full pipeline
app/traffic.py                          counting, congestion score, heatmap, ByteTrack video analysis
app/app.py                              Gradio demo (image + video)
models/                                 .pt / .pth / .onnx  ← from the run
results/                                metrics + figures   ← from the run
tests/test_traffic.py                   unit tests for the score and heatmap
tools/build_notebook.py                 regenerates the notebook from app/ sources
```

## Data and license
Dhaka-AI: Dhaka Traffic Detection Challenge dataset (ASM Shihavuddin, Mohammad Rifat Ahmmad Rashid; 3,953 images, 21 classes),
via DatasetNinja. DatasetNinja lists the license as CC0 1.0 in one place and GPL-2.0 in another. Check the original source
before commercial use. Known issue: some labels in the original data are noisy.

Code: MIT.
