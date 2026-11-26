"""Dhaka traffic analyzer: vehicle counts, congestion score and heatmap for images; ByteTrack counting for video.

The congestion score is a transparent heuristic (0-100), not a learned quantity:
  image : 60% road occupancy (area covered by vehicles) + 40% traffic load in PCU
  video : 45% occupancy + 20% PCU load + 35% slowness (how little tracked vehicles move)
Every constant below can be changed for a given camera.
"""
from __future__ import annotations

import shutil
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# Approximate passenger-car-unit style weights (editable; heavier / slower vehicles load the road more)
PCU = {
    "car": 1.0, "taxi": 1.0, "suv": 1.0, "policecar": 1.0, "ambulance": 1.0, "van": 1.2, "minivan": 1.2,
    "pickup": 1.2, "minibus": 2.0, "bus": 3.0, "truck": 3.0, "garbagevan": 3.0, "army vehicle": 2.0,
    "three wheelers (CNG)": 0.8, "auto rickshaw": 0.8, "human hauler": 1.2, "rickshaw": 1.0,
    "motorbike": 0.5, "scooter": 0.5, "bicycle": 0.5, "wheelbarrow": 0.5,
}
OCC_JAM = 0.35      # vehicles covering 35% of the frame (or ROI) ~ jammed
PCU_JAM = 40.0      # 40 PCU in view ~ jammed
V_FREE = 0.010      # free-flow motion: 1% of the frame diagonal per frame
LEVELS = [(30, "Low"), (55, "Moderate"), (75, "Heavy"), (101, "Jam")]
COLORS = {"Low": (80, 175, 76), "Moderate": (0, 193, 255), "Heavy": (0, 120, 255), "Jam": (40, 40, 220)}  # BGR


def level(score: float) -> str:
    return next(name for t, name in LEVELS if score < t)


def occupancy(xyxy: np.ndarray, shape, roi=None, scale: int = 4) -> float:
    """Fraction of the frame (or ROI polygon) covered by the union of vehicle boxes."""
    h, w = shape[:2]
    m = np.zeros((max(1, h // scale), max(1, w // scale)), np.uint8)
    for x1, y1, x2, y2 in (np.asarray(xyxy, float).reshape(-1, 4) / scale).astype(int):
        m[max(0, y1):max(y2, y1 + 1), max(0, x1):max(x2, x1 + 1)] = 1
    if roi is None:
        return float(m.mean())
    r = np.zeros_like(m)
    cv2.fillPoly(r, [(np.asarray(roi, float) / scale).astype(np.int32)], 1)
    return float((m & r).sum() / max(int(r.sum()), 1))


def congestion(xyxy, names, shape, slow: float | None = None, roi=None) -> dict:
    occ = occupancy(xyxy, shape, roi)
    pcu = float(sum(PCU.get(n, 1.0) for n in names))
    o, p = min(occ / OCC_JAM, 1.0), min(pcu / PCU_JAM, 1.0)
    s = 100 * (0.6 * o + 0.4 * p) if slow is None else 100 * (0.45 * o + 0.2 * p + 0.35 * slow)
    return {"score": round(s, 1), "level": level(s), "occupancy": round(occ, 4), "pcu": round(pcu, 1)}


def heat_add(heat: np.ndarray, xyxy, weights, scale: int = 4) -> None:
    for (x1, y1, x2, y2), wt in zip((np.asarray(xyxy, float).reshape(-1, 4) / scale).astype(int), weights):
        heat[max(0, y1):max(y2, y1 + 1), max(0, x1):max(x2, x1 + 1)] += wt


def heat_overlay(img: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    if heat.max() <= 0:
        return img.copy()
    h = cv2.GaussianBlur(heat, (0, 0), sigmaX=max(heat.shape) / 60)
    h = cv2.resize(h / h.max(), (img.shape[1], img.shape[0]))
    color = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET)
    a = (alpha * np.clip(h * 1.5, 0, 1))[..., None]
    return (img * (1 - a) + color * a).astype(np.uint8)


def banner(img: np.ndarray, lines: list[str], lvl: str) -> np.ndarray:
    out = img.copy()
    fs = max(0.5, img.shape[1] / 1600)
    th = int(28 * fs * 1.6)
    cv2.rectangle(out, (0, 0), (int(img.shape[1] * 0.42), th * len(lines) + 10), COLORS[lvl], -1)
    for i, t in enumerate(lines):
        cv2.putText(out, t, (10, th * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, int(2 * fs)),
                    cv2.LINE_AA)
    return out


class Analyzer:
    def __init__(self, weights: str, imgsz: int = 960, conf: float = 0.25, device=None):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.names = self.model.names
        self.imgsz, self.conf, self.device = imgsz, conf, device

    def _boxes(self, r):
        b = r.boxes
        xyxy = b.xyxy.cpu().numpy() if len(b) else np.zeros((0, 4))
        cls = [self.names[int(c)] for c in b.cls.cpu().numpy()] if len(b) else []
        return xyxy, cls

    def image(self, img_bgr: np.ndarray, roi=None) -> tuple[np.ndarray, np.ndarray, dict]:
        """Returns (annotated BGR, heatmap BGR, summary)."""
        r = self.model.predict(img_bgr, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        xyxy, cls = self._boxes(r)
        c = congestion(xyxy, cls, img_bgr.shape, roi=roi)
        heat = np.zeros((max(1, img_bgr.shape[0] // 4), max(1, img_bgr.shape[1] // 4)), np.float32)
        heat_add(heat, xyxy, [PCU.get(n, 1.0) for n in cls])
        counts = dict(Counter(cls).most_common())
        ann = banner(r.plot(line_width=2, font_size=12), [f"Congestion {c['score']:.0f}/100  {c['level']}",
                                                          f"Vehicles {len(cls)} | PCU {c['pcu']:.0f}"], c["level"])
        return ann, heat_overlay(img_bgr, heat), {"vehicles": len(cls), "counts": counts, **c}

    def video(self, src: str, out_path: str, line: float = 0.6, max_frames: int | None = None, roi=None) -> dict:
        """ByteTrack over a video: unique vehicles, line-crossing counts, congestion timeline, heatmap video."""
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        tmp = str(Path(out_path).with_suffix(".raw.mp4"))
        writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        ly, diag = line * h, float(np.hypot(w, h))
        heat = np.zeros((max(1, h // 4), max(1, w // 4)), np.float32)
        track_cls, prev, crossed, timeline, v_ema = {}, {}, {}, [], None
        stream = self.model.track(source=str(src), stream=True, persist=False, tracker="bytetrack.yaml",
                                  imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)
        for i, r in enumerate(stream):
            if max_frames and i >= max_frames:
                break
            xyxy, cls = self._boxes(r)
            ids = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else []
            moves = []
            for tid, box, name in zip(ids, xyxy, cls):
                track_cls.setdefault(tid, name)
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                if tid in prev:
                    px, py = prev[tid]
                    moves.append(np.hypot(cx - px, cy - py) / diag)
                    if (py - ly) * (cy - ly) < 0 and tid not in crossed:
                        crossed[tid] = track_cls[tid]
                prev[tid] = (cx, cy)
            if moves:
                v = float(np.median(moves))
                v_ema = v if v_ema is None else 0.9 * v_ema + 0.1 * v
            slow = None if v_ema is None else float(np.clip(1 - v_ema / V_FREE, 0, 1))
            c = congestion(xyxy, cls, (h, w), slow=slow, roi=roi)
            heat_add(heat, xyxy, [PCU.get(n, 1.0) for n in cls])
            timeline.append({"frame": i, "t": i / fps, "vehicles": len(cls), **c, "slowness": slow})
            fr = heat_overlay(r.plot(line_width=2, font_size=12), heat, alpha=0.35)
            cv2.line(fr, (0, int(ly)), (w, int(ly)), (255, 255, 255), 2)
            fr = banner(fr, [f"Congestion {c['score']:.0f}/100  {c['level']}",
                             f"In view {len(cls)} | unique {len(track_cls)} | crossed {len(crossed)}"], c["level"])
            writer.write(fr)
        writer.release()
        if shutil.which("ffmpeg"):       # H.264 so the video plays in browsers / Gradio
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-preset", "veryfast",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path], check=True)
            Path(tmp).unlink(missing_ok=True)
        else:
            shutil.move(tmp, out_path)
        scores = [t["score"] for t in timeline] or [0]
        return {"frames": len(timeline), "fps": fps, "unique_vehicles": len(track_cls),
                "unique_by_class": dict(Counter(track_cls.values()).most_common()),
                "crossed_line": len(crossed), "crossed_by_class": dict(Counter(crossed.values()).most_common()),
                "mean_score": round(float(np.mean(scores)), 1), "peak_score": round(float(np.max(scores)), 1),
                "level": level(float(np.mean(scores))), "timeline": timeline, "heatmap": heat}
