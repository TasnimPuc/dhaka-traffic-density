"""Gradio demo: Dhaka traffic density analyzer (image + video). Run: python app.py"""
import os
import tempfile

import cv2
import gradio as gr
import pandas as pd

from traffic import Analyzer


def build_demo(an: Analyzer) -> gr.Blocks:
    def on_image(img_rgb):
        if img_rgb is None:
            return None, None, {}
        ann, heat, s = an.image(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        return cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), s

    def on_video(path, seconds):
        if not path:
            return None, {}, None
        out = os.path.join(tempfile.mkdtemp(), "analyzed.mp4")
        fps = cv2.VideoCapture(path).get(cv2.CAP_PROP_FPS) or 25
        s = an.video(path, out, max_frames=int(seconds * fps))
        tl = pd.DataFrame(s.pop("timeline"))[["t", "score"]]
        s.pop("heatmap")
        return out, s, tl

    with gr.Blocks(title="Dhaka Traffic Density Analyzer") as demo:
        gr.Markdown("# 🚦 Dhaka Traffic Density Analyzer\nYOLO11 fine-tuned on Dhaka-AI (21 local vehicle classes, "
                    "incl. rickshaw and CNG) · congestion score 0–100 · heatmap · ByteTrack counting for video.")
        with gr.Tab("Image"):
            with gr.Row():
                inp = gr.Image(type="numpy", label="Traffic photo")
                out = gr.Image(label="Detections + score")
            with gr.Row():
                heat = gr.Image(label="Density heatmap")
                info = gr.JSON(label="Summary")
            gr.Button("Analyze", variant="primary").click(on_image, inp, [out, heat, info])
        with gr.Tab("Video"):
            with gr.Row():
                vin = gr.Video(label="Traffic video")
                vout = gr.Video(label="Tracked + heatmap")
            secs = gr.Slider(5, 120, value=20, step=5, label="Seconds to analyze")
            vinfo = gr.JSON(label="Counts")
            plot = gr.LinePlot(x="t", y="score", title="Congestion over time", y_lim=[0, 100])
            gr.Button("Analyze video", variant="primary").click(on_video, [vin, secs], [vout, vinfo, plot])
        gr.Markdown("Score = occupancy + PCU load (+ slowness for video). It is a heuristic; tune the constants in "
                    "`traffic.py` for a fixed camera.")
    return demo


if __name__ == "__main__":
    build_demo(Analyzer(os.environ.get("MODEL_PATH", "dhaka_yolo11s.pt"))).launch()
