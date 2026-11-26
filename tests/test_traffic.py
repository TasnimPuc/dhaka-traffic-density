import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from traffic import congestion, heat_add, heat_overlay, level, occupancy  # noqa: E402


def test_occupancy_union_not_sum():
    boxes = np.array([[0, 0, 100, 100], [0, 0, 100, 100]])           # same box twice
    assert abs(occupancy(boxes, (200, 200)) - 0.25) < 0.02


def test_score_grows_with_traffic():
    few = congestion(np.array([[0, 0, 50, 50]]), ["motorbike"], (480, 640))
    many = congestion(np.array([[i * 60, 100, i * 60 + 55, 300] for i in range(10)]), ["bus"] * 10, (480, 640))
    assert few["score"] < many["score"] <= 100 and many["level"] in ("Heavy", "Jam")
    assert congestion(np.zeros((0, 4)), [], (480, 640))["score"] == 0


def test_slowness_raises_video_score():
    b, n = np.array([[0, 0, 200, 200]]), ["car"]
    assert congestion(b, n, (480, 640), slow=1.0)["score"] > congestion(b, n, (480, 640), slow=0.0)["score"]


def test_levels_and_heatmap():
    assert [level(s) for s in (0, 40, 60, 99)] == ["Low", "Moderate", "Heavy", "Jam"]
    heat = np.zeros((120, 160), np.float32)
    heat_add(heat, np.array([[100, 100, 300, 300]]), [3.0])
    out = heat_overlay(np.zeros((480, 640, 3), np.uint8), heat)
    assert out.shape == (480, 640, 3) and out.max() > 0
