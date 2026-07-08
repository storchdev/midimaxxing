"""YOLO-based hand detection and multi-hand tracking across video frames.

Detects hand bounding boxes per frame with a YOLOv10 hand-detection model,
keeps only boxes intersecting the keyboard bbox, and tracks up to `num_hands`
identities across frames -- gap-filling short disappearances via linear
interpolation and using a Hungarian (min-cost) assignment so that multiple
hands disappearing/reappearing at different, possibly overlapping times don't
get cross-matched to the wrong identity.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import requests
from scipy.optimize import linear_sum_assignment

WEIGHTS_URL = (
    "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/"
    "hagrid_v2/models/YOLOv10n_hands.pt"
)

Box = tuple[float, float, float, float]  # x1, y1, x2, y2


def ensure_weights(path: Path = Path("weights/YOLOv10n_hands.pt"), url: str = WEIGHTS_URL) -> Path:
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(part_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    part_path.rename(path)
    return path


def load_model(path: Path = Path("weights/YOLOv10n_hands.pt")):
    from ultralytics import YOLO
    return YOLO(str(ensure_weights(path)))


def _intersects(box: Box, other: Box) -> bool:
    x1, y1, x2, y2 = box
    ox1, oy1, ox2, oy2 = other
    return x1 < ox2 and x2 > ox1 and y1 < oy2 and y2 > oy1


def filter_and_rank_detections(
    boxes: list[tuple[Box, float]],
    keyboard_bbox: Box,
    num_hands: int,
    conf_threshold: float,
) -> list[Box]:
    """boxes: list of (box, confidence). Drop boxes below conf_threshold or
    not intersecting keyboard_bbox, then keep the num_hands most confident.
    """
    survivors = [
        (box, conf) for box, conf in boxes
        if conf >= conf_threshold and _intersects(box, keyboard_bbox)
    ]
    survivors.sort(key=lambda bc: bc[1], reverse=True)
    return [box for box, _ in survivors[:num_hands]]


def _center(box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


@dataclass
class _Track:
    track_id: int
    box: Box
    last_seen_frame: int
    missed: int = 0


def _interpolate_box(box_a: Box, frame_a: int, box_b: Box, frame_b: int, frame: int) -> Box:
    t = (frame - frame_a) / (frame_b - frame_a)
    return tuple(a + (b - a) * t for a, b in zip(box_a, box_b))


def track_hands(
    frames: list[np.ndarray] | None = None,
    video_path: str | Path | None = None,
    hands: Literal["top", "bottom"] = "bottom",
    keyboard_bbox: Box | None = None,
    num_hands: int = 2,
    max_disappear_frames: int = 15,
    conf_threshold: float = 0.25,
    max_match_distance_px: float | None = None,
    weights_path: Path | None = None,
    raw_detections: list[list[tuple[Box, float]]] | None = None,
    cache_path: Path | None = None,
) -> dict[int, list[Box]]:
    """Track up to `num_hands` hands across frames.

    Either pass `raw_detections` (one list of (box, conf) per frame, already
    computed -- used by tests and for the on-disk cache path) or `video_path`
    to run YOLO from scratch.

    Returns frame index -> list of up to num_hands (x1,y1,x2,y2) boxes,
    including gap-filled/interpolated frames. Frames with no data are absent.
    """
    if keyboard_bbox is None:
        raise ValueError("keyboard_bbox is required")

    if raw_detections is None:
        raw_detections = _run_yolo(
            video_path, hands, weights_path, cache_path=cache_path,
        )

    if max_match_distance_px is None:
        kb_width = keyboard_bbox[2] - keyboard_bbox[0]
        max_match_distance_px = 0.15 * kb_width

    active: list[_Track] = []
    next_id = 0
    output: dict[int, list[Box]] = {}

    for f, dets in enumerate(raw_detections):
        boxes = filter_and_rank_detections(dets, keyboard_bbox, num_hands, conf_threshold)

        eligible = [t for t in active if t.missed <= max_disappear_frames]
        cost = np.full((len(eligible), len(boxes)), np.inf)
        for i, t in enumerate(eligible):
            tc = _center(t.box)
            for j, b in enumerate(boxes):
                cost[i, j] = _dist(tc, _center(b))

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        if eligible and boxes:
            row_idx, col_idx = linear_sum_assignment(cost)
            for r, c in zip(row_idx, col_idx):
                if cost[r, c] <= max_match_distance_px:
                    matched_tracks.add(r)
                    matched_dets.add(c)
                    track = eligible[r]
                    if track.missed > 0:
                        for gap_f in range(track.last_seen_frame + 1, f):
                            interp = _interpolate_box(
                                track.box, track.last_seen_frame, boxes[c], f, gap_f,
                            )
                            output.setdefault(gap_f, []).append(interp)
                    track.box = boxes[c]
                    track.last_seen_frame = f
                    track.missed = 0
                    output.setdefault(f, []).append(track.box)

        for j, b in enumerate(boxes):
            if j not in matched_dets:
                new_track = _Track(track_id=next_id, box=b, last_seen_frame=f, missed=0)
                next_id += 1
                active.append(new_track)
                output.setdefault(f, []).append(b)

        for i, t in enumerate(eligible):
            if i not in matched_tracks:
                t.missed += 1

        active = [t for t in active if t.missed <= max_disappear_frames]

    return output


def _run_yolo(
    video_path, hands, weights_path, cache_path: Path | None = None,
) -> list[list[tuple[Box, float]]]:
    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    import av

    model = load_model(weights_path) if weights_path else load_model()

    container = av.open(str(video_path))
    raw_detections: list[list[tuple[Box, float]]] = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="bgr24")
        if hands == "top":
            img = img[::-1, ::-1]
        results = model.predict(img, verbose=False)[0]
        dets = []
        for box, conf in zip(
            results.boxes.xyxy.cpu().numpy(), results.boxes.conf.cpu().numpy()
        ):
            dets.append((tuple(float(v) for v in box), float(conf)))
        raw_detections.append(dets)

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(raw_detections, f)

    return raw_detections
