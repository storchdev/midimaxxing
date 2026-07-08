"""Interactive keyboard-geometry picker.

Prompts the user to click the two corners of the piano keyboard, plus the
left edge (x-only) of every C key in the given note range. Always displays
the frame "hands-bottom" for the annotator, regardless of the video's native
orientation, then inverse-transforms clicks back into the original video's
native pixel space before returning.

Usage:
    python keyboard_picker.py video.mp4 --hands top
    python keyboard_picker.py video.mp4 --hands bottom --lowest-note C2 --highest-note C6
    python keyboard_picker.py video.mp4 --hands top --bbox 100,200,900,300   # reduced: markers only
    python keyboard_picker.py video.mp4 --no-markers                         # bbox-only mode (used by vit.py)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from keyboard_geometry import KeyboardGeometry, c_pitches, flip_bbox_180, flip_point_180

PathLike = str | Path


@dataclass
class PickerState:
    """Pure state machine for the click sequence, no matplotlib/GUI code.

    Phases:
        "corners" -- collecting the two keyboard bbox corners (or skipped
                     entirely if a known bbox was supplied).
        "markers" -- collecting one x-click per C key, in ascending-pitch
                     (left-to-right on screen) order.
        "done"    -- all points collected.
    """
    n_markers: int
    corners: list[tuple[float, float]] = field(default_factory=list)
    markers: list[float] = field(default_factory=list)
    phase: Literal["corners", "markers", "done"] = "corners"

    def __post_init__(self):
        if self.phase == "markers" and self.n_markers == 0:
            self.phase = "done"

    def add_click(self, x: float, y: float) -> None:
        if self.phase == "done":
            return
        if self.phase == "corners":
            self.corners.append((x, y))
            if len(self.corners) == 2:
                self.phase = "markers"
                if self.n_markers == 0:
                    self.phase = "done"
        elif self.phase == "markers":
            self.markers.append(x)
            if len(self.markers) == self.n_markers:
                self.phase = "done"

    def undo(self) -> None:
        if self.phase == "markers" and self.markers:
            self.markers.pop()
        elif self.phase == "markers" and not self.markers:
            # Cross back into corners phase.
            self.phase = "corners"
            if self.corners:
                self.corners.pop()
        elif self.phase == "corners" and self.corners:
            self.corners.pop()
        elif self.phase == "done":
            self.phase = "markers"
            if self.markers:
                self.markers.pop()
            elif self.corners:
                self.phase = "corners"
                self.corners.pop()

    def is_done(self) -> bool:
        return self.phase == "done"

    def bbox(self) -> tuple[int, int, int, int]:
        (x1, y1), (x2, y2) = self.corners
        return (
            int(min(x1, x2)), int(min(y1, y2)),
            int(max(x1, x2)), int(max(y1, y2)),
        )


_PICKER_SCRIPT = r'''
import sys, json
import av
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from keyboard_picker import PickerState

video_path = sys.argv[1]
hands = sys.argv[2]
n_markers = int(sys.argv[3])
known_bbox = json.loads(sys.argv[4]) if sys.argv[4] != "null" else None

container = av.open(video_path)
vs = container.streams.video[0]
container.seek(vs.duration // 2, stream=vs)
frame = next(container.decode(video=0))
img = frame.to_ndarray(format="rgb24")
h, w = img.shape[0], img.shape[1]

# Display hands-bottom: if the native video is hands-top, rotate 180 for viewing.
display_img = img[::-1, ::-1] if hands == "top" else img

state = PickerState(n_markers=n_markers)
if known_bbox is not None:
    state.phase = "markers"
    if n_markers == 0:
        state.phase = "done"

fig, ax = plt.subplots(figsize=(14, 8))
ax.imshow(display_img)
title = "Click TOP-LEFT then BOTTOM-RIGHT of the piano keys"
if known_bbox is not None:
    title = f"Click the left edge of each C key, left to right ({n_markers} total)"
ax.set_title(f"{title}  ({w}x{h})  [u=undo]")
plt.tight_layout()

artists = []


def redraw_title():
    if state.phase == "corners":
        ax.set_title(f"Click TOP-LEFT then BOTTOM-RIGHT of the piano keys ({len(state.corners)}/2)  [u=undo]")
    elif state.phase == "markers":
        ax.set_title(f"Click the left edge of each C key, left to right ({len(state.markers)}/{n_markers})  [u=undo]")
    else:
        ax.set_title("Done -- close window to continue")
    fig.canvas.draw_idle()


def on_click(event):
    if event.inaxes != ax or state.is_done():
        return
    x, y = event.xdata, event.ydata
    if state.phase == "corners":
        artist = ax.plot(x, y, "ro", markersize=8)[0]
        artists.append(artist)
        state.add_click(x, y)
    elif state.phase == "markers":
        artist = ax.axvline(x, color="lime", linewidth=1.5)
        artists.append(artist)
        state.add_click(x, y)
    redraw_title()
    if state.is_done():
        plt.close("all")


def on_key(event):
    if event.key == "u":
        if artists:
            artists.pop().remove()
        state.undo()
        redraw_title()


fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()

if not state.is_done():
    sys.exit("Picker closed before all points were collected")

if known_bbox is not None:
    display_bbox = tuple(known_bbox)
else:
    display_bbox = state.bbox()

# Inverse-transform from displayed coordinates back to native video coordinates.
if hands == "top":
    x1, y1 = display_bbox[0], display_bbox[1]
    x2, y2 = display_bbox[2], display_bbox[3]
    nx1, ny1 = w - 1 - x1, h - 1 - y1
    nx2, ny2 = w - 1 - x2, h - 1 - y2
    native_bbox = (min(nx1, nx2), min(ny1, ny2), max(nx1, nx2), max(ny1, ny2))
    native_markers = [w - 1 - x for x in state.markers]
else:
    native_bbox = display_bbox
    native_markers = list(state.markers)

result = {
    "bbox": [int(v) for v in native_bbox],
    "c_marker_xs": native_markers,
    "frame_width": w,
    "frame_height": h,
}
print(json.dumps(result))
'''


def _run_picker(video_path: PathLike, hands: str, n_markers: int, known_bbox=None) -> dict:
    bbox_arg = json.dumps(list(known_bbox)) if known_bbox is not None else "null"
    result = subprocess.run(
        [sys.executable, "-c", _PICKER_SCRIPT, str(video_path), hands, str(n_markers), bbox_arg],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=str(Path(__file__).resolve().parent),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Picker failed:\n{result.stderr.decode()}")
    return json.loads(result.stdout.decode().strip().split("\n")[-1])


def pick_keyboard_geometry(
    video_path: PathLike,
    hands: Literal["top", "bottom"],
    lowest_note: str = "A0",
    highest_note: str = "C8",
    known_bbox: tuple[int, int, int, int] | None = None,
) -> KeyboardGeometry:
    n_markers = len(c_pitches(lowest_note, highest_note))
    raw = _run_picker(video_path, hands, n_markers, known_bbox=known_bbox)
    return KeyboardGeometry(
        lowest_note=lowest_note,
        highest_note=highest_note,
        bbox=tuple(raw["bbox"]),
        c_marker_xs=raw["c_marker_xs"],
        hands=hands,
        frame_width=raw["frame_width"],
        frame_height=raw["frame_height"],
    )


def pick_bbox_only(video_path: PathLike) -> tuple[int, int, int, int]:
    """Bbox-only mode used by vit.py (no C markers, no --hands orientation
    logic since the corners are used exactly as clicked, native video space).
    """
    raw = _run_picker(video_path, hands="bottom", n_markers=0, known_bbox=None)
    return tuple(raw["bbox"])


def geometry_to_dict(geom: KeyboardGeometry) -> dict:
    return {
        "lowest_note": geom.lowest_note,
        "highest_note": geom.highest_note,
        "bbox": list(geom.bbox),
        "c_marker_xs": geom.c_marker_xs,
        "hands": geom.hands,
        "frame_width": geom.frame_width,
        "frame_height": geom.frame_height,
    }


def geometry_from_dict(d: dict) -> KeyboardGeometry:
    return KeyboardGeometry(
        lowest_note=d["lowest_note"],
        highest_note=d["highest_note"],
        bbox=tuple(d["bbox"]),
        c_marker_xs=list(d["c_marker_xs"]),
        hands=d["hands"],
        frame_width=d["frame_width"],
        frame_height=d["frame_height"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--hands", choices=["top", "bottom"], default="bottom")
    parser.add_argument("--lowest-note", default="A0")
    parser.add_argument("--highest-note", default="C8")
    parser.add_argument("--bbox", default=None, help="x1,y1,x2,y2 -- reduced mode, only C-markers get collected")
    parser.add_argument("--no-markers", action="store_true", help="bbox-only mode, used by vit.py")
    args = parser.parse_args()

    if args.no_markers:
        bbox = pick_bbox_only(args.video)
        print(json.dumps({"bbox": list(bbox)}))
        return

    known_bbox = None
    if args.bbox:
        known_bbox = tuple(int(v) for v in args.bbox.split(","))

    geom = pick_keyboard_geometry(
        args.video, hands=args.hands,
        lowest_note=args.lowest_note, highest_note=args.highest_note,
        known_bbox=known_bbox,
    )
    print(json.dumps(geometry_to_dict(geom)))


if __name__ == "__main__":
    main()
