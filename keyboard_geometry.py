"""Pure geometry/math for mapping piano keys to pixel x-ranges.

No I/O, no GUI. Used by keyboard_picker.py (to build a KeyboardGeometry from
clicks) and prune_midi.py (to turn it into per-pitch pixel ranges).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

_NOTE_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")
_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def parse_note(name: str) -> int:
    """Parse a note name like 'A0' or 'C#4' into a MIDI pitch number."""
    m = _NOTE_RE.match(name.strip())
    if not m:
        raise ValueError(f"Invalid note name: {name!r}")
    letter, accidental, octave_str = m.groups()
    semitone = _SEMITONES[letter.upper()]
    if accidental == "#":
        semitone += 1
    elif accidental == "b":
        semitone -= 1
    octave = int(octave_str)
    return (octave + 1) * 12 + semitone


def pitch_to_note(pitch: int) -> str:
    """Inverse of parse_note, always spelled with sharps (no flats)."""
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = pitch // 12 - 1
    return f"{names[pitch % 12]}{octave}"


def c_pitches(lowest: str, highest: str) -> list[int]:
    """Every C pitch (pitch % 12 == 0) between lowest and highest, ascending."""
    lo, hi = parse_note(lowest), parse_note(highest)
    if lo > hi:
        raise ValueError(f"lowest ({lowest}) must be <= highest ({highest})")
    start = lo + (-lo) % 12  # smallest pitch >= lo with pitch % 12 == 0
    return list(range(start, hi + 1, 12))


@dataclass
class KeyboardGeometry:
    lowest_note: str
    highest_note: str
    bbox: tuple[int, int, int, int]
    c_marker_xs: list[float]
    hands: Literal["top", "bottom"]
    frame_width: int
    frame_height: int


def key_x_range(geom: KeyboardGeometry, pitch: int) -> tuple[float, float]:
    """Return the (lo, hi) pixel x-range spanned by `pitch`'s key.

    Anchors are (c_pitch, c_marker_x) pairs, index-aligned in ascending-pitch
    click order -- x may ascend or descend depending on display orientation,
    so the slope between anchors is signed and never re-sorted by x.
    """
    c_pitch_list = c_pitches(geom.lowest_note, geom.highest_note)
    anchors = list(zip(c_pitch_list, geom.c_marker_xs))
    if not anchors:
        raise ValueError("No C markers in geometry")

    if len(anchors) == 1:
        lo_pitch = parse_note(geom.lowest_note)
        hi_pitch = parse_note(geom.highest_note)
        x1, x2 = geom.bbox[0], geom.bbox[2]
        span = hi_pitch - lo_pitch
        slope = (x2 - x1) / span if span else 0.0
        p0, x0 = anchors[0]
    else:
        # Bracket pitch between the nearest two anchors; extrapolate off the
        # first/last pair if pitch is outside the anchor range.
        if pitch <= anchors[0][0]:
            (p0, x0), (p1, x1) = anchors[0], anchors[1]
        elif pitch >= anchors[-1][0]:
            (p0, x0), (p1, x1) = anchors[-2], anchors[-1]
        else:
            (p0, x0), (p1, x1) = anchors[0], anchors[1]
            for i in range(len(anchors) - 1):
                lo_p, lo_x = anchors[i]
                hi_p, hi_x = anchors[i + 1]
                if lo_p <= pitch <= hi_p:
                    p0, x0, p1, x1 = lo_p, lo_x, hi_p, hi_x
                    break
        slope = (x1 - x0) / (p1 - p0)

    center = x0 + slope * (pitch - p0)
    half = abs(slope) / 2
    return (center - half, center + half)


def flip_point_180(x: float, y: float, w: int, h: int) -> tuple[float, float]:
    return (w - 1 - x, h - 1 - y)


def flip_bbox_180(bbox: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1 = flip_point_180(bbox[0], bbox[1], w, h)
    x2, y2 = flip_point_180(bbox[2], bbox[3], w, h)
    return (
        int(min(x1, x2)), int(min(y1, y2)),
        int(max(x1, x2)), int(max(y1, y2)),
    )


def flip_geometry(geom: KeyboardGeometry) -> KeyboardGeometry:
    """Flip bbox and each c_marker_xs element 180°, elementwise (no re-sort)."""
    flipped_bbox = flip_bbox_180(geom.bbox, geom.frame_width, geom.frame_height)
    flipped_markers = [
        geom.frame_width - 1 - x for x in geom.c_marker_xs
    ]
    other_hands = "bottom" if geom.hands == "top" else "top"
    return KeyboardGeometry(
        lowest_note=geom.lowest_note,
        highest_note=geom.highest_note,
        bbox=flipped_bbox,
        c_marker_xs=flipped_markers,
        hands=other_hands,
        frame_width=geom.frame_width,
        frame_height=geom.frame_height,
    )
