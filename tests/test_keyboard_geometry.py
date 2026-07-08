import pytest

from keyboard_geometry import (
    KeyboardGeometry,
    c_pitches,
    flip_bbox_180,
    flip_geometry,
    flip_point_180,
    key_x_range,
    parse_note,
    pitch_to_note,
)


def test_parse_note_round_trip():
    for name, pitch in [("A0", 21), ("C4", 60), ("C8", 108), ("C#4", 61)]:
        assert parse_note(name) == pitch
    assert pitch_to_note(60) == "C4"
    assert pitch_to_note(21) == "A0"
    assert pitch_to_note(parse_note("C4")) == "C4"


def test_c_pitches_default_full_range():
    pitches = c_pitches("A0", "C8")
    assert pitches == [24, 36, 48, 60, 72, 84, 96, 108]


def test_c_pitches_custom_narrow_range():
    pitches = c_pitches("C3", "C5")
    assert pitches == [48, 60, 72]


def make_geom(c_marker_xs, lowest="C3", highest="C5", bbox=(0, 0, 1000, 100),
              hands="bottom", w=1000, h=1000):
    return KeyboardGeometry(
        lowest_note=lowest, highest_note=highest, bbox=bbox,
        c_marker_xs=c_marker_xs, hands=hands, frame_width=w, frame_height=h,
    )


def test_interpolation_exact_at_c_markers():
    geom = make_geom([100.0, 300.0, 500.0])
    lo, hi = key_x_range(geom, 60)  # C4, middle marker
    assert lo < 300.0 < hi


def test_interpolation_mid_octave_key():
    geom = make_geom([100.0, 300.0, 500.0])  # C3=48@100, C4=60@300, C5=72@500
    lo, hi = key_x_range(geom, 54)  # F#3, halfway between C3 and C4
    center = (lo + hi) / 2
    assert abs(center - 200.0) < 1e-6


def test_extrapolation_below_lowest_c():
    geom = make_geom([100.0, 300.0, 500.0])
    lo, hi = key_x_range(geom, 45)  # A2, below C3
    center = (lo + hi) / 2
    assert center < 100.0


def test_extrapolation_above_highest_c():
    geom = make_geom([100.0, 300.0, 500.0])
    lo, hi = key_x_range(geom, 76)  # E5, above C5
    center = (lo + hi) / 2
    assert center > 500.0


def test_flip_round_trip_identity():
    geom = make_geom([100.0, 300.0, 500.0], bbox=(50, 20, 900, 80), w=1000, h=1000)
    flipped = flip_geometry(geom)
    back = flip_geometry(flipped)
    assert back.bbox == geom.bbox
    assert back.c_marker_xs == pytest.approx(geom.c_marker_xs)
    assert back.hands == geom.hands


def test_flip_point_and_bbox():
    x, y = flip_point_180(10, 20, 100, 200)
    assert (x, y) == (89, 179)
    bbox = flip_bbox_180((10, 20, 30, 40), 100, 200)
    assert bbox == (69, 159, 89, 179)


def test_interpolation_with_descending_marker_xs():
    # Simulates --hands top: user clicks left-to-right (ascending pitch) on
    # screen, but the display was rotated 180°, so stored native x's descend.
    geom = make_geom([500.0, 300.0, 100.0], hands="top")  # C3=500, C4=300, C5=100
    lo3, hi3 = key_x_range(geom, 48)
    lo4, hi4 = key_x_range(geom, 60)
    lo5, hi5 = key_x_range(geom, 72)
    assert lo3 < hi3
    assert lo4 < hi4
    assert lo5 < hi5
    # Pitch order still ascends 48 -> 60 -> 72, but x descends 500 -> 300 -> 100.
    assert (lo3 + hi3) / 2 > (lo4 + hi4) / 2 > (lo5 + hi5) / 2


def test_single_marker_degenerate_fallback():
    geom = make_geom([500.0], lowest="B3", highest="D4", bbox=(400, 0, 600, 100))
    lo, hi = key_x_range(geom, 60)  # C4, the only C marker
    assert lo < 500.0 < hi
