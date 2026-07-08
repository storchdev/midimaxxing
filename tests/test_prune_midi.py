import pretty_midi

from keyboard_geometry import KeyboardGeometry
from prune_midi import prune_midi

FPS = 30


def make_geom(c_marker_xs=(100.0, 300.0, 500.0), lowest="C3", highest="C5",
              bbox=(0, 0, 1000, 200), hands="bottom"):
    return KeyboardGeometry(
        lowest_note=lowest, highest_note=highest, bbox=bbox,
        c_marker_xs=list(c_marker_xs), hands=hands,
        frame_width=1000, frame_height=1000,
    )


def note(pitch, start, end=None, velocity=80):
    return pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end or start + 0.5)


def test_note_dropped_when_no_hand_overlaps_at_onset():
    geom = make_geom()
    n = note(pitch=60, start=1.0)  # C4 -> x=300
    onset_frame = round(1.0 * FPS)
    hand_track = {onset_frame: [(700.0, 0, 750.0, 50)]}  # far from x=300

    kept = prune_midi([n], geom, hand_track, hands="bottom")
    assert kept == []


def test_note_kept_when_hand_overlaps_at_onset():
    geom = make_geom()
    n = note(pitch=60, start=1.0)  # C4 -> x=300
    onset_frame = round(1.0 * FPS)
    hand_track = {onset_frame: [(280.0, 0, 320.0, 50)]}

    kept = prune_midi([n], geom, hand_track, hands="bottom")
    assert len(kept) == 1
    assert kept[0].pitch == 60


def test_onset_only_check_ignores_later_hand_movement():
    geom = make_geom()
    n = note(pitch=60, start=1.0, end=5.0)  # long sustained note
    onset_frame = round(1.0 * FPS)
    # Hand is present at onset (reachable) but this is the only tracked frame
    # -- later frames have no data, which must not affect the decision since
    # only the onset frame is inspected.
    hand_track = {onset_frame: [(280.0, 0, 320.0, 50)]}

    kept = prune_midi([n], geom, hand_track, hands="bottom")
    assert len(kept) == 1
    assert kept[0].end == 5.0


def test_fail_open_when_no_hand_track_data_at_onset():
    geom = make_geom()
    n = note(pitch=60, start=1.0)
    hand_track = {}  # nothing tracked anywhere

    kept = prune_midi([n], geom, hand_track, hands="bottom")
    assert len(kept) == 1


def test_reach_margin_flips_borderline_case():
    geom = make_geom()
    n = note(pitch=60, start=1.0)  # C4 -> x=300, key half-width small
    onset_frame = round(1.0 * FPS)
    # Hand box just outside the raw key range but within a generous margin.
    hand_track = {onset_frame: [(320.0, 0, 340.0, 50)]}

    kept_no_margin = prune_midi([n], geom, hand_track, hands="bottom", margin_px=0.0)
    kept_with_margin = prune_midi([n], geom, hand_track, hands="bottom", margin_px=50.0)

    assert kept_no_margin == []
    assert len(kept_with_margin) == 1


def test_note_outside_annotated_range_always_kept():
    geom = make_geom(lowest="C3", highest="C5")  # pitches 48-72
    n = note(pitch=30, start=1.0)  # far below range
    hand_track = {}

    kept = prune_midi([n], geom, hand_track, hands="bottom")
    assert len(kept) == 1


def test_geometry_flipped_when_orientation_mismatches_tracking():
    # Geometry stored in "bottom" space; tracking happened in "top" space.
    geom = make_geom(c_marker_xs=(100.0, 300.0, 500.0), hands="bottom")
    n = note(pitch=60, start=1.0)  # native x=300 -> flipped x = 1000-1-300 = 699
    onset_frame = round(1.0 * FPS)
    hand_track = {onset_frame: [(680.0, 0, 720.0, 50)]}

    kept = prune_midi([n], geom, hand_track, hands="top")
    assert len(kept) == 1
