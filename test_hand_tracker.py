import pytest

from hand_tracker import filter_and_rank_detections, track_hands

KEYBOARD_BBOX = (0, 0, 1000, 1000)


def left_box(f):
    x = 100 + 10 * f
    return (x, 100, x + 10, 110)


def right_box(f):
    x = 600 - 10 * f
    return (x, 100, x + 10, 110)


def approx_box(a, b, tol=1e-6):
    return all(abs(a[i] - b[i]) < tol for i in range(4))


def test_continuous_no_gap_two_hand_tracking():
    raw = []
    for f in range(6):
        raw.append([(left_box(f), 0.9), (right_box(f), 0.9)])

    out = track_hands(raw_detections=raw, keyboard_bbox=KEYBOARD_BBOX, num_hands=2)

    for f in range(6):
        boxes = out[f]
        assert len(boxes) == 2
        assert any(approx_box(b, left_box(f)) for b in boxes)
        assert any(approx_box(b, right_box(f)) for b in boxes)


def test_single_hand_gap_interpolated_and_reattached():
    raw = []
    for f in range(8):
        if 3 <= f <= 5:
            raw.append([])  # hand disappeared for frames 3,4,5
        else:
            raw.append([(left_box(f), 0.9)])

    out = track_hands(raw_detections=raw, keyboard_bbox=KEYBOARD_BBOX,
                       num_hands=1, max_disappear_frames=5)

    for f in [3, 4, 5]:
        assert f in out
        assert len(out[f]) == 1
        assert approx_box(out[f][0], left_box(f), tol=1e-6)


def test_overlapping_disappearances_do_not_cross_match():
    # left present 0-2, missing 3-6, present 7-9
    # right present 0-3, missing 4-7, present 8-9
    raw = []
    for f in range(10):
        dets = []
        if f <= 2 or f >= 7:
            dets.append((left_box(f), 0.9))
        if f <= 3 or f >= 8:
            dets.append((right_box(f), 0.9))
        raw.append(dets)

    out = track_hands(raw_detections=raw, keyboard_bbox=KEYBOARD_BBOX,
                       num_hands=2, max_disappear_frames=5)

    for f in [3, 4, 5, 6]:
        assert any(approx_box(b, left_box(f)) for b in out[f]), f"frame {f} missing left interpolation"
    for f in [4, 5, 6, 7]:
        assert any(approx_box(b, right_box(f)) for b in out[f]), f"frame {f} missing right interpolation"

    # Frames 4-6: right hand is genuinely absent (mid-gap), so its only
    # appearance in out[f] must come from its own interpolated trajectory,
    # not from a mismatch that duplicates/hijacks the left hand's box.
    for f in [4, 5, 6]:
        assert not any(approx_box(b, left_box(f)) and approx_box(b, right_box(f)) for b in out[f])
        assert len(out[f]) == 2


def test_gap_exceeding_budget_expires_track_no_interpolation():
    raw = []
    for f in range(12):
        if 3 <= f <= 9:  # 7-frame gap, exceeds max_disappear_frames=5
            raw.append([])
        else:
            raw.append([(left_box(f), 0.9)])

    out = track_hands(raw_detections=raw, keyboard_bbox=KEYBOARD_BBOX,
                       num_hands=1, max_disappear_frames=5)

    for f in range(3, 10):
        assert f not in out
    assert 10 in out
    assert approx_box(out[10][0], left_box(10))


def test_spurious_detection_does_not_hijack_reappearance():
    raw = []
    for f in range(8):
        dets = []
        if f not in (3, 4, 5):
            dets.append((left_box(f), 0.9))
        if f == 4:
            # Far-away spurious detection during the legitimate gap.
            dets.append(((900, 900, 910, 910), 0.3))
        raw.append(dets)

    out = track_hands(raw_detections=raw, keyboard_bbox=KEYBOARD_BBOX,
                       num_hands=2, max_disappear_frames=5, max_match_distance_px=50)

    for f in [3, 4, 5]:
        assert any(approx_box(b, left_box(f)) for b in out[f]), f"frame {f} true hand not interpolated correctly"


def test_filter_and_rank_drops_non_intersecting_and_caps_by_confidence():
    dets = [
        ((10, 10, 20, 20), 0.9),   # inside keyboard, high conf
        ((15, 15, 25, 25), 0.8),   # inside keyboard, medium conf
        ((30, 30, 40, 40), 0.95),  # inside keyboard, highest conf
        ((2000, 2000, 2010, 2010), 0.99),  # outside keyboard bbox entirely
    ]
    kept = filter_and_rank_detections(dets, KEYBOARD_BBOX, num_hands=2, conf_threshold=0.5)
    assert len(kept) == 2
    assert (30, 30, 40, 40) in kept
    assert (10, 10, 20, 20) in kept
