from keyboard_picker import PickerState


def test_full_phase_progression():
    state = PickerState(n_markers=3)
    assert state.phase == "corners"
    state.add_click(10, 10)
    assert state.phase == "corners"
    state.add_click(90, 90)
    assert state.phase == "markers"
    assert state.bbox() == (10, 10, 90, 90)

    state.add_click(20, 0)
    state.add_click(50, 0)
    assert not state.is_done()
    state.add_click(80, 0)
    assert state.is_done()
    assert state.markers == [20, 50, 80]


def test_reduced_mode_starts_in_markers_phase():
    state = PickerState(n_markers=2, phase="markers")
    assert state.phase == "markers"
    state.add_click(30, 0)
    assert not state.is_done()
    state.add_click(60, 0)
    assert state.is_done()


def test_zero_markers_reduced_mode_is_immediately_done():
    state = PickerState(n_markers=0, phase="markers")
    assert state.is_done()


def test_undo_within_markers_phase():
    state = PickerState(n_markers=2)
    state.add_click(0, 0)
    state.add_click(100, 100)
    state.add_click(5, 0)
    assert state.markers == [5]
    state.undo()
    assert state.markers == []
    assert state.phase == "markers"


def test_undo_crosses_back_into_corners_phase():
    state = PickerState(n_markers=2)
    state.add_click(0, 0)
    state.add_click(100, 100)
    assert state.phase == "markers"
    assert state.markers == []
    state.undo()  # nothing to pop in markers -> revert to corners, pop a corner
    assert state.phase == "corners"
    assert state.corners == [(0, 0)]


def test_undo_after_done_reverts_to_markers():
    state = PickerState(n_markers=1)
    state.add_click(0, 0)
    state.add_click(100, 100)
    state.add_click(50, 0)
    assert state.is_done()
    state.undo()
    assert state.phase == "markers"
    assert state.markers == []


def test_undo_on_empty_state_is_noop():
    state = PickerState(n_markers=2)
    state.undo()
    assert state.phase == "corners"
    assert state.corners == []
