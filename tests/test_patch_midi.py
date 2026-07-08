import numpy as np
import pretty_midi

from patch_midi import FPS, LOWEST_PITCH, estimate_offset, patch_notes


def make_roll(n_frames: int, n_keys: int = 88) -> np.ndarray:
    return np.zeros((n_keys, n_frames), dtype=bool)


def press(roll: np.ndarray, pitch: int, start_frame: int, end_frame: int) -> None:
    roll[pitch - LOWEST_PITCH, start_frame:end_frame] = True


def test_oversustained_note_is_shortened():
    # Audio says the note (pedal-rung) lasts 0.0 - 3.0s, but the video shows
    # the key physically released at 1.0s.
    note = pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=3.0)
    roll = make_roll(n_frames=4 * FPS)
    press(roll, 60, 0, 1 * FPS)  # key down 0.0 - 1.0s, then released

    patched = patch_notes([note], roll, visible_keys=(21, 108))

    assert len(patched) == 1
    assert patched[0].start == note.start
    assert abs(patched[0].end - 1.0) < 1e-6
    assert patched[0].pitch == note.pitch
    assert patched[0].velocity == note.velocity


def test_note_matching_video_is_untouched():
    note = pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=1.0)
    roll = make_roll(n_frames=2 * FPS)
    press(roll, 60, 0, int(1.0 * FPS) + 5)  # still pressed through note end

    patched = patch_notes([note], roll, visible_keys=(21, 108))

    assert patched[0].end == note.end


def test_single_frame_flicker_is_ignored():
    note = pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=2.0)
    roll = make_roll(n_frames=3 * FPS)
    press(roll, 60, 0, 3 * FPS)
    # One-frame dropout at frame 30 shouldn't count as a confirmed release.
    roll[60 - LOWEST_PITCH, 30] = False

    patched = patch_notes([note], roll, visible_keys=(21, 108),
                          min_confirm_frames=3)

    assert patched[0].end == note.end


def test_note_outside_visible_keys_is_untouched():
    note = pretty_midi.Note(velocity=80, pitch=30, start=0.0, end=3.0)
    roll = make_roll(n_frames=4 * FPS)  # all False, would look oversustained

    patched = patch_notes([note], roll, visible_keys=(60, 84))

    assert patched[0].end == note.end


def test_estimate_offset_recovers_known_lag():
    fps = FPS
    true_offset = 2.0  # video_time = audio_time - offset
    onset_times = [0.5, 1.5, 3.0, 4.2, 6.0]
    notes = [
        pretty_midi.Note(velocity=80, pitch=60, start=t, end=t + 0.5)
        for t in onset_times
    ]

    n_frames = int((max(onset_times) + 5) * fps)
    roll = make_roll(n_frames=n_frames)
    for t in onset_times:
        video_frame = int(round((t - true_offset) * fps))
        press(roll, 60, video_frame, video_frame + 5)

    offset, score = estimate_offset(notes, roll, search_window=10.0)

    assert abs(offset - true_offset) < 1e-6
    assert score > 0
