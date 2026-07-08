"""Patch an audio-derived MIDI transcription using a video-derived pianoroll.

Audio-only piano transcription can't distinguish "the key is still down" from
"the sustain pedal is still ringing the note out", so notes tend to get
oversustained. A separate video pipeline (see vit.py / the ppan package)
predicts, per 1/30s frame, which keys are visually pressed. This script uses
that signal only to shorten notes: if the video shows a confident key-up
while the audio MIDI still has the note held, the note is truncated there.
Onset time, pitch, and velocity from the audio transcription are never
touched -- audio remains the source of truth for everything except how long
a note rings past its true physical release.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pretty_midi
from scipy.ndimage import binary_dilation

FPS = 30
LOWEST_PITCH = 21  # A0
HIGHEST_PITCH = 108  # C8


def load_video_pianoroll(path: Path) -> np.ndarray:
    """Load a (88, T) boolean pianoroll from a ppan .pkl or a .mid fallback.

    Row 0 corresponds to MIDI pitch 21 (A0), row 87 to MIDI pitch 108 (C8),
    matching the layout ppan.evaluate.evaluate_on_dataset writes out.
    """
    path = Path(path)
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            roll = pickle.load(f)
        return np.asarray(roll).astype(bool)
    if path.suffix in (".mid", ".midi"):
        pm = pretty_midi.PrettyMIDI(str(path))
        roll = pm.get_piano_roll(fs=FPS)[LOWEST_PITCH:HIGHEST_PITCH + 1] > 0
        return roll
    raise ValueError(f"Unsupported video pianoroll file type: {path.suffix}")


def estimate_offset(
    audio_notes: list[pretty_midi.Note],
    video_roll: np.ndarray,
    search_window: float = 15.0,
    fps: int = FPS,
) -> tuple[float, float]:
    """Cross-correlate audio note onsets against video key-down rising edges
    to estimate the time offset (video_time = audio_time - offset) that
    best aligns the two timelines. Returns (offset_seconds, score).
    """
    if not audio_notes:
        return 0.0, 0.0

    duration = max(n.end for n in audio_notes) + search_window
    n_bins = int(np.ceil(duration * fps)) + 1

    audio_signal = np.zeros(n_bins)
    for note in audio_notes:
        b = int(round(note.start * fps))
        if 0 <= b < n_bins:
            audio_signal[b] += 1

    rising = np.zeros_like(video_roll, dtype=bool)
    rising[:, 1:] = (~video_roll[:, :-1]) & video_roll[:, 1:]
    rising[:, 0] = video_roll[:, 0]
    video_signal = rising.sum(axis=0).astype(float)

    max_lag = int(round(search_window * fps))
    best_lag = 0
    best_score = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        # video_signal shifted by lag against audio_signal, i.e. we test
        # aligning video frame (i - lag) with audio bin i.
        if lag >= 0:
            a = audio_signal[lag:]
            v = video_signal[: len(a)]
        else:
            v = video_signal[-lag:]
            a = audio_signal[: len(v)]
        n = min(len(a), len(v))
        if n == 0:
            continue
        score = float(np.dot(a[:n], v[:n]))
        if score > best_score:
            best_score = score
            best_lag = lag

    offset_seconds = best_lag / fps
    return offset_seconds, best_score


def patch_notes(
    audio_notes: list[pretty_midi.Note],
    video_roll: np.ndarray,
    visible_keys: tuple[int, int],
    offset: float = 0.0,
    fps: int = FPS,
    search_margin: float = 1.0,
    min_confirm_frames: int = 3,
    min_shrink_sec: float = 0.05,
    min_note_dur: float = 0.03,
) -> list[pretty_midi.Note]:
    """Return a new list of notes with offsets shortened wherever the video
    pianoroll shows a confident key-up before the audio-predicted note end.
    """
    lo, hi = visible_keys
    n_frames = video_roll.shape[1]
    patched = []

    for note in audio_notes:
        new_note = pretty_midi.Note(
            velocity=note.velocity, pitch=note.pitch,
            start=note.start, end=note.end,
        )
        patched.append(new_note)

        if not (lo <= note.pitch <= hi):
            continue

        row_idx = note.pitch - LOWEST_PITCH
        if not (0 <= row_idx < video_roll.shape[0]):
            continue
        row = video_roll[row_idx]

        start_frame = max(0, int(round((note.start - offset) * fps)))
        end_frame = int(round((note.end - offset) * fps))
        search_end = min(n_frames, end_frame + int(round(search_margin * fps)))
        if start_frame >= search_end:
            continue

        key_up_frame = _find_confirmed_key_up(
            row, start_frame, search_end, min_confirm_frames
        )
        if key_up_frame is None:
            continue

        candidate_end = key_up_frame / fps + offset
        if candidate_end >= note.end - min_shrink_sec:
            continue

        new_end = max(candidate_end, note.start + min_note_dur)
        if new_end < new_note.end:
            new_note.end = new_end

    return patched


def _find_confirmed_key_up(
    row: np.ndarray, start_frame: int, search_end: int, min_confirm_frames: int
) -> int | None:
    """First frame index >= start_frame where `row` goes False and stays
    False for at least min_confirm_frames consecutive frames. None if no
    such run exists before search_end.
    """
    run_len = 0
    for f in range(start_frame, search_end):
        if not row[f]:
            run_len += 1
            if run_len == min_confirm_frames:
                return f - min_confirm_frames + 1
        else:
            run_len = 0
    return None


def compute_reachable_mask(
    video_roll: np.ndarray,
    margin_keys: int,
    hold_frames: int,
    fallback_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive an approximate "hands could be here" mask from the video
    pianoroll, without any pixel-level hand detection.

    Returns (reachable, evidence):
    - reachable[key, frame]: a key within margin_keys semitones of a
      visually-active key, within hold_frames in time (bridges brief
      per-key video dropouts). Dilating around each active key individually
      (rather than an envelope across all active keys) means that when both
      hands are active far apart at once, the empty register between them
      is not marked reachable.
    - evidence[frame]: whether there was ANY video activity nearby in time
      (any pitch), using a larger fallback window. Frames without evidence
      mean "unknown", never "nothing reachable".
    """
    reach_struct = np.ones((2 * margin_keys + 1, 2 * hold_frames + 1), dtype=bool)
    reachable = binary_dilation(video_roll, structure=reach_struct)

    any_active = video_roll.any(axis=0, keepdims=True)
    time_struct = np.ones((1, 2 * fallback_frames + 1), dtype=bool)
    evidence = binary_dilation(any_active, structure=time_struct)[0]

    return reachable, evidence


def should_prune_note(
    row_idx: int,
    start_frame: int,
    end_frame: int,
    reachable: np.ndarray,
    evidence: np.ndarray,
    min_evidence_frames: int,
) -> bool:
    """Whether a note should be dropped entirely: true only if there's
    enough nearby video evidence across the note's whole span, and every
    evidence-bearing frame says the pitch was unreachable.
    """
    n_frames = len(evidence)
    start_frame = max(0, start_frame)
    end_frame = min(n_frames, end_frame)
    if start_frame >= end_frame:
        return False

    ev = evidence[start_frame:end_frame]
    if ev.sum() < min_evidence_frames:
        return False

    reach = reachable[row_idx, start_frame:end_frame]
    return not reach[ev].any()


def prune_notes(
    audio_notes: list[pretty_midi.Note],
    video_roll: np.ndarray,
    visible_keys: tuple[int, int],
    offset: float = 0.0,
    fps: int = FPS,
    margin_keys: int = 6,
    hold_sec: float = 0.5,
    fallback_sec: float = 2.0,
    min_evidence_frames: int = 3,
    verbose: bool = False,
) -> list[pretty_midi.Note]:
    """Drop whole notes that fall outside the video's hand-reachable key
    range for their entire duration -- a conservative first pass against
    stray audio-model false positives, run before patch_notes.
    """
    lo_key, hi_key = visible_keys
    hold_frames = max(0, round(hold_sec * fps))
    fallback_frames = max(hold_frames, round(fallback_sec * fps))
    reachable, evidence = compute_reachable_mask(
        video_roll, margin_keys=margin_keys, hold_frames=hold_frames,
        fallback_frames=fallback_frames,
    )

    kept = []
    for note in audio_notes:
        if not (lo_key <= note.pitch <= hi_key):
            kept.append(note)  # outside camera crop -- video has no opinion
            continue

        row_idx = note.pitch - LOWEST_PITCH
        if not (0 <= row_idx < video_roll.shape[0]):
            kept.append(note)
            continue

        start_frame = max(0, int(round((note.start - offset) * fps)))
        end_frame = int(round((note.end - offset) * fps))
        if should_prune_note(row_idx, start_frame, end_frame, reachable,
                             evidence, min_evidence_frames):
            if verbose:
                name = pretty_midi.note_number_to_name(note.pitch)
                print(f"  pruned: t={note.start:.3f}s  {name} ({note.pitch})  "
                      f"dur={note.end - note.start:.3f}s")
            continue

        kept.append(note)

    return kept


def patch_midi(
    audio_midi_path: Path,
    video_roll_path: Path,
    output_path: Path,
    visible_keys: tuple[int, int],
    offset: float | None,
    search_window: float,
    search_margin: float,
    min_confirm_frames: int,
    min_shrink_sec: float,
    min_note_dur: float,
    prune: bool,
    prune_margin_keys: int,
    prune_hold_sec: float,
    prune_fallback_sec: float,
    min_prune_evidence_frames: int,
    verbose: bool = False,
) -> None:
    audio_pm = pretty_midi.PrettyMIDI(str(audio_midi_path))
    video_roll = load_video_pianoroll(video_roll_path)

    audio_notes = [n for inst in audio_pm.instruments for n in inst.notes]

    if offset is None:
        offset, score = estimate_offset(audio_notes, video_roll, search_window)
        print(f"Auto-aligned offset: {offset:.3f}s (score={score:.1f})")
    else:
        print(f"Using manual offset: {offset:.3f}s")

    if prune:
        survivors = prune_notes(
            audio_notes,
            video_roll,
            visible_keys=visible_keys,
            offset=offset,
            margin_keys=prune_margin_keys,
            hold_sec=prune_hold_sec,
            fallback_sec=prune_fallback_sec,
            min_evidence_frames=min_prune_evidence_frames,
            verbose=verbose,
        )
    else:
        survivors = audio_notes

    n_pruned = len(audio_notes) - len(survivors)
    print(f"Pruned {n_pruned}/{len(audio_notes)} notes")

    patched = patch_notes(
        survivors,
        video_roll,
        visible_keys=visible_keys,
        offset=offset,
        search_margin=search_margin,
        min_confirm_frames=min_confirm_frames,
        min_shrink_sec=min_shrink_sec,
        min_note_dur=min_note_dur,
    )

    n_shortened = sum(
        1 for orig, new in zip(survivors, patched) if new.end < orig.end
    )
    print(f"Shortened {n_shortened}/{len(survivors)} notes")

    out_pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes = patched
    out_pm.instruments.append(inst)
    out_pm.write(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_midi", type=Path)
    parser.add_argument("video_pianoroll", type=Path,
                        help=".pkl from ppan.evaluate.evaluate_on_dataset, or a .mid")
    parser.add_argument("output_midi", type=Path)
    parser.add_argument("--visible-keys", type=int, nargs=2,
                        default=[LOWEST_PITCH, HIGHEST_PITCH],
                        metavar=("LOW", "HIGH"),
                        help="MIDI pitch range visible in the video's camera crop "
                             f"(default: {LOWEST_PITCH} {HIGHEST_PITCH}, full 88 keys)")
    parser.add_argument("--offset", type=float, default=None,
                        help="Manual video-vs-audio offset in seconds; "
                             "skips auto-alignment if given")
    parser.add_argument("--search-window", type=float, default=15.0)
    parser.add_argument("--search-margin", type=float, default=1.0)
    parser.add_argument("--min-confirm-frames", type=int, default=3)
    parser.add_argument("--min-shrink-sec", type=float, default=0.05)
    parser.add_argument("--min-note-dur", type=float, default=0.03)
    parser.add_argument("--prune", action=argparse.BooleanOptionalAction, default=True,
                        help="Drop whole audio notes outside the video's hand-reachable "
                             "key range for their entire duration, before release-"
                             "shortening (default: enabled; use --no-prune to disable)")
    parser.add_argument("--prune-margin-keys", type=int, default=6,
                        help="Semitones of slack added around visually-active keys to "
                             "approximate hand reach (default: 6)")
    parser.add_argument("--prune-hold-sec", type=float, default=0.5,
                        help="Temporal window (seconds, each direction) used to bridge "
                             "brief per-key video dropouts (default: 0.5)")
    parser.add_argument("--prune-fallback-sec", type=float, default=2.0,
                        help="Larger fallback window (seconds, each direction) used only "
                             "to check whether ANY video evidence exists nearby; if none "
                             "is found even here, the note is kept (default: 2.0)")
    parser.add_argument("--min-prune-evidence-frames", type=int, default=3,
                        help="Minimum evidence-bearing frames within a note's span "
                             "required before a prune decision is made (default: 3)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print each pruned note's start time, pitch, and duration")
    args = parser.parse_args()

    patch_midi(
        audio_midi_path=args.audio_midi,
        video_roll_path=args.video_pianoroll,
        output_path=args.output_midi,
        visible_keys=tuple(args.visible_keys),
        offset=args.offset,
        search_window=args.search_window,
        search_margin=args.search_margin,
        min_confirm_frames=args.min_confirm_frames,
        min_shrink_sec=args.min_shrink_sec,
        min_note_dur=args.min_note_dur,
        prune=args.prune,
        prune_margin_keys=args.prune_margin_keys,
        prune_hold_sec=args.prune_hold_sec,
        prune_fallback_sec=args.prune_fallback_sec,
        min_prune_evidence_frames=args.min_prune_evidence_frames,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
