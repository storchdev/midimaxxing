"""Prune audio-transcribed MIDI notes that no hand could physically reach.

Runs before patch_midi.py in the full (non-audio-only) pipeline. Uses a
YOLO hand-detection model tracked across the video (hand_tracker.py) plus
the annotated keyboard geometry (keyboard_geometry.py / keyboard_picker.py)
to work out which keys were reachable at each note's onset frame, and drops
any note whose key no tracked hand was near at onset. Fail-open: if hand
data is missing/uncertain at a note's onset, the note is kept. Reachability
is checked only at the note's onset frame, not across the held duration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pretty_midi

from hand_tracker import track_hands
from keyboard_geometry import KeyboardGeometry, flip_geometry, key_x_range, parse_note
from keyboard_picker import geometry_from_dict

FPS = 30


def prune_midi(
    notes: list[pretty_midi.Note],
    geometry: KeyboardGeometry,
    hand_track: dict[int, list[tuple[float, float, float, float]]],
    hands: str,
    offset: float = 0.0,
    fps: int = FPS,
    margin_px: float = 0.0,
) -> list[pretty_midi.Note]:
    """Return the subset of `notes` reachable by a tracked hand at onset."""
    if geometry.hands != hands:
        geometry = flip_geometry(geometry)

    lowest_pitch = parse_note(geometry.lowest_note)
    highest_pitch = parse_note(geometry.highest_note)

    kept = []
    for note in notes:
        if not (lowest_pitch <= note.pitch <= highest_pitch):
            kept.append(note)
            continue

        onset_frame = round((note.start - offset) * fps)
        hand_boxes = hand_track.get(onset_frame)
        if not hand_boxes:
            kept.append(note)  # fail-open: no hand data at onset
            continue

        lo, hi = key_x_range(geometry, note.pitch)
        lo -= margin_px
        hi += margin_px

        reachable = any(
            box[0] <= hi and box[2] >= lo
            for box in hand_boxes
        )
        if reachable:
            kept.append(note)

    return kept


def _default_margin_px(geometry: KeyboardGeometry) -> float:
    lowest_pitch = parse_note(geometry.lowest_note)
    highest_pitch = parse_note(geometry.highest_note)
    span = geometry.bbox[2] - geometry.bbox[0]
    n_keys = max(highest_pitch - lowest_pitch, 1)
    avg_key_width = abs(span) / n_keys
    return avg_key_width / 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_midi", type=Path)
    parser.add_argument("keyboard_geometry", type=Path, help="JSON from keyboard_picker.py")
    parser.add_argument("video", type=Path)
    parser.add_argument("output_midi", type=Path)
    parser.add_argument("--hands", choices=["top", "bottom"], required=True)
    parser.add_argument("--offset", type=float, default=None,
                         help="video-vs-audio offset in seconds")
    parser.add_argument("--video-pkl", type=Path, default=None,
                         help="video pianoroll .pkl, used to auto-estimate --offset if omitted")
    parser.add_argument("--num-hands", type=int, default=2)
    parser.add_argument("--max-disappear-frames", type=int, default=15)
    parser.add_argument("--reach-margin-px", type=float, default=None)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--max-match-distance-px", type=float, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    args = parser.parse_args()

    with open(args.keyboard_geometry) as f:
        geometry = geometry_from_dict(json.load(f))

    audio_pm = pretty_midi.PrettyMIDI(str(args.audio_midi))
    notes = [n for inst in audio_pm.instruments for n in inst.notes]

    offset = args.offset
    if offset is None:
        if args.video_pkl is not None:
            from patch_midi import estimate_offset, load_video_pianoroll
            video_roll = load_video_pianoroll(args.video_pkl)
            offset, score = estimate_offset(notes, video_roll)
            print(f"Auto-aligned offset: {offset:.3f}s (score={score:.1f})")
        else:
            offset = 0.0
            print("Warning: no --offset or --video-pkl given, defaulting offset to 0.0")

    margin_px = args.reach_margin_px
    if margin_px is None:
        margin_px = _default_margin_px(geometry)

    hand_track = track_hands(
        video_path=args.video,
        hands=args.hands,
        keyboard_bbox=geometry.bbox if geometry.hands == args.hands
        else flip_geometry(geometry).bbox,
        num_hands=args.num_hands,
        max_disappear_frames=args.max_disappear_frames,
        conf_threshold=args.yolo_conf,
        max_match_distance_px=args.max_match_distance_px,
        weights_path=args.weights,
    )

    pruned = prune_midi(
        notes, geometry, hand_track, hands=args.hands,
        offset=offset, margin_px=margin_px,
    )

    print(f"Kept {len(pruned)}/{len(notes)} notes")

    out_pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes = pruned
    out_pm.instruments.append(inst)
    out_pm.write(str(args.output_midi))


if __name__ == "__main__":
    main()
