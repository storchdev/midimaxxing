#!/usr/bin/env python3
"""Transcribe piano video to MIDI.

Modes:
    full (default)  audio transcription + visual inference + MIDI patching
    audio-only      audio transcription only (no GPU/ViT required)

Usage:
    python transcribe.py --hands top video.mp4                   # full, interactive bbox
    python transcribe.py --hands top video.mp4 x1,y1,x2,y2      # full, given crop
    python transcribe.py --hands bottom video.mp4                # full, flips video first
    python transcribe.py --audio-only video.mp4                  # audio only
    python transcribe.py --hands top video.mp4 --patch-args --offset 0.5

Prerequisites:
    - Docker container already running: bash rundocker.sh
    - Visual inference deps (full mode only): uv sync
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from keyboard_geometry import flip_bbox_180, parse_note


def run(cmd):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], check=True)


def _nvenc_available() -> bool:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True, text=True,
    )
    return "h264_nvenc" in result.stdout


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe piano video to MIDI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video", type=Path, help="Input video file (mp4, mov, etc.)")
    parser.add_argument(
        "bbox",
        nargs="?",
        help="Piano key crop as x1,y1,x2,y2 (full mode only). Omit for interactive picker.",
    )
    parser.add_argument(
        "--hands",
        choices=["top", "bottom"],
        help=(
            "Where the player's hands appear in the frame. "
            "'top' = hands at top of frame. "
            "'bottom' = hands at bottom of frame; video will be rotated 180° before visual inference. "
            "Required for full pipeline."
        ),
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Skip visual inference and MIDI patching; output audio transcription directly.",
    )
    parser.add_argument(
        "--patch-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra arguments forwarded to patch_midi.py (e.g. --offset 0.5).",
    )
    parser.add_argument(
        "--keyboard-range", nargs=2, default=["A0", "C8"], metavar=("LOW", "HIGH"),
        help="Note-name range of the annotated keyboard (default: A0 C8, full 88-key).",
    )
    parser.add_argument("--num-hands", type=int, default=2,
                         help="Number of hands to track per frame, kept by YOLO "
                              "confidence (default: 2)")
    parser.add_argument(
        "--no-prune", action="store_true",
        help="Skip the hand-reachability pruning pass before patching.",
    )
    parser.add_argument("--max-disappear-frames", type=int, default=15,
                         help="Max consecutive frames a tracked hand can go "
                              "undetected by YOLO and still be linearly "
                              "interpolated across the gap (default: 15)")
    parser.add_argument("--reach-margin-px", type=float, default=None,
                         help="Extra pixels of slack added to each key's x-range "
                              "when testing hand overlap (default: half the "
                              "average key width, computed by prune_midi.py)")
    parser.add_argument("--yolo-conf", type=float, default=0.25,
                         help="Minimum YOLO detection confidence to keep a hand "
                              "box (default: 0.25)")
    parser.add_argument("--max-match-distance-px", type=float, default=None,
                         help="Max center-to-center pixel distance allowed when "
                              "matching a hand detection to an existing track "
                              "across frames (default: 0.15 * keyboard bbox width)")
    parser.add_argument(
        "--prune-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra arguments forwarded to prune_midi.py.",
    )
    args = parser.parse_args()

    stem = args.video.stem

    if args.audio_only:
        output_midi = Path("midi") / f"{stem}.mid"
        print("=== Audio transcription (Docker) ===")
        run([sys.executable, "audio_transcribe.py", args.video, output_midi])
        print(f"\nDone! MIDI: {output_midi}")
        return

    if args.hands is None:
        parser.error("--hands is required (use 'top' or 'bottom'). Use --audio-only to skip visual inference.")

    do_prune = not args.no_prune

    audio_midi = Path("midi") / f"{stem}_audio.mid"
    video_pkl = Path("midi") / f"{stem}.pkl"
    keyboard_json = Path("midi") / f"{stem}_keyboard.json"
    pruned_midi = Path("midi") / f"{stem}_pruned.mid"
    output_midi = Path("midi") / f"{stem}_final.mid"

    n_steps = 5 if do_prune else 3
    step = 1

    print(f"=== Step {step}/{n_steps}: Audio transcription (Docker) ===")
    run([sys.executable, "audio_transcribe.py", args.video, audio_midi])
    step += 1

    geometry = None
    vit_bbox_arg = args.bbox
    if do_prune:
        print(f"\n=== Step {step}/{n_steps}: Keyboard geometry ===")
        step += 1
        from keyboard_picker import geometry_from_dict, geometry_to_dict, pick_keyboard_geometry

        if keyboard_json.exists():
            print(f"Using cached {keyboard_json}")
            with open(keyboard_json) as f:
                geometry = geometry_from_dict(json.load(f))
        else:
            keyboard_json.parent.mkdir(parents=True, exist_ok=True)
            known_bbox = None
            if args.bbox:
                known_bbox = tuple(int(v) for v in args.bbox.split(","))
            geometry = pick_keyboard_geometry(
                args.video, hands=args.hands,
                lowest_note=args.keyboard_range[0], highest_note=args.keyboard_range[1],
                known_bbox=known_bbox,
            )
            with open(keyboard_json, "w") as f:
                json.dump(geometry_to_dict(geometry), f)

        if args.hands == "bottom":
            vit_bbox = flip_bbox_180(geometry.bbox, geometry.frame_width, geometry.frame_height)
        else:
            vit_bbox = geometry.bbox
        vit_bbox_arg = ",".join(str(v) for v in vit_bbox)

    print(f"\n=== Step {step}/{n_steps}: Visual inference (ViT) ===")
    step += 1
    tmpdir = None
    try:
        if args.hands == "bottom":
            # Use a temp dir but keep the original filename so vit.py derives
            # the correct dataset_name (and saves midi/<stem>.pkl, not midi/tmpXXX.pkl).
            tmpdir = Path(tempfile.mkdtemp())
            flipped_path = tmpdir / args.video.name
            print("Rotating video 180°...")
            encoder = ["-c:v", "h264_nvenc"] if _nvenc_available() else []
            run(["ffmpeg", "-y", "-i", args.video, "-vf", "hflip,vflip", *encoder, "-c:a", "copy", flipped_path])
            vit_input = flipped_path
        else:
            vit_input = args.video

        vit_cmd = [sys.executable, "vit.py", vit_input]
        if vit_bbox_arg:
            vit_cmd.append(vit_bbox_arg)
        run(vit_cmd)
    finally:
        if tmpdir and tmpdir.exists():
            shutil.rmtree(tmpdir)

    offset = None
    patch_input_midi = audio_midi
    if do_prune:
        print(f"\n=== Step {step}/{n_steps}: Pruning unreachable notes ===")
        step += 1
        import pretty_midi
        from patch_midi import estimate_offset, load_video_pianoroll
        audio_pm_notes = pretty_midi.PrettyMIDI(str(audio_midi))
        notes = [n for inst in audio_pm_notes.instruments for n in inst.notes]
        video_roll = load_video_pianoroll(video_pkl)
        offset, score = estimate_offset(notes, video_roll)
        print(f"Auto-aligned offset: {offset:.3f}s (score={score:.1f})")

        prune_cmd = [
            sys.executable, "prune_midi.py", audio_midi, keyboard_json, args.video, pruned_midi,
            "--hands", args.hands,
            "--num-hands", args.num_hands,
            "--max-disappear-frames", args.max_disappear_frames,
            "--yolo-conf", args.yolo_conf,
        ]
        if args.reach_margin_px is not None:
            prune_cmd += ["--reach-margin-px", args.reach_margin_px]
        if args.max_match_distance_px is not None:
            prune_cmd += ["--max-match-distance-px", args.max_match_distance_px]
        if "--offset" not in args.prune_args:
            prune_cmd += ["--offset", offset]
        prune_cmd += args.prune_args
        run(prune_cmd)
        patch_input_midi = pruned_midi

    print(f"\n=== Step {step}/{n_steps}: Patching MIDI ===")
    patch_cmd = [sys.executable, "patch_midi.py", patch_input_midi, video_pkl, output_midi]
    if offset is not None and "--offset" not in args.patch_args:
        patch_cmd += ["--offset", offset]
    if "--visible-keys" not in args.patch_args:
        lo = parse_note(args.keyboard_range[0])
        hi = parse_note(args.keyboard_range[1])
        patch_cmd += ["--visible-keys", lo, hi]
    patch_cmd += args.patch_args
    run(patch_cmd)

    print(f"\nDone! Final MIDI: {output_midi}")


if __name__ == "__main__":
    main()
