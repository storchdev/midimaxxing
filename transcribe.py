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
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


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

    audio_midi = Path("midi") / f"{stem}_audio.mid"
    video_pkl = Path("midi") / f"{stem}.pkl"
    output_midi = Path("midi") / f"{stem}_final.mid"

    print("=== Step 1/3: Audio transcription (Docker) ===")
    run([sys.executable, "audio_transcribe.py", args.video, audio_midi])

    print(f"\n=== Step 2/3: Visual inference (ViT) ===")
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
        if args.bbox:
            vit_cmd.append(args.bbox)
        run(vit_cmd)
    finally:
        if tmpdir and tmpdir.exists():
            shutil.rmtree(tmpdir)

    print(f"\n=== Step 3/3: Patching MIDI ===")
    run([sys.executable, "patch_midi.py", audio_midi, video_pkl, output_midi, *args.patch_args])

    print(f"\nDone! Final MIDI: {output_midi}")


if __name__ == "__main__":
    main()
