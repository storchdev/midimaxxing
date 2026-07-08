# midimaxxing

Transcribe piano video/audio to MIDI. Combines two transcription approaches:

- **Audio** — ByteDance's [piano-transcription](https://github.com/bytedance/piano_transcription) model running in Docker ([Replicate image](https://replicate.com/bytedance/piano-transcription), [paper](https://arxiv.org/pdf/2010.01815)). Accurate onsets, pitches, and velocities.
- **Visual** — A vision transformer (ViT) that watches which keys are physically pressed frame-by-frame ([ONF-VPT](https://chromeilion.github.io/onf_vpt/), [paper](https://arxiv.org/pdf/2411.09037v2)).

The two outputs are merged by `patch_midi.py`: onsets and velocities come from the audio model, but note releases are shortened wherever the video shows the finger has already lifted. The key release cue cannot come from audio alone because pedal can sustain notes after the player has let go of the key. Therefore we use a visual cue. The result matches what a player piano or Synthesia roll would show.

If you want the MIDI to sound as close to the original recording as possible (sustain and all), use audio-only. If you want releases that match the physical key presses like in standard midi/Synthesia recordings, use the full pipeline.

## Requirements

**Platform:** Linux or WSL2. NVIDIA DALI (used for visual inference) does not support Windows or macOS natively.

**Software:** [Docker](https://docs.docker.com/engine/install/) with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), [uv](https://github.com/astral-sh/uv).

**Hardware:**

| Mode | Requirement |
|---|---|
| Full pipeline | NVIDIA GPU — audio container uses `--gpus=all`, visual inference requires CUDA via NVIDIA DALI |
| Audio-only | NVIDIA GPU for the Docker container; no GPU needed on the host beyond that |

NVENC is used automatically for the 180° flip step if available, but is not required.

## Quick start

```bash
# 1. Start the Docker container (keep it running across sessions)
bash rundocker.sh

# 2. Install visual inference dependencies
uv sync

# 3. Run the full pipeline on a video
python transcribe.py --hands top videos/my_recording.mp4

# Or skip visual inference (audio only)
python transcribe.py --audio-only videos/my_recording.mp4
```

Output lands in `midi/`.

## transcribe.py

```
python transcribe.py --hands top video.mp4                    # full pipeline, interactive bbox picker
python transcribe.py --hands top video.mp4 x1,y1,x2,y2       # full pipeline, pre-specified crop
python transcribe.py --hands bottom video.mp4                 # full pipeline, upside-down recording
python transcribe.py --audio-only video.mp4                   # audio transcription only
python transcribe.py --hands top video.mp4 --patch-args --offset 0.5
```

**Full pipeline** produces `midi/<stem>_final.mid`:
1. Audio transcription → `midi/<stem>_audio.mid`
2. Keyboard geometry (interactive, cached) → `midi/<stem>_keyboard.json` (skipped with `--no-prune`)
3. Visual inference → `midi/<stem>.pkl`, `midi/<stem>_video.mid`
4. Pruning: drop audio-hallucinated notes no hand could reach → `midi/<stem>_pruned.mid` (skipped with `--no-prune`)
5. MIDI patching → `midi/<stem>_final.mid`

**Audio-only** produces `midi/<stem>.mid` directly. No GPU or visual dependencies needed beyond `requests`.

### --hands

Use `--hands top` if the player's hands are at the top of the frame, or `--hands bottom` if they're at the bottom (player POV, synthesia style). The visual model requires hands at the top; `--hands bottom` rotates the video 180° before inference and cleans up the temp file automatically. The keyboard picker always displays the frame hands-bottom for annotation regardless of `--hands`, so the picture never looks upside-down.

`--hands` is required for the full pipeline so you don't silently get wrong predictions.

### bbox

The bbox is the pixel crop of the piano keys in the video frame (`x1,y1,x2,y2`). If omitted, an interactive picker opens in your browser so you can click the two corners (plus, when pruning is enabled, the left edge of every C key in `--keyboard-range`).

The picker (`keyboard_picker.py`) is a small local web GUI, not a desktop window: it starts a server on `127.0.0.1` and opens a browser tab, so no X11/display is needed on the machine running the pipeline. If you're running over SSH, forward the port instead of opening a browser there, e.g. `ssh -L 8000:localhost:8000 host` then `python keyboard_picker.py video.mp4 --hands top --port 8000 --no-browser` and open `http://localhost:8000` locally. Click to place points, press `u` to undo the last one.

### Pruning unreachable notes

The full pipeline runs a hand-reachability pruning pass by default: a YOLO hand-detection model tracks both hands across the video, and any audio-transcribed note whose key no hand was near at its onset gets dropped before patching. This catches notes the audio model hallucinated outright (patching can only shorten notes, never remove them).

The keyboard picker collects `--keyboard-range LOW HIGH` (default `A0 C8`, full 88-key) as note names, plus the left edge of every C key, so key positions can be interpolated from pixel coordinates. Geometry is cached to `midi/<stem>_keyboard.json` — delete it to re-annotate.

Flags:

| Flag | Default | Meaning |
|---|---|---|
| `--no-prune` | off | Skip pruning entirely (matches pre-pruning behavior); the keyboard-geometry step is also skipped. |
| `--keyboard-range LOW HIGH` | `A0 C8` | Note-name range of the annotated keyboard; also forwarded to `patch_midi.py --visible-keys`. |
| `--num-hands` | `2` | Number of hands to track per frame, kept by YOLO confidence. |
| `--max-disappear-frames` | `15` | Max consecutive frames a tracked hand can go undetected by YOLO and still be linearly interpolated across the gap; beyond this the track expires and those frames have no hand data (fail-open: notes there are kept). |
| `--reach-margin-px` | half the average key width | Extra pixels of slack added to each key's x-range when testing hand overlap, to tolerate annotation/detection jitter. |
| `--yolo-conf` | `0.25` | Minimum YOLO detection confidence to keep a hand box. |
| `--max-match-distance-px` | `0.15 × keyboard bbox width` | Max center-to-center pixel distance allowed when matching a hand detection to an existing track across frames; farther pairs are treated as unmatched (new/expired track). |
| `--prune-args ...` | — | Extra arguments forwarded verbatim to `prune_midi.py` (e.g. `--weights path/to.pt`, `--offset 0.5`). |

`--weights` (path to the YOLOv10n hand-detection model) isn't exposed directly on `transcribe.py` — pass it via `--prune-args --weights path/to.pt` (everything after `--prune-args` is forwarded as-is to `prune_midi.py`). If omitted, weights are downloaded automatically to `weights/YOLOv10n_hands.pt` on first use.

## Other scripts

| Script | Purpose |
|---|---|
| `rundocker.sh` | Starts the ByteDance transcription container. Run once; leave it running. |
| `audio_transcribe.py` | Audio → MIDI via the Docker container. Called by `transcribe.py` but works standalone: `python audio_transcribe.py input.mp4 [output.mid]` |
| `vit.py` | Video → per-frame pianoroll via ViT. Saves `midi/<stem>.pkl` and `midi/<stem>_video.mid`. Usage same as transcribe.py sans `--audio-only`. |
| `keyboard_picker.py` | Browser-based GUI for the keyboard bbox + C-key markers. `python keyboard_picker.py video.mp4 --hands top`. |
| `keyboard_geometry.py` | Pure note-name/pixel-geometry math used by the picker and `prune_midi.py`. |
| `hand_tracker.py` | YOLO hand detection + multi-hand tracking across frames. |
| `prune_midi.py` | Drops audio-transcribed notes unreachable by any tracked hand. Run `python prune_midi.py --help` for options. |
| `patch_midi.py` | Merges audio MIDI with video pianoroll. Run `python patch_midi.py --help` for options. |
| `flip180.sh` | Rotates a video 180° for recordings where the camera is upside-down. |
