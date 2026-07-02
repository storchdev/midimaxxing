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

**Full pipeline** produces `midi/<stem>_final.mid` via three steps:
1. Audio transcription → `midi/<stem>_audio.mid`
2. Visual inference → `midi/<stem>.pkl` (per-frame key predictions)
3. MIDI patching → `midi/<stem>_final.mid`

**Audio-only** produces `midi/<stem>.mid` directly. No GPU or visual dependencies needed beyond `requests`.

### --hands

Use `--hands top` if the player's hands are at the top of the frame, or `--hands bottom` if they're at the bottom (player POV, synthesia style). The visual model requires hands at the top; `--hands bottom` rotates the video 180° before inference and cleans up the temp file automatically. The bbox picker (if used) opens on the already-flipped frame, so the crop coordinates you click will be correct.

`--hands` is required for the full pipeline so you don't silently get wrong predictions.

### bbox

The bbox is the pixel crop of the piano keys in the video frame (`x1,y1,x2,y2`). If omitted, an interactive picker opens so you can click the two corners.

## Other scripts

| Script | Purpose |
|---|---|
| `rundocker.sh` | Starts the ByteDance transcription container. Run once; leave it running. |
| `audio_transcribe.py` | Audio → MIDI via the Docker container. Called by `transcribe.py` but works standalone: `python audio_transcribe.py input.mp4 [output.mid]` |
| `vit.py` | Video → per-frame pianoroll via ViT. Saves `midi/<stem>.pkl` and `midi/<stem>.mid`. Usage same as transcribe.py sans `--audio-only`. |
| `patch_midi.py` | Merges audio MIDI with video pianoroll. Run `python patch_midi.py --help` for options. |
| `flip180.sh` | Rotates a video 180° for recordings where the camera is upside-down. |
