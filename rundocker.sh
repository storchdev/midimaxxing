#!/usr/bin/env bash
set -e

GPU_FLAGS="--gpus=all"
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "No NVIDIA GPU detected; starting the container on CPU." >&2
    echo "(Whether this image supports CPU inference is untested -- if it fails, an NVIDIA GPU + NVIDIA Container Toolkit is required.)" >&2
    GPU_FLAGS=""
fi

docker run --name piano-worker -d -p 5000:5000 $GPU_FLAGS r8.im/bytedance/piano-transcription@sha256:8978296ce461e1fd8caae879d59063bc8009f57b734c1c8a2c7b19de0016fd35
