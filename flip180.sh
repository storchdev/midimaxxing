#!/usr/bin/env bash
set -e
input="$1"
output="${2:-${input%.*}_flipped.${input##*.}}"
ffmpeg -i "$input" -vf "hflip,vflip" -c:a copy "$output"
echo "Saved to $output"
