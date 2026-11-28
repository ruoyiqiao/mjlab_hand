#!/bin/bash
# Rebuild the README task gallery (media/tasks/<clip>.webp) from policy checkpoints.
#
# Usage: tools/allegro4/make_gallery.sh <ckpt_dir> <out_dir> [clip ...]
#
# <ckpt_dir>/<clip>.pt is the checkpoint of each clip; gallery_clips.tsv lists its task, the seed and iteration it
# comes from, and the recording, rendering and encoding settings. Needs the locked environment
# (uv sync --locked) and Blender on PATH (the gallery was rendered with Blender 5.1.1). Recording runs on the
# CPU and is deterministic: a checkpoint always gives the same trajectory, and the frames match the README clips
# up to Cycles sampling noise.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ckpt_dir=$1
out=$2
shift 2
mkdir -p "$out/rec" "$out/frames"
grep -v '^#' "$here/gallery_clips.tsv" | while IFS=$'\t' read -r clip task seed iter rec render webp; do
  [ $# -eq 0 ] || [[ " $* " == *" $clip "* ]] || continue
  echo "== $clip ($task, seed $seed, iteration $iter)"
  # word splitting of the argument columns is intended
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES= uv run python "$here/record_scene.py" "$task" "$ckpt_dir/$clip.pt" "$out/rec/$clip.npz" $rec < /dev/null
  # shellcheck disable=SC2086
  blender -b -P "$here/render_blender.py" -- "$out/rec/$clip.npz" "$out/frames/$clip" $render < /dev/null > "$out/frames/$clip.log" 2>&1
  # shellcheck disable=SC2086
  uv run python "$here/make_webp.py" "$out/frames/$clip" "$out/$clip.webp" $webp < /dev/null
done
