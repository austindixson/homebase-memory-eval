#!/bin/sh
set -eu
if test "$#" -ne 2; then
  echo 'Usage: serve_qwen.sh /path/to/mlx-serve /path/to/Qwen3-14B-8bit/snapshot' >&2
  exit 2
fi
binary_path=$1
model_snapshot=$2
actual_sha=$(shasum -a 256 "$binary_path" | awk '{print $1}')
if test "$actual_sha" != f6b32efcbbaa3d2d82baa0948071a5037eef7ded868902ab296ba182749e902e; then
  echo 'Server binary differs from the evaluated MLX Serve 26.9.5 binary.' >&2
  exit 1
fi
if test "$(basename "$model_snapshot")" != da33cf28f06636847fd9e93e0a03d819b84cb55e; then
  echo 'Model directory must be the pinned Hugging Face snapshot revision.' >&2
  exit 1
fi
exec "$binary_path" --model "$model_snapshot" --serve --host 127.0.0.1 --port 11240 \
  --pld --no-drafter --ctx-size 131072 \
  --config-overrides '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768},"max_position_embeddings":131072}' \
  --no-decode-attn-quant --kv-quant off --prefix-cache-entries 64 --prefix-cache-mem 12GB \
  --max-tokens 1024 --temp 0 --timeout 900 --no-vision
