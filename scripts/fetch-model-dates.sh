#!/usr/bin/env bash
# Fetch release dates (createdAt) for all evaluated models from the Hugging Face API.
# Usage: ./scripts/fetch-model-dates.sh

set -euo pipefail

models=(
  "meta-llama/Llama-3.1-70B"
  "moonshotai/Kimi-K2.5"
  "openai/gpt-oss-120b"
  "pentagoniac/llamarine"
  "Qwen/Qwen3.5-397B-A17B-FP8"
  "Qwen/Qwen3.5-122B-A10B-FP8"
  "Qwen/Qwen3.5-35B-A3B"
  "Qwen/Qwen3.5-27B"
  "Qwen/Qwen3.5-9B"
  "Qwen/Qwen3.5-4B"
  "Qwen/Qwen3-4B-Instruct-2507"
  "google/gemma-4-31B-it"
  "google/gemma-4-E4B-it"
)

for model in "${models[@]}"; do
  created=$(curl -sf "https://huggingface.co/api/models/$model" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('createdAt','NOT FOUND'))" 2>/dev/null || true)
  if [ -z "${created:-}" ]; then
    created="NOT FOUND"
  fi
  echo "$model -> $created"
done
