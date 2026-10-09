#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 {5s|60s|240s} CHECKPOINT [agf infer options]" >&2
  exit 2
fi
PHASE="$1"
CHECKPOINT="$2"
shift 2
AGF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
case "$PHASE" in
  5s) FRAMES=21; PROMPTS=vbench_5s_official_unique_944.txt ;;
  60s) FRAMES=240; PROMPTS=vbench_60s_official_stratified_240.txt ;;
  240s) FRAMES=960; PROMPTS=mgvb_extended_240s_diverse_120.txt ;;
  *) echo "Unknown phase: $PHASE" >&2; exit 2 ;;
esac
exec "${AGF_PYTHON:-python}" -m agf infer \
  --config "$AGF_REPO_ROOT/configs/inference.yaml" \
  --checkpoint "$CHECKPOINT" --prompts "$AGF_REPO_ROOT/prompts/benchmark/$PROMPTS" \
  --latent-frames "$FRAMES" --seed 100000 --output "videos/$PHASE" "$@"
