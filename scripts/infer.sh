#!/usr/bin/env bash
set -euo pipefail

AGF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${AGF_PYTHON:-python}" -m agf infer \
  --config "$AGF_REPO_ROOT/configs/inference.yaml" \
  --prompts "$AGF_REPO_ROOT/prompts/examples.txt" "$@"
