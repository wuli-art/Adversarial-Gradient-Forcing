#!/usr/bin/env bash
set -euo pipefail

AGF_DATA_ROOT="${AGF_DATA_ROOT:-.}"
AGF_MODEL_DIR="${AGF_WAN_MODELS_DIR:-$AGF_DATA_ROOT/wan_models}"

hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir "$AGF_MODEL_DIR/Wan2.1-T2V-1.3B"
hf download Wan-AI/Wan2.1-T2V-14B --local-dir "$AGF_MODEL_DIR/Wan2.1-T2V-14B"
hf download gdhe17/Self-Forcing checkpoints/ode_init.pt --local-dir "$AGF_DATA_ROOT"
hf download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir "$AGF_DATA_ROOT/prompts"
