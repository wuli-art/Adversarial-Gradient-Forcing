#!/usr/bin/env bash
set -euo pipefail
exec "${AGF_PYTHON:-python}" -m agf.utils.recipe "$@"
