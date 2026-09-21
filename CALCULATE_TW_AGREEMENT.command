#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
AGREEMENT_PYTHON="/opt/miniconda3/envs/akl-benchmark/bin/python"
if [[ ! -x "$AGREEMENT_PYTHON" ]]; then AGREEMENT_PYTHON="$(command -v python3)"; fi
if [[ "$#" -gt 0 ]]; then
  TW_JSON_PATH="$1"
else
  read -r -p "Drag the returned teacher JSON here, then press Return: " TW_JSON_PATH
  # macOS Terminal inserts shell escapes when a file is dragged. Decode those
  # escapes as characters, without eval or executing the supplied path.
  TW_JSON_PATH="$($AGREEMENT_PYTHON -c 'import shlex,sys; p=shlex.split(sys.argv[1]); print(p[0] if len(p)==1 else sys.argv[1])' "$TW_JSON_PATH")"
fi
"$AGREEMENT_PYTHON" scripts/31_receive_independent_labels.py --tim-json "$TW_JSON_PATH"
read -r -p "Calculation finished. Press Return to close. " AGREEMENT_CLOSE_REPLY
