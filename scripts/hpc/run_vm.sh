#!/usr/bin/env bash
# Direct standalone VM execution: bash scripts/hpc/run_vm.sh NAME -- COMMAND...
set -euo pipefail
if [[ $# -lt 3 || "$2" != "--" || ! "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo 'Usage: bash scripts/hpc/run_vm.sh RUN_NAME -- COMMAND [ARG...]' >&2
  exit 2
fi
RUN_NAME="$1"; shift 2
KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$KIT_DIR"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_${RUN_NAME}_$$"
RUN_OUT="$KIT_DIR/outputs/benchmarks/$RUN_ID"
mkdir -p "$KIT_DIR/outputs/benchmarks" "$KIT_DIR/logs/hpc"
mkdir "$RUN_OUT"
export BENCHMARK_OUTPUT_DIR="$RUN_OUT"
{
  printf 'run_id: %s\nstarted_utc: %s\nhost: %s\n' "$RUN_ID" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname)"
  printf 'command:'; printf ' %q' "$@"; printf '\n'
  printf 'git_commit: '; git rev-parse HEAD || true
  printf 'git_status:\n'; git status --short || true
} > "$RUN_OUT/run_metadata.txt"
python --version > "$RUN_OUT/python_version.txt" 2>&1 || true
python -m pip freeze > "$RUN_OUT/pip_freeze.txt" 2>&1 || true
nvidia-smi > "$RUN_OUT/nvidia_smi.txt" 2>&1 || true
if command -v conda >/dev/null 2>&1; then conda list --explicit > "$RUN_OUT/conda_explicit.txt" 2>&1 || true; fi
RUN_COMMAND_STARTED="$(date +%s)"
set +e
{ time "$@"; } 2>&1 | tee "$RUN_OUT/run.log" "$KIT_DIR/logs/hpc/$RUN_ID.log"
RUN_STATUS="${PIPESTATUS[0]}"
set -e
RUN_COMMAND_FINISHED="$(date +%s)"
printf '{"wall_seconds":%s,"exit_code":%s,"resolution_seconds":1,"scope":"command including tee; environment inventory excluded"}\n' "$((RUN_COMMAND_FINISHED-RUN_COMMAND_STARTED))" "$RUN_STATUS" > "$RUN_OUT/timing.json"
printf 'finished_utc: %s\nexit_code: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RUN_STATUS" >> "$RUN_OUT/run_metadata.txt"
printf 'Outputs and logs: %s\n' "$RUN_OUT"
exit "$RUN_STATUS"
