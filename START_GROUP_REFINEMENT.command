#!/bin/bash
cd "$(dirname "$0")"
task_python=/opt/miniconda3/envs/akl-benchmark/bin/python
if [ ! -x "$task_python" ]; then
  echo 'The akl-benchmark Python environment is required (geopandas, shapely).'
  exit 1
fi
echo 'Tree-group proposal editor: http://127.0.0.1:8903/'
if curl --silent --fail http://127.0.0.1:8903/api/data >/dev/null; then
  open http://127.0.0.1:8903/
  exit 0
fi
"$task_python" scripts/26_refine_tree_groups.py --serve &
task_server_pid=$!
trap 'kill "$task_server_pid" 2>/dev/null' EXIT
for task_attempt in {1..20}; do
  if curl --silent --fail http://127.0.0.1:8903/ >/dev/null; then
    open http://127.0.0.1:8903/
    break
  fi
  sleep 0.3
done
wait "$task_server_pid"
