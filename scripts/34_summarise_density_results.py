#!/usr/bin/env python3
"""Summarise audited, training-only SegmentAnyTree density runs.

Each --item binds a density fraction to one result directory and one immutable
benchmark run ID.  Explicit binding avoids accidentally selecting stale runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_item(text: str) -> tuple[float, Path, str]:
    try:
        fraction_text, result_dir, run_id = text.split("::", 2)
        fraction = float(fraction_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "item must be FRACTION::RESULT_DIR::INFERENCE_RUN_ID"
        ) from exc
    if not 0 < fraction <= 1:
        raise argparse.ArgumentTypeError("fraction must be in (0, 1]")
    return fraction, Path(result_dir), run_id


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def summarise(items, benchmarks_root: Path) -> list[dict]:
    rows = []
    seen = set()
    for fraction, result_dir, run_id in sorted(items, reverse=True):
        if fraction in seen:
            raise ValueError(f"duplicate fraction: {fraction}")
        seen.add(fraction)
        reconciled = result_dir / "predictions/reconciled"
        reconciliation_path = reconciled / "reconciliation.json"
        audit_path = reconciled / "output_audit.json"
        timing_path = benchmarks_root / run_id / "timing.json"
        reconciliation = load(reconciliation_path)
        audit = load(audit_path)
        timing = load(timing_path)

        checks = {
            "run_id_matches": timing.get("run_id") == run_id,
            "inference_exit_zero": timing.get("exit_code") == 0,
            "reconciled": reconciliation.get("status") == "RECONCILED",
            "point_count_equal": audit.get("point_count_equal") is True,
            "coordinate_coverage_ok": audit.get("bidirectional_coordinate_coverage_ok") is True,
            "coordinate_identity": audit.get("max_output_to_input_distance_m") == 0,
            "epsg2193": audit.get("epsg2193_coordinate_consistency") is True,
            "no_label_conflicts": reconciliation.get("duplicate_prediction_label_conflicts") == 0,
            "no_uint16_warning": audit.get("uint16_boundary_warning") is False,
            "no_benchmark_scores": audit.get("benchmark_scores_created") is False,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"fraction {fraction} failed: {', '.join(failed)}")

        rows.append(
            {
                "fraction": fraction,
                "input_points": audit["input_points"],
                "upstream_prediction_points": reconciliation["upstream_prediction_points"],
                "reconciled_points": audit["prediction_points"],
                "removed_cartesian_join_rows": reconciliation["removed_cartesian_join_rows"],
                "positive_instances": audit["positive_instances"],
                "max_instance_id": audit["max_instance_id"],
                "elapsed_seconds": timing["elapsed_seconds"],
                "elapsed_gpu_hours": timing["elapsed_seconds"] / 3600,
                "finished_utc": timing["finished_utc"],
                "inference_run_id": run_id,
                "input_sha256": audit["input_sha256"],
                "reconciled_output_sha256": audit["prediction_sha256"],
                "reconciliation_report_sha256": sha256(reconciliation_path),
                "audit_report_sha256": sha256(audit_path),
                "output_epsg": reconciliation["output_epsg"],
                "max_coordinate_error_m": audit["max_output_to_input_distance_m"],
                "audit_passed": True,
                "benchmark_scores_created": False,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item", action="append", type=parse_item, required=True)
    parser.add_argument("--benchmarks-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = summarise(args.item, args.benchmarks_root)
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "density_run_summary.csv"
    json_path = args.output / "density_run_summary.json"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "status": "AUDITED_TRAINING_TILE_TECHNICAL_RUNS",
        "scope": "Pretrained inference on one training-only tile; not held-out benchmark scores.",
        "fractions": [row["fraction"] for row in rows],
        "runs": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
