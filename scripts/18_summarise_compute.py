#!/usr/bin/env python
"""Report measured run throughput and explicit area/cost assumptions."""
import argparse
import json
import math
from pathlib import Path

from benchmark.common import sha256, write_json


def compute_summary(wall_seconds,area_km2,hourly_price=None,billing_seconds=None):
    if not math.isfinite(wall_seconds) or wall_seconds<=0 or not math.isfinite(area_km2) or area_km2<=0:raise ValueError('positive wall time and processed area required')
    if hourly_price is not None and (not math.isfinite(hourly_price) or hourly_price<0):raise ValueError('invalid hourly price')
    if billing_seconds is not None and (not math.isfinite(billing_seconds) or billing_seconds<wall_seconds):raise ValueError('billed time must include the measured run')
    cost_seconds=billing_seconds if billing_seconds is not None else wall_seconds
    return {'wall_seconds':wall_seconds,'processed_area_km2':area_km2,
            'wall_seconds_per_km2':wall_seconds/area_km2,
            'wall_hours_per_km2':wall_seconds/3600/area_km2,
            'km2_per_hour':area_km2*3600/wall_seconds,
            'cost_per_km2':cost_seconds/3600*hourly_price/area_km2 if hourly_price is not None else None,
            'cost_basis':'provider billed seconds' if billing_seconds is not None else 'runtime-only estimate; excludes idle VM time',
            'hourly_price':hourly_price,'billing_seconds':billing_seconds}


def wall_seconds_from_timing(timing):
    """Accept both the VM run wrapper and an older generic timing receipt."""
    values=[timing[k] for k in ('elapsed_seconds','wall_seconds') if k in timing]
    if not values:
        raise ValueError('timing receipt needs elapsed_seconds or wall_seconds')
    if len(values)==2 and values[0]!=values[1]:
        raise ValueError('conflicting elapsed_seconds and wall_seconds')
    return values[0]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--timing',type=Path,required=True)
    ap.add_argument('--area-record',type=Path,required=True,help='JSON processed_area_km2, footprint_description, measurement_record')
    ap.add_argument('--hourly-price',type=float)
    ap.add_argument('--currency')
    ap.add_argument('--billing-seconds',type=float)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():ap.error('use a fresh output file')
    if args.hourly_price is not None and not args.currency:ap.error('currency required with hourly price')
    timing=json.loads(args.timing.read_text());area=json.loads(args.area_record.read_text())
    if timing.get('exit_code')!=0:raise ValueError('failed run is not a successful throughput measurement')
    if not area.get('measurement_record') or not area.get('footprint_description'):raise ValueError('area measurement/provenance required')
    summary=compute_summary(wall_seconds_from_timing(timing),area['processed_area_km2'],args.hourly_price,args.billing_seconds)
    summary.update(currency=args.currency,timing_sha256=sha256(args.timing),area_record_sha256=sha256(args.area_record),
        footprint_description=area['footprint_description'],area_measurement_record=area['measurement_record'],
        note='Use an explicit area denominator for the processed input. A coordinate-envelope rectangle is only a technical throughput proxy; do not divide full-tile runtime by the small reference-site area. Declare preprocessing, I/O, inference and container startup inclusions. Runtime estimates are not invoices.')
    args.output.parent.mkdir(parents=True,exist_ok=True);write_json(args.output,summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
