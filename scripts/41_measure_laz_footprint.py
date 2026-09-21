#!/usr/bin/env python
"""Hash a complete input LAZ and measure its coordinate-envelope area.

This is a technical throughput denominator, not the actual vegetated area or
the union of detector-visited cells. It must not be substituted for site area.
"""
import argparse
import json
from pathlib import Path

import laspy
import numpy as np

from benchmark.common import sha256, write_json


def measure(path):
    with laspy.open(path) as stream:
        header=stream.header
        crs=header.parse_crs()
        if crs is None or crs.to_epsg()!=2193:
            raise ValueError('EPSG:2193 input required for square-metre area')
        xy_min=np.asarray(header.mins[:2],dtype=float)
        xy_max=np.asarray(header.maxs[:2],dtype=float)
        dimensions=xy_max-xy_min
        if not np.isfinite(dimensions).all() or (dimensions<=0).any() or header.point_count<=0:
            raise ValueError('positive finite XY extent and nonempty LAZ required')
        return {
            'processed_area_km2':float(np.prod(dimensions)/1e6),
            'footprint_description':'EPSG:2193 XY coordinate-envelope rectangle of the exact filtered full-density input LAZ; technical tile-throughput proxy, not site area or occupied-cell union',
            'measurement_record':'laspy.open(input).header.mins/maxs, width × height / 1,000,000; gaps and empty corners within envelope are included',
            'input_file':str(path),
            'input_sha256':sha256(path),
            'point_count':int(header.point_count),
            'xy_min_m':xy_min.tolist(),
            'xy_max_m':xy_max.tolist(),
            'xy_extent_m':dimensions.tolist(),
            'crs':'EPSG:2193',
            'limitation':'Envelope area is only a throughput proxy. A final operational cost must use the same complete footprint for all models, define overlap reduction and include appropriate billed time.'
        }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():ap.error('use a fresh output file')
    result=measure(args.input)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,result)
    print(json.dumps({'input':str(args.input),'area_km2':result['processed_area_km2'],'sha256':result['input_sha256']}))


if __name__=='__main__':main()
