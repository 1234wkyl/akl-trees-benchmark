#!/usr/bin/env python
"""Pinned, isolated SAT invocation; held-out prediction needs an inference-only protocol."""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark.common import ROOT, sha256, site_definitions, write_json
from benchmark.inference_scenario import validate_inference_scenario


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,default=ROOT.parent/'SegmentAnyTree')
    ap.add_argument('--input',type=Path,required=True,help='folder containing one geographic LAZ only')
    ap.add_argument('--image',required=True,help='immutable local sha256 image ID or repo@sha256 digest')
    ap.add_argument('--output',type=Path,help='defaults to BENCHMARK_OUTPUT_DIR/sat')
    ap.add_argument('--execute',action='store_true',help='requires Docker/NVIDIA on the VM; otherwise write plan only')
    ap.add_argument('--inference-scenario',type=Path,
                    help='explicit assumption-only pretrained prediction protocol; no scoring permission')
    args=ap.parse_args()
    if not re.fullmatch(r'(?:[^\s]+@)?sha256:[0-9a-f]{64}',args.image):
        ap.error('pin an immutable sha256 image ID/digest, not a mutable tag')
    if args.execute and args.image.endswith('0'*64):
        ap.error('placeholder image digest cannot be executed')
    if args.output is None:
        if not os.environ.get('BENCHMARK_OUTPUT_DIR'):
            ap.error('supply --output or run under run_vm.sh')
        args.output=Path(os.environ['BENCHMARK_OUTPUT_DIR'])/'sat'
    out=args.output.resolve()
    if out.exists():ap.error('use a fresh output folder')
    source=args.source.resolve();input_dir=args.input.resolve()
    expected=json.loads((ROOT/'config/segmentanytree_source.json').read_text())
    scenario=validate_inference_scenario(args.inference_scenario) if args.inference_scenario else None
    if scenario and scenario['image']!=args.image:raise ValueError('scenario immutable image differs')
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if commit!=expected['commit']:raise ValueError('unexpected source commit')
    changes=subprocess.check_output(['git','-C',str(source),'diff','--name-only','HEAD'],text=True).splitlines()
    if any(p!=expected['checkpoint'] for p in changes):raise ValueError('upstream tracked source has unrecorded edits')
    weight=source/expected['checkpoint']
    if weight.stat().st_size!=expected['checkpoint_bytes'] or sha256(weight)!=expected['checkpoint_sha256']:
        raise ValueError('checkpoint size/hash mismatch')
    if input_dir==source or source in input_dir.parents or input_dir in out.parents:
        raise ValueError('keep input, source and output directories separate')
    files=sorted(input_dir.iterdir())
    laz=[p for p in files if p.is_file() and p.suffix.lower()=='.laz']
    # Density folders carry an index sidecar: upstream must see only the LAZ.
    if len(laz)!=1 or any(p.suffix not in {'.laz','.npy'} for p in files):
        raise ValueError('input folder requires one LAZ and optional index sidecar only')
    import laspy
    from shapely.geometry import box
    with laspy.open(laz[0]) as f:
        h=f.header;crs=h.parse_crs()
        if crs is None or crs.to_epsg()!=2193:raise ValueError('geographic EPSG:2193 LAZ required')
        bounds=box(*h.mins[:2],*h.maxs[:2])
        overlaps=[s for s in site_definitions().values() if box(*s['bbox_epsg2193']).intersection(bounds).area>0]
        if not overlaps or (not scenario and any(s['split']!='train' for s in overlaps)):
            raise ValueError('this development runner refuses held-out tiles; freeze and final run manifest required first')
    if any(p.is_symlink() for p in source.rglob('*')):
        raise ValueError('inspect upstream symlinks before copying executable code')
    out.mkdir(parents=True)
    code=out/'code';safe_input=out/'input'
    docker_cmd=['docker','run','--rm','--gpus','all',
                '--user',f'{os.getuid()}:{os.getgid()}',
                '--env','HOME=/tmp','--env','XDG_CACHE_HOME=/tmp/.cache',
                '--tmpfs','/home/datascience:rw,mode=1777',
                '--mount',f'type=bind,src={code},dst=/home/nibio/mutable-outside-world',
                '--mount',f'type=bind,src={weight},dst=/home/nibio/mutable-outside-world/model_file/PointGroup-PAPER.pt,readonly',
                '--mount',f'type=bind,src={safe_input},dst=/benchmark-input,readonly',
                '--mount',f'type=bind,src={out},dst=/benchmark-output',
                '--workdir','/home/nibio/mutable-outside-world',
                '--entrypoint','bash',args.image,'-c',
                'set -e; python3 benchmark_preflight.py; bash run_inference.sh /benchmark-input /benchmark-output/predictions false']
    plan={'status':'PLANNED ONLY','source_commit':commit,'checkpoint':expected,
          'input_file':str(laz[0]),'input_sha256':sha256(laz[0]),'image':args.image,
          'training_sites':[s['site_id'] for s in overlaps if s['split']=='train'],
          'prediction_site_ids':[s['site_id'] for s in overlaps],
          'docker_command_argv':docker_cmd,'clean_output_argument':'false',
          'container_cache':'isolated tmpfs; no shared cache mounted',
          'upstream_default_entrypoint_overridden':True,
          'held_out_inference_permitted':bool(scenario),
          'inference_scenario_sha256':sha256(args.inference_scenario) if scenario else None,
          'evaluation_permitted':False,
          'notes':'Source code is copied per run; checkpoint and input are readonly. Input sidecars are not sent to upstream. Architecture loading and container compatibility are unverified until VM execution.'}
    write_json(out/'plan.json',plan)
    if not args.execute:
        print(f'Plan saved: {out}/plan.json; no Docker or GPU called');return
    inspect=json.loads(subprocess.check_output(['docker','image','inspect',args.image],text=True))[0]
    write_json(out/'image_inspect.json',inspect)
    shutil.copytree(source,code,ignore=shutil.ignore_patterns('.git','*.pt','__pycache__','outputs','data_for_test_results'))
    (code/'model_file/PointGroup-PAPER.pt').touch()
    shutil.copy2(ROOT/'scripts/hpc/sat_container_preflight.py',code/'benchmark_preflight.py')
    safe_input.mkdir();shutil.copy2(laz[0],safe_input/laz[0].name)
    if sha256(safe_input/laz[0].name)!=plan['input_sha256']:raise ValueError('staging changed input')
    plan['status']='RUNNING';plan['started_utc']=datetime.now(timezone.utc).isoformat();write_json(out/'plan.json',plan)
    result=subprocess.run(docker_cmd)
    plan.update(status='PROCESS COMPLETED' if result.returncode==0 else 'FAILED',
                exit_code=result.returncode,finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(out/'plan.json',plan)
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
