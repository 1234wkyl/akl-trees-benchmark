#!/usr/bin/env python
"""Sequential smoke density runs under the standalone VM logging wrapper."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark.common import ROOT, sha256


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--image',required=True)
    ap.add_argument('--execute',action='store_true')
    args=ap.parse_args()
    manifest=json.loads(args.manifest.read_text())
    if manifest.get('held_out_sites')!=[] or manifest.get('thinning_unit')!='individual return record, not laser pulse':raise ValueError('training-only verified density manifest required')
    # Hash checks precede every actual or planned model invocation.
    for run in manifest['runs']:
        path=(args.manifest.parent/run['input_file']).resolve()
        if sha256(path)!=run['sha256']:raise ValueError(f'density input changed: {path}')
        name=f"density_train_seed{manifest['seed']}_fraction{run['fraction']}"
        cmd=['bash',str(ROOT/'scripts/hpc/run_vm.sh'),name,'--',sys.executable,
             str(ROOT/'scripts/hpc/run_sat_container.py'),'--input',str(path.parent),'--image',args.image,'--execute']
        print(json.dumps({'fraction':run['fraction'],'command_argv':cmd}),flush=True)
        if args.execute:subprocess.run(cmd,check=True,cwd=ROOT)
    if not args.execute:print('Commands only; no GPU/container called. Add --execute on VM after smoke succeeds.')


if __name__=='__main__':main()
