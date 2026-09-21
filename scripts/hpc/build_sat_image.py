#!/usr/bin/env python
"""Prepare a small recorded upstream Docker build context; build on VM only."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark.common import ROOT, sha256, write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,default=ROOT.parent/'SegmentAnyTree')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--execute',action='store_true')
    args=ap.parse_args()
    if args.output.exists():ap.error('use a fresh output folder')
    expected=json.loads((ROOT/'config/segmentanytree_source.json').read_text())
    commit=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()
    if commit!=expected['commit']:raise ValueError('unexpected source version')
    if any(p.is_symlink() for p in args.source.rglob('*')):raise ValueError('unreviewed source symlink')
    context=args.output.resolve()/'context'
    shutil.copytree(args.source,context,ignore=shutil.ignore_patterns('.git','*.pt','__pycache__'))
    command=['docker','build','--progress=plain','--tag',f'akl-sat:{commit[:12]}',str(context)]
    report={'status':'BUILD PREPARED ONLY','source_commit':commit,
            'dockerfile_sha256':sha256(context/'Dockerfile'),'command_argv':command,
            'warning':'Upstream Dockerfile contains unpinned transitive/git dependencies. Capture actual image ID, digest and package versions; rebuilding is not guaranteed identical. Old CUDA stack must be tested on reserved GPU.'}
    write_json(args.output/'build.json',report)
    if args.execute:
        with (args.output/'build.log').open('w') as f:
            result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
        report['exit_code']=result.returncode;report['status']='BUILT' if result.returncode==0 else 'FAILED'
        if result.returncode==0:
            report['image']=json.loads(subprocess.check_output(['docker','image','inspect',f'akl-sat:{commit[:12]}'],text=True))[0]
        write_json(args.output/'build.json',report)
        raise SystemExit(result.returncode)
    print(f'Build context prepared: {context}; no image built locally')


if __name__=='__main__':main()
