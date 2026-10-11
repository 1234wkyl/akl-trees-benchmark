#!/usr/bin/env python3
"""Audited compact prediction queue for a separate fixed weak SAT checkpoint.

Reuse hash-verified original inputs, never previous model outputs. This queue
does not read reference polygons or scores and cannot tune its checkpoint.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from benchmark.common import ROOT,sha256,write_json


def run(name,argv):
    print('+',' '.join(map(str,argv)),flush=True)
    subprocess.run(['bash',str(ROOT/'scripts/hpc/run_vm.sh'),name,'--',*map(str,argv)],cwd=ROOT,check=True)


def compact_prediction(prediction, raw_prediction):
    """Delete only new redundant intermediates after a successful coordinate audit.

    Preserve raw final LAZ and reconciled output (outside this folder). No old
    experiment, staged input, checkpoint or upstream source can be deleted.
    """
    prediction=prediction.resolve();raw_prediction=raw_prediction.resolve()
    if not prediction.is_relative_to(ROOT/'outputs/nectar') or prediction.name!='prediction':
        raise ValueError('compact cleanup restricted to new Nectar prediction folders')
    if not raw_prediction.is_relative_to(prediction) or not raw_prediction.is_file():
        raise ValueError('preserved raw prediction must exist inside output')
    files=list(prediction.rglob('*'))
    if any(f.is_symlink() for f in files):raise ValueError('cleanup refuses symbolic links')
    removed=[]
    for f in files:
        if f.is_file() and f.resolve()!=raw_prediction:
            removed.append({'path':str(f.relative_to(prediction)),'bytes':f.stat().st_size,'sha256':sha256(f)})
    write_json(prediction.parent/'intermediate_cleanup.json',{
       'status':'READY','reason':'Bound disk growth in the new experiment after exact coordinate audit',
       'preserved_raw_prediction':str(raw_prediction),'preserved_raw_sha256':sha256(raw_prediction),
       'removed':removed,'removed_bytes':sum(f['bytes'] for f in removed)})
    for item in removed:(prediction/item['path']).unlink()
    for f in sorted(prediction.rglob('*'),key=lambda p:len(p.parts),reverse=True):
        if f.is_dir() and not any(f.iterdir()):f.rmdir()
    assert raw_prediction.is_file()
    receipt=prediction.parent/'intermediate_cleanup.json'
    completed=json.loads(receipt.read_text());completed['status']='COMPLETE';write_json(receipt,completed)
    print('Compacted audited intermediates:',sum(f['bytes'] for f in removed),'bytes',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--producer-queue',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--training-receipt',type=Path,required=True)
    p.add_argument('--author-lock',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--image',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    queue=json.loads(a.producer_queue.read_text());training=json.loads(a.training_receipt.read_text())
    lock=json.loads(a.author_lock.read_text());cfg=json.loads((ROOT/'config/segmentanytree_source.json').read_text())
    if queue['status']!='COMPLETED' or any(t['status']!='AUDITED' for t in queue['tiles']):raise ValueError('completed verified original input queue required')
    if (training['status']!='COMPLETE' or training['output_checkpoint_sha256']!=sha256(a.checkpoint)
        or training['reference_json_sha256']!=lock['source_json_sha256']
        or training['validation_test_used_for_gradients'] is not False
        or lock['status']!='author_locked'
        or sha256(a.author_lock.parent/Path(lock['source_json']).name)!=lock['source_json_sha256']):
        raise ValueError('fixed training-only weak checkpoint and author reference lock required')
    commit=subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()
    if commit!=cfg['commit'] or sha256(a.source/cfg['checkpoint'])!=cfg['checkpoint_sha256']:raise ValueError('official source/checkpoint changed')
    image=json.loads(subprocess.check_output(['docker','image','inspect',a.image],text=True))[0]
    if image['Id']!=a.image or queue['image']!=a.image:raise ValueError('same immutable CUDA image required')
    if a.output.exists() and not a.resume:p.error('fresh output or explicit resume required')
    a.output.mkdir(parents=True,exist_ok=a.resume)
    path=a.output/'queue_receipt.json'
    fixed={'checkpoint_sha256':sha256(a.checkpoint),'training_receipt_sha256':sha256(a.training_receipt),
           'author_lock_sha256':sha256(a.author_lock),'producer_queue_sha256':sha256(a.producer_queue),
           'image':a.image,'source_commit':commit,'reference_labels_used_for_prediction':False,
           'scope':'Separate exploratory weak-label model; previously exposed held-out sites',
           'source_code_sha256':sha256(Path(__file__)), 'compact_intermediates_after_audit':True}
    rows=[]
    if path.exists():
        previous=json.loads(path.read_text())
        if any(previous[k]!=v for k,v in fixed.items()):raise ValueError('resume model/input/protocol differs')
        rows=previous['tiles']
    def save(status='RUNNING'):
        write_json(path,{**fixed,'status':status,'updated_utc':datetime.now(timezone.utc).isoformat(),
                        'expected_tiles':[t['tile'] for t in queue['tiles']],'tiles':rows,
                        'heldout_results_previously_exposed':True,'model_settings_changed_during_queue':False})
    save()
    for old in queue['tiles']:
        tile=old['tile'];prior=next((r for r in rows if r['tile']==tile),None)
        if prior:
            if prior['status']!='AUDITED':raise ValueError('inspect partial tile before retry: '+tile)
            for key in ['reconciled_prediction','coordinate_audit']:
                if sha256(prior[key])!=prior[key+'_sha256']:raise ValueError('completed output changed: '+tile)
            continue
        source=Path(old['input'])
        if sha256(source)!=old['input_sha256']:raise ValueError('original staged input changed')
        dest=a.output/source.stem
        if dest.exists():raise ValueError('unreceipted tile directory exists')
        if shutil.disk_usage(a.output).free<4*1024**3:raise ValueError('less than 4 GiB disk headroom; stop before new tile')
        dest.mkdir();code=dest/'code';prediction=dest/'prediction'
        shutil.copytree(a.source,code,ignore=shutil.ignore_patterns('.git','*.pt','__pycache__','outputs','data_for_test_results'))
        (code/'model_file/PointGroup-PAPER.pt').touch()
        prediction.mkdir()
        row={'tile':tile,'input':str(source.resolve()),'input_sha256':sha256(source),'status':'RUNNING'}
        rows.append(row);save()
        cmd=['docker','run','--rm','--gpus','all','--user',f'{os.getuid()}:{os.getgid()}',
             '--env','HOME=/tmp','--env','OMP_NUM_THREADS=4','--tmpfs','/home/datascience:rw,mode=1777',
             '--mount',f'type=bind,src={code.resolve()},dst=/home/nibio/mutable-outside-world',
             '--mount',f'type=bind,src={a.checkpoint.resolve()},dst=/home/nibio/mutable-outside-world/model_file/PointGroup-PAPER.pt,readonly',
             '--mount',f'type=bind,src={source.parent.resolve()},dst=/benchmark-input,readonly',
             '--mount',f'type=bind,src={prediction.resolve()},dst=/benchmark-output',
             '--workdir','/home/nibio/mutable-outside-world','--entrypoint','bash',a.image,'-c',
             'set -e; bash run_inference.sh /benchmark-input /benchmark-output false']
        write_json(dest/'inference_plan.json',{'argv':cmd,**fixed,'input_sha256':sha256(source)})
        run('weak_fixed_'+source.stem,cmd)
        produced=list((prediction/'final_results').glob('*.laz'))
        if len(produced)!=1:raise ValueError('expected one completed raw prediction')
        reconciled=dest/'reconciled.laz';report=dest/'coordinate_reconciliation.json';audit=dest/'output_audit.json'
        run('weak_reconcile_'+source.stem,[sys.executable,ROOT/'scripts/33_reconcile_sat_output.py',
            '--input',source,'--prediction',produced[0],'--output',reconciled,'--report',report])
        run('weak_audit_'+source.stem,[sys.executable,ROOT/'scripts/14_audit_sat_output.py',
            '--prediction',reconciled,'--input',source,'--output',audit])
        check=json.loads(audit.read_text())
        # The exact audit CLI owns its complete acceptance check and exit code.
        row.update(status='AUDITED',raw_prediction=str(produced[0].resolve()),raw_prediction_sha256=sha256(produced[0]),
            reconciled_prediction=str(reconciled.resolve()),reconciled_prediction_sha256=sha256(reconciled),
            coordinate_audit=str(audit.resolve()),coordinate_audit_sha256=sha256(audit),
            coordinate_reconciliation_sha256=sha256(report))
        compact_prediction(prediction,produced[0])
        row['intermediate_cleanup_sha256']=sha256(dest/'intermediate_cleanup.json')
        save();print('AUDITED:',tile,flush=True)
    save('COMPLETED');print('Completed weak checkpoint queue',flush=True)


if __name__=='__main__':main()
