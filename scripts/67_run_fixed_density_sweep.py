#!/usr/bin/env python3
"""Serial fixed density variants; stop on incomplete producer or low disk space."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
from benchmark.common import ROOT,sha256,write_json
from benchmark.inference_scenario import validate_inference_scenario

def recorded(name,cmd):subprocess.run(['bash',str(ROOT/'scripts/hpc/run_vm.sh'),name,'--',*map(str,cmd)],cwd=ROOT,check=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--classical-base',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    prep=json.loads((a.base/'preparation_receipt.json').read_text());protocol=json.loads((a.base/'density_protocol.json').read_text())
    if prep['density_protocol_sha256']!=sha256(a.base/'density_protocol.json') or protocol['fractions']!=[.75,.5,.25,.125] or protocol['seed']!=42:raise ValueError('fixed September retention design required')
    if any(sha256(ROOT/n)!=h for n,h in protocol['driver_script_sha256'].items()):raise ValueError('fixed density driver code changed')
    finished=[]
    for f in protocol['fractions']:
        root=a.base/f'fraction_{f:g}';scenario=root/'inference_scenario.json';r=validate_inference_scenario(scenario)
        if sha256(root/'queue_plan.json')!=prep['queue_plan_sha256'][str(f)]:raise ValueError('prepared queue changed')
        if sha256(root/'postprocessing_lock.json')!=protocol['postprocessing_lock_sha256']:raise ValueError('postprocessing lock changed')
        if shutil.disk_usage(ROOT).free<20*1024**3:raise ValueError('less than 20 GiB free; stop before a new variant, preserve receipts')
        cmd=[sys.executable,ROOT/'scripts/48_run_training_tile_queue.py','--base',root,'--source',a.source,'--image',r['image'],'--inference-scenario',scenario,'--execute']
        for tile in protocol['tile_names']:cmd+=['--tile',tile]
        q=root/'queue_receipt.json'
        if not q.exists() or json.loads(q.read_text())['status']!='COMPLETED':recorded(f'density_{f:g}_pretrained_queue',cmd)
        complete=json.loads(q.read_text())
        if (complete['expected_tiles']!=protocol['tile_names'] or len(complete['tiles'])!=19
                or any(t['status']!='AUDITED' for t in complete['tiles'])
                or complete['inference_scenario_sha256']!=sha256(scenario)):
            raise ValueError('completed audited fixed-density queue required')
        assembly=root/'assembled_sites_v1'
        if not (assembly/'assembly_receipt.json').exists():
            recorded(f'density_{f:g}_assemble_sites',[sys.executable,ROOT/'scripts/58_assemble_assumed_heldout_predictions.py',
                '--base',root,'--output',assembly,*(['--resume'] if assembly.exists() else [])])
        result=root/'hypothetical_comparison_v1'
        if not (result/'hypothetical_comparison.json').exists():
            recorded(f'density_{f:g}_conditional_evaluation',[sys.executable,ROOT/'scripts/61_evaluate_assumed_reference.py',
                '--inference-scenario',scenario,'--reference',ROOT/'outputs/reference_20261006_assumed_adjudicated/prefreeze_evaluation_reference.gpkg',
                '--sat-assembly',assembly,'--classical-predictions',a.classical_base/'test_standardized/predictions.gpkg',
                '--classical-receipt',a.classical_base/'prediction_receipt.json','--output',result])
        finished.append({'fraction':f,'comparison_sha256':sha256(result/'hypothetical_comparison.json')})
        write_json(a.base/'sweep_receipt.json',{'status':'COMPLETE' if len(finished)==4 else 'RUNNING',
            'completed':finished,'density_protocol_sha256':sha256(a.base/'density_protocol.json'),
            'reference_frozen':False,'model_settings_changed':False,'classical_inputs_thinned':False})
    print('All four fixed density conditions inferred, assembled and conditionally scored; full-density result reused.')
if __name__=='__main__':main()
