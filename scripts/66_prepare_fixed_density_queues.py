#!/usr/bin/env python3
"""Stage the four predeclared nested retention conditions; reuse full-density predictions."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
from benchmark.common import ROOT,sha256,write_json
from benchmark.inference_scenario import validate_inference_scenario

FRACTIONS=[.75,.5,.25,.125]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full-base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();scenario=a.full_base/'inference_scenario.json';r=validate_inference_scenario(scenario)
    if a.output.exists():p.error('fresh density experiment root required')
    q=json.loads((a.full_base/'queue_receipt.json').read_text())
    if q['status']!='COMPLETED' or len(q['tiles'])!=19 or any(x['status']!='AUDITED' for x in q['tiles']):raise ValueError('completed full-density queue required')
    lock=json.loads((a.full_base/'postprocessing_lock.json').read_text())
    if any(sha256(ROOT/n)!=h for n,h in lock['script_sha256'].items()):raise ValueError('fixed postprocessing code changed')
    a.output.mkdir(parents=True)
    protocol={'status':'PREDECLARED RETENTION SWEEP; MODEL AND CROWN SETTINGS UNCHANGED',
        'fractions':FRACTIONS,'full_density_reused':str(a.full_base.resolve()),
        'full_queue_sha256':sha256(a.full_base/'queue_receipt.json'),'inference_scenario_sha256':sha256(scenario),
        'seed':42,'thinning_unit':'individual return record, not laser pulse','count_rule':'max(1,floor(N*fraction))',
        'predeclared_design':'The same .75/.5/.25/.125 fractions and PCG64 permutation seed 42 were already used by the September training-site pilot. No retention condition is selected using October test results.',
        'geometry_settings':r['point_to_crown_conversion'],'classical_inputs_thinned':False,
        'classical_role':'Fixed full-density CHM comparator only; not a classical density-response experiment',
        'reference_frozen':False,'actual_independent_agreement_available':False,
        'tile_names':q['expected_tiles'],
        'driver_script_sha256':{n:sha256(ROOT/n) for n in ('scripts/13_prepare_density.py',
            'scripts/66_prepare_fixed_density_queues.py','scripts/67_run_fixed_density_sweep.py',
            'scripts/68_report_fixed_density_sweep.py')},
        'postprocessing_lock_sha256':sha256(a.full_base/'postprocessing_lock.json')}
    write_json(a.output/'density_protocol.json',protocol)
    roots={}
    for f in FRACTIONS:
        root=a.output/f'fraction_{f:g}';root.mkdir();roots[f]=root
        shutil.copy2(scenario,root/'inference_scenario.json');shutil.copy2(a.full_base/'postprocessing_lock.json',root/'postprocessing_lock.json')
    staged={f:[] for f in FRACTIONS}
    for row in q['tiles']:
        source=Path(row['input']);stem=Path(row['tile']).stem
        if sha256(source)!=row['input_sha256']:raise ValueError('full-density input changed')
        tmp=a.output/'nested_inputs'/stem
        subprocess.run(['bash',str(ROOT/'scripts/hpc/run_vm.sh'),'prepare_nested_density_'+stem,'--',sys.executable,
            str(ROOT/'scripts/13_prepare_density.py'),'--input',str(source),'--fractions',*map(str,FRACTIONS),
            '--seed','42','--inference-scenario',str(scenario),'--output',str(tmp)],cwd=ROOT,check=True)
        manifest=json.loads((tmp/'manifest.json').read_text())
        if manifest['source_sha256']!=row['input_sha256'] or manifest['source_unchanged'] is not True:raise ValueError('nested source binding failed')
        original=json.loads((a.full_base/stem/'staging/provenance.json').read_text())
        for run in manifest['runs']:
            f=run['fraction'];dest=roots[f]/stem/'staging';(dest/'input_laz').mkdir(parents=True)
            path=(tmp/run['input_file']).resolve();target=dest/'input_laz'/source.name
            shutil.copy2(path,target);shutil.copy2(path.parent/'source_point_indices.npy',dest/'source_point_indices.npy')
            if sha256(target)!=run['sha256']:raise ValueError('copied density input changed')
            record=dict(original);record.update(status='FIXED DENSITY VARIANT STAGED; NO MODEL RUN',
                input_laz=str(target.resolve()),input_laz_sha256=run['sha256'],retained_points=run['points'],
                retained_after_noise_points=manifest['source_points'],retention_fraction=f,
                thinning_removed_points=manifest['source_points']-run['points'],
                density_protocol_sha256=sha256(a.output/'density_protocol.json'),
                density_manifest_sha256=sha256(tmp/'manifest.json'),source_point_indices_sha256=run['point_indices_sha256'],
                source_filtered_input_sha256=row['input_sha256'],native_record_subset_verified=True)
            write_json(dest/'provenance.json',record)
            staged[f].append({'tile':row['tile'],'input_sha256':run['sha256'],'retained_points':run['points']})
    for f,root in roots.items():
        write_json(root/'queue_plan.json',{'status':'FIXED DENSITY QUEUE PREPARED; NO INFERENCE',
            'tiles':q['expected_tiles'],'fraction':f,'seed':42,'scenario_sha256':sha256(scenario),
            'density_protocol_sha256':sha256(a.output/'density_protocol.json'),'staged':staged[f],
            'model_input_points':sum(x['retained_points'] for x in staged[f]),'full_density_not_rerun':True})
    write_json(a.output/'preparation_receipt.json',{'status':'ALL FOUR FIXED DENSITY INPUT QUEUES COMPLETE',
        'fractions':FRACTIONS,'tiles_per_fraction':19,'density_protocol_sha256':sha256(a.output/'density_protocol.json'),
        'queue_plan_sha256':{str(f):sha256(root/'queue_plan.json') for f,root in roots.items()},
        'reference_frozen':False,'model_settings_changed':False})
    print('76 nested variant inputs prepared; full-density inference reused:',a.output)
if __name__=='__main__':main()
