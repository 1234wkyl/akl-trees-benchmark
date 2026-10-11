#!/usr/bin/env python3
"""Prepare current replacement masks/greenness for fixed held-out classical prediction."""
import argparse,importlib.util,json,shutil,subprocess,sys
from pathlib import Path
from shapely.geometry import box
from benchmark.common import ROOT,sha256,site_definitions,write_json
from benchmark.inference_scenario import validate_inference_scenario

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','template-plan','seed-cache','green-source','green-adapter','classical-selection','inference-scenario','output'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--osm-endpoints',nargs='+',default=['https://overpass-api.de/api/interpreter','https://overpass.private.coffee/api/interpreter'])
    p.add_argument('--osm-subdivisions',type=int,choices=[1,2,4],default=2)
    p.add_argument('--osm-maxsize-mib',type=int,choices=[16,32,64,128,256,512])
    p.add_argument('--osm-busy-retries',type=int,choices=[0,1,2,3],default=0)
    a=p.parse_args();r=validate_inference_scenario(a.inference_scenario)
    if a.output.exists():p.error('fresh input directory required; preserve partial failed acquisitions')
    selection=json.loads(a.classical_selection.read_text());sites=site_definitions()
    if selection.get('provisional_reconstruction') is not True or set(selection['site_ids'])!={sid for sid,s in sites.items() if s['split']=='train'}:
        raise ValueError('fixed all-training provisional classical selection required')
    spec=importlib.util.spec_from_file_location('adapter',ROOT/'scripts/51_attempt_historical_crowns.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    source,_,hashes=m.source_module(a.source)
    template=json.loads(a.template_plan.read_text());grid=template['parameters']['original_global_bbox']
    chosen={sid:s for sid,s in sites.items() if s['split']!='train'}
    cells=m.select_grid_cells(grid,source.CHUNK_SIZE_M,[box(*s['bbox_epsg2193']).buffer(16) for s in chosen.values()])
    masks=m.select_grid_cells(grid,source.MAX_OSM_BBOX_M,[box(*c).buffer(source.CROWN_HALO_M) for c in cells])
    a.output.mkdir(parents=True)
    (a.output/'executed_preparation_source.py').write_bytes(Path(__file__).read_bytes())
    plan={'status':'FIXED PROVISIONAL CLASSICAL HELD-OUT INPUT PREPARATION; NO SCORING',
        'selected_site_ids':list(chosen),'original_chunks':cells,'osm_query_cells':masks,
        'parameters':selection['selected']['parameters'],'original_global_bbox':grid,
        'classical_selection_sha256':sha256(a.classical_selection),'inference_scenario_sha256':sha256(a.inference_scenario),
        'source_file_hashes':hashes,'green_source_sha256':sha256(a.green_source),'green_adapter_sha256':sha256(a.green_adapter),
        'osm_subdivisions':a.osm_subdivisions,'osm_endpoints':a.osm_endpoints,
        'osm_busy_retries':a.osm_busy_retries,'osm_maxsize_mib':a.osm_maxsize_mib,'acquisition_adapter_sha256':sha256(ROOT/'scripts/51_attempt_historical_crowns.py'),
        'historical_inputs_verified':False,'formal_reference_frozen':False,'evaluation_permitted':False}
    write_json(a.output/'plan.json',plan)
    shutil.copytree(a.seed_cache,a.output/'current_osm_cache')
    m.fetch_current_masks(source,masks,a.output/'current_osm_cache',a.output,a.osm_subdivisions,a.osm_endpoints,'GET',25,a.osm_maxsize_mib,a.osm_busy_retries)
    m.load_masks(source,masks,a.output/'current_osm_cache')
    command=[sys.executable,str(a.green_adapter),'--source',str(a.green_source),'--source-sha256',sha256(a.green_source),
        '--supplied-scripts',str(a.source/'scripts'),'--plan',str(a.output/'plan.json'),
        '--output',str(a.output/'current_greenness_complete')]
    subprocess.run(['bash',str(ROOT/'scripts/hpc/run_vm.sh'),'classical_heldout_current_greenness','--',*command],cwd=ROOT,check=True)
    write_json(a.output/'input_receipt.json',{'status':'COMPLETE CURRENT REPLACEMENT INPUTS; NOT HISTORICAL SNAPSHOT',
        'selected_site_ids':list(chosen),'plan_sha256':sha256(a.output/'plan.json'),
        'greenness_provenance_sha256':sha256(a.output/'current_greenness_complete/provenance.json'),
        'source_files':{str(f.relative_to(a.output)):sha256(f) for f in a.output.rglob('*') if f.is_file()},
        'evaluation_permitted':False})
    print('All fixed classical prediction inputs complete:',a.output)
if __name__=='__main__':main()
