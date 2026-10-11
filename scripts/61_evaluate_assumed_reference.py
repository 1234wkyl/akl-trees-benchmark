#!/usr/bin/env python3
"""Score a completed hypothetical test comparison without creating a formal freeze.

This is a separate assumption analysis. It cannot produce an official benchmark
receipt or bypass the supervisor-approved freeze required by script 12.
"""
import argparse,importlib.util,json
from pathlib import Path
import geopandas as gpd
from shapely.geometry import box
from benchmark.common import ROOT,sha256,site_definitions,write_json
from benchmark.evaluation import aggregate,paired_site_bootstrap,score_site
from benchmark.inference_scenario import validate_inference_scenario


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('inference-scenario','reference','sat-assembly','classical-predictions','classical-receipt','output'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--policy',type=Path,default=ROOT/'config/evaluation_draft.json')
    a=p.parse_args();scenario=validate_inference_scenario(a.inference_scenario)
    if a.output.exists():p.error('fresh hypothetical analysis directory required')
    policy=json.loads(a.policy.read_text());sites={sid:s for sid,s in site_definitions().items() if s['split']=='test'}
    supported={'matching':'greedy_iou_then_distance_then_ids','distance_reference':'centroids_after_site_clipping',
        'bootstrap_unit':'paired_site','bootstrap_stratification':'none','crs':'EPSG:2193',
        'false_positive_precision_definition':'same_scored_detection_precision'}
    if any(policy.get(k)!=v for k,v in supported.items()):raise ValueError('unsupported fixed evaluation policy')
    if (policy['match_tolerance_m']!=2.0 or policy['minimum_match_iou']!=0.0 or policy['random_seed']!=42
            or policy['bootstrap_iterations']!=2000 or policy['prediction_ignore_rule']!='unmatched_centroid_in_group_or_uncertain'):
        raise ValueError('do not change the fixed operational policy for the hypothetical comparison')
    ref_receipt=json.loads((a.reference.parent/'provenance.json').read_text())
    if (ref_receipt['output_gpkg_sha256']!=sha256(a.reference)
            or ref_receipt['source_json_sha256']!=scenario['reference_json_sha256']):
        raise ValueError('derived scoring geometry does not bind the scenario reference')
    assembly=json.loads((a.sat_assembly/'assembly_receipt.json').read_text());sat=a.sat_assembly/'test_predictions.gpkg'
    if (assembly['status']!='ASSUMPTION-ONLY HELD-OUT PREDICTIONS ASSEMBLED; NO SCORING'
            or assembly['inference_scenario_sha256']!=sha256(a.inference_scenario)
            or assembly['output_sha256']['test']!=sha256(sat)
            or {r['site_id'] for r in assembly['sites'] if r['split']=='test'}!=set(sites)):
        raise ValueError('completed fixed prediction assembly required')
    classic=json.loads(a.classical_receipt.read_text())
    if (classic['status']!='FIXED PROVISIONAL CLASSICAL HELD-OUT PREDICTIONS; NO SCORING'
            or classic['scenario_sha256']!=sha256(a.inference_scenario)
            or set(classic['site_ids'])!={sid for sid,s in site_definitions().items() if s['split']!='train'}):
        raise ValueError('completed fixed classical prediction receipt required')
    standard=json.loads((a.classical_predictions.parent/'provenance.json').read_text())
    if (standard['output_sha256']!=sha256(a.classical_predictions) or standard['input_sha256']!=classic['crowns_sha256']
            or set(standard['requested_site_ids'])!=set(sites)):
        raise ValueError('classical standardized test crowns do not bind the fixed producer')
    spec=importlib.util.spec_from_file_location('official_evaluator',ROOT/'scripts/12_evaluate.py')
    ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    references={}
    for layer in ('single_trees','tree_groups','uncertainty_regions'):
        frame=gpd.read_file(a.reference,layer=layer)
        if (frame.crs is None or frame.crs.to_epsg()!=2193 or not frame.geometry.is_valid.all()
                or frame.geometry.isna().any() or frame.geometry.is_empty.any()
                or not frame.geometry.geom_type.isin(['Polygon','MultiPolygon']).all()
                or frame.duplicated(['site_id','label_id']).any()):raise ValueError('valid unique EPSG:2193 reference required')
        references[layer]={sid:[(str(r.label_id),r.geometry.intersection(box(*s['bbox_epsg2193'])))
            for _,r in frame[frame.site_id==sid].iterrows()] for sid,s in sites.items()}
    def evaluate(path):
        predictions,outside=ev.load_polygons(path,'predictions',sites,'prediction_id')
        scores={sid:score_site(references['single_trees'][sid],predictions[sid],
            [g for _,g in references['tree_groups'][sid]],[g for _,g in references['uncertainty_regions'][sid]],
            policy['match_tolerance_m'],policy['minimum_match_iou'],policy['prediction_ignore_rule']) for sid in sites}
        return scores,outside
    model,mo=evaluate(sat);baseline,bo=evaluate(a.classical_predictions)
    a.output.mkdir(parents=True)
    write_json(a.output/'hypothetical_comparison.json',{'status':'HYPOTHETICAL TEST COMPARISON; NOT AN APPROVED BENCHMARK',
        'reference_frozen':False,'actual_independent_agreement_available':False,
        'held_out_results_exposed':True,'model_changes_after_viewing_not_permitted':True,
        'note':'Conditional on the author reference being the eventual adjudicated outcome. No actual inter-rater agreement or historical certification is asserted.',
        'site_ids':list(sites),'pretrained_summary':aggregate(list(model.values())),
        'classical_provisional_summary':aggregate(list(baseline.values())),
        'pretrained_by_site':model,'classical_by_site':baseline,
        'by_stratum':{label:{'pretrained':aggregate([model[s] for s,v in sites.items() if v['stratum']==label]),
            'classical':aggregate([baseline[s] for s,v in sites.items() if v['stratum']==label])} for label in {v['stratum'] for v in sites.values()}},
        'bootstrap':paired_site_bootstrap(model,baseline,2000,42),'outside_site_predictions':{'pretrained':mo,'classical':bo},
        'sources':{n:sha256(v) for n,v in [('reference',a.reference),('policy',a.policy),('scenario',a.inference_scenario),
            ('sat',sat),('classical',a.classical_predictions),('sat_assembly',a.sat_assembly/'assembly_receipt.json'),('classical_receipt',a.classical_receipt)]},
        'acceptance_declared':False})
    print('Hypothetical comparison complete; official benchmark and inter-rater records unchanged.')
if __name__=='__main__':main()
