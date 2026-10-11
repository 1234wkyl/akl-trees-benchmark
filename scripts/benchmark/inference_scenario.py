"""An assumption-only prediction protocol cannot certify reference agreement."""
import hashlib,json
from pathlib import Path
from .common import ROOT,sha256,site_definitions


def training_signature(path):
    data=json.loads(Path(path).read_text())['sites'];sites=site_definitions()
    selected={sid:sorted([{'id':l['id'],'class':l['cls'],'world':l['world']} for l in data[sid]['labels']],key=lambda x:x['id'])
              for sid,s in sites.items() if s['split']=='train'}
    return hashlib.sha256(json.dumps(selected,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_inference_scenario(path):
    r=json.loads(Path(path).read_text())
    if (r.get('mode')!='ASSUMED_REFERENCE_PRETRAINED_INFERENCE_ONLY'
            or r.get('formal_reference_frozen') is not False
            or r.get('independent_labels_received') is not False
            or r.get('evaluation_permitted') is not False
            or r.get('model_configuration_locked_for_this_scenario') is not True):
        raise ValueError('require explicit assumption-only inference record; no formal freeze/agreement claim')
    if r.get('site_definitions_sha256')!=sha256(ROOT/'data/sites/reference_sites.json'):
        raise ValueError('scenario site definitions changed')
    current=ROOT/r['reference_json'];old=ROOT/r['original_selection_reference_json']
    if sha256(current)!=r['reference_json_sha256'] or sha256(old)!=r['original_selection_reference_json_sha256']:
        raise ValueError('scenario reference versions changed')
    if training_signature(current)!=r['training_geometry_signature'] or training_signature(old)!=r['training_geometry_signature']:
        raise ValueError('training labels differ from the conversion-selection reference; repeat training selection first')
    if r.get('sat_config_sha256')!=sha256(ROOT/'config/segmentanytree_source.json'):
        raise ValueError('pretrained source/checkpoint configuration changed')
    selection=ROOT/r['training_conversion_selection']
    if sha256(selection)!=r['training_conversion_selection_sha256']:
        raise ValueError('training conversion selection changed')
    chosen=json.loads(selection.read_text())
    if set(chosen['site_ids'])!={sid for sid,s in site_definitions().items() if s['split']=='train'}:
        raise ValueError('conversion must have used all and only training sites')
    if r.get('conversion_config_id')!=chosen['selected_config_id']:
        raise ValueError('scenario does not retain the training-selected conversion')
    return r
