#!/usr/bin/env python
"""Validate actual refinement export and regenerate manual cut drafts."""
import argparse, importlib.util, json
from pathlib import Path
import geopandas as gpd
from shapely.geometry import LineString, shape
from shapely.ops import split, unary_union

spec=importlib.util.spec_from_file_location('refinement',Path(__file__).with_name('26_refine_tree_groups.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

def validate_export(directory=None):
    out=Path(directory).resolve() if directory else r.OUT/'export'
    if not out.is_relative_to(r.OUT): raise ValueError('Export must be in the local refinement directory')
    data=json.loads((out/'crown_labels_refinement_proposal.json').read_text())
    frame=gpd.read_file(out/'reference_refinement_proposal.gpkg',layer='all_objects')
    assert frame.label_id.is_unique and frame.geometry.is_valid.all()
    assert not frame.geometry.is_empty.any()
    assert data['refinement_provenance']['source_sha256']==r.sha(r.SOURCE)
    decision_file=out/'refinement_decisions.json'
    if not decision_file.exists(): decision_file=r.OUT/'decisions.json'
    ds=json.loads(decision_file.read_text())
    assert data['refinement_provenance']['decisions_sha256']==r.sha(decision_file)
    assert data['refinement_provenance']['reference_frozen'] is False
    accepted=None
    acceptance_hash=data['refinement_provenance'].get('author_acceptance_sha256')
    if acceptance_hash:
        acceptance_file=out/'author_acceptance.json'
        assert r.sha(acceptance_file)==acceptance_hash
        acceptance=json.loads(acceptance_file.read_text())
        assert acceptance['source_sha256']==r.sha(r.SOURCE)
        accepted={lid for lid,d in acceptance['accepted_decisions'].items() if ds.get(lid)==d}
        assert set(data['refinement_provenance']['accepted_parent_ids'])==accepted
    assert len(ds)==228
    audits=[]
    for lid,d in ds.items():
        r.validate(lid,d)
        if d['action']=='retain':continue
        sid,_=r.OBJECTS[lid];p=r.parent_geom(lid)
        children=frame[frame.parent_label_id==lid]
        union=unary_union(children.geometry)
        delta=union.symmetric_difference(p).area
        assert delta<1e-5,(lid,delta)
        overlap=sum(children.geometry.area)-union.area
        assert overlap<1e-5,(lid,overlap)
        for q in children.geometry:
            assert q.difference(p).area<1e-5
        for _,row in children[children['class']=='tree'].iterrows():
            for old in r.LABELS['sites'][sid]['labels']:
                if old['cls']!='tree':continue
                other=r.parent_geom(old['id'])
                assert row.geometry.intersection(other).area<=.1*min(row.geometry.area,other.area),(row.label_id,old['id'])
        audits.append(dict(parent=lid,action=d['action'],new_single_crowns=int(sum(children['class']=='tree')),
            unresolved_components=int(sum(children['class']=='tree_group')),area_conservation_difference_m2=delta,
            overlapping_children_area_m2=overlap))
    # All original single trees remain present, with their original identities.
    original_ids={l['id'] for rec in r.LABELS['sites'].values() for l in rec['labels'] if l['cls']=='tree'}
    assert original_ids.issubset(set(frame[frame['class']=='tree'].label_id))
    expected=219+sum(1 if d['action']=='single' else len(r.validate(lid,d)) if d['action']=='split' else 0 for lid,d in ds.items())
    assert sum(frame['class']=='tree')==expected
    assert sum(frame['class']=='uncertain')==4
    assert sum(frame['class']=='not_tree')==41
    for sid,rec in data['sites'].items():
        for l in rec['labels']:
            q=shape(l['geometry'])
            saved=frame[frame.label_id==l['id']].geometry.iloc[0]
            assert q.symmetric_difference(saved).area<1e-8
            assert abs(l['centroid_epsg2193'][0]-q.centroid.x)<1e-7
            assert abs(l['centroid_epsg2193'][1]-q.centroid.y)<1e-7
            if l['refinement_status']!='source_annotation':
                assert l['labeller']=='Codex_AI_refinement_of_XZ'
                if accepted is not None:
                    expected_status='author_accepted_pre_freeze_revision' if l['parent_label_id'] in accepted else 'AI_visual_proposal_pending_author_review'
                    assert l['refinement_status']==expected_status
                    if l['parent_label_id'] in accepted:assert l['accepted_by']=='Xi Zhang'
            if 'refinement_status' in frame:
                assert frame[frame.label_id==l['id']].refinement_status.iloc[0]==l['refinement_status']
            if l['cls']!='tree':assert l['excluded_from_single_tree_evaluation']
    receipt=dict(passed=True,checks=['source hash unchanged','228 group decisions present','unique IDs',
        'valid nonempty geometry','parent area conserved','child crowns do not overlap',
        'child crowns inside parent','no substantial overlap with original single crowns',
        'all 219 original singles preserved','all four uncertainties remain excluded',
        'all 41 negatives preserved','JSON/GPKG geometries agree','centroids recomputed for every geometry',
        'AI refinement attribution distinct from source XZ'],parents=audits,
        ecological_identity_validated=False,author_review_required=json.loads((out/'summary.json').read_text())['author_review_required'],
        source_sha256=r.sha(r.SOURCE))
    (out/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt

def regenerate_drafts():
    """Reproduce human/AI-chosen cut lines; these are not detected treetops."""
    old=json.loads((r.OUT/'manual_split_drafts.json').read_text());ds=[]
    sides={'S007-mspep4dv':'top','S014-mss0nwrf':'both','S017-mss1nobm':'right'}
    for d in old:
        sid,_=r.OBJECTS[d['id']]
        pieces=r.components(split(r.parent_geom(d['id']),LineString(r.px_world(sid,d['manual_cut_pixels']))))
        if sides[d['id']]=='top':pieces=[max(pieces,key=lambda q:q.centroid.y)]
        if sides[d['id']]=='right':pieces=[max(pieces,key=lambda q:q.centroid.x)]
        new=dict(d,crowns=[r.world_px(sid,p) for p in pieces]);r.validate(d['id'],new);ds.append(new)
    # A separate file prevents overwriting the original visual decisions.
    p=r.OUT/'manual_split_drafts_regenerated.json';p.write_text(json.dumps(ds,indent=2)+'\n')
    return {'regenerated':str(p),'manual_cut_lines':True,'automatic_tree_detection':False}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--regenerate-drafts',action='store_true');ap.add_argument('--export-directory');a=ap.parse_args()
    print(json.dumps(regenerate_drafts() if a.regenerate_drafts else validate_export(a.export_directory),indent=2))
