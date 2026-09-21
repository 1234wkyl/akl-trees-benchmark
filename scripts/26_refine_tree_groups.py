#!/usr/bin/env python
"""Manual RGB/CHM-supported group refinement. Never overwrites source labels.

Run: python scripts/26_refine_tree_groups.py --serve
Then open http://127.0.0.1:8903/. Click vertices, Add crown, record reason, Save.
Coordinates are on the original 400-pixel annotation grid. Unassigned parent
geometry remains tree_group; proposals require author review before freeze.
"""
import argparse, copy, csv, hashlib, json
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Polygon, box, mapping
from shapely.ops import unary_union

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/labels/crown_labels_XZ_2026-08-13.json'
OUT=ROOT/'docs/group_refinement_2026-09-18'
SITES={s['site_id']:s for s in json.loads((ROOT/'data/sites/reference_sites.json').read_text())['sites']}
LABELS=json.loads(SOURCE.read_text())
OBJECTS={l['id']:(sid,l) for sid,s in LABELS['sites'].items() for l in s['labels']}
REVIEW_LOCK=Lock()
class ReviewServer(ThreadingHTTPServer):
    request_queue_size=64

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read_decisions():
    p=OUT/'decisions.json'
    return json.loads(p.read_text()) if p.exists() else {}
def accepted_decisions(ds):
    """Acceptance is tied to each exact reviewed decision, never a future edit."""
    p=OUT/'author_acceptance.json'
    if not p.exists(): return set()
    record=json.loads(p.read_text())
    if record['source_sha256']!=sha(SOURCE): raise ValueError('Acceptance source changed')
    return {lid for lid,d in record['accepted_decisions'].items() if ds.get(lid)==d}
def parent_geom(lid):
    sid,l=OBJECTS[lid]
    return Polygon(l['world']).buffer(0).intersection(box(*SITES[sid]['bbox_epsg2193']))
def px_world(sid,pts):
    x0,y0,x1,y1=SITES[sid]['bbox_epsg2193']
    return [(x0+(x+.5)*(x1-x0)/400,y1-(y+.5)*(y1-y0)/400) for x,y in pts]
def world_px(sid,p):
    x0,y0,x1,y1=SITES[sid]['bbox_epsg2193']
    return [[(x-x0)*400/(x1-x0)-.5,(y1-y)*400/(y1-y0)-.5] for x,y in list(p.exterior.coords)[:-1]]
def components(p):
    if p.is_empty: return []
    if p.geom_type=='Polygon': return [p]
    return [q for part in p.geoms for q in components(part)]
def validate(lid,d):
    sid,l=OBJECTS[lid]
    if l['cls']!='tree_group': raise ValueError('Only original tree groups can be refined')
    if d['action'] not in ['retain','single','split']: raise ValueError('Invalid action')
    if len(d['reason'].strip())<15: raise ValueError('Specific RGB/CHM evidence required')
    if d['action']!='split': return []
    if not d['crowns']: raise ValueError('No crowns drawn')
    parent=parent_geom(lid); result=[]
    for points in d['crowns']:
        p=Polygon(px_world(sid,points))
        if not p.is_valid or p.area<1: raise ValueError('Crown must be valid and >=1 m2')
        if p.difference(parent).area>p.area*.15: raise ValueError('Too much crown outside original group')
        q=p.intersection(parent)
        if q.geom_type!='Polygon' or len(q.interiors): raise ValueError('Draw a simple crown without holes')
        if any(q.intersection(other).area>0.01 for other in result): raise ValueError('Child crowns overlap')
        result.append(q)
    return result
def export(new_version=False):
    ds=read_decisions(); out=OUT/'export'
    if out.exists():
        if not new_version: raise ValueError('Existing version preserved; use --new-version')
        out=OUT/('export_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    accepted=accepted_decisions(ds)
    records=[];lineage=[];derived=copy.deepcopy(LABELS)
    old_review=json.loads((ROOT/'docs/progress_2026-09-18/reference_review/browser_decisions.json').read_text())
    for sid,rec in derived['sites'].items():
        active=[]
        for l in rec['labels']:
            lid=l['id']; d=ds.get(lid); p=parent_geom(lid)
            pieces=[(lid,l['cls'],p,'unchanged')]
            if d:
                children=validate(lid,d)
                if d['action']=='single': pieces=[(lid+'-single','tree',p,'reclassified_group')]
                if d['action']=='split':
                    pieces=[(f'{lid}-c{i+1:02}','tree',q,'split_child') for i,q in enumerate(children)]
                    remaining=p.difference(unary_union(children))
                    pieces.extend((f'{lid}-r{i+1:02}','tree_group',q,'unresolved_remainder') for i,q in enumerate(components(remaining)) if q.area>1e-8)
                    assert abs(sum(q.area for _,_,q,_ in pieces)-p.area)<1e-5
            for new_id,cls,q,kind in pieces:
                # GeoJSON/GPKG preserve holes; JSON displays rings explicitly.
                nl=copy.deepcopy(l)
                nl.update(id=new_id,cls=cls,world=list(q.exterior.coords)[:-1] if q.geom_type=='Polygon' else l['world'],
                    px=world_px(sid,q) if q.geom_type=='Polygon' else l['px'],area_m2=q.area,
                    centroid_epsg2193=[q.centroid.x,q.centroid.y],
                    geometry=mapping(q),parent_label_id=lid,
                    excluded_from_single_tree_evaluation=cls!='tree',
                    refinement_status='AI_visual_proposal_pending_author_review' if kind!='unchanged' else 'source_annotation')
                if kind!='unchanged':
                    if lid in accepted:
                        nl['refinement_status']='author_accepted_pre_freeze_revision'
                        nl['accepted_by']='Xi Zhang'
                    for key in ['ellipse','cx','cy','rx','ry','angle']: nl.pop(key,None)
                    nl['refinement_reason']=d['reason']
                    nl['source_labeller']=l.get('labeller','XZ')
                    nl['source_created']=l.get('created')
                    nl['labeller']='Codex_AI_refinement_of_XZ'
                    nl['created']=d.get('reviewed_at')
                if cls=='uncertain': nl['review']=old_review.get(lid)
                active.append(nl)
                row=dict(label_id=new_id,parent_label_id=lid,site_id=sid,split=SITES[sid]['split'],stratum=SITES[sid]['stratum'],
                    **{'class':cls},kind=kind,area_m2=q.area,reason=d['reason'] if d else '',
                    refinement_status=nl['refinement_status'],accepted_by=nl.get('accepted_by',''),geometry=q)
                records.append(row)
                if kind!='unchanged': lineage.append({k:v for k,v in row.items() if k!='geometry'})
        rec['labels']=active
    out.mkdir(parents=True)
    (out/'refinement_decisions.json').write_bytes((OUT/'decisions.json').read_bytes())
    if (OUT/'author_acceptance.json').exists():
        (out/'author_acceptance.json').write_bytes((OUT/'author_acceptance.json').read_bytes())
    derived['refinement_provenance']=dict(source_sha256=sha(SOURCE),decisions_sha256=sha(OUT/'decisions.json'),
        model_detections_used=False,reference_frozen=False,reviewer='Codex AI visual refinement requested by Xi',
        independent_rater=False,original_XZ_TW_comparison_must_precede_adjudication=True,
        geometry_policy='repair/clip parent; manual child crowns clipped to parent; all positive-area remainders retained; GeoJSON geometry authoritative for holes')
    acceptance_file=OUT/'author_acceptance.json'
    derived['refinement_provenance']['author_acceptance_sha256']=sha(acceptance_file) if acceptance_file.exists() else None
    derived['refinement_provenance']['accepted_parent_ids']=sorted(accepted)
    (out/'crown_labels_refinement_proposal.json').write_text(json.dumps(derived,indent=2)+'\n')
    frame=gpd.GeoDataFrame(records,crs=2193)
    frame.to_file(out/'reference_refinement_proposal.gpkg',layer='all_objects',driver='GPKG')
    for cls in ['tree','tree_group','not_tree','uncertain']:
        frame[frame['class']==cls].to_file(out/'reference_refinement_proposal.gpkg',layer=cls,driver='GPKG')
    frame.drop(columns='geometry').to_csv(out/'object_audit.csv',index=False)
    with (out/'parent_child_lineage.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(lineage[0]) if lineage else ['label_id']);w.writeheader();w.writerows(lineage)
    frame.groupby(['split','stratum','class']).size().unstack('class',fill_value=0).to_csv(out/'counts_by_split_stratum.csv')
    summary=dict(source_single_trees=219,proposal_classes=dict(Counter(frame['class'])),
        single_trees_by_split=frame[frame['class']=='tree'].groupby('split').size().to_dict(),
        actions=dict(Counter(d['action'] for d in ds.values())),reviewed_groups=len(ds),source_groups=228,
        source_sha256=sha(SOURCE),reference_frozen=False,
        author_review_required=any(d['action']!='retain' and lid not in accepted for lid,d in ds.items()),
        author_accepted_changed_parents=len(accepted),
        pending_changed_parents=sum(d['action']!='retain' and lid not in accepted for lid,d in ds.items()),
        export_directory=str(out.relative_to(ROOT)))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary

class Handler(BaseHTTPRequestHandler):
    def send(self,obj,typ='application/json'):
        raw=obj if isinstance(obj,bytes) else json.dumps(obj).encode()
        self.send_response(200);self.send_header('Content-Type',typ);self.end_headers();self.wfile.write(raw)
    def do_GET(self):
        path=self.path.split('?')[0]
        if path=='/': return self.send((ROOT/'labeller/group_refinement.html').read_bytes(),'text/html')
        if path=='/api/data': return self.send(dict(labels=LABELS,sites=SITES,decisions=read_decisions()))
        if path=='/gallery': return self.send((OUT/'visual_review/review_proposals.html').read_bytes(),'text/html; charset=utf-8')
        if path.count('/')==1 and path.endswith('.png') and path[1:-4] in OBJECTS:
            p=OUT/'visual_review'/path[1:]
            if p.exists(): return self.send(p.read_bytes(),'image/png')
        parts=path.strip('/').split('/')
        if len(parts)==3 and parts[0]=='chips' and parts[1] in SITES and parts[2] in ['rgb.png','chm.png']:
            return self.send((ROOT/'data/sites/chips'/parts[1]/parts[2]).read_bytes(),'image/png')
        self.send_error(404)
    def do_POST(self):
        # Images load concurrently; serialize saves/exports to preserve versions.
        with REVIEW_LOCK: self.handle_post()
    def handle_post(self):
        try:
            d=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path=='/api/save':
                validate(d['id'],d);ds=read_decisions()
                d['reviewed_at']=datetime.now(timezone.utc).isoformat();ds[d['id']]=d
                OUT.mkdir(parents=True,exist_ok=True)
                temporary=OUT/'decisions.tmp'
                temporary.write_text(json.dumps(ds,indent=2)+'\n')
                temporary.replace(OUT/'decisions.json')
                return self.send({'saved':d['id']})
            if self.path=='/api/export': return self.send(export(new_version=True))
            self.send_error(404)
        except Exception as e:
            self.send_response(400);self.end_headers();self.wfile.write(str(e).encode())

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--serve',action='store_true');ap.add_argument('--export',action='store_true');ap.add_argument('--new-version',action='store_true');ap.add_argument('--port',type=int,default=8903);a=ap.parse_args()
    if a.export: print(json.dumps(export(new_version=a.new_version),indent=2))
    elif a.serve: ReviewServer(('127.0.0.1',a.port),Handler).serve_forever()
