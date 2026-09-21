#!/usr/bin/env python
"""Local visual review UI; saves decisions and fresh exports inside Starter_Kit.

No detector outputs are loaded. Original annotations are never overwritten.
Run with the project's geospatial Python; browse http://127.0.0.1:8902/.
"""
import argparse
import copy
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

import geopandas as gpd
from shapely.geometry import Polygon, box

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/labels/crown_labels_XZ_2026-08-13.json'
OUT = ROOT / 'docs/progress_2026-09-18/reference_review'
OUT.mkdir(parents=True, exist_ok=True)
LABELS = json.loads(SOURCE.read_text())
SITES = {s['site_id']: s for s in json.loads((ROOT / 'data/sites/reference_sites.json').read_text())['sites']}
OBJECTS = {l['id']: (sid, l) for sid, r in LABELS['sites'].items() for l in r['labels']}
DECISIONS = OUT / 'browser_decisions.json'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def decisions():
    return json.loads(DECISIONS.read_text()) if DECISIONS.exists() else {}


def export():
    reviewed = copy.deepcopy(LABELS)
    ds = decisions()
    rows = []
    for sid, rec in reviewed['sites'].items():
        for lab in rec['labels']:
            d = ds.get(lab['id'])
            must_review = lab['cls'] in ['tree_group', 'uncertain'] or Polygon(lab['world']).buffer(0).area > 900
            if must_review and d is None:
                raise ValueError('Review incomplete: ' + lab['id'])
            lab['original_class'] = lab['cls']
            lab['excluded_from_single_tree_evaluation'] = lab['cls'] in ['tree_group', 'uncertain']
            if d:
                if d['action'] in ['tree', 'tree_group', 'not_tree']:
                    lab['cls'] = d['action']
                lab['excluded_from_single_tree_evaluation'] = d['action'] == 'exclude' or lab['cls'] != 'tree'
                lab['review'] = d
            p = Polygon(lab['world'])
            repaired = p if p.is_valid else p.buffer(0)
            clipped = repaired.intersection(box(*SITES[sid]['bbox_epsg2193']))
            rows.append(dict(label_id=lab['id'], site_id=sid, stratum=SITES[sid]['stratum'], split=SITES[sid]['split'],
                original_class=lab['original_class'], **{'class': lab['cls']}, labeller='XZ',
                reviewed=d is not None, review_action=d['action'] if d else 'not_flagged',
                review_reason=d['reason'] if d else '', area_before_clip_m2=repaired.area,
                raw_geometry_valid=p.is_valid, geometry=clipped))
    reviewed['exported_at'] = datetime.now(timezone.utc).isoformat()
    reviewed['review_provenance'] = {'source': str(SOURCE.relative_to(ROOT)), 'source_sha256': sha(SOURCE),
        'reviewer': 'Codex AI assisted visual review requested by Xi; not an independent human rater',
        'decision_file_sha256': sha(DECISIONS), 'reference_frozen': False,
        'geometry_policy': 'raw px/world unchanged; exported geospatial layers repair buffer(0) and clip to site',
        'model_detections_used': False}
    dest = OUT / 'crown_labels_XZ_reviewed_2026-09-18.json'
    dest.write_text(json.dumps(reviewed, indent=2) + '\n')
    frame = gpd.GeoDataFrame(rows, crs=2193)
    gpkg = OUT / 'reference_reviewed_2026-09-18.gpkg'
    if gpkg.exists():
        raise ValueError('Export already exists; preserve the previous export and use a new version')
    frame.to_file(gpkg, layer='all_reviewed_objects', driver='GPKG')
    for name, cls in [('single_trees','tree'), ('tree_groups','tree_group'), ('uncertainty_regions','uncertain'), ('hard_negatives','not_tree')]:
        frame[frame['class'] == cls].to_file(gpkg, layer=name, driver='GPKG')
    frame.drop(columns='geometry').to_csv(OUT / 'object_review_audit.csv', index=False)
    counts = frame.groupby(['stratum','split','class']).size().unstack('class', fill_value=0)
    counts.to_csv(OUT / 'counts_by_split_stratum.csv')
    summary = {'objects': len(frame), 'classes': dict(Counter(frame['class'])), 'reviewed_objects': len(ds),
        'single_trees_by_split': frame[frame['class']=='tree'].groupby('split').size().to_dict(),
        'large_polygons_reviewed': sum(r['area_before_clip_m2'] > 900 and r['reviewed'] for r in rows),
        'uncertain_excluded': sum(r['original_class']=='uncertain' and r['review_action']=='exclude' for r in rows),
        'raw_source_unchanged': sha(SOURCE)==reviewed['review_provenance']['source_sha256'],
        'reference_frozen': False, 'independent_agreement_available': False,
        'outputs': {p.name:sha(p) for p in [dest,gpkg,DECISIONS]},
        'scope': 'Class/large-object AI-assisted visual QA. No new crown splits inferred; no independent ground truth or freeze claimed.'}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


class Handler(BaseHTTPRequestHandler):
    def send(self, data, typ='application/json'):
        raw = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.send_response(200); self.send_header('Content-Type',typ); self.end_headers(); self.wfile.write(raw)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/':
            return self.send((ROOT/'labeller/review.html').read_bytes(),'text/html; charset=utf-8')
        if path == '/api/data':
            return self.send({'labels': LABELS, 'sites': SITES, 'decisions': decisions(), 'source_sha256':sha(SOURCE)})
        parts = path.strip('/').split('/')
        if len(parts)==3 and parts[0]=='chips' and parts[1] in SITES and parts[2] in ['rgb.png','chm.png']:
            return self.send((ROOT/'data/sites/chips'/parts[1]/parts[2]).read_bytes(),'image/png')
        self.send_error(404)

    def do_POST(self):
        try:
            data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path=='/api/decision':
                ds=decisions()
                for lid in data['ids']:
                    if lid not in OBJECTS or data['action'] not in ['retain','exclude','tree','tree_group','not_tree']:
                        raise ValueError('Invalid review request')
                    if len(data['reason'].strip())<10: raise ValueError('Record a specific review reason')
                    ds[lid]={'action':data['action'],'reason':data['reason'],'reviewed_at_utc':datetime.now(timezone.utc).isoformat(),
                        'reviewer':'Codex AI assisted visual review requested by Xi','evidence':'RGB + CHM chips; model detections hidden'}
                DECISIONS.write_text(json.dumps(ds,indent=2)+'\n')
                return self.send({'saved':len(ds)})
            if self.path=='/api/export': return self.send(export())
            self.send_error(404)
        except Exception as exc:
            self.send_response(400);self.end_headers();self.wfile.write(str(exc).encode())


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8902);args=ap.parse_args()
    print(f'Review http://127.0.0.1:{args.port}/; original labels preserved',flush=True)
    HTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
