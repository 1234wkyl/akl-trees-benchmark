#!/usr/bin/env python3
"""Read-only RGB/CHM overlays of all test errors under the October reference assumption."""
import argparse,html,json,os
from pathlib import Path
import geopandas as gpd
from benchmark.common import ROOT,sha256,site_definitions,write_json


def path_geometry(geometry,bbox):
    x0,y0,x1,y1=bbox
    def ring(coords):
        points=[(400*(x-x0)/(x1-x0),400*(y1-y)/(y1-y0)) for x,y,*_ in coords]
        return 'M '+' L '.join(f'{x:.3f},{y:.3f}' for x,y in points)+' Z'
    polygons=[geometry] if geometry.geom_type=='Polygon' else list(geometry.geoms)
    return ' '.join(ring(p.exterior.coords)+' '+' '.join(ring(h.coords) for h in p.interiors) for p in polygons)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['comparison','reference','sat-predictions','classical-predictions','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--author-lock',type=Path,help='explicit author lock; does not imply supervisor approval')
    p.add_argument('--classical-label',default='Classical provisional')
    p.add_argument('--sat-label',default='SegmentAnyTree pretrained')
    a=p.parse_args()
    if a.output.exists():p.error('fresh read-only gallery required')
    r=json.loads(a.comparison.read_text())
    if a.author_lock:
        lock=json.loads(a.author_lock.read_text())
        if lock['status']!='author_locked' or lock['reference_gpkg_sha256']!=sha256(a.reference):
            raise ValueError('matching author lock required')
    if r['reference_frozen'] is not False:raise ValueError('explicit conditional comparison required')
    for key,path in [('reference',a.reference),('sat',a.sat_predictions),('classical',a.classical_predictions)]:
        if r['sources'][key]!=sha256(path):raise ValueError('comparison source changed: '+key)
    refs=gpd.read_file(a.reference,layer='all_objects')
    preds=[gpd.read_file(a.classical_predictions,layer='predictions'),gpd.read_file(a.sat_predictions,layer='predictions')]
    defs=site_definitions();a.output.mkdir(parents=True);cards=[]
    for sid in r['site_ids']:
        chunks=[];bbox=defs[sid]['bbox_epsg2193']
        for name,frame,scores in zip([a.classical_label,a.sat_label],preds,[r['classical_by_site'],r['pretrained_by_site']]):
            score=scores[sid];fn=set(score['unmatched_reference_ids']);fp=set(score['false_positive_ids']);ignored=set(score['ignored_prediction_ids'])
            svg=['<svg viewBox="0 0 400 400" role="img" aria-label="'+html.escape(sid+' '+name)+'">']
            rgb=html.escape(os.path.relpath(ROOT/'data/sites/chips'/sid/'rgb.png',a.output),quote=True)
            chm=html.escape(os.path.relpath(ROOT/'data/sites/chips'/sid/'chm.png',a.output),quote=True)
            svg.append(f'<image class="background" href="{rgb}" data-rgb="{rgb}" data-chm="{chm}" x="0" y="0" width="400" height="400"/>')
            for _,v in refs[refs.site_id==sid].iterrows():
                oid=str(v.label_id);kind=v['class'];col=('#ffe535' if oid in fn else '#2cccf4') if kind=='tree' else {'tree_group':'#c466f8','uncertain':'#ff9c29','not_tree':'#dddddd'}[kind]
                label='missed single reference' if oid in fn else 'reference '+kind
                svg.append(f'<path class="reference" d="{path_geometry(v.geometry,bbox)}" fill="{col}" fill-opacity="0.07" fill-rule="evenodd" stroke="{col}" stroke-width="1.1" stroke-dasharray="3 2"><title>{html.escape(oid+": "+label)}</title></path>')
            for _,v in frame[frame.site_id==sid].iterrows():
                oid=str(v.prediction_id);status='false positive' if oid in fp else 'ignored' if oid in ignored else 'matched prediction'
                col='#ff3939' if oid in fp else '#bbbbbb' if oid in ignored else '#24ed73'
                svg.append(f'<path class="prediction" data-status="{status}" d="{path_geometry(v.geometry,bbox)}" fill="none" stroke="{col}" stroke-width="1.3"><title>{html.escape(oid+": "+status)}</title></path>')
            svg.append('</svg>');chunks.append('<section><h3>'+name+f"</h3><p>TP {score['tp']} · FP {score['fp']} · FN {score['fn']} · ignored {score['ignored']}</p>"+''.join(svg)+'</section>')
        cards.append(f'<article id="{sid}"><h2>{sid} — {html.escape(defs[sid]["stratum"])}</h2><div class="pair">'+''.join(chunks)+'</div></article>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Conditional test error review</title><style>body{font:15px system-ui;max-width:1100px;margin:24px auto;background:#f5f6f8;color:#182635}header{position:sticky;top:0;background:#fff;padding:14px;z-index:1;border:1px solid #ccc}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}article{background:#fff;padding:18px;margin-top:22px}svg{width:100%;border:1px solid #888}button{padding:7px 12px;margin-right:8px}@media(max-width:650px){.pair{grid-template-columns:1fr}}</style><header><h1>Conditional test error review</h1><p>6 October reference assumption. Independent annotations and formal freeze pending. Read-only; no model or label editing.</p><p>Predictions: green matched · red false positive · grey ignored. Dashed reference: yellow missed single · cyan other single · purple group · orange uncertain · pale grey non-tree.</p><button onclick="bg('rgb')">RGB</button><button onclick="bg('chm')">CHM</button><label><input type="checkbox" checked onchange="toggle('.prediction',this.checked)"> Predictions</label> <label><input type="checkbox" checked onchange="toggle('.reference',this.checked)"> Reference</label> <label><input type="checkbox" checked onchange="toggle('[data-status=ignored]',this.checked)"> Ignored</label><p>Hover a polygon for its ID. Classification is defined by the fixed scoring rule, not an independent biological judgment. Inspect imagery before assigning an error cause. No selective examples are omitted.</p></header>'''+''.join(cards)+'''<script>function bg(n){document.querySelectorAll('.background').forEach(e=>e.setAttribute('href',e.dataset[n]))}function toggle(s,on){document.querySelectorAll(s).forEach(e=>e.style.display=on?'':'none')}</script></html>'''
    if r.get('actual_independent_agreement_available'):
        page=page.replace('6 October reference assumption. Independent annotations and formal freeze pending.',
            'Revised author reference, 10 October 2026. Independent annotations received; author harmonisation completed, with remaining interpretations documented. Formal freeze pending.')
    if a.author_lock:
        page=page.replace('Revised author reference, 10 October 2026. Independent annotations received; author harmonisation completed, with remaining interpretations documented. Formal freeze pending.',
            'Author-confirmed reference, 10 October 2026. Supervisor approval is not asserted. Previous test exposure and annotation revision are disclosed; this is a conditional comparison.')
    (a.output/'index.html').write_text(page)
    write_json(a.output/'gallery_receipt.json',{'status':'READ-ONLY CONDITIONAL ERROR REVIEW; NO LABEL OR MODEL CHANGES','sites':r['site_ids'],
        'comparison_sha256':sha256(a.comparison),'reference_sha256':sha256(a.reference),
        'prediction_sha256':{'sat':sha256(a.sat_predictions),'classical':sha256(a.classical_predictions)},
        'author_lock_sha256':sha256(a.author_lock) if a.author_lock else None,
        'gallery_sha256':sha256(a.output/'index.html'),'source_sha256':sha256(Path(__file__)),
        'chips_sha256':{f'{sid}/{kind}.png':sha256(ROOT/'data/sites/chips'/sid/f'{kind}.png') for sid in r['site_ids'] for kind in ['rgb','chm']},
        'display_mapping':'100 m EPSG:2193 site bbox to 400 x 400 chip; y reversed; polygon holes retained using evenodd fill','reference_frozen':False})
    print('All eight fixed test-site error overlays created:',a.output/'index.html')
if __name__=='__main__':main()
