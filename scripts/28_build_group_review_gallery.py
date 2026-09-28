#!/usr/bin/env python
"""Build a portable RGB/CHM gallery of every changed group, with provenance."""
import argparse,html,importlib.util,json
from pathlib import Path
from PIL import Image,ImageDraw
spec=importlib.util.spec_from_file_location('refinement',Path(__file__).with_name('26_refine_tree_groups.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
out=r.OUT/'visual_review';out.mkdir(exist_ok=True)
ap=argparse.ArgumentParser();ap.add_argument('--export-directory',type=Path);args=ap.parse_args()
directory=args.export_directory or r.OUT/'export'
data=json.loads((directory/'crown_labels_refinement_proposal.json').read_text())
summary=json.loads((directory/'summary.json').read_text())
ds=json.loads((directory/'refinement_decisions.json').read_text()) if (directory/'refinement_decisions.json').exists() else r.read_decisions()
disposition_path=r.OUT/'author_review_round2_2026-09-22.json'
disposition=json.loads(disposition_path.read_text()) if disposition_path.exists() else {}
rejected=set(disposition.get('rejected_parent_ids',[]))
comparison_dir=r.ROOT/disposition['rejected_proposal_export'] if rejected else None
comparison_data=json.loads((comparison_dir/'crown_labels_refinement_proposal.json').read_text()) if rejected else None
original_proposals=json.loads((r.OUT/'second_pass/new_decisions.json').read_text()) if rejected else {}
cards=[]
for lid in sorted(set(lid for lid,d in ds.items() if d['action']!='retain')|rejected):
    is_rejected=lid in rejected
    d=original_proposals[lid] if is_rejected else ds[lid]
    sid,parent=r.OBJECTS[lid]
    displayed=comparison_data if is_rejected else data
    children=[l for l in displayed['sites'][sid]['labels'] if l['parent_label_id']==lid]
    if not children: raise ValueError(f'No comparison children for {lid}')
    xs=[p[0] for p in parent['px']];ys=[p[1] for p in parent['px']]
    x0=max(0,int(min(xs))-12);y0=max(0,int(min(ys))-12)
    x1=min(400,int(max(xs))+13);y1=min(400,int(max(ys))+13)
    canvas=Image.new('RGB',(920,1020),'white');draw=ImageDraw.Draw(canvas)
    draw.text((12,4),lid+' | '+d['action']+' | ORIGINAL RGB / CHM',(0,0,0))
    caption='DECLINED SUGGESTION (not in current JSON)' if is_rejected else 'ACCEPTED REVISION (pre-freeze)'
    draw.text((12,510),caption+': cyan singles; magenta unresolved group',(0,0,0))
    for i,name in enumerate(['rgb','chm','rgb','chm']):
        im=Image.open(r.ROOT/'data/sites/chips'/sid/(name+'.png')).convert('RGB')
        drawim=ImageDraw.Draw(im)
        drawim.line([tuple(p) for p in parent['px']]+[tuple(parent['px'][0])],fill='#ffe000',width=1)
        for child in (children if i>=2 else []):
            col='#00ffff' if child['cls']=='tree' else '#ff00ff'
            points=[tuple(p) for p in child['px']]
            drawim.line(points+[points[0]],fill=col,width=1)
        scale=min(440/(x1-x0),440/(y1-y0))
        crop=im.crop((x0,y0,x1,y1)).resize((round((x1-x0)*scale),round((y1-y0)*scale)),Image.Resampling.NEAREST)
        canvas.paste(crop,(10+460*(i%2),30+510*(i//2)))
    status='Not accepted; original tree_group retained' if is_rejected else 'Accepted working revision; reference not frozen'
    draw.text((12,995),'YELLOW parent; CYAN singles; MAGENTA remainder. '+status,(0,0,0))
    filename=lid+'.png';canvas.save(out/filename)
    child_heading='Comparison-only children: ' if is_rejected else 'Current children: '
    cards.append(f'<article><h3>{sid} {lid}</h3><p>{html.escape(status)}</p><img src="{filename}"><p>{html.escape(d["reason"])}</p><p>'+child_heading+', '.join(l['id']+' ('+l['cls']+')' for l in children)+'</p></article>')
counts=summary['single_trees_by_split']
header=f"Original: 219 singles. Current accepted working version: {summary['proposal_classes']['tree']} (train {counts['train']} / val {counts['val']} / test {counts['test']}). Accepted changes: {summary.get('author_accepted_changed_parents',0)}; declined suggestions shown for comparison: {len(rejected)}. Not frozen. Original XZ/TW comparison must precede adjudication."
(out/'review_proposals.html').write_text('<!doctype html><meta charset="utf-8"><title>Tree-group refinement review</title><style>body{font:16px system-ui;max-width:960px;margin:30px auto}img{max-width:100%}article{border-top:1px solid #aaa;margin-top:25px}</style><h1>Tree-group refinement review</h1><p>'+header+'</p>'+''.join(cards))
print(json.dumps({'changed_groups':len(cards),'gallery':str(out/'review_proposals.html')}))
