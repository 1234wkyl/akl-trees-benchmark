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
cards=[]
for lid,d in ds.items():
    if d['action']=='retain':continue
    sid,parent=r.OBJECTS[lid]
    children=[l for l in data['sites'][sid]['labels'] if l['parent_label_id']==lid]
    xs=[p[0] for p in parent['px']];ys=[p[1] for p in parent['px']]
    x0=max(0,int(min(xs))-12);y0=max(0,int(min(ys))-12)
    x1=min(400,int(max(xs))+13);y1=min(400,int(max(ys))+13)
    canvas=Image.new('RGB',(920,1020),'white');draw=ImageDraw.Draw(canvas)
    draw.text((12,4),lid+' | '+d['action']+' | ORIGINAL RGB / CHM',(0,0,0))
    draw.text((12,510),'PROPOSED RGB / CHM: cyan singles; magenta unresolved group',(0,0,0))
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
    status=children[0]['refinement_status']
    draw.text((12,995),'YELLOW parent; CYAN singles; MAGENTA remainder. '+status,(0,0,0))
    filename=lid+'.png';canvas.save(out/filename)
    cards.append(f'<article><h3>{sid} {lid}</h3><p>{html.escape(status)}</p><img src="{filename}"><p>{html.escape(d["reason"])}</p><p>Children: '+', '.join(l['id']+' ('+l['cls']+')' for l in children)+'</p></article>')
counts=summary['single_trees_by_split']
header=f"Original: 219 singles. This version: {summary['proposal_classes']['tree']} (train {counts['train']} / val {counts['val']} / test {counts['test']}). Accepted changed parents: {summary.get('author_accepted_changed_parents',0)}; pending: {summary.get('pending_changed_parents',len(cards))}. Not frozen. Original XZ/TW comparison must precede adjudication."
(out/'review_proposals.html').write_text('<!doctype html><meta charset="utf-8"><title>Tree-group refinement proposals</title><style>body{font:16px system-ui;max-width:960px;margin:30px auto}img{max-width:100%}article{border-top:1px solid #aaa;margin-top:25px}</style><h1>Tree-group refinement proposals</h1><p>'+header+'</p>'+''.join(cards))
print(json.dumps({'changed_groups':len(cards),'gallery':str(out/'review_proposals.html')}))
