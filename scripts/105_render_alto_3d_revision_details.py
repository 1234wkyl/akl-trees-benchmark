#!/usr/bin/env python3
"""Render exact before/after returns and auxiliary ALTO profiles for inspection."""
import importlib.util,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'outputs/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from benchmark.common import ROOT,sha256,write_json,site_definitions

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
 out=ROOT/'outputs/alto_3d_weak_revision_20261011/revised';audit=json.loads((out/'instance_3d_audit.json').read_text())
 ply=load('ply105',ROOT.parent/'SegmentAnyTree/torch_points3d/modules/KPConv/plyutils.py');method=load('method103',ROOT/'scripts/103_refine_weak_labels_with_alto_3d.py')
 defs=site_definitions();panels=[]
 for sid,ident in [('S008',25),('S010',42),('S019',10),('S010',17),('S010',62)]:
  before=ROOT/'outputs/confirmed_reference_20261010/screened_weak_labels'/sid/(sid+'_screened_weak.ply');after=out/'labels'/sid/before.name
  b=ply.read_ply(str(before));a=ply.read_ply(str(after));xyz=np.column_stack([b[k] for k in ('X','Y','Z')]);selected=(b['treeID']==ident)&(b['semantic_seg']==2)
  retained=selected&(a['semantic_seg']==2)&(a['treeID']==ident);withdrawn=selected&~retained
  candidate=json.loads((ROOT/'outputs/confirmed_reference_20261010/alto_group_candidates'/sid/'viewer_points.json').read_text());cla=np.asarray(candidate['points'])[:,5]
  ground,supported,_=method.terrain_height(xyz,cla);hag=xyz[:,2]-ground
  pts=xyz[selected];lo=pts.min(0);hi=pts.max(0);ctx=(xyz[:,0]>=lo[0]-3)&(xyz[:,0]<=hi[0]+3)&(xyz[:,1]>=lo[1]-3)&(xyz[:,1]<=hi[1]+3)
  rec=next(r for r in audit if r['site_id']==sid and r['treeID']==ident);fig,axs=plt.subplots(1,4,figsize=(17,4.8),layout='constrained')
  origin=np.array(candidate['origin_xyz_m']);bb=defs[sid]['bbox_epsg2193'];extent=(bb[0]-origin[0],bb[2]-origin[0],bb[1]-origin[1],bb[3]-origin[1])
  axs[0].imshow(plt.imread(ROOT/'data/sites/chips'/sid/'rgb.png'),extent=extent)
  axs[0].scatter(xyz[retained,0],xyz[retained,1],s=9,c='#1596b2');axs[0].scatter(xyz[withdrawn,0],xyz[withdrawn,1],s=18,c='#e86826',marker='x')
  axs[0].set_xlim(lo[0]-3,hi[0]+3);axs[0].set_ylim(lo[1]-3,hi[1]+3);axs[0].set_aspect('equal');axs[0].set_title('RGB: blue retained; orange unknown')
  reliable_count=int((selected&supported).sum());vertical=hag if reliable_count>=10 else xyz[:,2]
  shown=supported if reliable_count>=10 else np.ones(len(xyz),bool)
  for ax,dim,title in [(axs[1],0,'East'),(axs[2],1,'North')]:
   mask=ctx&shown;ax.scatter(xyz[mask,dim],vertical[mask],s=2,c='#b7bec5',alpha=.4)
   mask=retained&shown;ax.scatter(xyz[mask,dim],vertical[mask],s=10,c='#1596b2')
   mask=withdrawn&shown;ax.scatter(xyz[mask,dim],vertical[mask],s=22,c='#e86826',marker='x')
   if rec['alto_canopy_top_m'] and reliable_count>=10:ax.axhline(rec['alto_canopy_top_m'],c='green',ls='--',label='Current ALTO top')
   ax.set_title(title+(' / estimated HAG' if reliable_count>=10 else ' / local Z: terrain unsupported'));ax.set_ylabel('Estimated HAG (m)' if reliable_count>=10 else 'Local recorded Z (m; not HAG)');ax.grid(alpha=.2)
  mask=selected&supported;vals=hag[mask];bins=np.arange(np.floor(min(0,vals.min())),np.ceil(vals.max())+2) if len(vals) else np.arange(0,2)
  if len(vals):axs[3].hist(vals,bins=bins,density=True,orientation='horizontal',color='#bbbbbb',alpha=.5,label='Original assigned returns')
  else:axs[3].text(.03,.9,'No supported LAZ HAG here',transform=axs[3].transAxes,fontsize=8)
  vals=hag[retained&supported]
  if len(vals):axs[3].hist(vals,bins=bins,density=True,orientation='horizontal',histtype='step',color='#1596b2',lw=2,label='Revised assigned returns')
  profile=rec['historical_foliage_profile']
  if profile and sum(profile)>0:axs[3].plot(np.array(profile)/sum(profile),np.arange(len(profile))+.5,c='#aa5c96',label='Historical ALTO profile (old footprint)')
  if rec['alto_canopy_top_m']:axs[3].axhline(rec['alto_canopy_top_m'],c='green',ls='--')
  axs[3].legend(fontsize=7);axs[3].set_xlabel('Normalized return frequency per metre');axs[3].set_ylabel('Height above ground (m)');axs[3].set_title('Auxiliary vertical evidence')
  fig.suptitle(f'{sid} ID {ident}: {withdrawn.sum()} ownership targets withdrawn; {retained.sum()} retained\nSource ID {rec["alto_id"]}; current top {rec["alto_canopy_top_m"]} m. Historical profile has an earlier spatial basis. No field certification.',fontsize=11)
  name=f'{sid}_ID{ident}_3d_detail.png';fig.savefig(out/name,dpi=150);plt.close(fig);panels.append(name)
 html=(out/'index.html').read_text();html=html.split('<h2>Detailed before/after and 3D evidence</h2>')[0].removesuffix('</html>')+'</html>';html=html.replace('../../labeller/point_review.html','../../../labeller/point_review.html')
 html=html.replace('</html>','<h2>Detailed before/after and 3D evidence</h2><p>Blue = retained candidate ownership; orange = ownership withdrawn to unknown. Grey = original height distribution/context. Purple = historical Alto profile; its old footprint can differ from the current crown. Green = current Alto canopy top. Estimated HAG is shown only where nearby ground supports a stable local plane.</p>'+''.join(f'<img src="{p}">' for p in panels)+'</html>');(out/'index.html').write_text(html)
 write_json(out/'detail_figure_receipt.json',{'figures':{p:sha256(out/p) for p in panels},'script_sha256':sha256(Path(__file__)),'labels_changed':False})

if __name__=='__main__':main()
