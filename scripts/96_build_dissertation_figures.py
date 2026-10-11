#!/usr/bin/env python3
"""Build original dissertation figures from the confirmed benchmark evidence.

This is presentation-only: it does not revise labels, predictions or scoring.
PNG files are for Word; SVG files preserve editable vector graphics.
"""
import argparse
import csv
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / 'outputs/.matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(__file__).resolve().parents[1] / 'outputs/.plot_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon as PatchPolygon
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import Polygon, box
from shapely.affinity import translate
from benchmark.common import ROOT, sha256, site_definitions, write_json

SPLITS = ['train', 'val', 'test']
SPLIT_COLORS = ['#0072B2', '#E69F00', '#009E73']
CLASSES = ['tree', 'tree_group', 'not_tree', 'uncertain']
CLASS_COLORS = ['#0072B2', '#AA4499', '#666666', '#E69F00']
STRATA = ['open_street_trees', 'park_open_grown', 'dense_broadleaf',
          'conifer_dominant', 'palm_present', 'small_subcanopy', 'canopy_edge', 'hard_negative_built']
SHORT = ['Street', 'Park', 'Broadleaf', 'Conifer', 'Palm', 'Small', 'Edge', 'Built']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--composition', type=Path, required=True)
    ap.add_argument('--metrics', type=Path, required=True)
    ap.add_argument('--weak-root', type=Path, required=True)
    ap.add_argument('--weak-metrics',type=Path,help='Completed hash-bound weak report metric CSV')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--completion-draft', action='store_true',
                    help='Use completed-study wording with an explicit result-update note in the workflow diagram; does not create missing results.')
    a = ap.parse_args()
    if a.output.exists(): ap.error('A fresh output directory is required.')
    a.output.mkdir(parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                        'axes.titlesize':11, 'axes.labelsize':10, 'svg.fonttype':'none'})
    refs = gpd.read_file(a.reference, layer='all_objects')
    assert refs.crs.to_epsg() == 2193
    assert refs['class'].value_counts().to_dict() == {'tree':458, 'tree_group':142, 'not_tree':41, 'uncertain':4}
    definitions = site_definitions()
    composition = list(csv.DictReader(a.composition.open()))
    metrics = list(csv.DictReader(a.metrics.open()))
    evidence = {str(p.resolve()):sha256(p) for p in [a.reference,a.composition,a.metrics,ROOT/'data/sites/reference_sites.json']}
    files=[]; details={}

    def save(fig, name):
        for ext in ('png','svg'):
            p=a.output/(name+'.'+ext)
            fig.savefig(p, dpi=300, facecolor='white', bbox_inches='tight', pad_inches=.08)
            files.append(p)
        plt.close(fig)

    # Exact site centroids; no invented background geography.
    fig,ax=plt.subplots(figsize=(6.7,5.6),layout='constrained')
    coords=np.array([v['centroid_epsg2193'] for v in definitions.values()])
    origin=np.floor(coords.min(axis=0)/1000)*1000
    for split,color,marker in zip(SPLITS,SPLIT_COLORS,['o','^','s']):
        subset=[v for v in definitions.values() if v['split']==split]
        xy=(np.array([v['centroid_epsg2193'] for v in subset])-origin)/1000
        ax.scatter(xy[:,0],xy[:,1],s=65,color=color,marker=marker,edgecolors='white',linewidths=.7,
                   label={'train':'Training','val':'Validation','test':'Test'}[split]+' (8 sites)',zorder=4)
    # Display offsets separate neighbouring labels; no site coordinates change.
    destinations={'S001':(10.6,23.2),'S017':(10.8,25.3),'S024':(16.8,21.9),
                  'S014':(12.4,27.3),'S016':(17.2,26.3),'S015':(20,22.5),
                  'S019':(23.4,24.6),'S013':(13.4,21.55),'S002':(14.5,19.1),
                  'S003':(17.7,20.6),'S021':(10.1,20.3),'S023':(7.8,18.1)}
    for i,(sid,v) in enumerate(definitions.items()):
        x,y=(np.array(v['centroid_epsg2193'])-origin)/1000
        if sid in destinations:
            ax.annotate(sid,(x,y),xytext=destinations[sid],textcoords='data',ha='center',fontsize=8,
                        arrowprops={'arrowstyle':'-','color':'#888888','lw':.6,'shrinkB':5},
                        path_effects=[pe.withStroke(linewidth=2,foreground='white')],zorder=5)
        else:
            ax.annotate(sid,(x,y),xytext=(5,5 if i%2 else -11),textcoords='offset points',fontsize=8,
                        path_effects=[pe.withStroke(linewidth=2,foreground='white')],zorder=5)
    ax.set_aspect('equal',adjustable='datalim');ax.margins(.14)
    ax.set(xlabel=f'Easting relative to {origin[0]:,.0f} m (km)',ylabel=f'Northing relative to {origin[1]:,.0f} m (km)',
           title='Auckland benchmark sites · NZTM2000 / EPSG:2193')
    ax.grid(alpha=.18);ax.legend(loc='upper left',fontsize=9,frameon=True)
    ax.annotate('N',xy=(.94,.93),xytext=(.94,.83),xycoords='axes fraction',ha='center',fontsize=11,
                arrowprops={'arrowstyle':'-|>','color':'black','lw':1.3})
    save(fig,'figure_3_1_site_distribution')
    details['site_map']={'coordinate_origin_epsg2193':origin.tolist(),'site_ids':list(definitions),'basemap':'none; exact site centroids only'}

    # Four real class examples, median-area polygon within fixed illustrative sites.
    selections=[]
    fig,axes=plt.subplots(2,4,figsize=(7.4,4.2),layout='constrained')
    for row,(cls,sid,color) in enumerate(zip(CLASSES,['S004','S009','S018','S014'],CLASS_COLORS)):
        subset=refs[(refs['class']==cls)&(refs.site_id==sid)].copy()
        region=box(*definitions[sid]['bbox_epsg2193'])
        interior=subset[subset.geometry.distance(region.boundary)>.5]
        if len(interior):subset=interior
        subset['area']=subset.geometry.area
        subset['distance_to_median']=(subset.area-subset.area.median()).abs()
        chosen=subset.sort_values(['distance_to_median','label_id']).iloc[0]
        geom=chosen.geometry;bb=definitions[sid]['bbox_epsg2193'];full=(bb[0],bb[2],bb[1],bb[3])
        x0,y0,x1,y1=geom.bounds;side=min(100,max(18,x1-x0+10,y1-y0+10))
        cx,cy=(x0+x1)/2,(y0+y1)/2;cx=np.clip(cx,bb[0]+side/2,bb[2]-side/2);cy=np.clip(cy,bb[1]+side/2,bb[3]-side/2)
        rgb=ROOT/'data/sites/chips'/sid/'rgb.png';chm=ROOT/'data/sites/chips'/sid/'chm.tif'
        evidence[str(rgb.resolve())]=sha256(rgb);evidence[str(chm.resolve())]=sha256(chm)
        for col in range(2):
            ax=axes[col,row]
            if col==0:ax.imshow(plt.imread(rgb),extent=full)
            else:
                with rasterio.open(chm) as src:im=ax.imshow(src.read(1,masked=True),extent=full,cmap='viridis',vmin=0,vmax=30)
            for poly in [geom] if geom.geom_type=='Polygon' else geom.geoms:
                x,y=poly.exterior.xy;ax.plot(x,y,color=color,lw=2.0,path_effects=[pe.withStroke(linewidth=3.3,foreground='white')])
            ax.set(xlim=(cx-side/2,cx+side/2),ylim=(cy-side/2,cy+side/2));ax.set_xticks([]);ax.set_yticks([])
            ax.set_title(f'({chr(97+col*4+row)}) {sid} · {cls}\n'+('RGB' if col==0 else 'CHM'),fontsize=9)
            scale=5 if side<30 else 10
            bx=cx-side*.42;by=cy-side*.40
            ax.plot([bx,bx+scale],[by,by],color='white',lw=3,path_effects=[pe.withStroke(linewidth=5,foreground='black')])
            ax.text(bx+scale/2,by+side*.045,f'{scale} m',color='white',ha='center',fontsize=8,
                    path_effects=[pe.withStroke(linewidth=2,foreground='black')])
        selections.append({'class':cls,'site_id':sid,'label_id':str(chosen.label_id),'area_m2':float(geom.area),'crop_side_m':float(side),'selection':'median-area example among polygons >0.5 m from site edge when available'})
    fig.colorbar(im,ax=axes[1,:],fraction=.03,pad=.02,label='CHM (m)',ticks=[0,10,20,30])
    save(fig,'figure_4_1_annotation_classes');details['annotation_examples']=selections

    fig,(ax,hm)=plt.subplots(1,2,figsize=(7.4,3.7),gridspec_kw={'width_ratios':[1,1.55]},layout='constrained')
    bottom=np.zeros(3,dtype=int)
    for cls,color in zip(CLASSES,CLASS_COLORS):
        n=np.array([sum(int(r['current']) for r in composition if r['class']==cls and r['split']==split) for split in SPLITS])
        ax.bar(np.arange(3),n,bottom=bottom,color=color,label=cls,width=.65)
        for j,v in enumerate(n):
            if v>=10:ax.text(j,bottom[j]+v/2,str(v),ha='center',va='center',color='white',fontsize=9)
        bottom+=n
    for j,v in enumerate(bottom):ax.text(j,v+4,str(v),ha='center',fontsize=9)
    ax.set_xticks(range(3),['Train','Validation','Test']);ax.set_ylabel('Active polygons');ax.set_ylim(0,260);ax.set_title('(a) All annotation classes')
    ax.legend(loc='upper left',bbox_to_anchor=(-.05,-.13),fontsize=8,ncol=2,frameon=False)
    values=np.array([[next(int(r['current']) for r in composition if r['class']=='tree' and r['stratum']==s and r['split']==split) for split in SPLITS] for s in STRATA])
    hm.imshow(values,cmap='Blues',vmin=0,vmax=45,aspect='auto')
    for i in range(8):
        for j in range(3):hm.text(j,i,str(values[i,j]),ha='center',va='center',fontsize=10,color='white' if values[i,j]>=25 else 'black')
    hm.set_xticks(range(3),['Train','Validation','Test']);hm.set_yticks(range(8),SHORT);hm.set_title('(b) Individual trees by stratum')
    save(fig,'figure_4_2_reference_composition');details['composition']={'class_totals':refs['class'].value_counts().to_dict(),'tree_matrix':values.tolist(),'strata':STRATA}

    # Original diagram of the executed workflow, not an upstream architecture claim.
    fig,ax=plt.subplots(figsize=(6.9,7.3));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def node(x,y,w,h,text,color='#EFF4F7',size=9.5):
        ax.add_patch(Rectangle((x-w/2,y-h/2),w,h,facecolor=color,edgecolor='#63717A',lw=1))
        ax.text(x,y,text,ha='center',va='center',fontsize=size)
    def arrow(start,end):ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=11,color='#46535B',lw=1.1))
    node(.5,.935,.94,.095,'Verified inputs\n39 LAZ tiles · supplied CHM / imagery · frozen database')
    for x in [.17,.5,.83]:arrow((.5,.887),(x,.82))
    node(.17,.754,.29,.13,'Classical reconstruction\nFrozen component checks\nCurrent CHM detector\n3 height trials: training only')
    node(.5,.754,.29,.13,'Pretrained SAT\nFixed source and weights\nNested retention levels\n100 / 75 / 50 / 25 / 12.5%')
    node(.83,.754,.29,.13,'Weak SAT pilot\nTraining polygons + ALTO\nScreened point candidates\n80 fixed training updates')
    arrow((.17,.689),(.17,.626));arrow((.5,.689),(.5,.626));arrow((.83,.689),(.83,.626))
    node(.17,.57,.29,.112,'Classical polygons\nTraining-selected 3.0 m\nFixed detector / masks')
    node(.665,.57,.62,.112,'SAT coordinate reconciliation and audit\nExact-XYZ tile merge · 10 m site context\nInstance convex hull + 1.0 m buffer')
    arrow((.17,.514),(.5,.456));arrow((.665,.514),(.5,.456))
    node(.5,.395,.94,.12,'Common site scoring · EPSG:2193\n147 train / 141 validation / 170 test individual trees\n2 m centroid match · crown IoU · ambiguity exclusions')
    arrow((.5,.335),(.5,.274))
    node(.5,.218,.94,.112,'Paired whole-site bootstrap · 2,000 draws · seed 42\nStratified recall · density response · error examples · timings')
    workflow_note = ('Training selects settings; validation reports a separate split.\n'
                     'Earlier test exposure and reference revision are disclosed.\n')
    workflow_note += ('All fixed predictions, site comparisons and timing reports completed.' if a.weak_metrics else
                     '(Weak numerical results and completion receipt to update.)'
                      if a.completion_draft else
                      'Weak held-out inference is running; its result is pending.')
    ax.text(.5,.095,workflow_note,ha='center',va='center',fontsize=9.5)
    fig.tight_layout(pad=.3);save(fig,'figure_5_1_benchmark_workflow')

    # Existing screened real point candidates; zero is a legitimate instance ID.
    records=[json.loads(p.read_text()) for p in sorted(a.weak_root.glob('*/screening_record.json'))]
    rec=max(records,key=lambda r:(r['retained_alto_ids'],r['site_id']));sid=rec['site_id']
    ply_path=a.weak_root/sid/(sid+'_screened_weak.ply');record_path=a.weak_root/sid/'screening_record.json'
    assert sha256(ply_path)==rec['screened_ply_sha256']
    reader=ROOT.parent/'SegmentAnyTree/torch_points3d/models/panoptic/ply.py'
    spec=importlib.util.spec_from_file_location('ply_figures',reader);ply=importlib.util.module_from_spec(spec);spec.loader.exec_module(ply)
    raw=ply.read_ply(str(ply_path));xyz=np.column_stack([raw[k] for k in ['X','Y','Z']]);sem=raw['semantic_seg'];ids=raw['treeID']
    rng=np.random.default_rng(42);keep=[]
    for value,cap in [(0,7000),(1,7000),(2,10000)]:
        ix=np.flatnonzero(sem==value);keep.extend(ix if len(ix)<=cap else rng.choice(ix,cap,replace=False))
    keep=np.sort(np.array(keep));points=xyz[keep];ss=sem[keep];ii=ids[keep]
    colors=np.tile(np.array([.25,.25,.25, .22]),(len(points),1));colors[ss==1]=[.70,.70,.70,.25]
    tree_ids=sorted(set(ii[ss==2]));palette=list(plt.get_cmap('tab20').colors)+list(plt.get_cmap('Dark2').colors)
    assert len(tree_ids)<=len(palette)
    for j,ident in enumerate(tree_ids):colors[(ss==2)&(ii==ident)]=[*palette[j],1]
    fig=plt.figure(figsize=(7.2,3.9),layout='constrained');a2=fig.add_subplot(121);a3=fig.add_subplot(122,projection='3d')
    order=np.argsort(ss==2,kind='stable');points,colors=points[order],colors[order]
    a2.scatter(points[:,0],points[:,1],s=1.5,c=colors,rasterized=True);a2.set_aspect('equal');a2.set(xlabel='Local X (m)',ylabel='Local Y (m)',title=f'(a) {sid}: top view')
    a3.scatter(points[:,0],points[:,1],points[:,2],s=1.7,c=colors,depthshade=False,rasterized=True)
    a3.set(xlabel='Local X (m)',ylabel='Local Y (m)',title='(b) Original LiDAR returns');a3.text2D(.87,.81,'Local Z (m)',transform=a3.transAxes,fontsize=8,rotation=90);a3.view_init(elev=23,azim=-65);a3.set_box_aspect((1,1,.6));a3.tick_params(labelsize=8)
    fig.legend(handles=[Line2D([],[],marker='o',ls='',color='#555555',label='Unknown target'),Line2D([],[],marker='o',ls='',color='#BBBBBB',label='Known background'),Line2D([],[],marker='o',ls='',color='#0072B2',label='Candidate IDs: distinct colours')],loc='outside lower center',ncol=3,fontsize=8)
    save(fig,'figure_5_2_weak_point_candidates')
    evidence.update({str(ply_path.resolve()):sha256(ply_path),str(record_path.resolve()):sha256(record_path),str(reader.resolve()):sha256(reader)})
    details['point_example']={'site':sid,'selection':'largest retained ALTO-ID count among training sites','coordinate_origin_xyz_m':rec['coordinate_origin_xyz_m'],'original_points':len(xyz),'displayed_points':len(points),'display_sampling_seed':42,'original_ids':rec['retained_original_ids'],'alto_ids':rec['retained_alto_ids'],'semantic_schema':{'0':'unknown','1':'background','2':'tree candidate'},'local_z':'original acquisition elevation minus stored Z origin; not height above ground','colors':'distinct candidate-ID colours from tab20 plus Dark2','point_ground_truth_certified':False}

    fig,axes=plt.subplots(2,2,figsize=(6.8,5.7),layout='constrained')
    reference=Polygon([(1.3,3),(2,2),(3.5,1.6),(4.6,2.6),(4.4,4.1),(3.1,4.7),(1.7,4.3)])
    def outline(ax,g,color,ls='-',fill=None,hatch=None):
        ax.add_patch(PatchPolygon(np.array(g.exterior.coords),closed=True,fill=fill is not None,facecolor=fill or 'none',edgecolor=color,lw=1.6,ls=ls,hatch=hatch))
    for index,ax in enumerate(axes.flat):
        ax.set(xlim=(0,10),ylim=(0,7));ax.set_aspect('equal');ax.set_xticks([0,5,10]);ax.set_yticks([0,5]);ax.tick_params(labelsize=8);ax.set_xlabel('Illustrative distance (m)',fontsize=8)
        if index<2:
            pred=translate(reference,xoff=1 if index==0 else 3.3,yoff=.3)
            outline(ax,reference,'#222222','--');outline(ax,pred,'#009E73' if index==0 else '#CC3311')
            c,p=reference.centroid,pred.centroid;distance=c.distance(p)
            ax.plot([c.x,p.x],[c.y,p.y],color='#555555',lw=1);ax.scatter([c.x,p.x],[c.y,p.y],s=18,color=['black','#009E73' if index==0 else '#CC3311'])
            ax.text(5,6.0,f'Centroid distance {distance:.2f} m',ha='center',fontsize=9)
            ax.set_title('(a) Match: TP = 1' if index==0 else '(b) Outside 2 m: FP = 1, FN = 1',fontsize=10)
        else:
            ambiguity=box(1,1,8.5,5.3);pred=translate(reference,xoff=1.0)
            color='#AA4499' if index==2 else '#666666'
            outline(ax,ambiguity,color,'--',fill='#F5EEF5' if index==2 else '#F0F0F0',hatch='///' if index==2 else 'xx')
            outline(ax,pred,'#888888' if index==2 else '#CC3311');ax.scatter([pred.centroid.x],[pred.centroid.y],s=22,color='#888888' if index==2 else '#CC3311')
            ax.text(5,6,'No eligible single-tree match',ha='center',fontsize=9)
            ax.set_title('(c) Group / uncertain: ignored = 1' if index==2 else '(d) not_tree: FP = 1',fontsize=10)
    save(fig,'figure_5_3_matching_and_ambiguity');details['scoring_diagram']={'synthetic_illustration':True,'threshold_m':2,'minimum_detection_iou':0,'sequence':'single matching first, then unmatched centroid exclusions','reference_boundary':'black dashed','prediction_boundary':'green TP / red FP / grey ignored'}

    # Fixed test results and existing CIs, no weak-model result guessed.
    rows=[r for r in metrics if r['split']=='test'];classical=next(r for r in rows if r['model'].startswith('Classical'))
    sat=sorted([r for r in rows if r['model']=='SAT'],key=lambda r:-float(r['retained_fraction']))
    ordered=[classical]+sat;labels=['Classical 3 m']+[f'SAT {float(r["retained_fraction"])*100:g}%' for r in sat]
    if a.weak_metrics:
        wm=list(csv.DictReader(a.weak_metrics.open()))
        ordered.append(next(r for r in wm if r['split']=='test' and r['model']=='SegmentAnyTree weak pilot'));labels.append('SAT weak pilot')
        evidence[str(a.weak_metrics.resolve())]=sha256(a.weak_metrics)
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.0),layout='constrained')
    for ax,(metric,title,target) in zip(axes.flat,[('precision','(a) Detection precision',.85),('recall','(b) Detection recall',.85),('f1','(c) Detection F1',None),('median_matched_iou','(d) Median matched crown IoU',.60)]):
        for i,r in enumerate(ordered):
            value=float(r[metric]);lo=float(r[metric+'_lower']);hi=float(r[metric+'_upper'])
            ax.errorbar(value,i,xerr=np.array([[value-lo],[hi-value]]),fmt='s' if i==0 else 'o',color='#444444' if i==0 else '#0072B2',capsize=3,markersize=4,lw=1.2)
        if target is not None:ax.axvline(target,color='#AA4499',ls='--',lw=1)
        if metric=='precision':ax.axvline(.95,color='#888888',ls=':',lw=1)
        ax.set_yticks(range(len(ordered)),labels,fontsize=8.5);ax.invert_yaxis();ax.set_xlim(-.02,1.02);ax.set_title(title,fontsize=10);ax.grid(axis='x',alpha=.2)
    fig.legend(handles=[Line2D([],[],color='#AA4499',ls='--',label='Declared metric target'),Line2D([],[],color='#888888',ls=':',label='Additional precision target: 0.95')],loc='outside lower center',ncol=2,fontsize=8)
    save(fig,'figure_6_1_model_intervals');details['result_intervals']={'split':'test','reference_trees':170,'bootstrap':'2000 paired whole-site draws, seed 42','models':labels,'weak_metrics_included':bool(a.weak_metrics),'prior_test_exposure_disclosed':True}

    write_json(a.output/'receipt.json',{'run_id':datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_dissertation_figures','presentation_only':True,'completion_draft':a.completion_draft,'source_hashes':evidence,'script_sha256':sha256(Path(__file__)),'details':details,'outputs':{p.name:sha256(p) for p in files}})
    print(json.dumps({'figures':len(files)//2,'output':str(a.output),'annotation_examples':selections,'point_example':sid},indent=2))


if __name__=='__main__':main()
