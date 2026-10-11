#!/usr/bin/env python3
"""Describe fixed crown errors by observed context and export dissertation figures."""
import argparse,csv,json
from datetime import datetime,timezone
from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import box
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmark.common import ROOT,sha256,site_definitions,write_json


def save_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('reference','comparison','sat','classical','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--weak-comparison',type=Path)
    p.add_argument('--weak',type=Path)
    a=p.parse_args()
    if a.output.exists():p.error('fresh output required')
    d=json.loads(a.comparison.read_text())
    for name,path in [('reference',a.reference),('sat',a.sat),('classical',a.classical)]:
        if sha256(path)!=d['sources'][name]:raise ValueError('source mismatch: '+name)
    models=[('Classical 3 m','classical_by_site'),('SAT pretrained','pretrained_by_site')]
    if bool(a.weak)!=bool(a.weak_comparison):p.error('both weak inputs required')
    if a.weak:
        weak=json.loads(a.weak_comparison.read_text())
        if weak['reference_sha256']!=sha256(a.reference) or weak['prediction_sha256']['weak']!=sha256(a.weak):
            raise ValueError('weak prediction/reference hash mismatch')
        d['weak_by_site']=weak['weak_by_site'];models.append(('SAT weak pilot','weak_by_site'))
    a.output.mkdir(parents=True)
    refs=gpd.read_file(a.reference,layer='all_objects');defs=site_definitions()
    density_path=ROOT/'docs/progress_2026-09-18/numerical_reproduction/site_point_density.csv'
    densities={r['site_id']:float(r['density_all_returns']) for r in csv.DictReader(density_path.open())}
    plan_path=ROOT/'docs/project_status_2026-09-18/tile_coverage_preflight/coverage_plan.json'
    plan=json.loads(plan_path.read_text());tiles={r['tile']:box(*r['bbox_epsg2193']) for r in plan['tiles']}
    site_tiles={r['site_id']:r['required_tiles'] for r in plan['sites']}
    rows=[];context={};chips={}
    for sid in d['site_ids']:
        region=box(*defs[sid]['bbox_epsg2193'])
        chm_path=ROOT/'data/sites/chips'/sid/'chm.tif';chips[sid]=sha256(chm_path)
        with rasterio.open(chm_path) as src:
            grid=src.read(1,masked=True);valid=~np.ma.getmaskarray(grid)&np.isfinite(grid.data)
            elevated=float(np.sum(valid&(grid.data>=3))/np.sum(valid))
            context[sid]={'density':densities[sid],'elevated_fraction':elevated}
            for r in refs[(refs.site_id==sid)&(refs['class']=='tree')].itertuples():
                geom=r.geometry.intersection(region)
                mask=geometry_mask([geom.__geo_interface__],grid.shape,src.transform,invert=True,all_touched=False)&valid
                vals=grid.data[mask];height=float(np.quantile(vals,.95)) if len(vals) else None
                hb='unavailable' if height is None else '<5 m' if height<5 else '5–<10 m' if height<10 else '10–<20 m' if height<20 else '≥20 m'
                de='<3 returns/m²' if densities[sid]<3 else '3–<6 returns/m²' if densities[sid]<6 else '≥6 returns/m²'
                cb='<25% elevated' if elevated<.25 else '25–<50% elevated' if elevated<.5 else '≥50% elevated'
                se=geom.distance(region.boundary)<5
                te=min(geom.centroid.distance(tiles[t].boundary) for t in site_tiles[sid])<10
                for model,key in models:
                    score=d[key][sid];match=next((x for x in score['matches'] if x['reference_id']==str(r.label_id)),None)
                    rows.append({'model':model,'site_id':sid,'reference_id':str(r.label_id),'stratum':defs[sid]['stratum'],
                        'crown_area_m2':geom.area,'chm_p95_m':height,'height_band':hb,'site_unique_return_density':densities[sid],
                        'density_band':de,'site_elevated_fraction':elevated,'elevated_band':cb,
                        'site_edge':'<5 m from site boundary' if se else 'site interior',
                        'tile_edge':'<10 m from source tile boundary' if te else 'tile interior',
                        'matched':int(match is not None),'matched_iou':match['iou'] if match else None})
    save_csv(a.output/'reference_error_context.csv',rows)
    grouped=[]
    for field in ('height_band','density_band','elevated_band','site_edge','tile_edge'):
        for model,_ in models:
            for value in dict.fromkeys(r[field] for r in rows):
                part=[r for r in rows if r['model']==model and r[field]==value];n=len(part);tp=sum(r['matched'] for r in part)
                grouped.append({'model':model,'variable':field,'band':value,'reference_trees':n,'matched':tp,'missed':n-tp,'recall':tp/n if n else None,'distinct_sites':len(set(r['site_id'] for r in part))})
    save_csv(a.output/'context_recall.csv',grouped)
    # All false positives are counted; non-tree regions are selective context only.
    fps=[]
    predictions={'Classical 3 m':gpd.read_file(a.classical,layer='predictions'),'SAT pretrained':gpd.read_file(a.sat,layer='predictions')}
    if a.weak:predictions['SAT weak pilot']=gpd.read_file(a.weak,layer='predictions')
    for model,key in models:
        frame=predictions[model]
        for sid in d['site_ids']:
            score=d[key][sid];neg=refs[(refs.site_id==sid)&(refs['class']=='not_tree')]
            for pid in score['false_positive_ids']:
                geom=frame[(frame.site_id==sid)&(frame.prediction_id.astype(str)==pid)].geometry.iloc[0]
                fps.append({'model':model,'site_id':sid,'prediction_id':pid,'centroid_in_selected_non_tree':int(any(g.covers(geom.centroid) for g in neg.geometry))})
    save_csv(a.output/'false_positive_context.csv',fps)
    # Transparent descriptive example selection: most SAT FP, highest classical
    # recall with >=10 singles, small-canopy stratum, canopy-edge stratum.
    selected=[max(d['site_ids'],key=lambda s:d['pretrained_by_site'][s]['fp']),
              max((s for s in d['site_ids'] if d['classical_by_site'][s]['n_reference']>=10),key=lambda s:d['classical_by_site'][s]['tp']/d['classical_by_site'][s]['n_reference']),
              next(s for s in d['site_ids'] if defs[s]['stratum']=='small_subcanopy'),
              next(s for s in d['site_ids'] if defs[s]['stratum']=='canopy_edge')]
    selected=list(dict.fromkeys(selected))
    for sid in selected:
        bb=defs[sid]['bbox_epsg2193'];extent=(bb[0],bb[2],bb[1],bb[3]);region=box(*bb)
        fig,axes=plt.subplots(2,len(models),figsize=(4.2*len(models),8.2),layout='constrained')
        for col,(model,key) in enumerate(models):
            score=d[key][sid];missed=set(score['unmatched_reference_ids']);fp=set(score['false_positive_ids']);ignored=set(score['ignored_prediction_ids'])
            for row,kind in enumerate(('RGB','CHM')):
                ax=axes[row,col]
                if kind=='RGB':ax.imshow(plt.imread(ROOT/'data/sites/chips'/sid/'rgb.png'),extent=extent)
                else:
                    with rasterio.open(ROOT/'data/sites/chips'/sid/'chm.tif') as src:ax.imshow(src.read(1,masked=True),extent=extent,cmap='viridis',vmin=0,vmax=30)
                def draw(g,color,style,lw):
                    g=g.intersection(region)
                    polys=[g] if g.geom_type=='Polygon' else list(g.geoms) if g.geom_type=='MultiPolygon' else []
                    for poly in polys:
                        xx,yy=poly.exterior.xy;ax.plot(xx,yy,color=color,linestyle=style,lw=lw)
                for rr in refs[refs.site_id==sid].itertuples():
                    cls=getattr(rr,'_2',None)
                    kindref=refs.loc[rr.Index,'class']
                    color=('#ffe535' if str(rr.label_id) in missed else '#00c8ee') if kindref=='tree' else '#bb77ee' if kindref=='tree_group' else '#ff9900' if kindref=='uncertain' else '#dddddd'
                    draw(rr.geometry,color,'--',.8)
                for rr in predictions[model][predictions[model].site_id==sid].itertuples():
                    pid=str(rr.prediction_id);draw(rr.geometry,'#ff3030' if pid in fp else '#bbbbbb' if pid in ignored else '#00e56a','-',.85)
                ax.set(xlim=(bb[0],bb[2]),ylim=(bb[1],bb[3]));ax.set_xticks([]);ax.set_yticks([])
                ax.set_title(f'{model} — {kind}\nTP {score["tp"]}, FP {score["fp"]}, FN {score["fn"]}',fontsize=10)
        fig.suptitle(f'{sid}  {defs[sid]["stratum"].replace("_"," ")}\nSolid: green match, red FP, grey ignored. Dashed: yellow FN, cyan single, purple group.',fontsize=10)
        fig.savefig(a.output/f'error_panel_{sid}.png',dpi=200);plt.close(fig)
    table_path=ROOT/'outputs/confirmed_reference_20261010/results_current_classical/model_metrics.csv'
    models=list(csv.DictReader(table_path.open()));sr=[r for r in models if r['split']=='test' and r['model']=='SAT']
    sr.sort(key=lambda r:float(r['retained_fraction']))
    fig,ax=plt.subplots(figsize=(6.8,3.8),layout='constrained')
    x=[100*float(r['retained_fraction']) for r in sr];y=[float(r['f1']) for r in sr]
    ax.plot(x,y,'o-',color='#165e8c',label='SegmentAnyTree pretrained')
    ax.fill_between(x,[float(r['f1_lower']) for r in sr],[float(r['f1_upper']) for r in sr],alpha=.18,color='#165e8c',label='95% whole-site bootstrap CI')
    ax.set(xlabel='Retained non-noise source records (%)',ylabel='Test detection F1',ylim=(0,.4));ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(a.output/'density_f1.png',dpi=200);plt.close(fig)
    receipt={'run_id':datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_contextual_error_report','status':'COMPLETE',
      'reference_sha256':sha256(a.reference),'comparison_sha256':sha256(a.comparison),'source_code_sha256':sha256(Path(__file__)),
      'density_csv_sha256':sha256(density_path),'coverage_plan_sha256':sha256(plan_path),'chm_sha256':chips,
      'variables':'Crown CHM 95th percentile; all-class unique site-return density; CHM>=3m elevated-surface fraction incl buildings; 5m polygon-to-site-edge; 10m centroid-to-nearest intersecting raw tile boundary',
      'bands_selected_after_test_exposure':True,'grouping_is_descriptive':True,'causal_inference':False,
      'precision_by_reference_band_reported':False,'examples':selected,'reference_model_rows':len(rows),
      'outputs':{f.name:sha256(f) for f in a.output.iterdir() if f.is_file()}}
    if a.weak:receipt.update(weak_comparison_sha256=sha256(a.weak_comparison),weak_predictions_sha256=sha256(a.weak))
    write_json(a.output/'receipt.json',receipt)
    print(json.dumps({'run_id':receipt['run_id'],'rows':len(rows),'bands':grouped,'examples':selected},indent=2))

if __name__=='__main__':main()
