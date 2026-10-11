#!/usr/bin/env python3
"""Diagnose fresh-vs-retained seed pixel ownership without changing either run."""
import argparse,importlib.util,json,sqlite3,csv
from pathlib import Path
import numpy as np
import rasterio
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import box
from benchmark.common import ROOT,write_json,sha256


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('source','chm','database','recovery-root','reconstruction','output'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():ap.error('fresh output required')
    spec=importlib.util.spec_from_file_location('crown_adapter',ROOT/'scripts/51_attempt_historical_crowns.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    source,_,_=m.source_module(a.source)
    plan=json.loads((a.recovery_root/'training_crowns_historical_order/plan.json').read_text())
    comparisons=list(csv.DictReader((a.reconstruction/'comparison/comparison_all126.csv').open()))
    bad={r['frozen_id'] for r in comparisons if r['all_four_match']!='True'}
    frozen=sqlite3.connect(a.database.resolve().as_uri()+'?mode=ro&immutable=1',uri=True);frozen.row_factory=sqlite3.Row
    fresh=sqlite3.connect((a.reconstruction/'reconstructed_inventory.sqlite').resolve().as_uri()+'?mode=ro&immutable=1',uri=True);fresh.row_factory=sqlite3.Row
    buildings,nonveg,_=m.load_masks(source,plan['osm_query_cells'],a.recovery_root/'current_osm_cache')
    rows=[];site_seeds={}
    with rasterio.open(a.chm) as raster, rasterio.open(a.recovery_root/'current_greenness_complete/greenness.vrt') as gli:
        for index,cell in enumerate(plan['original_chunks']):
            old=m.historical_chunk_seeds(frozen,cell,source,plan['parameters']['original_global_bbox'])
            if not any(s['tree_id'] in bad for s in old):continue
            new=m.historical_chunk_seeds(fresh,cell,source,plan['parameters']['original_global_bbox'])
            image,transform=source.read_chm_window(raster,cell)
            green=source.read_gli_for_chunk(gli,cell,image.shape,transform)
            mask=np.isfinite(image)&(image>=source.CANOPY_MIN_HEIGHT_M)
            exclusions=[g for g in buildings+nonveg if g.intersects(box(*cell).buffer(source.CROWN_HALO_M))]
            mask &= ~source.rasterize_buildings(exclusions,image.shape,transform)
            mask &= ndimage.uniform_filter(np.nan_to_num(green,nan=-1),size=source.CROWN_GLI_WINDOW_PX)>=source.CROWN_GLI_THRESHOLD
            labels,_=ndimage.label(mask);sizes=np.bincount(labels.ravel());keep=sizes>=source.MIN_CROWN_AREA_M2;keep[0]=False;mask=keep[labels]
            rr,cc=np.nonzero(mask);xy=np.column_stack([transform.c+(cc+.5)*transform.a,transform.f+(rr+.5)*transform.e])
            old_xy=np.array([[s['x_2193'],s['y_2193']] for s in old]);new_xy=np.array([[s['x_2193'],s['y_2193']] for s in new])
            kd_old=cKDTree(old_xy);kd_new=cKDTree(new_xy)
            od,oi=kd_old.query(xy,k=2,workers=-1);nd,ni=kd_new.query(xy,k=2,workers=-1)
            # k=2 can order equidistant neighbors differently from the source's
            # k=1 query. Use k=1 ownership and k=2 only for the distance-gap probe.
            od1,oi1=kd_old.query(xy,k=1,workers=-1);nd1,ni1=kd_new.query(xy,k=1,workers=-1)
            od[:,0]=od1;oi[:,0]=oi1;nd[:,0]=nd1;ni[:,0]=ni1
            radii_old=np.array([source.crown_radius_for_tree(s['chm_local_max_2m_m']) for s in old]);radii_new=np.array([source.crown_radius_for_tree(s['chm_local_max_2m_m']) for s in new])
            # Ownership identity: preserve canonical IDs; inferred IDs link only
            # for diagnosis by sub-micrometre coordinates after both runs finish.
            original_index={s['tree_id']:i for i,s in enumerate(old)}
            near,matching=kd_old.query(new_xy,k=1)
            mapped=np.array([original_index.get(s['tree_id'],int(j) if d<=1e-6 else -1) for s,d,j in zip(new,near,matching)])
            old_owner=np.where(od[:,0]<=radii_old[oi[:,0]],oi[:,0],-1)
            new_owner=np.where(nd[:,0]<=radii_new[ni[:,0]],mapped[ni[:,0]],-1)
            bad_indices={i for i,s in enumerate(old) if s['tree_id'] in bad}
            changed=(old_owner!=new_owner)&(np.isin(old_owner,list(bad_indices))|np.isin(new_owner,list(bad_indices)))
            for k in np.flatnonzero(changed):
                rows.append({'x':float(xy[k,0]),'y':float(xy[k,1]),'old_owner':old[old_owner[k]]['tree_id'] if old_owner[k]>=0 else None,
                    'new_owner_linked_frozen_id':old[new_owner[k]]['tree_id'] if new_owner[k]>=0 else None,
                    'new_generated_id':new[ni[k,0]]['tree_id'],
                    'old_nearest_distance_m':float(od[k,0]),'new_nearest_distance_m':float(nd[k,0]),
                    'old_first_second_gap_m':float(od[k,1]-od[k,0]),'new_first_second_gap_m':float(nd[k,1]-nd[k,0]),
                    'near_equidistant_both':bool(od[k,1]-od[k,0]<=1e-7 and nd[k,1]-nd[k,0]<=1e-7)})
            print('Ownership diagnostic chunk',index+1,'changed target pixels',int(changed.sum()),flush=True)
    frozen.close();fresh.close();a.output.mkdir(parents=True)
    with (a.output/'changed_pixel_ownership.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['x','y']);w.writeheader();w.writerows(rows)
    summary={'status':'READ-ONLY SEED OWNERSHIP DIAGNOSIS; NO PARAMETER OR PIXEL CHANGE',
        'changed_target_pixels':len(rows),'near_equidistant_both':sum(r['near_equidistant_both'] for r in rows),
        'mismatching_frozen_ids':sorted(bad),'probe_comparison_sha256':sha256(a.reconstruction/'comparison/comparison_all126.csv'),
        'note':'Distance gaps diagnose tie sensitivity; this does not identify the historical SciPy/PROJ runtime or license a per-object correction.'}
    write_json(a.output/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
