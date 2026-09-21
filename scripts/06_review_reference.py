#!/usr/bin/env python
"""Read-only audit of the canonical reference and local benchmark handoff."""
from pathlib import Path
from collections import Counter, defaultdict
import csv, json, hashlib, math
import numpy as np
import laspy, rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
from shapely.geometry import Polygon, box
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'Xi_Zhang_Benchmark'
OUT = ROOT / 'docs/review_2026-09-17'
OUT.mkdir(parents=True, exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def writecsv(name, rows):
    if not rows: return
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

source=ROOT/'data/labels/crown_labels_XZ_2026-08-13.json'
labels=json.loads(source.read_text())
sites={s['site_id']:s for s in json.loads((ROOT/'data/sites/reference_sites.json').read_text())['sites']}
rows=[]; geoms={}; flags=[]; repairs=[]
for sid,rec in labels['sites'].items():
    if sid not in sites:
        if rec['labels']: raise ValueError('Labels have unknown site '+sid)
        continue
    for lab in rec['labels']:
        original=Polygon(lab['world']); p=original if original.is_valid else original.buffer(0);geoms[lab['id']]=p
        if not original.is_valid:
            repairs.append(dict(site_id=sid,label_id=lab['id'],cls=lab['cls'],original_area=original.area,repaired_area=p.area,change_m2=p.area-original.area,parts=1 if p.geom_type=='Polygon' else len(p.geoms)))
        r=dict(site_id=sid,label_id=lab['id'],cls=lab['cls'],stratum=sites[sid]['stratum'],split=sites[sid]['split'],area_m2=round(p.area,4),valid=original.is_valid,complete=rec['complete'],outside_area_m2=round(p.difference(box(*sites[sid]['bbox_epsg2193'])).area,4))
        rows.append(r)
        reason=[]
        if not original.is_valid:reason.append('invalid raw geometry; exporter uses buffer(0)')
        if p.area>900:reason.append('area above 900 m2')
        if p.area<1:reason.append('area below 1 m2')
        if lab['cls']=='uncertain':reason.append('uncertain label')
        if r['outside_area_m2']>0.01:reason.append('extends outside site')
        if reason:flags.append(dict(r,reason='; '.join(reason)))
writecsv('all_objects.csv',rows);writecsv('flagged_objects.csv',flags)
(OUT/'geometry_repairs.json').write_text(json.dumps(repairs,indent=2))
counts=Counter(r['cls'] for r in rows)
matrix=[]
for st in sorted({s['stratum'] for s in sites.values()}):
    for sp in ['train','val','test']:
        rr=[r for r in rows if r['stratum']==st and r['split']==sp]
        c=Counter(r['cls'] for r in rr)
        matrix.append(dict(stratum=st,split=sp,sites=','.join(sid for sid,s in sites.items() if s['stratum']==st and s['split']==sp),tree=c['tree'],tree_group=c['tree_group'],not_tree=c['not_tree'],uncertain=c['uncertain'],total=len(rr)))
writecsv('counts_by_split_stratum.csv',matrix)
overlaps=[]
for sid in sites:
    rr=[r for r in rows if r['site_id']==sid]
    for i,a in enumerate(rr):
        for b in rr[i+1:]:
            p,q=geoms[a['label_id']],geoms[b['label_id']]
            inter=p.intersection(q).area
            if inter>0.35*min(p.area,q.area):overlaps.append(dict(site_id=sid,a=a['label_id'],b=b['label_id'],intersection_m2=round(inter,4)))
writecsv('heavy_overlaps.csv',overlaps)
for n,r in enumerate(flags):
    sid=r['site_id']; meta=json.loads((ROOT/f'data/sites/chips/{sid}/meta.json').read_text())
    im=Image.new('RGB',(800,430),'white');d=ImageDraw.Draw(im)
    for j,layer in enumerate(['rgb','chm']):
        chip=Image.open(ROOT/f'data/sites/chips/{sid}/{layer}.png').convert('RGB'); im.paste(chip,(j*400,30))
        p=geoms[r['label_id']]
        for part in ([p] if p.geom_type=='Polygon' else p.geoms):
            pts=[(j*400+(x-meta['bbox_epsg2193'][0])/meta['resolution_m']-.5,30+(meta['bbox_epsg2193'][3]-y)/meta['resolution_m']-.5) for x,y in part.exterior.coords]
            d.line(pts,fill='red',width=3)
    d.text((5,5),f"{r['label_id']} {r['cls']} {r['area_m2']:.1f} m2",fill='black')
    im.save(OUT/f"flag_{n+1:02d}_{sid}.png")
for page,start in enumerate(range(0,len(flags),6)):
    sheet=Image.new('RGB',(1200,650),'white')
    for j,r in enumerate(flags[start:start+6]):
        im=Image.open(OUT/f"flag_{start+j+1:02d}_{r['site_id']}.png");im.thumbnail((600,216));sheet.paste(im,((j%2)*600,(j//2)*216))
    sheet.save(OUT/f'flag_sheet_{page+1}.png')

# Verify each supplied content checksum independently of the handoff validation.
manifest=json.loads((DATA/'manifest.json').read_text());checks=[]
for f in manifest['files']:
    p=DATA/f['path'];checks.append(dict(path=f['path'],exists=p.exists(),size_ok=p.stat().st_size==f['bytes'] if p.exists() else False,sha256_ok=sha(p)==f['sha256'] if p.exists() else False))
writecsv('asset_checksums.csv',checks)
required=(ROOT/'data/sites/required_tiles.txt').read_text().splitlines()
required=[x.strip() for x in required if x.strip() and not x.startswith('#')]
assert set(required)=={p.name for p in (DATA/'laz').glob('*.laz')}

# Density is all returns per horizontal site area, half-open XY bounds;
# count all classifications/returns and exclude duplicate XYZ across tile overlaps.
point_sets={sid:set() for sid in sites}; classcounts={sid:Counter() for sid in sites}; tileinfo=[]
for p in sorted((DATA/'laz').glob('*.laz')):
    with laspy.open(p) as fh:
        h=fh.header; relevant=[sid for sid,s in sites.items() if box(*s['bbox_epsg2193']).intersects(box(h.mins[0],h.mins[1],h.maxs[0],h.maxs[1]))]
        total=0;cc=Counter()
        for pts in fh.chunk_iterator(500000):
            total+=len(pts);cls=np.asarray(pts.classification);u,c=np.unique(cls,return_counts=True);cc.update(dict(zip(map(int,u),map(int,c))))
            x,y,z=np.asarray(pts.x),np.asarray(pts.y),np.asarray(pts.z)
            for sid in relevant:
                x0,y0,x1,y1=sites[sid]['bbox_epsg2193'];m=(x>=x0)&(x<x1)&(y>=y0)&(y<y1)
                for a,b,c,k in zip(x[m],y[m],z[m],cls[m]):
                    key=(round(float(a),4),round(float(b),4),round(float(c),4))
                    if key not in point_sets[sid]:point_sets[sid].add(key);classcounts[sid][int(k)]+=1
        assert total==h.point_count
        crs=h.parse_crs()
        tileinfo.append(dict(file=p.name,points=total,crs=str(crs),point_format=h.point_format.id,dimensions=','.join(h.point_format.dimension_names),classes=dict(cc),status='full_decompression_passed'))
    print('Read',p.name,total,flush=True)
(OUT/'laz_validation.json').write_text(json.dumps(tileinfo,indent=2))
density=[]
for sid,s in sites.items():
    x0,y0,x1,y1=s['bbox_epsg2193'];area=(x1-x0)*(y1-y0)
    density.append(dict(site_id=sid,split=s['split'],stratum=s['stratum'],unique_xyz_returns=len(point_sets[sid]),area_m2=area,density_all_returns=len(point_sets[sid])/area,vegetation_returns=sum(classcounts[sid][k] for k in [3,4,5]),first_return_density='not measured',sampling_proxy_density=s['point_density']))
writecsv('site_point_density.csv',density)
chm=[]
with rasterio.open(DATA/'chm/reference_sites.vrt') as ds:
    for sid,s in sites.items():
        win=from_bounds(*s['bbox_epsg2193'],ds.transform)
        a=ds.read(1,window=win,masked=True)
        with rasterio.open(ROOT/f'data/sites/chips/{sid}/chm.tif') as chip:
            b=chip.read(1);dest=np.zeros_like(b)
            reproject(a.filled(ds.nodata),dest,src_transform=ds.window_transform(win),src_crs=ds.crs,dst_transform=chip.transform,dst_crs=chip.crs,resampling=Resampling.bilinear)
        equal=np.array_equal(dest,b)
        chm.append(dict(site_id=sid,native_valid_pixels=int(a.count()),native_total_pixels=a.size,matches_starter_chip=bool(equal),max_abs_difference_m=float(np.abs(dest-b).max())))
writecsv('chm_site_checks.csv',chm)
dsvals=np.array([r['density_all_returns'] for r in density]);stats=dict(zip(['min','q1','median','q3','max'],map(float,np.quantile(dsvals,[0,.25,.5,.75,1]))))
summary=dict(counts=dict(counts),tree_by_split={sp:sum(r['cls']=='tree' and r['split']==sp for r in rows) for sp in ['train','val','test']},flagged_count=len(flags),large_count=sum(r['area_m2']>900 for r in rows),invalid_count=sum(not r['valid'] for r in rows),outside_count=sum(r['outside_area_m2']>0.01 for r in rows),heavy_overlaps=len(overlaps),assets_checked=len(checks),asset_failures=sum(not(r['exists'] and r['size_ok'] and r['sha256_ok']) for r in checks),laz_tiles=len(tileinfo),laz_total_points=sum(r['points'] for r in tileinfo),chm_checks=chm,density_stats=stats,canonical_label_sha256=sha(source),reference_gpkg_sha256=sha(ROOT/'data/exports/reference_crowns.gpkg'))
import geopandas as gpd
exported=gpd.read_file(ROOT/'data/exports/reference_crowns.gpkg',layer='reference_crowns')
summary.update(export_invalid_count=int((~exported.geometry.is_valid).sum()),max_repair_area_change_m2=max((abs(r['change_m2']) for r in repairs),default=0),repair_multipart_count=sum(r['parts']>1 for r in repairs))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
lines=['# Reference and asset review 17 September 2026','','The canonical annotation JSON is preserved. The reference set is pending freeze and supervisor review.','','## Counts','', '| Stratum | Split | Site | Tree | Group | Not tree | Uncertain | Total |','|---|---|---|--:|--:|--:|--:|--:|']
for r in matrix:lines.append('| '+ ' | '.join(str(r[k]) for k in ['stratum','split','sites','tree','tree_group','not_tree','uncertain','total'])+' |')
lines+=['','## Quality checks','',f"{summary['invalid_count']} invalid geometries; {summary['outside_count']} objects extend beyond site bounds; {len(overlaps)} heavy overlap pairs. {summary['large_count']} polygons exceed 900 m2. All large polygons must be reviewed using the accompanying RGB/CHM overlays before freeze.",'','Four uncertain polygons remain in the original export. Exclude them from evaluable reference objects pending adjudication; their regions require an explicit ignore policy for predictions, rather than turning them into background negatives. Tree groups count as annotation effort and are reported separately from individual-tree metrics.','','## Asset verification','',f"Verified {len(checks)} manifest files by size and SHA-256; failures: {summary['asset_failures']}. Fully decompressed {len(tileinfo)} LAZ files ({summary['laz_total_points']:,} returns). CHM coverage and agreement with existing Starter Kit CHM chips are in chm_site_checks.csv.",'','## Density','',f"Unique XYZ returns per 10,000 m2 site (all classes, all return numbers): median {stats['median']:.4f}; IQR {stats['q1']:.4f}–{stats['q3']:.4f}; range {stats['min']:.4f}–{stats['max']:.4f} returns/m2. Bounds are half-open; XYZ duplicates across tiles are excluded at 0.0001 m precision. This is observed return density, not pulse density or nominal flight density.",'','## Frozen baseline limitation','','The frozen 30 July database itself is absent locally. Cached detections and historical scripts are supplied, but source_reference/PROVENANCE.json does not verify exact script correspondence with the frozen snapshot. Reproducing a frozen baseline requires snapshot detections/crowns or the database and verified parameter provenance. Do not substitute a rebuilt live baseline.']
recommendation=OUT/'freeze_recommendation.md'
if recommendation.exists():
    lines+=['',recommendation.read_text().replace('# Additional findings','## Additional findings',1)]
(OUT/'REVIEW.md').write_text('\n'.join(lines).rstrip()+'\n')
print(json.dumps(summary,indent=2))
