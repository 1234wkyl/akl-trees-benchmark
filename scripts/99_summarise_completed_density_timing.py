#!/usr/bin/env python3
"""Summarise recorded successful wrappers for the four completed thinning conditions."""
import argparse,csv,json,hashlib
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sources',type=Path,required=True);p.add_argument('--density-root',type=Path,required=True);p.add_argument('--full-density-cost',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('fresh output required')
 sources=json.loads(a.sources.read_text());full=json.loads(a.full_density_cost.read_text())['pretrained'];area=full['raw_header_envelope_union_km2'];full_tiles={r['tile'] for r in full['tile_timings']};results=[];bindings={}
 for fraction in ['0.75','0.5','0.25','0.125']:
  qp=a.density_root/f'fraction_{fraction}'/'queue_receipt.json';q=json.loads(qp.read_text());bindings[fraction]=sha(qp)
  assert q['status']=='COMPLETED' and len(q['tiles'])==19 and all(r['status']=='AUDITED' for r in q['tiles'])
  assert {r['tile'] for r in q['tiles']}==full_tiles
  timings=[]
  for tile in sorted(full_tiles):
   matches=[r for r in sources['records'] if r['fraction']==fraction and r['tile']==tile and r['timing']['exit_code']==0]
   if len(matches)!=1:raise ValueError(f'Expected one successful fixed wrapper for {fraction}/{tile}, found {len(matches)}')
   timings.append(matches[0])
  seconds=sum(r['timing']['wall_seconds'] for r in timings)
  results.append({'fraction':float(fraction),'tiles':19,'wall_seconds':seconds,'allocated_device_wall_hours':seconds/3600,'raw_header_envelope_union_km2':area,'allocated_device_wall_hours_per_header_envelope_km2':seconds/3600/area,'runs':timings})
 a.output.mkdir(parents=True)
 with (a.output/'density_timing.csv').open('w',newline='') as f:
  rows=[{k:v for k,v in r.items() if k!='runs'} for r in results];w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 receipt={'scope':'Successful prediction-wrapper allocation throughput; not provider billing, CUDA-kernel time or area-matched classical/GPU cost','source_sha256':sha(a.sources),'full_density_cost_sha256':sha(a.full_density_cost),'queue_sha256':bindings,'script_sha256':sha(Path(__file__)),'results':results,'output_sha256':sha(a.output/'density_timing.csv')}
 (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 for row in rows:print(json.dumps(row))
if __name__=='__main__':main()
