#!/usr/bin/env python
"""Build an isolated blind annotation package from existing chips only."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sites', nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if len(args.sites) != len(set(args.sites)):
        ap.error('duplicate site IDs')
    sites = {s['site_id']: s for s in json.loads((ROOT/'data/sites/reference_sites.json').read_text())['sites']}
    if any(s not in sites for s in args.sites):
        ap.error('unknown site ID')
    out = args.output.resolve()
    if out.exists() or out.with_suffix('.zip').exists():
        ap.error('choose a fresh output folder; existing packages are preserved')
    index = []
    for sid in args.sites:
        source = ROOT/'data/sites/chips'/sid
        dest = out/'data/sites/chips'/sid
        dest.mkdir(parents=True)
        for name in ['rgb.png', 'chm.png', 'chm.tif']:
            shutil.copy2(source/name, dest/name)
        meta = json.loads((source/'meta.json').read_text())
        for key in ['existing_detection_count', 'point_count']:
            meta.pop(key, None)
        (dest/'meta.json').write_text(json.dumps(meta, indent=2)+'\n')
        index.append({k: sites[sid][k] for k in ['site_id', 'stratum', 'split']})
    (out/'data/sites/chips/index.json').write_text(json.dumps(index, indent=2)+'\n')
    html = (ROOT/'labeller/index.html').read_text()
    key = 'akl_crown_labels_TW_blind_'+'_'.join(args.sites)+'_v1'
    html = html.replace('"akl_crown_labels_v1"', json.dumps(key))
    html = html.replace('  try { dets = await (await fetch(`${base}/existing_detections.json`)).json(); }\n  catch { dets = []; }', '  dets = [];')
    html = html.replace('<input type="checkbox" id="showdet"> Show existing model detections', '<input type="checkbox" id="showdet" disabled> Model detections unavailable for independent annotation')
    html = html.replace('"akl_labeller_name"', json.dumps(key+'_who'))
    html = html.replace('localStorage.getItem('+json.dumps(key+'_who')+') || ""', 'localStorage.getItem('+json.dumps(key+'_who')+') || "TW"')
    assert 'fetch(`${base}/existing_detections.json`)' not in html
    (out/'labeller').mkdir()
    (out/'labeller/index.html').write_text(html)
    ids = ' and '.join(args.sites)
    protocol = (ROOT/'docs/review_2026-09-17/independent_annotation_TW/README.md').read_text()
    rules = protocol.split('## Annotation rules', 1)[1]
    readme = f'''# Independent crown annotation for {ids}

Please independently annotate {ids} using the supplied RGB/CHM chips and rules below. The two sites are typical benchmark examples. This package contains no XZ labels, model detections, model-derived object counts or synthetic labels. Do not inspect the benchmark reference labels or site-count review before saving your independent export.

## Start

From this folder run `python3 -m http.server 8901 --bind 127.0.0.1`, then open http://127.0.0.1:8901/labeller/index.html. Python 3 is the only server requirement. Keep TW as the labeller, annotate both sites, mark both complete and export JSON. Return the original independent JSON before comparing or adjudicating labels. Browser storage is isolated from XZ and older blind packages.

## Annotation rules{rules}
'''
    (out/'README.md').write_text(readme.rstrip()+'\n')
    launcher = out/'START_LABELLER.command'
    launcher.write_text('#!/bin/bash\ncd "$(dirname "$0")"\npython3 -m http.server 8901 --bind 127.0.0.1\n')
    launcher.chmod(0o755)
    hashes = []
    for path in sorted(out.rglob('*')):
        if path.is_file():
            hashes.append(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+path.relative_to(out).as_posix())
    (out/'SHA256SUMS.txt').write_text('\n'.join(hashes)+'\n')
    archive = out.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                z.write(path, Path(out.name)/path.relative_to(out))
    print(archive)


if __name__ == '__main__':
    main()
