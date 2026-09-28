#!/usr/bin/env python3
"""Create a compact, hash-bound archive of training-development evidence."""
import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source, destination_root, repository_root):
    relative = source.resolve().relative_to(repository_root.resolve())
    target = destination_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True,
                        help='fresh .tar.gz path; a sibling staging directory is created temporarily')
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    if args.output.exists() or args.output.suffixes[-2:] != ['.tar', '.gz']:
        parser.error('--output must be a fresh .tar.gz path')
    stage = args.output.parent / (args.output.name[:-7] + '_staging')
    if stage.exists():
        parser.error('staging directory already exists')
    queue_root = args.work_root / 'training_multitile_v1'
    receipt_path = queue_root / 'queue_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('status') != 'COMPLETED' or not all(row.get('status') == 'AUDITED' for row in receipt['tiles']):
        raise ValueError('completed audited queue receipt required')

    selected = []
    selected += [queue_root/name for name in
                 ('queue_receipt.json', 'queue_process_20260928.status', 'queue_console_20260928.log')]
    for tile_dir in sorted(path for path in queue_root.iterdir() if path.is_dir() and path.name.startswith('pc_')):
        for path in tile_dir.rglob('*'):
            if path.is_file() and path.suffix.lower() in {'.json', '.txt', '.yaml', '.yml'}:
                selected.append(path)
    for directory in (args.work_root/'training_site_merges_v1',
                      args.work_root/'training_conversion_prefreeze249_v1', args.reference_dir):
        selected += [path for path in directory.rglob('*') if path.is_file()]
    for name in ('20_sat_crown_candidates.py', '35_compare_crown_conversion_candidates.py',
                 '40_audit_training_reference_points.py', '43_audit_training_misses.py',
                 '46_merge_training_site_tiles.py', '47_prepare_training_tile.py',
                 '48_run_training_tile_queue.py', '49_summarise_training_conversion_grid.py',
                 '50_archive_training_development.py'):
        selected.append(repository_root/'scripts'/name)

    run_dirs = set()
    for line in (queue_root/'queue_console_20260928.log').read_text(errors='replace').replace('\r', '\n').splitlines():
        if line.startswith('Outputs and logs: '):
            path = Path(line.split(': ', 1)[1])
            if path.is_dir():
                run_dirs.add(path)
    benchmark_root = repository_root/'outputs/benchmarks'
    patterns = ('20260923T18*prepare_training*', '20260923T195506Z_sat_pretrained*',
                '20260928T15*merge*', '20260928T15*crown_conversion*',
                '20260928T155*summarise*', '20260928T155*point_support*',
                '20260928T155*miss_audit*')
    for pattern in patterns:
        run_dirs.update(path for path in benchmark_root.glob(pattern) if path.is_dir())
    for directory in sorted(run_dirs):
        selected += [path for path in directory.rglob('*') if path.is_file()]

    unique = sorted({path.resolve() for path in selected})
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise ValueError(f'missing selected evidence: {missing[:5]}')
    stage.mkdir(parents=True)
    copied = [copy_file(path, stage, repository_root) for path in unique]
    omitted = []
    for row in receipt['tiles']:
        for key in ('raw_prediction', 'reconciled_prediction'):
            path = Path(row[key])
            omitted.append({'tile': row['tile'], 'kind': key, 'path': str(path),
                            'bytes': path.stat().st_size, 'sha256': row[f'{key}_sha256']})
    recorded = {(row['path'], row['sha256']) for row in omitted}
    for merge_path in sorted((args.work_root/'training_site_merges_v1').glob('S*/merge_receipt.json')):
        merge = json.loads(merge_path.read_text())
        for row in merge['tiles']:
            path = Path(row['prediction'])
            key = (str(path), row['prediction_sha256'])
            if key not in recorded:
                omitted.append({'tile': row['tile'], 'kind': 'reconciled_prediction',
                                'path': str(path), 'bytes': path.stat().st_size,
                                'sha256': row['prediction_sha256']})
                recorded.add(key)
    manifest = {
        'status': 'COMPACT TRAINING-DEVELOPMENT EVIDENCE ARCHIVE',
        'repository_root': str(repository_root),
        'included_file_count': len(copied),
        'included': {str(path.relative_to(stage)): sha256(path) for path in copied},
        'large_point_outputs_not_duplicated': omitted,
        'omission_rule': ('Full-tile prediction LAZ files remain on the VM and are bound by the completed queue receipt. '
                          'The compact archive includes their audits, hashes, logs and derived site-context inputs.'),
    }
    manifest_path = stage/'ARCHIVE_MANIFEST.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, 'w:gz') as archive:
        archive.add(stage, arcname=stage.name)
    result = {'archive': str(args.output), 'archive_sha256': sha256(args.output),
              'archive_bytes': args.output.stat().st_size,
              'included_file_count': len(copied) + 1,
              'omitted_large_point_files': len(omitted)}
    shutil.rmtree(stage)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
