#!/usr/bin/env python
"""Read-only intake checks; matching row counts do not establish a frozen identity."""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def inspect(database, manifest, crowns=None):
    """No table mutations, extension loading, or production script execution."""
    if not database.is_file():
        raise ValueError('download the separate frozen database first')
    wal = Path(str(database) + '-wal')
    if wal.exists() and wal.stat().st_size:
        raise ValueError('nonempty WAL exists: obtain a standalone checkpointed frozen copy')
    expected = json.loads(manifest.read_text())
    if database.stat().st_size != expected['database_bytes']:
        raise ValueError('database size differs from supplied frozen manifest')
    digest = sha(database)
    # A supplied, immutable snapshot with an absent/empty WAL needs no sidecar
    # changes. immutable=1 avoids creating/updating the supplied SHM file.
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro&immutable=1', uri=True) as conn:
        conn.execute('PRAGMA query_only=ON')
        checks = [r[0] for r in conn.execute('PRAGMA quick_check')]
        if checks != ['ok']:
            raise ValueError('SQLite quick_check failed: ' + str(checks[:5]))
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        tables = {}
        for name in expected['critical_tables']:
            if name not in names:
                raise ValueError('critical table missing: ' + name)
            columns = [{'name': r[1], 'type': r[2]} for r in conn.execute('PRAGMA table_info(' + quote(name) + ')')]
            count = conn.execute('SELECT COUNT(*) FROM ' + quote(name)).fetchone()[0]
            wanted = expected['tables'][name]['rows']
            if count != wanted:
                raise ValueError(f'{name}: row count {count} differs from {wanted}')
            tables[name] = {'rows': count, 'columns': columns,
                            'manifest_logical_checksum': expected['tables'][name].get('checksum'),
                            'logical_checksum_verified': False}
        crown_columns = [x['name'] for x in tables.get('tree_crown_pilot', {}).get('columns', [])]
    result = {
        'status': 'structural intake checks passed; frozen identity and run provenance not certified',
        'database_file': database.name, 'database_bytes': database.stat().st_size,
        'database_sha256_observed': digest, 'manifest_sha256': sha(manifest),
        'expected_frozen_at_utc': expected.get('frozen_at_utc'), 'sqlite_quick_check': checks,
        'critical_tables': tables, 'read_only': True,
        'sqlite_open_mode': 'mode=ro&immutable=1; query_only=ON',
        'wal_bytes': wal.stat().st_size if wal.exists() else None,
        'logical_checksum_note': 'Original decimal table checksum algorithm is not supplied. Matching counts and a newly recorded file hash do not verify those fingerprints.',
        'possible_crown_geometry_columns': [n for n in crown_columns if any(k in n.lower() for k in ['geom', 'wkt', 'wkb', 'geojson'])],
        'crown_file': None, 'core_metrics_computed': False,
        'unresolved_dependencies': ['authoritative snapshot/table-checksum verification',
                                    '30 July code, parameters and run provenance',
                                    'same-snapshot crown geometry or exact regeneration evidence',
                                    'approved evaluation matching and ignore rules; frozen reference'],
    }
    if crowns:
        import pyogrio
        if not crowns.is_file():
            raise ValueError('specified crown layer does not exist')
        layers = []
        for name, geometry in pyogrio.list_layers(crowns):
            info = pyogrio.read_info(crowns, layer=name)
            layers.append({'layer': name, 'geometry_schema': info['geometry_type'],
                           'properties': dict(zip(info['fields'].tolist(), info['dtypes'].tolist())),
                           'crs': info['crs'],
                           'bounds_in_source_crs': list(info['total_bounds']) if info['total_bounds'] is not None else None})
        result['crown_file'] = {'file': crowns.name, 'bytes': crowns.stat().st_size,
                                'sha256_observed': sha(crowns), 'layers': layers,
                                'correspondence_to_snapshot_verified': False}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--database', required=True, type=Path)
    ap.add_argument('--manifest', type=Path, default=ROOT.parent / 'Xi_Zhang_Benchmark/provenance/frozen_database_manifest.json')
    ap.add_argument('--crowns', type=Path, help='same-snapshot crown polygon file, if available')
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('use a fresh output directory')
    try:
        result = inspect(args.database, args.manifest, args.crowns)
    except (ValueError, sqlite3.DatabaseError) as exc:
        ap.error(str(exc))
    args.output.mkdir(parents=True)
    (args.output / 'intake_report.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output / 'README.md').write_text('''# Frozen baseline intake only

The database was opened read-only. Its observed SHA-256, critical schemas/counts and SQLite structural check are recorded. The supplied manifest contains decimal logical table checksums, whose algorithm is not included here: this tool does not pretend to verify them. Obtain the original snapshot verifier or an authoritative whole-file hash. A matched file size and row counts alone are not proof of a frozen snapshot.

Historical scripts store crown metrics in SQLite and write polygon geometry separately to tree_crowns_pilot.geojson. Inspect the actual downloaded schema; obtain the corresponding frozen crown polygon layer if geometry is absent, or verify exact regeneration inputs/code/parameters. An observed polygon-file hash records identity for this copy, not its historical correspondence. This tool does not rebuild a live baseline, overwrite inputs, freeze labels, compute model metrics, or certify the July 30 run.
''')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
