import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def site_definitions():
    return {s['site_id']: s for s in json.loads(
        (ROOT / 'data/sites/reference_sites.json').read_text())['sites']}


def validate_freeze(path, reference, policy):
    """No script grants freeze approval; read an explicit, hash-bound record."""
    if path is None:
        raise ValueError('a supervisor-approved reference freeze record is required')
    record = json.loads(Path(path).read_text())
    if record.get('status') != 'frozen' or not record.get('supervisor_confirmation_record'):
        raise ValueError('reference freeze is not confirmed')
    checks = {'reference_sha256': reference, 'evaluation_policy_sha256': policy,
              'site_definitions_sha256': sha256(ROOT / 'data/sites/reference_sites.json')}
    if any(record.get(k) != v for k, v in checks.items()):
        raise ValueError('freeze record does not match reference, policy and site hashes')
    return record
