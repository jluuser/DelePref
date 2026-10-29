"""Normalize the small frozen reports required by the training preflight."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


NAMES = (
    'base_validation_report.json', 'b20_validation_report.json',
    'b20_supplement_audit.json', 'condition_catalog_summary.json',
    'confirmation_manifest_lock.json', 'proposal_validation_summary.json',
)


def normalize(value):
    if isinstance(value, dict):
        return {normalize(k): normalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if isinstance(value, str) and value.startswith(('/public/home/', '/public/nfs-data/')):
        return 'research-source/' + PurePosixPath(value).name
    return value


def build(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    provenance = []
    for name in NAMES:
        raw = (source / name).read_bytes()
        data = normalize(json.loads(raw.decode('utf-8-sig')))
        payload = (json.dumps(data, indent=2, sort_keys=True) + '\n').encode('utf8')
        if any(x in payload for x in (b'/public/home/', b'/public/nfs-data/', b'github_pat_', b'ghp_')):
            raise ValueError('Unnormalized sensitive marker in ' + name)
        target = destination / name
        if target.exists():
            raise FileExistsError('Refusing to overwrite: ' + str(target))
        target.write_bytes(payload)
        provenance.append(dict(path=name, source_sha256=hashlib.sha256(raw).hexdigest(),
                               normalized_sha256=hashlib.sha256(payload).hexdigest()))
    (destination / 'metadata_provenance.json').write_text(
        json.dumps(dict(schema='delepref-training-metadata-v1', files=provenance), indent=2) + '\n',
        encoding='utf8')
    print('Prepared {} training metadata files'.format(len(provenance)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    build(args.source, args.out)
