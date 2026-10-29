"""Export the frozen primary benchmark structures and reference coordinates.

This packaging command reads existing records and artifacts; it does not run
any model or change the research checkout. The input layout is the archived
2026-10 research project, while the resulting archive uses portable paths.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_json(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False, allow_nan=False) + '\n').encode('utf8')


def normalize(value, root):
    if isinstance(value, dict):
        return {normalize(k, root): normalize(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize(v, root) for v in value]
    if isinstance(value, str):
        prefix = str(root).rstrip('/') + '/'
        if value.startswith(prefix):
            return 'research-source/' + value[len(prefix):]
        if value.startswith('/public/nfs-data/'):
            return 'external-source/' + Path(value).name
        if value.startswith('/public/home/'):
            return 'external-source/' + Path(value).name
    return value


def add_bytes(tar, name, payload):
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ''
    info.mtime = 1792886400
    tar.addfile(info, io.BytesIO(payload))


def add_file(tar, root, source, name):
    source = source.resolve()
    if not source.is_relative_to(root) or not source.is_file():
        raise ValueError('Artifact must be an existing file under the research root: ' + str(source))
    digest = sha256_file(source)
    info = tar.gettarinfo(str(source), arcname=name)
    info.uid = info.gid = 0
    info.uname = info.gname = ''
    info.mode = 0o644
    info.mtime = 1792886400
    with source.open('rb') as stream:
        tar.addfile(info, stream)
    return dict(path=name, bytes=source.stat().st_size, sha256=digest)


def build(root, benchmark_path, output):
    root = Path(root).resolve()
    benchmark_path, output = Path(benchmark_path), Path(output).resolve()
    if output.is_relative_to(root) or output.exists():
        raise ValueError('Use a new output path outside the research source')
    with gzip.open(benchmark_path, 'rt', encoding='utf8') as stream:
        benchmark = json.load(stream)
    rows = benchmark['per_output_measurements']
    groups = {
        'general': 'reports/recomb_confirm_v2_General100_structures',
        'application': 'reports/recomb_confirm_v2_application_structures_hardware_v3',
    }
    ids = defaultdict(set)
    statuses = Counter()
    for row in rows:
        statuses[row['prediction_status']] += 1
        if row['prediction_status'] == 'SUCCEEDED':
            ids['general' if row['cohort'] == 'general' else 'application'].add(row['physical_id'])
    if len(rows) != 3474 or len(ids['general']) != 1666 or len(ids['application']) != 803:
        raise ValueError('Unexpected frozen benchmark inventory')

    panel_path = root / 'reports/recomb_confirm_v2_general_native_v2/General100_panel.json'
    panel = json.loads(panel_path.read_text(encoding='utf8'))
    raygun_path = root / 'reports/recomb_raygun_canonical_20261025/strict_recalculation.json'
    raygun = json.loads(raygun_path.read_text(encoding='utf8'))
    if len(panel) != 100:
        raise ValueError('Expected 100 general reference constructs')
    output.parent.mkdir(parents=True, exist_ok=True)
    records, native = [], []
    with tarfile.open(output, 'w:gz', compresslevel=1) as tar:
        for group, rel in groups.items():
            directory = root / rel
            for pid in sorted(ids[group]):
                source_record = directory / 'records' / (pid + '.json')
                source_bytes = source_record.read_bytes()
                record = json.loads(source_bytes.decode('utf8'))
                if record['task_id'] != pid or record['status'] != 'SUCCEEDED':
                    raise ValueError('Prediction record identity/status mismatch: ' + pid)
                files = []
                for ext in ('.npz', '.cif', '.pdb'):
                    source = directory / 'artifacts' / (pid + ext)
                    if source.is_file():
                        item = add_file(tar, root, source, 'predictions/{}/{}{}'.format(group, pid, ext))
                        if ext == '.npz' and record.get('output_sha256') != item['sha256']:
                            raise ValueError('Prediction checksum mismatch: ' + pid)
                        files.append(item)
                if not files:
                    raise ValueError('No structure artifact: ' + pid)
                normalized = canonical_json(normalize(record, root))
                record_path = 'records/{}/{}.json'.format(group, pid)
                add_bytes(tar, record_path, normalized)
                records.append(dict(group=group, physical_id=pid, files=files,
                    record=record_path, original_record_sha256=hashlib.sha256(source_bytes).hexdigest(),
                    normalized_record_sha256=hashlib.sha256(normalized).hexdigest()))
        for row in panel:
            source = Path(row['wt_native_json']).resolve()
            if not source.is_relative_to(root):
                raise ValueError('General reference outside research root')
            raw = source.read_bytes()
            if hashlib.sha256(raw).hexdigest() != row['wt_native_json_sha256']:
                raise ValueError('General native reference checksum mismatch')
            payload = canonical_json(normalize(json.loads(raw.decode('utf8')), root))
            name = 'references/general/' + source.name
            add_bytes(tar, name, payload)
            native.append(dict(cohort='General100', protein_id=row['protein_id'], path=name,
                original_sha256=hashlib.sha256(raw).hexdigest(), normalized_sha256=hashlib.sha256(payload).hexdigest()))
        for name, expected in sorted(raygun['input_assets_sha256'].items()):
            if '/references/' not in name or not name.endswith('.json'):
                continue
            source = Path(name).resolve()
            if not source.is_relative_to(root):
                raise ValueError('Application reference outside research root')
            raw = source.read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError('Application native reference checksum mismatch')
            cohort = 'Binder60' if 'Binder60_constructs' in name else 'Enzyme80'
            payload = canonical_json(normalize(json.loads(raw.decode('utf8')), root))
            path = 'references/{}/{}'.format(cohort, source.name)
            add_bytes(tar, path, payload)
            native.append(dict(cohort=cohort, path=path, original_sha256=expected,
                               normalized_sha256=hashlib.sha256(payload).hexdigest()))
        if Counter(x['cohort'] for x in native) != {'General100':100, 'Enzyme80':80, 'Binder60':60}:
            raise ValueError('Unexpected native reference inventory')
        manifest = dict(schema='delepref-primary-structures-v1',
            benchmark_manifest_sha256=sha256_file(benchmark_path), logical_rows=len(rows),
            prediction_statuses=dict(statuses), physical_predictions=len(records),
            original_reference_paths='Normalized to research-source/ or external-source/; original hashes retained.',
            predictions=records, native_references=native)
        add_bytes(tar, 'manifest.json', canonical_json(manifest))
    print(json.dumps(dict(path=str(output), bytes=output.stat().st_size,
        sha256=sha256_file(output), logical_rows=len(rows), physical_predictions=len(records),
        reference_count=len(native), statuses=dict(statuses)), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--benchmark', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    build(a.root, a.benchmark, a.out)
