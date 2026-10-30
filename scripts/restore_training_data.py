"""Prepare a portable training configuration from the published dataset.

The published pair shards remain gzip-compressed. Compact per-shard WT
sequence tables are derived from the frozen condition catalog and selected
UniRef manifests; the original candidate-query shards are not reconstructed.
This command does not train a model or download the ESMC backbone.
"""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil


REPO = Path(__file__).resolve().parents[1]
TRAINING_MANIFEST_SHA256 = '8455b818140f8c7be73c344a8525b320c00109b24161fd782f8da22b68316e37'
AUDIT_PROVENANCE_SHA256 = 'a6b7be44bcde9752e3d744651d060ab396cfddb1c1a9235a19a3f0b4eff74863'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_dataset(source):
    manifest_path = source / 'training_manifest.json'
    if sha(manifest_path) != TRAINING_MANIFEST_SHA256:
        raise ValueError('Training manifest checksum mismatch')
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    if manifest.get('schema') != 'delepref-training-release-v1' or manifest['shards'] != 600:
        raise ValueError('Unexpected training manifest')
    for item in manifest['files'] + manifest['metadata']:
        path = (source / item['path']).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError('Missing or unsafe dataset file: ' + item['path'])
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Training file checksum mismatch: ' + item['path'])
    audit = source / 'metadata' / 'audit'
    provenance_path = audit / 'metadata_provenance.json'
    if sha(provenance_path) != AUDIT_PROVENANCE_SHA256:
        raise ValueError('Audit metadata provenance checksum mismatch')
    provenance = json.loads(provenance_path.read_text(encoding='utf8'))
    if provenance.get('schema') != 'delepref-training-metadata-v1':
        raise ValueError('Unexpected audit metadata provenance')
    for item in provenance['files']:
        path = (audit / item['path']).resolve()
        if not path.is_relative_to(audit.resolve()) or not path.is_file():
            raise ValueError('Missing or unsafe audit metadata file: ' + item['path'])
        if sha(path) != item['normalized_sha256']:
            raise ValueError('Audit metadata checksum mismatch: ' + item['path'])
    return manifest


def read_sequences(source):
    sequences = {}
    for filename in ('formal_300k_manifest.csv.gz', 'formal_val_20k_manifest.csv.gz'):
        with gzip.open(source / 'sequences' / filename, 'rt', encoding='utf8', newline='') as stream:
            for row in csv.DictReader(stream):
                seq = row['sequence']
                if hashlib.sha1(seq.encode('utf8')).hexdigest() != row['sequence_sha1']:
                    raise ValueError('Selected WT SHA1 mismatch: ' + row['protein_id'])
                old = sequences.setdefault(row['protein_id'], seq)
                if old != seq:
                    raise ValueError('Conflicting WT sequence: ' + row['protein_id'])
    return sequences


def decompress(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(source, 'rb') as inp, destination.open('xb') as out:
        shutil.copyfileobj(inp, out, length=8 * 1024 * 1024)


def restore(source, destination, encoder_snapshot):
    import yaml
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Use a new output directory: ' + str(destination))
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('Dataset and restored output must be separate directories')
    manifest = verify_dataset(source)
    sequences = read_sequences(source)
    print('Verified {} shards and loaded {} WT sequences'.format(manifest['shards'], len(sequences)), flush=True)
    destination.mkdir(parents=True)
    condition_path = destination / 'condition_catalog/conditions.csv'
    decompress(source / 'metadata/condition_catalog/conditions.csv.gz', condition_path)
    groups = defaultdict(dict)
    with condition_path.open(encoding='utf8', newline='') as stream:
        for row in csv.DictReader(stream):
            key = (row['source'], int(row['shard_index']))
            pid = row['protein_id']; seq = sequences.get(pid)
            if seq is None:
                raise ValueError('WT not found in selected sequence manifests: ' + pid)
            digest = hashlib.sha256(seq.encode('utf8')).hexdigest()
            if digest != row['sequence_sha256'] or len(seq) != int(row['wildtype_length']):
                raise ValueError('WT identity differs from condition catalog: ' + pid)
            groups[key][pid] = (seq, digest)
    for group in ('base', 'b20'):
        with gzip.open(source / ('metadata/indexes/' + group + '.csv.gz'),
                       'rt', encoding='utf8', newline='') as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            rows = list(reader)
        if len(rows) != 300:
            raise ValueError('Expected 300 index entries: ' + group)
        index_path = destination / group / 'indexes/preference_pair_shard_index.csv'
        index_path.parent.mkdir(parents=True, exist_ok=True)
        for row in rows:
            shard = int(row['shard_index'])
            key = (group, shard)
            if key not in groups:
                raise ValueError('No training conditions for shard: ' + str(key))
            sequence_file = destination / group / 'sequences' / (row['shard_id'] + '.csv')
            sequence_file.parent.mkdir(parents=True, exist_ok=True)
            with sequence_file.open('x', encoding='utf8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['protein_id','wildtype_sequence','wildtype_sha256'])
                writer.writeheader()
                for pid, (seq, digest) in sorted(groups[key].items()):
                    writer.writerow(dict(protein_id=pid, wildtype_sequence=seq, wildtype_sha256=digest))
            pair_path = source / 'pairs' / group / (row['shard_id'] + '.csv.gz')
            if not pair_path.is_file():
                raise ValueError('Pair shard missing: ' + str(pair_path))
            row['source_candidate_csv'] = str(sequence_file)
            row['output_csv'] = str(pair_path)
        with index_path.open('x', encoding='utf8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    for rel in ('metadata/splits/train_dev_manifest.csv.gz',
                'metadata/proposal_validation/proteins.csv.gz',
                'metadata/proposal_validation/cold_start_reference.csv.gz'):
        decompress(source / rel, destination / rel.removesuffix('.gz'))
    config = yaml.safe_load((REPO / 'configs/deletion_preference_student.yaml').read_text(encoding='utf8'))
    audit = source / 'metadata/audit'
    mapping = {
        'base_pair_index': destination / 'base/indexes/preference_pair_shard_index.csv',
        'b20_pair_index': destination / 'b20/indexes/preference_pair_shard_index.csv',
        'base_validation_report': audit / 'base_validation_report.json',
        'b20_validation_report': audit / 'b20_validation_report.json',
        'b20_supplement_audit': audit / 'b20_supplement_audit.json',
        'condition_catalog': condition_path,
        'condition_catalog_summary': audit / 'condition_catalog_summary.json',
        'train_dev_manifest': destination / 'metadata/splits/train_dev_manifest.csv',
        'confirmation_lock': audit / 'confirmation_manifest_lock.json',
        'proposal_validation_manifest': destination / 'metadata/proposal_validation/proteins.csv',
        'proposal_validation_manifest_summary': audit / 'proposal_validation_summary.json',
        'cold_start_reference': destination / 'metadata/proposal_validation/cold_start_reference.csv',
    }
    missing = [str(p) for p in mapping.values() if not p.is_file()]
    if missing:
        raise ValueError('Missing restored training inputs: ' + ', '.join(missing))
    config['data'].update({key: str(value) for key, value in mapping.items()})
    config['model']['esmc_snapshot'] = str(Path(encoder_snapshot).resolve())
    config['proposal_validation']['teacher_model_snapshot'] = str(Path(encoder_snapshot).resolve())
    config['training']['output_dir'] = str(destination / 'runs')
    config['training']['checkpoint_dir'] = str(destination / 'checkpoints')
    config_path = destination / 'training_config.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf8')
    print(json.dumps(dict(status='PREPARED', pairs=manifest['pairs'], shards=600,
                          WT_sequences=len(sequences), config=str(config_path)), indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, help='Downloaded training/ directory')
    p.add_argument('--out', required=True, help='New restored output directory')
    p.add_argument('--encoder-snapshot', required=True, help='Pinned ESMC model directory')
    a = p.parse_args()
    restore(a.dataset, a.out, a.encoder_snapshot)
