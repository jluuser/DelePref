"""Check that a published training snapshot can be staged without raw queries."""
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import pytest
import yaml

from scripts.restore_training_data import restore, verify_dataset
import scripts.restore_training_data as training_restore


def gz_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)


def fixture_dataset(root):
    sequence = 'ACDEFGHIKLMNPQRSTVWY'
    digest = hashlib.sha256(sequence.encode()).hexdigest()
    seq_row = dict(protein_id='UniRef50_test', sequence=sequence,
                   sequence_sha1=hashlib.sha1(sequence.encode()).hexdigest())
    metadata = []
    def register(path, target):
        raw = path.read_bytes()
        target.append(dict(path=str(path.relative_to(root)).replace('\\', '/'),
                           bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    for name in ('formal_300k_manifest.csv.gz', 'formal_val_20k_manifest.csv.gz'):
        path = root / 'sequences' / name
        gz_csv(path, list(seq_row), [seq_row]); register(path, metadata)
    conditions = []
    files = []
    for group in ('base', 'b20'):
        index = []
        for shard in range(300):
            name = 'pair_v3_{:06d}'.format(shard)
            path = root / 'pairs' / group / (name + '.csv.gz')
            gz_csv(path, ['protein_id'], [dict(protein_id='UniRef50_test')]); register(path, files)
            index.append(dict(shard_id=name, shard_index=shard, source_candidate_csv='old',
                              output_csv='old', num_pairs=1))
            conditions.append(dict(source=group, shard_index=shard, protein_id='UniRef50_test',
                                   sequence_sha256=digest, wildtype_length=len(sequence)))
        path = root / 'metadata/indexes' / (group + '.csv.gz')
        gz_csv(path, list(index[0]), index); register(path, metadata)
    path = root / 'metadata/condition_catalog/conditions.csv.gz'
    gz_csv(path, list(conditions[0]), conditions); register(path, metadata)
    for name in ('splits/train_dev_manifest.csv.gz', 'proposal_validation/proteins.csv.gz',
                 'proposal_validation/cold_start_reference.csv.gz'):
        path = root / 'metadata' / name
        gz_csv(path, ['protein_id'], [dict(protein_id='UniRef50_test')]); register(path, metadata)
    audit = root / 'metadata/audit'; audit.mkdir(parents=True)
    audit_files = []
    for name in ('base_validation_report.json', 'b20_validation_report.json',
                 'b20_supplement_audit.json', 'condition_catalog_summary.json',
                 'confirmation_manifest_lock.json', 'proposal_validation_summary.json'):
        (audit / name).write_text('{}', encoding='utf8')
        audit_files.append(dict(path=name, normalized_sha256=hashlib.sha256(b'{}').hexdigest()))
    (audit / 'metadata_provenance.json').write_text(json.dumps(dict(
        schema='delepref-training-metadata-v1', files=audit_files)), encoding='utf8')
    (root / 'training_manifest.json').write_text(json.dumps(dict(
        schema='delepref-training-release-v1', shards=600, pairs=600, files=files, metadata=metadata)), encoding='utf8')


def test_restore_builds_portable_indexes_and_detects_corruption(tmp_path, monkeypatch):
    dataset = tmp_path / 'dataset'; dataset.mkdir(); fixture_dataset(dataset)
    monkeypatch.setattr(training_restore, 'TRAINING_MANIFEST_SHA256',
                        hashlib.sha256((dataset / 'training_manifest.json').read_bytes()).hexdigest())
    monkeypatch.setattr(training_restore, 'AUDIT_PROVENANCE_SHA256',
                        hashlib.sha256((dataset / 'metadata/audit/metadata_provenance.json').read_bytes()).hexdigest())
    out = tmp_path / 'restored'
    restore(dataset, out, tmp_path / 'encoder')
    config = yaml.safe_load((out / 'training_config.yaml').read_text(encoding='utf8'))
    with Path(config['data']['base_pair_index']).open(encoding='utf8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 300
    assert Path(rows[0]['output_csv']).is_file()
    with Path(rows[0]['source_candidate_csv']).open(encoding='utf8', newline='') as stream:
        assert next(csv.DictReader(stream))['wildtype_sequence'] == 'ACDEFGHIKLMNPQRSTVWY'
    with pytest.raises(FileExistsError):
        restore(dataset, out, tmp_path / 'encoder')
    audit = dataset / 'metadata/audit/base_validation_report.json'
    audit.write_text('{"modified": true}', encoding='utf8')
    with pytest.raises(ValueError, match='Audit metadata checksum mismatch'):
        verify_dataset(dataset.resolve())
    audit.write_text('{}', encoding='utf8')
    pair = dataset / 'pairs/base/pair_v3_000000.csv.gz'
    pair.write_bytes(b'not-the-frozen-data')
    with pytest.raises(ValueError, match='checksum mismatch'):
        verify_dataset(dataset.resolve())
