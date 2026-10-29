"""Stage the original preference corpus for public distribution.

Reads the frozen source indexes and shards, checks source sizes, and writes
individually compressed files plus SHA256 metadata. Does not retrain a model.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path


CORRECTED = 'data/processed/preference_training/final_readiness/corrected/'
INDEXES = {
    'base': CORRECTED + 'base_pair_v3_absolute_utility/indexes/preference_pair_shard_index.csv',
    'b20': CORRECTED + 'b20_supplement/pairs/indexes/preference_pair_shard_index.csv',
}
METADATA = {
    'sequences/formal_300k_manifest.csv.gz': 'data/processed/preference_training/uniref50_manifest/uniref50_formal_300k_manifest.csv',
    'sequences/formal_val_20k_manifest.csv.gz': 'data/processed/preference_training/uniref50_manifest/uniref50_formal_val_20k_manifest.csv',
    'metadata/splits/train_dev_manifest.csv.gz': CORRECTED + 'splits/v3_sequence_hash_v1/train_dev_manifest.csv',
    'metadata/condition_catalog/conditions.csv.gz': CORRECTED + 'condition_catalog/v1/conditions.csv',
    'metadata/proposal_validation/proteins.csv.gz': CORRECTED + 'proposal_validation/v1/proteins.csv',
    'metadata/proposal_validation/cold_start_reference.csv.gz': CORRECTED + 'proposal_validation/v1/cold_start_reference.csv',
}
FORBIDDEN = (b'-----BEGIN PRIVATE KEY-----', b'-----BEGIN OPENSSH PRIVATE KEY-----',
             b'github_pat_', b'ghp_')


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_source(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Missing or out-of-root source: ' + relative)
    return path


def compress(job):
    source, target, expected_size, normalize_manifest = job
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.partial')
    if target.exists():
        raise FileExistsError('Refusing to overwrite: ' + str(target))
    raw_hash = hashlib.sha256()
    with source.open('rb') as input_stream, partial.open('wb') as raw_output:
        with gzip.GzipFile(filename='', mode='wb', compresslevel=1, fileobj=raw_output, mtime=0) as zipped:
            tail = b''
            for block in iter(lambda: input_stream.read(8 * 1024 * 1024), b''):
                raw_hash.update(block)
                check = tail + block
                if any(marker in check for marker in FORBIDDEN):
                    raise ValueError('Potential credential marker in source: ' + source.name)
                tail = check[-100:]
                if normalize_manifest:
                    # CSV fields can span blocks, so normalized manifests are handled separately.
                    raise ValueError('Use normalize_sequences for sequence manifests')
                zipped.write(block)
    if expected_size is not None and source.stat().st_size != expected_size:
        raise ValueError('Source size differs from frozen index: ' + source.name)
    partial.replace(target)
    return dict(source_bytes=source.stat().st_size, source_sha256=raw_hash.hexdigest(),
                bytes=target.stat().st_size, sha256=file_hash(target))


def normalize_sequences(source, target):
    """Preserve selected sequences while replacing the research-machine path."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError('Refusing to overwrite: ' + str(target))
    partial = target.with_name(target.name + '.partial')
    with source.open('r', encoding='utf8', newline='') as inp, partial.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', compresslevel=1, fileobj=raw, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding='utf8', newline='') as text:
                reader = csv.DictReader(inp)
                writer = csv.DictWriter(text, fieldnames=reader.fieldnames)
                writer.writeheader()
                for row in reader:
                    row['source_fasta'] = 'UniRef50 (original release identity not fully recovered)'
                    writer.writerow(row)
    partial.replace(target)
    return dict(source_bytes=source.stat().st_size, source_sha256=file_hash(source),
                bytes=target.stat().st_size, sha256=file_hash(target),
                normalization='source_fasta field replaced with a source description')


def build(root, output, workers=2, check_only=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError('Output must be outside the research source tree')
    jobs, indexes, count = [], {}, 0
    for group, relative in INDEXES.items():
        source_index = checked_source(root, relative)
        with source_index.open(newline='', encoding='utf8') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 300 or {int(r['shard_index']) for r in rows} != set(range(300)):
            raise ValueError('Expected 300 indexed shards for ' + group)
        indexes[group] = dict(source_sha256=file_hash(source_index), shards=len(rows))
        for row in rows:
            source = checked_source(root, row['output_csv'])
            if row['status'] != 'written':
                raise ValueError('Source index records an incomplete shard')
            if source.stat().st_size != int(row['output_size_bytes']):
                raise ValueError('Source size differs from frozen index: ' + source.name)
            name = 'pairs/{}/{}.csv.gz'.format(group, row['shard_id'])
            jobs.append((name, source, int(row['output_size_bytes']), int(row['num_pairs'])))
            count += int(row['num_pairs'])
    if count != 24386017:
        raise ValueError('Pair total differs from frozen corpus: ' + str(count))
    for destination, source in METADATA.items():
        checked_source(root, source)
    if check_only:
        print(json.dumps(dict(status='VALIDATED_INPUTS', pairs=count, shards=len(jobs),
                              metadata_files=len(METADATA))), flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(compress, (src, output / name, size, False)): (name, pairs)
                   for name, src, size, pairs in jobs}
        for finished in as_completed(futures):
            name, pairs = futures[finished]
            results.append(dict(path=name, pairs=pairs, **finished.result()))
            if len(results) % 25 == 0:
                print('Compressed {}/{} pair shards'.format(len(results), len(jobs)), flush=True)
    metadata = []
    for destination, relative in METADATA.items():
        source, target = checked_source(root, relative), output / destination
        if destination.startswith('sequences/'):
            record = normalize_sequences(source, target)
        else:
            record = compress((source, target, None, False))
        metadata.append(dict(path=destination, **record))
    for group, relative in INDEXES.items():
        src = checked_source(root, relative)
        target = output / ('metadata/indexes/' + group + '.csv.gz')
        metadata.append(dict(path=str(target.relative_to(output)).replace('\\', '/'),
                             **compress((src, target, None, False))))
    manifest = dict(schema='delepref-training-release-v1',
        pairs=count, shards=len(results), indices=indexes, files=sorted(results, key=lambda r:r['path']),
        metadata=metadata, description='Frozen preference pairs and selected UniRef50 WT sequence manifests',
        original_uniref_release='The exact upstream FASTA release is not fully recovered.',
        storage='Each CSV is gzip-compressed; SHA256 records cover source and released bytes.')
    target = output / 'training_manifest.json'
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf8')
    print(json.dumps(dict(status='COMPLETE', pairs=count, shards=len(results),
        compressed_bytes=sum(r['bytes'] for r in results + metadata),
        manifest_sha256=file_hash(target))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, help='Original research source root')
    p.add_argument('--out', required=True, help='New output directory outside source root')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    build(a.root, a.out, workers=a.workers, check_only=a.check_only)
