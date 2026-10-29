"""Build the portable paper-evidence archive from the frozen research records.

This does not run the model, fold proteins, or create experimental results.
Input directories must contain the original 2026-10-25 follow-up exports.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


RESEARCH_ROOT = '/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013/'
REPEAT_FILES = (
    'paired_comparisons.json', 'paired_per_WT_and_repeat.json',
    'candidate_pools.json', 'selected_outputs.json', 'per_output_metrics.json',
    'protocol.json', 'statistical_protocol.json', 'completion.json', 'sampling_audit.json',
)
RAYGUN_FILES = (
    'primary_tables.csv', 'primary_tables.json', 'per_protein_primary.csv',
    'canonical_sequences.fasta', 'strict_recalculation.json',
    'strict_protocol_before_readout.json', 'independent_readout_validation.json',
)


def normalize(value):
    if isinstance(value, dict):
        return {normalize(k): normalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if isinstance(value, str):
        return value.replace(RESEARCH_ROOT, 'research-source/')
    return value


def build(repeat_dir, raygun_dir, output):
    repeat_dir, raygun_dir, output = map(Path, (repeat_dir, raygun_dir, output))
    if output.exists():
        raise FileExistsError('Use a new output path: ' + str(output))
    entries = {}
    sources = []
    for group, directory, names in (
        ('completion', repeat_dir, REPEAT_FILES), ('raygun', raygun_dir, RAYGUN_FILES),
    ):
        for name in names:
            raw = (directory / name).read_bytes()
            sources.append(dict(name=group + '/' + name, bytes=len(raw),
                                original_sha256=hashlib.sha256(raw).hexdigest()))
            if name.endswith('.json'):
                obj = normalize(json.loads(raw.decode('utf-8-sig')))
                raw = (json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                  ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
            entries[group + '/' + name] = raw
    artifacts = sorted((repeat_dir / 'artifacts').glob('*.npz'))
    if len(artifacts) != 18:
        raise ValueError('Expected 18 follow-up structure artifacts; found {}'.format(len(artifacts)))
    for path in artifacts:
        raw = path.read_bytes()
        name = 'completion/structures/' + path.name
        entries[name] = raw
        sources.append(dict(name=name, bytes=len(raw), original_sha256=hashlib.sha256(raw).hexdigest()))
    manifest = dict(schema='delepref-paper-evidence-v1',
        description='Frozen post-hoc five-draw completion and strict Raygun identity evidence',
        source_path_normalization='Research-root prefixes in JSON replaced with research-source/; hashes refer to original inputs.',
        code_license='Author-written code is MIT; third-party data and outputs retain their source terms.',
        sources=sources)
    entries['manifest.json'] = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode('utf-8')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 25, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data, compresslevel=9)
    print(json.dumps(dict(path=str(output), bytes=output.stat().st_size,
                          sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                          entries=len(entries))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat', required=True)
    parser.add_argument('--raygun', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    build(args.repeat, args.raygun, args.out)
