"""Count sealed WT coordinate coverage without reading variant structures."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from common import AUTHORIZED_ROOT, read_jsonl, sha_file, sha_text, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if root != AUTHORIZED_ROOT.resolve():
        raise ValueError('unexpected experiment root')
    manifest = root / 'manifests/development_primary.jsonl'
    seal = json.loads((root / 'configs/execution_freeze.json').read_text())
    split_path = root / 'manifests/splits.json'
    if sha_file(manifest) != seal['input_hashes']['manifests/development_primary.jsonl'] or sha_file(split_path) != seal['input_hashes']['manifests/splits.json']:
        raise ValueError('frozen primary inputs changed')
    split = json.loads(split_path.read_text())
    records = {}
    for row in read_jsonl(manifest):
        wt = row['protein_id']
        if split['assignment'][wt] != 'validation':
            continue
        path = Path(row['wt_native_json'])
        if wt in records:
            if records[wt]['native_sha256'] != row['wt_native_json_sha256'] or records[wt]['sequence_sha256'] != sha_text(row['wt_sequence']):
                raise ValueError('WT coordinate/sequence identity differs across candidates')
            continue
        if sha_file(path) != row['wt_native_json_sha256']:
            raise ValueError('native coordinate identity changed')
        data = json.loads(path.read_text())
        if data['wt_sequence_sha256'] != sha_text(row['wt_sequence']):
            raise ValueError('native sequence differs')
        length = len(row['wt_sequence'])
        xyz = np.full((length, 3), np.nan)
        for index, value in data['coordinates'].items():
            position = int(index)
            if not 0 <= position < length or np.asarray(value).shape != (3,):
                raise ValueError('invalid native coordinate position')
            xyz[position] = value
        finite = np.isfinite(xyz).all(1)
        i, j = np.triu_indices(length, k=3)
        unknown = ~finite[i] | ~finite[j]
        records[wt] = {'protein_id': wt, 'sequence_length': length, 'known_CA': int(finite.sum()),
                       'CA_coverage': float(finite.mean()), 'missing_CA_positions': np.flatnonzero(~finite).tolist(),
                       'potential_sequence_separated_pairs': len(i), 'unmeasurable_pair_candidates': int(unknown.sum()),
                       'native_sha256': row['wt_native_json_sha256'], 'sequence_sha256': sha_text(row['wt_sequence'])}
    if len(records) != 10:
        raise ValueError('primary validation WT roster is incomplete')
    report = {'status': 'WT_REFERENCE_COORDINATES_AUDITED', 'WT': len(records), 'records': records,
              'manifest_sha256': sha_file(manifest), 'implementation_sha256': sha_file(__file__),
              'interpretation': 'Reference E includes only measurable WT distances <15A with sequence separation>=3. Unmeasurable WT pair candidates are recorded, not assigned geometric outcomes.',
              'variant_structure_reads': 0, 'GPU_calls': 0, 'structure_prediction_calls': 0}
    write_json(root / 'reports/cpu_development_diagnostics/reference_coordinate_coverage.json', report)
    print(json.dumps({'status': report['status'], 'WT': len(records), 'missing_CA': sum(len(r['missing_CA_positions']) for r in records.values())}))


if __name__ == '__main__':
    main()
