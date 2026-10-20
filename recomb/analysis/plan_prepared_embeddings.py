"""Prepare deduplicated confirmation cache inputs; zero embedding forwards."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from common import AUTHORIZED_ROOT, identity, read_jsonl, sha_file, sha_text, write_json, write_jsonl
from adapters.delepref_adapter import retained_mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if root != AUTHORIZED_ROOT.resolve():
        raise ValueError('unexpected experiment root')
    preparation = json.loads((root / 'reports/candidate_preparation.json').read_text())
    paths = {'matched': root / 'manifests/confirmation_candidates_prepared.jsonl',
             'pipeline': root / 'manifests/pipeline_candidates_prepared.jsonl'}
    combined, sequences, memberships = {}, {}, 0
    for kind, path in paths.items():
        if sha_file(path) != preparation[kind + '_manifest_sha256']:
            raise ValueError('prepared candidate pool changed')
        for row in read_jsonl(path):
            retained_mapping(row['wt_sequence'], row['sequence'], row['mask'])
            mask_hash = sha_text(''.join(str(int(x)) for x in row['mask']))
            sequence_hash = sha_text(row['sequence'])
            if (mask_hash != row['mask_sha256'] or sequence_hash != row['sequence_sha256'] or
                    sha_text(row['wt_sequence']) != row['wt_sequence_sha256'] or
                    identity((row['wt_sequence_sha256'], mask_hash, sequence_hash)) != row['candidate_id']):
                raise ValueError('prepared logical candidate identity differs')
            cid = row['candidate_id']
            membership = {'pool_kind': kind, 'group': row['group']}
            if cid in combined:
                old = combined[cid]
                for field in ('wt_sequence', 'sequence', 'mask', 'protein_id'):
                    if old[field] != row[field]:
                        raise ValueError('logical candidate collision')
                if membership in old['pool_memberships']:
                    raise ValueError('duplicate pool membership')
                old['pool_memberships'].append(membership)
            else:
                combined[cid] = {**row, 'pool_memberships': [membership]}
            memberships += 1
            for seq in (row['wt_sequence'], row['sequence']):
                sequences[sha_text(seq)] = len(seq)
    expected = preparation['combined_distinct_logical_candidates']
    if len(combined) != expected or memberships != preparation['matched_candidates'] + preparation['pipeline_candidates']:
        raise ValueError('prepared union does not preserve every logical pool membership')
    merged = root / 'manifests/confirmation_embedding_inputs_prepared.jsonl'
    write_jsonl(merged, [combined[key] for key in sorted(combined)])
    dispatch = json.loads((root / 'reports/dispatch_after_pilot.json').read_text())
    snapshot = Path(dispatch['code_snapshot'])
    if any(sha_file(snapshot / name) != checksum for name, checksum in dispatch['code_hashes'].items()):
        raise ValueError('computational snapshot changed')
    for shard in (0, 1):
        subprocess.run([sys.executable, str(snapshot / 'cache_plm.py'), '--protocol', str(root / 'configs/protocol_v2.json'),
                        '--manifest', str(merged), '--provenance', str(root / 'provenance/source_files.json'),
                        '--output', str(root / 'caches/embeddings'), '--shard', str(shard),
                        '--index-prefix', 'confirmation_prepared'], check=True)
    planned = [r for shard in (0, 1) for r in json.loads((root / f'caches/embeddings/confirmation_prepared_shard{shard}_plan.json').read_text())['sequences']]
    if len(planned) != len(sequences) or {r['sequence_sha256'] for r in planned} != set(sequences):
        raise ValueError('cache plans omit or duplicate sequences')
    report = {'status': 'CONFIRMATION_EMBEDDING_INPUTS_PREPARED_ONLY',
              'logical_candidates': len(combined), 'pool_memberships': memberships,
              'distinct_WT_and_variant_sequences': len(sequences),
              'planned_float32_bytes': sum(sequences.values()) * 1152 * 4,
              'merged_manifest_sha256': sha_file(merged),
              'pool_manifest_sha256': {kind: sha_file(path) for kind, path in paths.items()},
              'implementation_sha256': sha_file(__file__), 'code_snapshot': str(snapshot),
              'PLM_forward_calls': 0, 'structure_calls': 0, 'labels_read': False,
              'scores_or_final_selector_outputs_frozen': False,
              'interpretation': 'Prepared input union and dry cache plans; future computation remains gated on development evidence.'}
    write_json(root / 'reports/confirmation_embedding_preparation.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
