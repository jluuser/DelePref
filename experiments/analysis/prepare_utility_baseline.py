"""CPU-only original utility scoring from verified full residue embeddings.

This uses the historical scalar functions and six-decimal teacher serialization.
The source's missing `mean` binding affects junction detail only; the private
module copy receives an arithmetic mean binding, without changing any score.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from common import AUTHORIZED_ROOT, confined_path, read_jsonl, sha_file, write_json, write_jsonl
from adapters.delepref_adapter import retained_mapping, intervals_from_deletion_mask
from adapters.structure_adapter import coordinates_from_native_json
from build_pairs import reference_pairs
from train import Inputs


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot_sources(root, legacy_root):
    paths = {'teacher_scoring': legacy_root / 'scripts/teacher_scoring.py',
             'proposal_validation': legacy_root / 'src/delepref/proposal_validation.py'}
    record = {}
    for name, source in paths.items():
        checksum = sha_file(source)
        target = confined_path(root / 'provenance/utility_sources' / checksum / source.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha_file(target) != checksum:
            raise ValueError('utility source snapshot changed')
        if not target.exists():
            shutil.copyfile(source, target)
        record[name] = {'original_path': str(source), 'snapshot': str(target), 'sha256': checksum}
    seal = root / 'provenance/utility_baseline_sources.json'
    if seal.exists() and json.loads(seal.read_text()) != record:
        raise ValueError('previously frozen original utility source differs')
    write_json(seal, record)
    teacher = load_module('frozen_utility_teacher', record['teacher_scoring']['snapshot'])
    # The historical source refers to mean() only in the diagnostic detail dict.
    teacher.mean = statistics.fmean
    selection = load_module('frozen_utility_selection', record['proposal_validation']['snapshot'])
    return teacher, selection, record


def utility_scores(row, wt, edited, teacher, selection):
    kept = retained_mapping(row['wt_sequence'], row['sequence'], row['mask'])
    if wt.shape != (len(row['wt_sequence']), 1152) or edited.shape != (len(kept), 1152):
        raise ValueError('utility embedding shape mismatch')
    if not len(kept) or not np.isfinite(wt).all() or not np.isfinite(edited).all():
        raise ValueError('empty or nonfinite utility inputs')
    intervals = intervals_from_deletion_mask(row['mask'])
    original = torch.as_tensor(wt, device='cpu')
    variant = torch.as_tensor(edited, device='cpu')
    mapping = teacher.retained_residue_mapping(len(wt), intervals)
    if mapping != list(zip(kept.tolist(), range(len(kept)))):
        raise ValueError('historical utility alignment differs')
    with torch.inference_mode():
        global_shift = teacher.cosine_distance_unit(torch, original.mean(0), variant.mean(0))
        retained_shift = teacher.mean_retained_distance(torch, original, variant, mapping)
        junction, detail = teacher.junction_plausibility_proxy(torch, original, variant, len(wt), intervals)
    # Match teacher CSV precision before applying the original formal weights.
    views = {'global_similarity': float(teacher.fmt_score(teacher.clamp01(1 - 2 * global_shift))),
             'retained_similarity': float(teacher.fmt_score(teacher.clamp01(1 - 2 * retained_shift))),
             'junction_plausibility': float(teacher.fmt_score(junction))}
    result = {**views, 'utility': selection.teacher_utility(views),
              'local_cosine_similarity': 1 - 2 * retained_shift,
              'global_shift': global_shift, 'retained_shift': retained_shift,
              'junction_values': detail['junction_values']}
    if any(not math.isfinite(result[k]) for k in views.keys() | {'utility', 'local_cosine_similarity'}):
        raise ValueError('nonfinite utility score')
    return result


def run(root, protocol_path, manifest_path, indices, output):
    started = time.perf_counter()
    if root.resolve() != AUTHORIZED_ROOT.resolve():
        raise ValueError('unexpected experiment root')
    output = confined_path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(protocol_path.read_text())
    seal = json.loads((root / 'configs/execution_freeze.json').read_text())
    if sha_file(protocol_path) != seal['input_hashes']['configs/protocol_v2.json']:
        raise ValueError('protocol differs from frozen execution inputs')
    relative = str(manifest_path.resolve().relative_to(root.resolve()))
    if sha_file(manifest_path) != seal['input_hashes'][relative]:
        raise ValueError('development manifest differs from frozen execution inputs')
    teacher, selection, sources = snapshot_sources(root, Path(protocol['sources']['delepref']))
    inputs = Inputs(indices, sha_file(protocol_path))
    rows = read_jsonl(manifest_path)
    if len({r['candidate_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate logical candidate')
    references, result = {}, []
    for n, row in enumerate(rows, 1):
        pid = row['protein_id']
        if pid not in references:
            for field in ('wt_native_json', 'reference_path'):
                expected = row['wt_native_json_sha256' if field == 'wt_native_json' else 'reference_sha256']
                if sha_file(row[field]) != expected:
                    raise ValueError('WT reference identity changed')
            coords, _ = coordinates_from_native_json(row['wt_native_json'], row['wt_sequence'])
            references[pid] = reference_pairs(coords), reference_pairs(coords, 8.)
        wi, wj = references[pid][0]
        ci, cj = references[pid][1]
        if not len(wi) or not len(ci):
            raise ValueError('undefined static relationship/contact coverage')
        mask = np.asarray(row['mask'], dtype=bool)
        score = utility_scores(row, inputs.embedding(row['wt_sequence']), inputs.embedding(row['sequence']), teacher, selection)
        result.append({k: row[k] for k in ('candidate_id', 'protein_id', 'budget', 'generator')} |
                      score | {'static_relationship_coverage': float((~(mask[wi] | mask[wj])).mean()),
                               'direct_contact_loss': float((mask[ci] | mask[cj]).mean())})
        if n % 250 == 0:
            print(json.dumps({'status': 'UTILITY_PROGRESS', 'candidates': n, 'total': len(rows)}), flush=True)
    write_jsonl(output / 'candidates.jsonl', result)
    report = {'status': 'CPU_UTILITY_BASELINE_COMPLETE', 'candidates': len(result),
              'WT': len(references), 'seconds': time.perf_counter() - started,
              'protocol_sha256': sha_file(protocol_path), 'manifest_sha256': sha_file(manifest_path),
              'cache_index_sha256': {str(p): sha_file(p) for p in indices},
              'embedding_profile_hashes': sorted(inputs.profiles), 'sources': sources,
              'implementation_sha256': sha_file(__file__), 'output_sha256': sha_file(output / 'candidates.jsonl'),
              'weights': {'global_similarity': .35, 'retained_similarity': .45, 'junction_plausibility': .20},
              'serialization': 'original fmt_score six decimals for G/R/J before formal utility weights',
              'private_source_repair': 'arithmetic mean binding for junction detail only; shared source unchanged',
              'PLM_forward_calls': 0, 'structure_calls': 0, 'GPU_used': False,
              'variant_structures_or_labels_read': False, 'original_policy_checkpoint_used': False,
              'interpretation': 'Original PLM utility scalar baseline, not restored original full policy/scorer system'}
    write_json(output / 'report.json', report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'protocol', 'manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cache-index', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.root, args.protocol, args.manifest, args.cache_index, args.output)


if __name__ == '__main__':
    main()
