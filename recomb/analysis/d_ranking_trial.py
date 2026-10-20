"""Bounded D-ranking trial: sealed inputs, paired sampling, fixed-step training.

The original head, Inputs, logits and evaluation are imported from the immutable
snapshot only. CPU technical runs are separate from the nine formal Slurm runs.
"""
import argparse
from decimal import Decimal
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

AUTHORIZED_ROOT = Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013')
OLD_FREEZE = 'configs/prospective_replication_freeze.json'
OLD_FREEZE_SHA256 = '01ea67da5bd876ae9cd836c4e7db3ace90ba6e01d15c61040279bcd1d05999a0'
NEW_FREEZE = 'configs/d_ranking_trial_freeze.json'
SNAPSHOT = 'provenance/frozen_code/cec255e84a6562387a17e083713329a697a1c67ac842decc268aa8ae95b5c9c8/code'
SEEDS = (20261013, 20261014, 20261015)
BUDGETS = (.2, .3, .4)
SOURCES = ('evocompress', 'scisor')
MODES = {'D0': {'arm': 'D', 'lambda': 0.}, 'D_rank': {'arm': 'D', 'lambda': .1},
         'C0': {'arm': 'C', 'lambda': 0.}}
NEW_SOURCES = ('analysis/d_ranking_trial.py', 'analysis/tests/test_d_ranking_trial.py',
               'slurm/d_ranking_trial.sbatch')
DOCS = ('docs/superpowers/plans/2026-10-14-d-ranking-trial.md',
        'docs/research/2026-10-14-structural-risk-boundary.md')
CONTRACT = {
    'modes': MODES, 'seeds': list(SEEDS), 'budgets': list(BUDGETS), 'sources': list(SOURCES),
    'primary_candidates': 2520, 'train_wt': 50, 'validation_wt': 10, 'validation_candidates': 420,
    'candidates_per_condition': 7, 'steps': 2000, 'wt_per_step': 4,
    'candidates_per_sampled_condition': 2, 'relations_per_candidate': 128,
    'relation_updates_per_step': 1024, 'sampling': 'uniform_WT_then_budget_then_source_then_two_distinct_candidates',
    'relation_sampling': 'uniform_without_replacement_unless_fewer_than_128',
    'sampler_rng': 'independent_numpy_default_rng_seed_only', 'parameter_count': 133701,
    'learning_rate': .001, 'optimizer': 'AdamW', 'weight_decay': .001, 'dropout': .1,
    'tau': .05, 'ce': 'mean_of_eight_candidate_mean_CE',
    'rank': 'mean_of_four_weighted_softplus_losses_including_zero_weight_ties',
    'predicted_T': 'full_retained_reference_count/full_WT_reference_count * sampled_mean_expected_preservation',
    'truth_T': 'full_verified_known_training_labels_only',
    'validation_steps': [500, 1000, 1500, 2000], 'primary_step': 2000,
    'early_stopping': False, 'checkpoint_selection': False, 'initialization': 'from_scratch_same_seed',
    'max_concurrent_gpus': 2, 'gpu': 'NVIDIAA40', 'partition': 'gpu_gub',
    'cpus': 4, 'memory': '48G', 'time_limit': '00:45:00', 'automatic_retry': False,
    'PLM_forward_calls': 0, 'structure_calls': 0, 'variant_structure_files_read': 0,
    'confirmation_access': False, 'paper_changes': False,
    'compute_comparison': 'equal_candidates_relations_updates; C0_has_extra_WT_projection; record_actual_time',
    'primary_analysis': {'comparison': 'D_rank-D0', 'endpoint': 'mixed_selected_T',
        'averaging': 'budgets_then_seeds_within_WT', 'bootstrap_unit': '10_WT',
        'bootstrap_replicates': 10000, 'bootstrap_seed': 20261013,
        'continue_mean_min': .005, 'continue_95pct_lower_strict_min': 0.,
        'continue_positive_seed_count_min': 2, 'continue_source_means_min': 0.,
        'continue_each_mixed_budget_mean_min': -.005},
}


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def identity(value): return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_json(path): return json.loads(Path(path).read_text())


def read_jsonl(path):
    with Path(path).open() as handle: return [json.loads(x) for x in handle if x.strip()]


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_json(handle, value):
    handle.write(canonical(value)+'\n'); handle.flush()


def confined(root, relative):
    root = Path(root).resolve(); result = (root / relative).resolve()
    if root not in result.parents: raise ValueError('path escapes trial root')
    return result


def authorized_root(root):
    root = Path(root).resolve()
    if root != AUTHORIZED_ROOT.resolve(): raise ValueError('unexpected trial root')
    return root


def fresh_output(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    return path


def task_inventory():
    return [dict(mode=mode, seed=seed, **config,
                 output=f'checkpoints/d_ranking_trial/{mode}_s{seed}')
            for mode, config in MODES.items() for seed in SEEDS]


def ranking_auxiliary(pred_a, pred_b, truth_a, truth_b):
    a = torch.as_tensor(truth_a, device=pred_a.device, dtype=pred_a.dtype)
    b = torch.as_tensor(truth_b, device=pred_a.device, dtype=pred_a.dtype)
    gap = a-b
    return (gap.abs()/.05).clamp(max=1.) * F.softplus(-gap.sign()*(pred_a-pred_b)/.05)


def predicted_t(logits, coverage):
    if not 0 < coverage <= 1: raise ValueError('invalid full reference coverage')
    values = logits.new_tensor([1., .75, .5, .25, 0.])
    return float(coverage)*(logits.softmax(-1) @ values).mean()


def validate_pair(a, b, assignment):
    if a['candidate_id'] == b['candidate_id']: raise ValueError('pair candidates must be distinct')
    if any(a[k] != b[k] for k in ('protein_id', 'budget', 'generator')):
        raise ValueError('mixed condition pair forbidden')
    if assignment.get(a['protein_id']) != 'train': raise ValueError('ranking pair must be training WT only')


class PairedSampler:
    """All sampling uses this RNG, never the dropout/global RNG."""
    def __init__(self, rows, assignment, seed):
        self.rng = np.random.default_rng(seed); self.assignment = assignment; self.groups = {}
        seen = set()
        for row in rows:
            if assignment.get(row['protein_id']) != 'train': raise ValueError('sampler accepts train WT only')
            if row['candidate_id'] in seen: raise ValueError('duplicate candidate identity')
            seen.add(row['candidate_id'])
            if row['budget'] not in BUDGETS or row['generator'] not in SOURCES:
                raise ValueError('unfrozen sampling condition')
            self.groups.setdefault(row['protein_id'], {}).setdefault(row['budget'], {}).setdefault(row['generator'], []).append(row)
        self.proteins = sorted(self.groups)
        if not self.proteins: raise ValueError('no training WT')
        for budgets in self.groups.values():
            if set(budgets) != set(BUDGETS): raise ValueError('missing budget condition')
            for sources in budgets.values():
                if set(sources) != set(SOURCES): raise ValueError('missing source condition')
                for candidates in sources.values():
                    candidates.sort(key=lambda r: r['candidate_id'])
                    if len(candidates) != 7: raise ValueError('condition must contain seven candidates')

    def step(self, labels, step):
        proteins = self.rng.choice(self.proteins, 4, replace=len(self.proteins)<4)
        sampled = []; conditions = []
        for pid in proteins:
            budget = float(self.rng.choice(BUDGETS)); source = str(self.rng.choice(SOURCES))
            candidates = self.groups[str(pid)][budget][source]
            a, b = [candidates[int(i)] for i in self.rng.choice(len(candidates), 2, replace=False)]
            validate_pair(a, b, self.assignment)
            pair = []; records = []
            for row in (a, b):
                ids = labels[row['candidate_id']]['pair_ids']
                if not len(ids): raise ValueError('candidate has no known relations')
                selection = self.rng.choice(len(ids), 128, replace=len(ids)<128)
                actual_ids = np.asarray(ids[selection], dtype='<i8')
                records.append(dict(candidate_id=row['candidate_id'], relation_count=128,
                    relation_id_sha256=hashlib.sha256(actual_ids.tobytes()).hexdigest(), replacement=len(ids)<128))
                pair.append((row, selection))
            sampled.append(pair)
            conditions.append(dict(protein_id=str(pid), budget=budget, generator=source, candidates=records))
        return sampled, dict(step=step, relation_updates=1024, conditions=conditions)


def verify_candidate_labels(row, summary, arrays, reference_i, reference_j, kept):
    """Require complete identity-preserving mapping; derive T from full labels."""
    for key in ('candidate_id', 'protein_id', 'source', 'budget'):
        if row[key] != summary.get(key): raise ValueError('label summary identity differs: '+key)
    reference_i = np.asarray(reference_i); reference_j = np.asarray(reference_j); kept = np.asarray(kept)
    retained = np.isin(reference_i, kept) & np.isin(reference_j, kept)
    ids = np.flatnonzero(retained); total = len(reference_i)
    if not total or not len(ids): raise ValueError('empty full reference or retained map')
    expected = dict(kept=kept, reference_i=reference_i, reference_j=reference_j,
                    retained_mask=retained, pair_ids=ids, wt_i=reference_i[ids], wt_j=reference_j[ids],
                    edited_i=np.searchsorted(kept, reference_i[ids]), edited_j=np.searchsorted(kept, reference_j[ids]))
    for key, value in expected.items():
        if key not in arrays or not np.array_equal(arrays[key], value):
            raise ValueError('full retained label mapping failed: '+key)
    if 'unknown_pair_ids' not in arrays or len(arrays['unknown_pair_ids']) or summary.get('unknown_pairs') != 0:
        raise ValueError('unknown labels forbidden for point-T ranking')
    target = arrays['target']
    if (target.shape != ids.shape or not np.issubdtype(target.dtype, np.integer)
            or np.any(target<0) or np.any(target>4)):
        raise ValueError('invalid five-class target')
    contribution = float(np.array([1., .75, .5, .25, 0.])[target].sum())
    derived = dict(reference_pairs=total, retained_pairs=len(ids), known_pairs=len(ids),
        C=len(ids)/total, G_lower=contribution/len(ids), G_upper=contribution/len(ids),
        T_lower=contribution/total, T_upper=contribution/total)
    for key, value in derived.items():
        if key not in summary or not np.isclose(summary[key], value, rtol=0, atol=1e-12):
            raise ValueError('recomputed label summary differs: '+key)
    if summary.get('class_counts') != np.bincount(target, minlength=5).tolist():
        raise ValueError('recomputed label class counts differ')
    return dict(pair_ids=np.asarray(ids, dtype=np.int64), target=np.asarray(target, dtype=np.int64),
                C=derived['C'], T=derived['T_lower'], total=total)


def verify_hashes(root, hashes):
    for relative, expected in hashes.items():
        if sha_file(confined(root, relative)) != expected: raise ValueError('sealed input changed: '+relative)


def sealed_inputs(root):
    """The pinned prior freeze is the authority; fresh rehashes cannot reopen it."""
    root = authorized_root(root)
    old_path = root/OLD_FREEZE
    if sha_file(old_path) != OLD_FREEZE_SHA256: raise ValueError('pinned prospective freeze changed')
    old = read_json(old_path)
    if old['status'] != 'PROSPECTIVE_SEED_REPLICATION_FROZEN': raise ValueError('prior freeze incomplete')
    verify_hashes(root, old['input_hashes'])
    for name in ('prediction_diagnostic', 'decision_diagnostic'):
        item = old[name]
        if sha_file(confined(root, item['report'])) != item['sha256']: raise ValueError('old evidence changed')
    snapshot = confined(root, SNAPSHOT)
    if Path(old['code_snapshot']).resolve() != snapshot: raise ValueError('frozen code path differs')
    actual = {p.relative_to(snapshot).as_posix(): sha_file(p) for p in snapshot.rglob('*.py')}
    if actual != old['code_hashes']: raise ValueError('frozen code identity differs')
    protocol = read_json(root/'configs/protocol_v2.json')
    if (protocol['train']['seeds'] != list(SEEDS) or protocol['model']['dropout'] != .1
            or protocol['train']['weight_decay'] != .001
            or protocol['labels']['reference'] != 'known_wt_reference'
            or protocol['labels']['ca_cutoff_angstrom'] != 15.
            or protocol['labels']['min_sequence_separation'] != 3
            or protocol['labels']['error_thresholds'] != [.5, 1., 2., 4.]
            or protocol['labels']['preservation_values'] != [1., .75, .5, .25, 0.]):
        raise ValueError('original model/label/seed contract differs')
    return old


def load_runtime(root):
    snapshot = confined(root, SNAPSHOT)
    sys.path.insert(0, str(snapshot))
    names = ('common', 'model', 'build_pairs', 'train', 'evaluate')
    runtime = {}
    for name in names:
        module = importlib.import_module(name)
        if Path(module.__file__).resolve().parent != snapshot:
            raise ValueError('runtime module is not the frozen snapshot: '+name)
        runtime[name] = module
    return runtime


def verify_roster(rows, split, protocol_hash):
    if split['status'] != 'SPLIT_FROZEN' or split['protocol_sha256'] != protocol_hash:
        raise ValueError('split identity failed')
    if len(rows) != 2520 or len({r['candidate_id'] for r in rows}) != 2520:
        raise ValueError('primary candidate inventory differs')
    groups = {}; sets = {'train': set(), 'validation': set()}
    for row in rows:
        state = split['assignment'].get(row['protein_id'])
        if state not in sets: raise ValueError('unassigned primary WT')
        if row['source'] != 'e1_exact' or row['budget'] not in BUDGETS or row['generator'] not in SOURCES:
            raise ValueError('primary source/condition differs')
        sets[state].add(row['protein_id'])
        key = (row['protein_id'], row['budget'], row['generator'])
        groups[key] = groups.get(key, 0)+1
    if len(sets['train']) != 50 or len(sets['validation']) != 10 or any(n != 7 for n in groups.values()) or len(groups) != 360:
        raise ValueError('primary WT/condition roster differs')
    train = [r for r in rows if split['assignment'][r['protein_id']] == 'train']
    valid = [r for r in rows if split['assignment'][r['protein_id']] == 'validation']
    if len(valid) != 420: raise ValueError('validation candidate inventory differs')
    return train, valid


def audit_embeddings(root, rows, protocol_hash):
    """Stream identity/shape checks, without loading a PLM or retaining all arrays."""
    index = {}; sequences = {}
    for shard in (0, 1):
        for item in read_jsonl(root/f'caches/embeddings/full_shard{shard}_index.jsonl'):
            key = item['sequence_sha256']
            if key in index and index[key]['key'] != item['key']: raise ValueError('mixed embedding index profile')
            index[key] = item
    for row in rows:
        for name in ('wt_sequence', 'sequence'):
            sequence = row[name]; sequences[hashlib.sha256(sequence.encode()).hexdigest()] = len(sequence)
    profiles = set(); assets = {}
    for key, length in sorted(sequences.items()):
        if key not in index: raise ValueError('missing cached embedding')
        item = index[key]; path = Path(item['path']).resolve()
        if root/'caches/embeddings' not in path.parents: raise ValueError('embedding outside frozen cache')
        meta_path = path.with_suffix('.json'); meta = read_json(meta_path)
        checksum = sha_file(path)
        if meta.get('sequence_sha256') != key or meta.get('protocol_sha256') != protocol_hash or meta.get('output_sha256') != checksum:
            raise ValueError('embedding identity failed')
        for name, value in item.items():
            if name not in ('key', 'path', 'reused') and meta.get(name) != value:
                raise ValueError('embedding index metadata differs')
        a = np.load(path, mmap_mode='r', allow_pickle=False)
        if a.shape != (length, 1152) or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError('invalid cached embedding')
        profile = {k:v for k,v in meta.items() if k not in ('sequence_sha256','length','output_sha256','gpu','batch_seconds')}
        profiles.add(identity(profile))
        assets[path.relative_to(root).as_posix()] = checksum
        assets[meta_path.relative_to(root).as_posix()] = sha_file(meta_path)
    if len(profiles) != 1: raise ValueError('mixed ESMC profiles')
    return dict(sequences=len(sequences), profiles=sorted(profiles),
                assets_sha256=identity(assets), asset_hashes=assets)


def make_pinned_inputs(base, cache_indices, protocol_hash, root, embedding_audit):
    """Extend frozen Inputs with immutable prepare-time pins on first load."""
    root = Path(root).resolve(); pins = dict(embedding_audit.get('asset_hashes', {}))
    if not pins or identity(pins) != embedding_audit.get('assets_sha256'):
        raise ValueError('embedding asset pins missing or inconsistent')

    class PinnedInputs(base):
        def __init__(self):
            super().__init__(cache_indices, protocol_hash)
            self.pinned_sequences = set()

        def embedding(self, sequence):
            key = hashlib.sha256(sequence.encode()).hexdigest()
            if key in self.pinned_sequences:
                return super().embedding(sequence)
            path = Path(self.index[key]['path']).resolve()
            paths = (path, path.with_suffix('.json'))
            expected = []
            for asset in paths:
                if root not in asset.parents: raise ValueError('embedding asset escapes pinned root')
                pin = pins.get(asset.relative_to(root).as_posix())
                if not pin: raise ValueError('embedding asset pin missing')
                expected.append(pin)
            try:
                for asset, pin in zip(paths, expected):
                    if sha_file(asset) != pin: raise ValueError('prepare-time embedding asset pin differs')
                # Preserve the original protocol/index, shape, finite-value and
                # profile checks. Recheck the pins after its actual file read.
                result = super().embedding(sequence)
                for asset, pin in zip(paths, expected):
                    if sha_file(asset) != pin: raise ValueError('embedding asset changed during first load')
            except BaseException:
                self.arrays.pop(key, None)
                raise
            self.pinned_sequences.add(key)
            return result

    return PinnedInputs()


def verify_native_metadata(row, native, mapping):
    sequence_hash = hashlib.sha256(row['wt_sequence'].encode()).hexdigest()
    if (row['wt_sequence_sha256'] != sequence_hash or native.get('protein_id') != row['protein_id']
            or native.get('wt_sequence_sha256') != sequence_hash
            or native.get('reference_path') != row['reference_path']
            or native.get('reference_sha256') != row['reference_sha256']):
        raise ValueError('native reference metadata identity failed')
    if (mapping.get('sequence_sha256') != sequence_hash or mapping.get('path') != row['reference_path']
            or mapping.get('sha256') != row['reference_sha256']
            or mapping.get('chain') != row['reference_auth_chain']):
        raise ValueError('WT mapping metadata identity failed')
    # A SCISOR JSONL may differ from the group that produced the native export;
    # require its native source identity through the matching sealed occurrence.
    # Native group files are the WT authority even when a candidate's source is
    # a separate SCISOR roster. Their own hash is also checked in preflight.
    if not native.get('source_group_path') or not native.get('source_group_sha256'):
        raise ValueError('native WT source group identity missing')
    coordinates = native['coordinates']; indices = sorted(int(k) for k in coordinates)
    if (len(indices) != len(set(indices)) or any(i<0 or i>=len(row['wt_sequence']) for i in indices)
            or native.get('CA_count') != len(indices)
            or any(np.asarray(coordinates[str(i)]).shape != (3,) or not np.isfinite(coordinates[str(i)]).all() for i in indices)):
        raise ValueError('invalid native coordinate mapping')
    mapped = sorted(r['sequence_index'] for r in mapping['residue_map'])
    if mapped != indices or sorted(mapping['missing_positions']) != sorted(set(range(len(row['wt_sequence'])))-set(indices)):
        raise ValueError('native coordinate and sequence-index mapping differ')


def preflight(root, sealed_embeddings=None):
    """Complete CPU audit of the primary data; never opens variant structures."""
    root = authorized_root(root); old = sealed_inputs(root); runtime = load_runtime(root)
    protocol_hash = sha_file(root/'configs/protocol_v2.json')
    protocol = read_json(root/'configs/protocol_v2.json')
    execution = read_json(root/'configs/execution_freeze.json')
    label_audit_path = root/'reports/label_audit.json'; label_audit = read_json(label_audit_path)
    if (sha_file(label_audit_path) != execution['input_hashes']['reports/label_audit.json']
            or label_audit.get('status') != 'LABELS_VERIFIED' or label_audit.get('failures') != []
            or label_audit.get('protocol_sha256') != protocol_hash
            or label_audit.get('manifest_sha256') != execution['input_hashes']['manifests/development.jsonl']):
        raise ValueError('original label audit identity/status differs')
    rows = read_jsonl(root/'manifests/development_primary.jsonl'); split = read_json(root/'manifests/splits.json')
    train, valid = verify_roster(rows, split, protocol_hash)
    summaries_list = read_jsonl(root/'reports/label_candidates.jsonl')
    summaries = {r['candidate_id']: r for r in summaries_list}
    if len(summaries) != len(summaries_list): raise ValueError('duplicate label summary identity')
    references = {}; labels = {}; asset_hashes = {}
    for row in rows:
        pid = row['protein_id']
        if (hashlib.sha256(row['sequence'].encode()).hexdigest() != row['sequence_sha256']
                or hashlib.sha256(''.join('1' if d else '0' for d in row['mask']).encode()).hexdigest() != row['mask_sha256']
                or identity((row['wt_sequence_sha256'], row['mask_sha256'], row['sequence_sha256'])) != row['candidate_id']):
            raise ValueError('candidate sequence/mask identity failed')
        # Inputs checks these identities on first use; preflight checks every row.
        wt_path = Path(row['wt_native_json']); ref_path = Path(row['reference_path'])
        for path, expected in ((wt_path, row['wt_native_json_sha256']), (ref_path, row['reference_sha256']),
                               (Path(row['source_path']), row['source_sha256'])):
            key = str(path)
            if key not in asset_hashes: asset_hashes[key] = sha_file(path)
            if asset_hashes[key] != expected: raise ValueError('WT reference identity failed')
        if pid not in references:
            native = read_json(wt_path)
            mapping_path = root/f'caches/wt_references/{pid}.mapping.json'
            mapping = read_json(mapping_path); verify_native_metadata(row, native, mapping)
            asset_hashes[mapping_path.relative_to(root).as_posix()] = sha_file(mapping_path)
            source_group = Path(native['source_group_path']); source_group_hash = sha_file(source_group)
            if source_group_hash != native['source_group_sha256']: raise ValueError('native source group changed')
            asset_hashes[str(source_group)] = source_group_hash
            coords = np.full((len(row['wt_sequence']), 3), np.nan)
            for key, value in native['coordinates'].items(): coords[int(key)] = value
            wi, wj = runtime['build_pairs'].reference_pairs(coords, protocol['labels']['ca_cutoff_angstrom'],
                                                           protocol['labels']['min_sequence_separation'])
            references[pid] = (wi, wj, row['wt_sequence'], str(wt_path), str(ref_path), native, mapping)
        wi, wj, sequence, native_path, reference_path, native, mapping = references[pid]
        if sequence != row['wt_sequence'] or native_path != str(wt_path) or reference_path != str(ref_path):
            raise ValueError('inconsistent WT reference within primary roster')
        verify_native_metadata(row, native, mapping)
        kept = runtime['train'].retained_mapping(row['wt_sequence'], row['sequence'], row['mask'])
        cid = row['candidate_id']; summary = summaries[cid]; path = root/'labels/pairs'/(cid+'.npz')
        checksum = sha_file(path)
        if checksum != summary['labels_sha256']: raise ValueError('label NPZ identity failed: '+cid)
        with np.load(path, allow_pickle=False) as z:
            arrays = {key:z[key] for key in ('kept','reference_i','reference_j','retained_mask',
                'pair_ids','wt_i','wt_j','edited_i','edited_j','target','unknown_pair_ids')}
        labels[cid] = verify_candidate_labels(row, summary, arrays, wi, wj, kept)
        asset_hashes[path.relative_to(root).as_posix()] = checksum
    # Formal runs reuse the complete prepare-time scan. PinnedInputs compares
    # both files to its sealed per-asset hashes around the original load checks.
    embeddings = audit_embeddings(root, rows, protocol_hash) if sealed_embeddings is None else sealed_embeddings
    audit = dict(status='D_RANKING_CPU_PREFLIGHT_COMPLETE', candidates=len(rows),
        train_wt=len({r['protein_id'] for r in train}), validation_wt=len({r['protein_id'] for r in valid}),
        validation_candidates=len(valid), unknown_relations=0, label_and_reference_assets_sha256=identity(asset_hashes),
        embeddings=embeddings, label_audit_sha256=sha_file(label_audit_path),
        frozen_code_hashes=old['code_hashes'], old_freeze_sha256=OLD_FREEZE_SHA256,
        PLM_forward_calls=0, structure_calls=0, variant_structure_files_read=0, confirmation_access=False)
    return dict(root=root, old=old, runtime=runtime, rows=rows, train=train, valid=valid,
                split=split, labels=labels, summaries=summaries, audit=audit)


def prepare_freeze(root):
    bundle = preflight(root); root = bundle['root']; path = root/NEW_FREEZE
    result = dict(status='D_RANKING_TRIAL_FROZEN', scientific_scope='bounded_development_ranking_only',
        contract=CONTRACT, tasks=task_inventory(), prior_freeze_sha256=OLD_FREEZE_SHA256,
        input_hashes=bundle['old']['input_hashes'], frozen_code_hashes=bundle['old']['code_hashes'],
        code_snapshot=str(root/SNAPSHOT), source_hashes={p:sha_file(root/p) for p in NEW_SOURCES},
        document_hashes={p:sha_file(root/p) for p in DOCS}, preflight=bundle['audit'])
    # This is an immutable, single creation. Repeated preparation never overwrites it.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n')
        handle.flush(); os.fsync(handle.fileno())
    return dict(status=result['status'], path=str(path), sha256=sha_file(path), preflight=bundle['audit'])


def checked_freeze(root, freeze_hash):
    root = authorized_root(root); path = root/NEW_FREEZE
    if (not freeze_hash or len(freeze_hash) != 64 or sha_file(path) != freeze_hash):
        raise ValueError('explicit trial freeze identity differs')
    freeze = read_json(path)
    if (freeze.get('status') != 'D_RANKING_TRIAL_FROZEN' or freeze.get('contract') != CONTRACT
            or freeze.get('tasks') != task_inventory()
            or freeze.get('scientific_scope') != 'bounded_development_ranking_only'):
        raise ValueError('trial contract/inventory differs')
    if freeze.get('prior_freeze_sha256') != OLD_FREEZE_SHA256: raise ValueError('prior freeze identity differs')
    if set(freeze.get('source_hashes', {})) != set(NEW_SOURCES) or set(freeze.get('document_hashes', {})) != set(DOCS):
        raise ValueError('new source/document identity inventory incomplete')
    if Path(__file__).resolve() != root/'analysis/d_ranking_trial.py':
        raise ValueError('runner must execute from the frozen trial source path')
    verify_hashes(root, freeze['source_hashes']); verify_hashes(root, freeze['document_hashes'])
    old = sealed_inputs(root)
    if (freeze['input_hashes'] != old['input_hashes'] or freeze['frozen_code_hashes'] != old['code_hashes']
            or freeze['code_snapshot'] != str(root/SNAPSHOT)):
        raise ValueError('old sealed execution identities differ')
    return freeze


def checked_task(root, freeze_hash, task_index):
    freeze = checked_freeze(root, freeze_hash)
    if type(task_index) is not int or not 0 <= task_index < 9: raise ValueError('task index outside nine-task inventory')
    task = freeze['tasks'][task_index]
    if confined(root, task['output']).exists(): raise ValueError('trial output already exists; no duplicate/overwrite')
    return task, freeze


def ledger_directory(root, freeze_hash):
    if len(freeze_hash) != 64 or any(c not in '0123456789abcdef' for c in freeze_hash):
        raise ValueError('invalid ledger freeze identity')
    directory = confined(root, f'ledger/d_ranking_trial/{freeze_hash}')
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def exclusive_json(path, value):
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, 'w') as handle:
        handle.write(canonical(value)+'\n'); handle.flush(); os.fsync(handle.fileno())


def claim_task(root, freeze_hash, task_index):
    if type(task_index) is not int or not 0 <= task_index < 9: raise ValueError('invalid task claim index')
    path = ledger_directory(root, freeze_hash)/f'{task_index}.claim'
    exclusive_json(path, dict(task_index=task_index, slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                              slurm_array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID')))
    return path


def claim_array(root, freeze_hash, array_id):
    if not str(array_id).isdigit(): raise ValueError('explicit Slurm array ID required')
    directory = ledger_directory(root, freeze_hash)
    descriptor, temporary = tempfile.mkstemp(dir=directory, prefix='.array-owner-')
    temporary = Path(temporary); path = directory/'array_owner.json'
    try:
        with os.fdopen(descriptor, 'w') as handle:
            handle.write(canonical({'slurm_array_job_id':str(array_id)})+'\n')
            handle.flush(); os.fsync(handle.fileno())
        try: os.link(temporary, path)
        except FileExistsError: pass
        if read_json(path)['slurm_array_job_id'] != str(array_id):
            raise RuntimeError('different array already owns this trial freeze')
    finally: temporary.unlink(missing_ok=True)


def submit_array(root, freeze_hash):
    root = authorized_root(root)
    freeze = checked_freeze(root, freeze_hash)
    directory = ledger_directory(root, freeze_hash); claim = directory/'submission.claim'; receipt = directory/'submission.json'
    if claim.exists():
        return read_json(receipt) if receipt.exists() else dict(status='SUBMISSION_ALREADY_ATTEMPTED_OUTCOME_UNKNOWN', freeze_sha256=freeze_hash)
    for task in freeze['tasks']:
        if confined(root, task['output']).exists(): raise ValueError('trial output already exists')
    # Recheck every primary NPZ/reference/source before submission, reusing the
    # complete embedding audit sealed by prepare. Frozen Inputs revalidates each
    # embedding's pinned file/sidecar hashes, shape and profile on first load.
    if preflight(root, sealed_embeddings=freeze['preflight']['embeddings'])['audit'] != freeze['preflight']:
        raise ValueError('primary preflight identity changed')
    try: exclusive_json(claim, dict(freeze_sha256=freeze_hash))
    except FileExistsError:
        return read_json(receipt) if receipt.exists() else dict(status='SUBMISSION_ALREADY_ATTEMPTED_OUTCOME_UNKNOWN', freeze_sha256=freeze_hash)
    command = ['/opt/gridview/slurm/bin/sbatch', '--parsable', '--array=0-8%2',
        '--partition=gpu_gub', '--gres=gpu:NVIDIAA40:1', '--cpus-per-task=4', '--mem=48G',
        '--time=00:45:00', '--nodes=1', '--ntasks=1', '--no-requeue',
        '--export=ALL,D_RANKING_FREEZE_SHA256='+freeze_hash, str(root/'slurm/d_ranking_trial.sbatch')]
    env = {k:v for k,v in os.environ.items() if not k.startswith('SBATCH_')}
    env['D_RANKING_FREEZE_SHA256'] = freeze_hash
    result = dict(status='SUBMISSION_ATTEMPT_STARTED', freeze_sha256=freeze_hash, command=command)
    write_json(receipt, result)
    try:
        response = subprocess.run(command, check=True, text=True, capture_output=True, env=env)
        job_id = response.stdout.strip().split(';')[0]
        if not job_id.isdigit(): raise ValueError('unparseable sbatch outcome; do not resubmit')
        claim_array(root, freeze_hash, job_id)
        result.update(status='D_RANKING_TRIAL_SUBMITTED', job_id=job_id)
    except Exception as error:
        result.update(status='SUBMISSION_ERROR_OR_UNKNOWN_NO_AUTOMATIC_RETRY', error=repr(error))
        write_json(receipt, result); raise
    write_json(receipt, result)
    return result


def submission_receipt(root, freeze_hash, array_id, timeout=30.):
    """Allow a fast first task to wait for its controller's atomic receipt."""
    directory = ledger_directory(root, freeze_hash); deadline = time.monotonic()+timeout
    if not (directory/'submission.claim').is_file(): raise ValueError('missing one-shot submission claim')
    while True:
        receipt = read_json(directory/'submission.json')
        if receipt.get('status') != 'SUBMISSION_ATTEMPT_STARTED': break
        if time.monotonic() >= deadline: raise ValueError('submission receipt is still pending; no automatic retry')
        time.sleep(.05)
    if (receipt.get('status') != 'D_RANKING_TRIAL_SUBMITTED' or receipt.get('freeze_sha256') != freeze_hash
            or receipt.get('job_id') != str(array_id)):
        raise ValueError('Slurm allocation has no matching one-shot submission receipt')
    return receipt


def memory_mib(value):
    match = re.fullmatch(r'(\d+(?:\.\d+)?)([KMGT])', value.upper())
    if not match: raise ValueError('unsupported Slurm memory unit')
    scale = {'K': Decimal(1)/1024, 'M': Decimal(1), 'G': Decimal(1024), 'T': Decimal(1024)**2}
    return Decimal(match.group(1))*scale[match.group(2)]


def validate_slurm_resources(fields):
    """Check effective allocation across modern ReqTRES and legacy TRES."""
    tres = dict(token.split('=', 1) for token in
                (fields.get('ReqTRES') or fields.get('TRES', '')).split(',') if '=' in token)
    gpu_counts = [value for key, value in tres.items() if key == 'gres/gpu' or key.startswith('gres/gpu:')]
    typed_gpu_keys = [key for key in tres if key.startswith('gres/gpu:')]
    if (fields.get('Partition') != 'gpu_gub' or fields.get('TimeLimit') != '00:45:00'
            or fields.get('NumCPUs') != '4' or fields.get('NumNodes') != '1'
            or fields.get('Requeue') != '0' or tres.get('cpu') != '4' or tres.get('node') != '1'
            or not gpu_counts or any(count != '1' for count in gpu_counts) or len(typed_gpu_keys) > 1
            or memory_mib(tres.get('mem', '')) != Decimal(48*1024)):
        raise ValueError('effective Slurm resource contract differs')


def checked_slurm(root, freeze_hash, task_index):
    env = os.environ
    if not env.get('SLURM_JOB_ID', '').isdigit() or not env.get('SLURM_ARRAY_JOB_ID', '').isdigit():
        raise RuntimeError('formal training requires an explicit Slurm array allocation')
    if env.get('SLURM_ARRAY_TASK_ID') != str(task_index): raise ValueError('Slurm task index differs')
    if (env.get('SLURM_JOB_PARTITION') != 'gpu_gub' or env.get('SLURM_CPUS_PER_TASK') != '4'
            or env.get('SLURM_ARRAY_TASK_COUNT') != '9' or env.get('SLURM_RESTART_COUNT', '0') != '0'):
        raise ValueError('Slurm resource/array/restart contract differs')
    submission_receipt(root, freeze_hash, env['SLURM_ARRAY_JOB_ID'])
    # Verify effective allocation as well as the immutable requested resources.
    response = subprocess.run(['/opt/gridview/slurm/bin/scontrol', 'show', 'job', '-o', env['SLURM_JOB_ID']],
                              check=True, capture_output=True, text=True)
    fields = dict(token.split('=', 1) for token in response.stdout.split() if '=' in token)
    validate_slurm_resources(fields)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or 'A40' not in torch.cuda.get_device_name(0):
        raise RuntimeError('formal task requires exactly one A40 GPU')
    claim_array(root, freeze_hash, env['SLURM_ARRAY_JOB_ID'])
    return dict(environment={k:v for k,v in env.items() if k.startswith('SLURM_')}, job_fields=fields,
                gpu_name=torch.cuda.get_device_name(0))


def state_sha256(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def train_trial(bundle, mode, seed, output, technical=False, freeze_hash=None, slurm=None):
    """Two-step CPU check or exactly 2,000 steps; validation never selects weights."""
    root = bundle['root']; runtime = bundle['runtime']; output = Path(output)
    if mode not in MODES or seed not in SEEDS: raise ValueError('unfrozen mode or seed')
    technical_base = root/'technical/d_ranking_trial'
    formal_output = root/f'checkpoints/d_ranking_trial/{mode}_s{seed}'
    if technical:
        if technical_base not in output.resolve().parents: raise ValueError('technical output must be separate')
        device = 'cpu'; steps = 2
        # Exactly two training WTs, preserving every condition for each.
        proteins = sorted({r['protein_id'] for r in bundle['train']})[:2]
        train = [r for r in bundle['train'] if r['protein_id'] in proteins]
        valid = bundle['valid'][:1]
    else:
        if output.resolve() != formal_output.resolve() or freeze_hash is None or slurm is None:
            raise ValueError('formal execution identity missing')
        device = 'cuda'; steps = 2000; train = bundle['train']; valid = bundle['valid']
    torch.set_num_threads(4)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device == 'cuda': torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    model = runtime['model'].PairResponseHead()
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != 133701: raise ValueError('original head parameter count changed')
    initialization_sha256 = state_sha256(model); model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    sampler = PairedSampler(train, bundle['split']['assignment'], seed)
    inputs = make_pinned_inputs(runtime['train'].Inputs,
        [root/f'caches/embeddings/full_shard{s}_index.jsonl' for s in (0, 1)],
        sha_file(root/'configs/protocol_v2.json'), root, bundle['audit']['embeddings'])
    config = MODES[mode]; arm = config['arm']; coefficient = config['lambda']
    counts = {'training': {'project_calls': 0, 'project_residues': 0},
              'validation': {'project_calls': 0, 'project_residues': 0}}
    phase = ['training']
    def projection_hook(module, args, result):
        counts[phase[0]]['project_calls'] += 1
        counts[phase[0]]['project_residues'] += len(args[0])
    hook = model.project.register_forward_hook(projection_hook)
    fresh_output(output)
    provenance = dict(status='CPU_TECHNICAL_RUNNING' if technical else 'D_RANKING_TRIAL_RUNNING',
        technical=technical, mode=mode, arm=arm, seed=seed, ranking_lambda=coefficient,
        freeze_sha256=freeze_hash, contract=CONTRACT, input_audit=bundle['audit'],
        parameter_count=parameter_count, initialization_sha256=initialization_sha256,
        new_source_hashes={p:sha_file(root/p) for p in NEW_SOURCES},
        slurm=slurm, primary_step=None if technical else 2000,
        training_wt=sorted({r['protein_id'] for r in train}),
        validation_wt=sorted({r['protein_id'] for r in valid}),
        PLM_forward_calls=0, structure_calls=0, variant_structure_files_read=0)
    write_json(output/'run.json', provenance)
    write_json(output/'status.json', dict(status=provenance['status'], steps=0, technical=technical))
    elapsed = {'training_device_seconds': 0., 'validation_device_seconds': 0.,
               'training_wall_seconds': 0., 'validation_wall_seconds': 0.}
    validations = []; started = time.perf_counter()
    try:
        with (output/'sampling_trace.jsonl').open('x') as sampling_trace, (output/'train_trace.jsonl').open('x') as loss_trace:
            for step in range(1, steps+1):
                phase[0] = 'training'; model.train(); optimizer.zero_grad(set_to_none=True)
                wall_start = time.perf_counter()
                if device == 'cuda':
                    event_start, event_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    event_start.record()
                sampled, ledger = sampler.step(bundle['labels'], step)
                ces = []; ranks = []
                for pair in sampled:
                    predictions = []; truths = []
                    for row, selection in pair:
                        data, ids, total = inputs.candidate(row); lab = bundle['labels'][row['candidate_id']]
                        if (total != lab['total'] or not np.array_equal(ids, lab['pair_ids'])
                                or not np.array_equal(data['kept'], runtime['train'].retained_mapping(row['wt_sequence'], row['sequence'], row['mask']))):
                            raise ValueError('sampled labels differ from WT-only input full relation map')
                        # Pair endpoint indices must invert to the exact verified
                        # WT reference IDs, not merely have the same array length.
                        wi, wj = inputs.references[row['protein_id']][1]
                        if (not np.array_equal(data['kept'][data['i']], wi[ids])
                                or not np.array_equal(data['kept'][data['j']], wj[ids])):
                            raise ValueError('sampled WT-only endpoint mapping differs')
                        prediction = runtime['train'].logits(model, data, arm, device, selection)
                        target = torch.as_tensor(lab['target'][selection], device=device, dtype=torch.long)
                        ces.append(F.cross_entropy(prediction, target))
                        predictions.append(predicted_t(prediction, lab['C'])); truths.append(lab['T'])
                    ranks.append(ranking_auxiliary(predictions[0], predictions[1], truths[0], truths[1]))
                ce = torch.stack(ces).mean(); rank = torch.stack(ranks).mean(); loss = ce+coefficient*rank
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite training loss')
                loss.backward()
                grad_squared = sum(p.grad.detach().square().sum() for p in model.parameters() if p.grad is not None)
                grad_norm = float(grad_squared.sqrt())
                if not np.isfinite(grad_norm): raise FloatingPointError('nonfinite training gradient')
                optimizer.step()
                if device == 'cuda':
                    event_end.record(); event_end.synchronize()
                    elapsed['training_device_seconds'] += event_start.elapsed_time(event_end)/1000
                else: elapsed['training_device_seconds'] += time.perf_counter()-wall_start
                elapsed['training_wall_seconds'] += time.perf_counter()-wall_start
                append_json(sampling_trace, ledger)
                append_json(loss_trace, dict(step=step, ce=float(ce.detach()), ranking=float(rank.detach()),
                    loss=float(loss.detach()), gradient_norm=grad_norm, relation_updates=1024))
                if step == 1 or step%100 == 0 or step == steps:
                    print(canonical(dict(stage='TRAINING', mode=mode, seed=seed, step=step,
                        loss=float(loss.detach()), gradient_norm=grad_norm)), flush=True)
                    write_json(output/'status.json', dict(status=provenance['status'], steps=step,
                        technical=technical, latest_loss=float(loss.detach()), latest_gradient_norm=grad_norm))
                if not technical and step in (500, 1000, 1500, 2000):
                    phase[0] = 'validation'; wall_start = time.perf_counter()
                    event_start.record()
                    report = runtime['evaluate'].evaluate(model, valid, inputs, bundle['summaries'],
                        root/'labels/pairs', arm, device, output/f'validation_{step}')
                    event_end.record(); event_end.synchronize()
                    elapsed['validation_device_seconds'] += event_start.elapsed_time(event_end)/1000
                    elapsed['validation_wall_seconds'] += time.perf_counter()-wall_start
                    validations.append(dict(step=step, hierarchical_ce=report['hierarchical_ce'], primary=step==2000))
                    write_json(output/'validation_trace.json', validations)
        if technical:
            phase[0] = 'validation'; wall_start = time.perf_counter()
            report = runtime['evaluate'].evaluate(model, valid, inputs, bundle['summaries'],
                root/'labels/pairs', arm, device, output/'validation_technical_2')
            elapsed['validation_wall_seconds'] = time.perf_counter()-wall_start
            elapsed['validation_device_seconds'] = elapsed['validation_wall_seconds']
            validations.append(dict(step=2, hierarchical_ce=report['hierarchical_ce'], primary=False,
                                    technical_only=True, candidates=1))
        checkpoint = dict(model=model.state_dict(), step=steps, arm=arm, mode=mode,
            provenance=provenance, esmc_profile_hashes=sorted(inputs.profiles),
            validation_ce=None if technical else validations[-1]['hierarchical_ce'])
        torch.save(checkpoint, output/'final.pt')
        write_json(output/'status.json', dict(status='CPU_TECHNICAL_COMPLETE' if technical else 'D_RANKING_TRIAL_COMPLETE',
            steps=steps, technical=technical, mode=mode, seed=seed, parameter_count=parameter_count,
            relation_updates=steps*1024, candidate_forwards=steps*8, validations=validations,
            primary_step=None if technical else 2000, checkpoint_selection=False, early_stopping=False,
            sampling_trace_sha256=sha_file(output/'sampling_trace.jsonl'),
            initialization_sha256=initialization_sha256, final_state_sha256=state_sha256(model),
            actual_projection_counts=counts, elapsed=elapsed, total_wall_seconds=time.perf_counter()-started,
            actual_gpu_seconds=0. if technical else elapsed['training_device_seconds']+elapsed['validation_device_seconds'],
            freeze_sha256=freeze_hash, esmc_profile_hashes=sorted(inputs.profiles)))
    except BaseException as error:
        write_json(output/'failure.json', dict(status='FAILED_NO_AUTOMATIC_RETRY', error=repr(error),
                                             technical=technical, elapsed=elapsed))
        raise
    finally: hook.remove()
    return read_json(output/'status.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--freeze-sha256'); parser.add_argument('--task-index', type=int)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare-freeze', action='store_true')
    action.add_argument('--submit-array', action='store_true')
    action.add_argument('--execute', action='store_true')
    action.add_argument('--technical', action='store_true')
    parser.add_argument('--mode', choices=MODES, default='D_rank')
    parser.add_argument('--seed', type=int, choices=SEEDS, default=SEEDS[0])
    parser.add_argument('--technical-output', type=Path)
    args = parser.parse_args(); root = authorized_root(args.root)
    if not args.technical and (args.technical_output is not None or args.mode != 'D_rank' or args.seed != SEEDS[0]):
        parser.error('mode/seed/technical-output overrides are only for CPU technical checks')
    if args.prepare_freeze: result = prepare_freeze(root)
    elif args.submit_array: result = submit_array(root, args.freeze_sha256)
    elif args.technical:
        if args.freeze_sha256 is not None or args.task_index is not None:
            parser.error('technical checks do not use a formal task/freeze')
        output = args.technical_output or root/f'technical/d_ranking_trial/{args.mode}_s{args.seed}'
        result = train_trial(preflight(root), args.mode, args.seed, output, technical=True)
    else:
        task, freeze = checked_task(root, args.freeze_sha256, args.task_index)
        slurm = checked_slurm(root, args.freeze_sha256, args.task_index)
        claim_task(root, args.freeze_sha256, args.task_index)
        bundle = preflight(root, sealed_embeddings=freeze['preflight']['embeddings'])
        if bundle['audit'] != freeze['preflight']: raise ValueError('formal primary preflight identity changed')
        print(canonical(dict(stage='D_RANKING_TASK_VALIDATED', task=task, slurm=slurm)), flush=True)
        result = train_trial(bundle, task['mode'], task['seed'], root/task['output'],
                             freeze_hash=args.freeze_sha256, slurm=slurm)
    print(canonical(result), flush=True)


if __name__ == '__main__': main()
