"""CPU-only prospective seed audit and equal-probability three-seed ensemble.

Explicit --root, --freeze and --output are required. Completed artifacts are
validated even while other tasks are pending. Checkpoints are opaque bytes;
neither torch nor the GPU evaluator is imported. No seed or LR is selected here.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import development_review as core
import cpu_prediction_diagnostics as prediction
import review_scalar_baselines as scalar
from replication_task import first_seed_identities

common = core.common
SEEDS = (20261013, 20261014, 20261015)
ARMS = 'SABCD'
WAITING = 'REPLICATION_REVIEW_WAITING'
READY = 'THREE_SEED_REPLICATION_REVIEW_READY'
ARTIFACTS = ('run.json', 'status.json', 'train_trace.jsonl', 'validation_trace.jsonl')
EVALUATION_FILES = ('candidates.jsonl', 'proteins.jsonl', 'decisions.jsonl', 'report.json')
INTERPRETATION = (
    'Historical primary-validation training-seed stability analysis. Seed 20261013 and its '
    'LR choices were already inspected; seeds 20261014/20261015 prospectively use LR 0.001 '
    'for every arm. Model selection and effect estimation share validation data. All three '
    'seeds and all C-B/C-S/C-A/C-D results are retained. The same equal arithmetic mean of '
    'aligned five-class pair probabilities is used for S/A/B/C/D before CE, Ghat, That and '
    'candidate selection. This is not new independent WT evidence or confirmation. '
    'Bootstrap intervals describe WT/component uncertainty conditional on the fitted '
    'models; they do not quantify validation model-selection or training-seed uncertainty.')


def protected_output(root, output, inputs=()):
    """Reject symlink/reparse destinations, input overlap and other report trees."""
    root = Path(root).resolve()
    output = Path(output).absolute()
    dedicated = root / 'reports/replication_review'
    core.require(output == dedicated or dedicated in output.parents,
                 'output must stay within reports/replication_review')
    current = output
    while current != root and root in current.parents:
        core.require(not current.is_symlink() and not (hasattr(current, 'is_junction') and current.is_junction()),
                     'linked output directory is forbidden')
        current = current.parent
    core.require(output.resolve() == output, 'output resolves through a link')
    prediction.protected_output(root, output, inputs)
    # Existing file links must not survive as alternate write/read targets.
    if output.exists():
        for path in output.rglob('*'):
            core.require(not path.is_symlink() and not (hasattr(path, 'is_junction') and path.is_junction()),
                         'linked output entry is forbidden')
            if path.is_file():
                core.require(path.stat().st_nlink == 1, 'hard-linked output entry is forbidden')
    return output


def input_path(root, relative, output):
    path = Path(root) / relative
    core.require(Path(root) in path.resolve().parents, 'input escapes experiment root: ' + str(relative))
    scalar.reject_input_overlap(output, [path])
    return path


def checked_hash(path, expected, name):
    return prediction.checked_hash(path, expected, name)


def sealed_first_seed_predictions(report_path, report, learned, valid, root, output):
    """Recover original prediction byte identities through the frozen CPU report."""
    ledger_path = input_path(root, Path(report_path).parent / 'artifact_ledger.jsonl', output)
    checksum = checked_hash(ledger_path, report['artifact_sha256']['artifact_ledger.jsonl'],
                            'sealed first-seed prediction ledger')
    rows = [r for r in common.read_jsonl(ledger_path) if r.get('kind') == 'prediction']
    observed = core.keyed(rows, lambda r: (r['arm'], r['candidate_id']), 'sealed first-seed prediction')
    expected = {(arm, row['candidate_id']) for arm in ARMS for row in valid}
    core.equal(set(observed), expected, 'sealed first-seed prediction roster')
    for (arm, cid), row in observed.items():
        path = Path(learned['selected'][arm]['validation_output']) / (cid + '.npz')
        core.equal(row['path'], str(path), 'sealed first-seed prediction path')
        core.require(root in path.resolve().parents, 'sealed first-seed prediction escapes root')
        scalar.reject_input_overlap(output, [path])
        core.require(len(row['sha256']) == 64 and all(c in '0123456789abcdef' for c in row['sha256']),
                     'invalid sealed first-seed prediction digest')
    checked_hash(ledger_path, checksum, 'sealed first-seed prediction ledger after read')
    return observed, checksum


def read_context(root, freeze_path, output):
    """Validate the original seal and prospective freeze without launcher side effects."""
    root = Path(root).resolve()
    core.equal(root, common.AUTHORIZED_ROOT.resolve(), 'authorized experiment root')
    freeze_path = Path(freeze_path).resolve()
    core.equal(freeze_path, root / 'configs/prospective_replication_freeze.json', 'replication freeze path')
    output = protected_output(root, output, [freeze_path])
    freeze = core.read_json(freeze_path)
    core.equal(freeze['status'], 'PROSPECTIVE_SEED_REPLICATION_FROZEN', 'replication freeze status')
    for name, value in {'first_seed_inspected': SEEDS[0], 'prospective_seeds': list(SEEDS[1:]),
                        'scientific_scope': 'training_seed_stability_only_no_confirmation_folds',
                        'selected_learning_rates': {a: .001 for a in ARMS},
                        'confirmation_folding_authorized': False}.items():
        core.equal(freeze[name], value, 'freeze ' + name)
    required = {'configs/protocol_v2.json', 'configs/execution_freeze.json',
                'configs/development_diagnostics_first_seed.json', 'manifests/development_primary.jsonl',
                'manifests/splits.json', 'reports/label_candidates.jsonl',
                'caches/embeddings/full_shard0_index.jsonl', 'caches/embeddings/full_shard1_index.jsonl',
                'reports/dispatch_after_pilot.json', 'reports/development_review/report.json',
                'reports/cpu_development_diagnostics/reference_coordinate_coverage.json',
                'reports/prospective_replication_decision.json'}
    core.require(required <= set(freeze['input_hashes']), 'incomplete replication input seal')
    for relative, checksum in freeze['input_hashes'].items():
        checked_hash(input_path(root, relative, output), checksum, 'freeze input ' + relative)
    checked_hash(root / 'analysis/replication_task.py', freeze['implementation_sha256'], 'replication launcher implementation')
    checked_hash(root / 'slurm/prospective_replication.sbatch', freeze['launcher_sha256'], 'replication Slurm launcher')
    learned, identities = first_seed_identities(root)
    checked_hash(root / 'analysis/development_review.py', learned['identities']['review_code_sha256'], 'original core implementation')
    diagnostic_reports = {}
    for name, status in [('prediction_diagnostic', 'CPU_PREDICTION_DIAGNOSTICS_COMPLETE'),
                         ('decision_diagnostic', 'CPU_DECISION_DIAGNOSTICS_COMPLETE')]:
        evidence = freeze[name]
        path = input_path(root, evidence['report'], output)
        checked_hash(path, evidence['sha256'], name)
        report = core.read_json(path)
        core.equal(evidence['required_status'], status, name + ' required status')
        core.equal(report['status'], status, name + ' status')
        for field, value in {'validation_candidates': 420, 'validation_wt_components': learned['validation_wt_components'],
                             'seed': SEEDS[0], 'structure_calls': 0, 'PLM_forward_calls': 0,
                             'GPU_used': False, 'variant_structure_files_read': 0}.items():
            core.equal(report[field], value, name + ' ' + field)
        for field, value in identities.items():
            core.equal(report['identities'].get(field), value, name + ' identity ' + field)
        diagnostic_reports[name] = report
    snapshot = Path(freeze['code_snapshot']).resolve()
    core.require(root / 'provenance/frozen_code' in snapshot.parents, 'snapshot outside frozen provenance')
    scalar.disjoint_output(output, snapshot, 'snapshot')
    core.equal(str(snapshot), learned['identities']['code_snapshot'], 'original snapshot path')
    hashes = {p.relative_to(snapshot).as_posix(): common.sha_file(p) for p in snapshot.rglob('*.py')}
    core.equal(hashes, freeze['code_hashes'], 'frozen snapshot hashes')
    core.equal(hashes, identities['frozen_code_hashes'], 'original snapshot hashes')
    protocol = core.read_json(root / 'configs/protocol_v2.json')
    core.equal(Path(protocol['root']).resolve(), root, 'protocol root')
    core.equal(protocol['seed'], SEEDS[0], 'protocol first seed')
    core.equal(protocol['model']['arms'], list(ARMS), 'protocol arms')
    for key, value in {'seeds': list(SEEDS), 'learning_rates': [.0003, .001], 'max_steps': 3000,
                       'eval_every': 100, 'patience': 5, 'selection': 'hierarchical_protein_mean_validation_ce'}.items():
        core.equal(protocol['train'][key], value, 'protocol training ' + key)
    core.equal(protocol['statistics']['bootstrap_replicates'], 10000, 'bootstrap replicates')
    for key, value in {'ca_cutoff_angstrom': 15., 'min_sequence_separation': 3,
                       'contact_cutoff_angstrom': 8., 'error_thresholds': prediction.THRESHOLDS.tolist(),
                       'preservation_values': prediction.VALUES.tolist(),
                       'missing_coordinates': 'unknown_with_fixed_denominator_bounds'}.items():
        core.equal(protocol['labels'][key], value, 'frozen label definition ' + key)
    tasks = freeze['tasks']
    core.require(len(tasks) == 10 and {(r['arm'], r['seed']) for r in tasks} ==
                 {(a, s) for a in ARMS for s in SEEDS[1:]}, 'prospective task inventory differs')
    for task in tasks:
        core.equal(task['lr'], .001, 'prospective fixed LR')
        core.equal(task['output'], f"checkpoints/{task['arm']}_s{task['seed']}_lr0.001", 'prospective output')
    split = core.read_json(root / 'manifests/splits.json')
    core.equal(split['status'], 'SPLIT_FROZEN', 'split status')
    core.equal(split['protocol_sha256'], identities['protocol'], 'split protocol identity')
    primary = common.read_jsonl(root / 'manifests/development_primary.jsonl')
    core.keyed(primary, lambda r: r['candidate_id'], 'primary candidate')
    core.require(all(r['source'] == protocol['primary_source'] and r['generator'] != 'mixed' for r in primary),
                 'primary source differs')
    valid = [r for r in primary if split['assignment'][r['protein_id']] == 'validation']
    roster = {wt: split['component'][wt] for wt in sorted({r['protein_id'] for r in valid})}
    core.equal(len(valid), 420, 'validation candidate count')
    core.equal(len(roster), 10, 'validation WT count')
    core.equal(roster, learned['validation_wt_components'], 'original full WT/component roster')
    summaries = core.keyed(common.read_jsonl(root / 'reports/label_candidates.jsonl'), lambda r: r['candidate_id'], 'label summary')
    core.require(all(r['candidate_id'] in summaries for r in valid), 'missing validation labels')
    profiles = {tuple(sorted(r['esmc_profile_hashes'])) for r in learned['runs']}
    core.require(len(profiles) == 1 and bool(next(iter(profiles))), 'original ESMC profiles differ')
    original_predictions, ledger_hash = sealed_first_seed_predictions(
        freeze['prediction_diagnostic']['report'], diagnostic_reports['prediction_diagnostic'], learned, valid, root, output)
    identities.update(replication_freeze_sha256=common.sha_file(freeze_path), replication_review_sha256=common.sha_file(__file__),
                      first_seed_prediction_ledger_sha256=ledger_hash)
    return dict(root=root, freeze=freeze, output=output, protocol=protocol, learned=learned,
                identities=identities, valid=valid, roster=roster, summaries=summaries,
                snapshot=snapshot, profiles=list(next(iter(profiles))), paths={'labels': root / 'labels/pairs'},
                first_seed_predictions=original_predictions)


def inventory(context):
    runs = []
    for arm in ARMS:
        original = context['learned']['selected'][arm]
        core.equal(original['lr'], .001, 'first-seed fixed selected LR')
        core.equal(original['arm'], arm, 'first-seed selected arm')
        core.equal(original['seed'], SEEDS[0], 'first-seed selected identity')
        expected = min((r for r in context['learned']['runs'] if r['arm'] == arm), key=lambda r: (r['best_ce'], r['lr']))
        core.equal(original, expected, 'original selected run')
        runs.append({'arm': arm, 'seed': SEEDS[0], 'lr': .001,
                     'directory': str(context['root'] / f'checkpoints/{arm}_s{SEEDS[0]}_lr0.001'), '_sealed': original})
        for seed in SEEDS[1:]:
            task = next(r for r in context['freeze']['tasks'] if r['arm'] == arm and r['seed'] == seed)
            runs.append({'arm': arm, 'seed': seed, 'lr': .001, 'directory': str(context['root'] / task['output'])})
    return runs


def validate_identity(context, run, provenance):
    for key in ('protocol', 'manifest', 'splits', 'label_summary_sha256', 'cache_index_sha256'):
        core.equal(provenance[key], context['identities'][key], 'run input ' + key)
    computational = {k: v for k, v in context['freeze']['code_hashes'].items() if 'tests' not in Path(k).parts}
    core.equal(provenance['code'], computational, 'run frozen computational code')
    cfg = provenance['config']
    for key, value in {'arm': run['arm'], 'seed': run['seed'], 'lr': .001, 'technical': False,
                       'steps': None, 'device': 'cuda'}.items():
        core.equal(cfg[key], value, 'run config ' + key)
    for key, relative in [('protocol', 'configs/protocol_v2.json'), ('manifest', 'manifests/development_primary.jsonl'),
                          ('splits', 'manifests/splits.json'), ('label_summary', 'reports/label_candidates.jsonl'),
                          ('labels', 'labels/pairs')]:
        core.equal(Path(cfg[key]).resolve(), (context['root'] / relative).resolve(), 'run config path ' + key)
    core.equal(Path(cfg['output']).resolve(), Path(run['directory']).resolve(), 'run config output')
    core.equal(cfg['cache_index'], list(context['identities']['cache_index_sha256']), 'run cache paths')


def load_probability(path, lab, output, expected_hash=None):
    scalar.reject_input_overlap(output, [path])
    checksum = common.sha_file(path)
    if expected_hash is not None:
        core.equal(checksum, expected_hash, 'prediction hash')
    with np.load(path, allow_pickle=False) as archive:
        core.equal(set(archive.files), {'pair_ids', 'probabilities'}, 'prediction arrays')
        p = archive['probabilities']
        ids = prediction.integer_array(archive['pair_ids'], 'prediction pair_ids')
        prediction.validate_probabilities(p, ids, np.flatnonzero(lab['retained_mask']))
    checked_hash(path, checksum, 'prediction after read')
    return p, checksum


def candidate_metrics(row, summary, lab, probabilities):
    ids = np.flatnonzero(lab['retained_mask'])
    positions = np.searchsorted(ids, lab['pair_ids'])
    core.require(len(positions) > 0 and (positions < len(ids)).all() and
                 np.array_equal(ids[positions], lab['pair_ids']), 'known label prediction mapping differs')
    known = np.asarray(probabilities, dtype=float)[positions]
    y = lab['target']
    ce = float(-np.log(np.maximum(known[np.arange(len(y)), y], 1e-30)).mean())
    ghat = float((np.asarray(probabilities, dtype=float) @ prediction.VALUES).mean())
    c = len(ids) / summary['reference_pairs']
    core.equal(c, summary['C'], 'frozen retained coverage')
    return {k: row[k] for k in ('candidate_id', 'protein_id', 'budget', 'generator')} | {
        'ce': ce, 'valid_pairs': len(y), 'Ghat': ghat, 'C': c, 'That': c * ghat} | {
        k: summary[k] for k in ('G_lower', 'G_upper', 'T_lower', 'T_upper')}


def verify_run(context, run, labels):
    directory = Path(run['directory'])
    core.equal(directory.resolve(), directory.absolute(), 'run directory resolves through a link')
    core.require(context['root'] in directory.resolve().parents, 'run directory escapes experiment root')
    scalar.disjoint_output(context['output'], directory, 'training run')
    sealed = run.get('_sealed')
    if not (directory / 'status.json').exists():
        core.require(sealed is None, 'sealed first-seed completed status missing')
        return run | {'state': 'INCOMPLETE' if directory.exists() else 'PENDING'}
    scalar.reject_input_overlap(context['output'], [directory / name for name in ARTIFACTS])
    status = core.read_json(directory / 'status.json')
    if status.get('status') != 'DEVELOPMENT_RUN_COMPLETE':
        core.require(sealed is None, 'sealed first-seed status differs')
        return run | {'state': 'INCOMPLETE', 'observed_status': status.get('status')}
    hashes = {name: common.sha_file(directory / name) for name in ARTIFACTS}
    validate_identity(context, run, core.read_json(directory / 'run.json'))
    core.equal(status['arm'], run['arm'], 'status arm')
    core.equal(status['convergence_review_required'], run['arm'] in ('S', 'B'), 'convergence review flag')
    core.equal(status['esmc_profile_hashes'], context['profiles'], 'run ESMC profiles')
    trace = core.verify_trace(directory, status, context['protocol']['train'])
    compact = {k: v for k, v in trace.items() if k not in ('train_trace', 'validation_trace')}
    validation = directory / f"validation_{trace['best_step']}"
    scalar.reject_input_overlap(context['output'], [validation, directory / 'best.pt'])
    evaluation = core.verify_evaluation(validation, run['arm'], trace['best_ce'], context['valid'], context['summaries'], context['roster'])
    checkpoint = common.sha_file(directory / 'best.pt')
    core.require((directory / 'best.pt').stat().st_size > 0, 'empty checkpoint artifact')
    if sealed:
        core.equal(Path(sealed['directory']).resolve(), directory.resolve(), 'sealed first-seed directory')
        core.equal(sealed['artifact_hashes'], hashes, 'sealed first-seed artifact hashes')
        core.equal(sealed['output_hashes'], evaluation['output_hashes'], 'sealed first-seed evaluation hashes')
        core.equal(sealed['checkpoint_sha256'], checkpoint, 'sealed first-seed checkpoint hash')
        core.equal(Path(sealed['checkpoint_path']).resolve(), (directory / 'best.pt').resolve(), 'sealed checkpoint path')
        core.equal(Path(sealed['validation_output']).resolve(), validation.resolve(), 'sealed validation path')
        for field in ('steps', 'best_step', 'best_ce'):
            core.equal(sealed[field], compact[field], 'sealed first-seed convergence ' + field)
    expected = {r['candidate_id']: r for r in evaluation['candidates']}
    ledger = []
    for row in context['valid']:
        cid = row['candidate_id']
        path = validation / (cid + '.npz')
        prior_hash = None
        if run['seed'] == SEEDS[0]:
            prior = context['first_seed_predictions'][(run['arm'], cid)]
            core.equal(prior['path'], str(path), 'original prediction sealed path')
            prior_hash = prior['sha256']
        p, checksum = load_probability(path, labels[cid], context['output'], prior_hash)
        recomputed = candidate_metrics(row, context['summaries'][cid], labels[cid], p)
        for key in ('ce', 'Ghat', 'That', 'C'):
            prediction.close(recomputed[key], expected[cid][key], cid + ' recomputed ' + key)
        ledger.append({'arm': run['arm'], 'seed': run['seed'], 'candidate_id': cid, 'path': str(path), 'sha256': checksum})
    for name, checksum in hashes.items():
        checked_hash(directory / name, checksum, 'run artifact after read ' + name)
    checked_hash(directory / 'best.pt', checkpoint, 'opaque checkpoint after read')
    for name, checksum in evaluation['output_hashes'].items():
        checked_hash(validation / name, checksum, 'validation output after read ' + name)
    return {k: v for k, v in run.items() if k != '_sealed'} | compact | {
        'state': 'COMPLETE', 'artifact_hashes': hashes, 'checkpoint_path': str(directory / 'best.pt'),
        'checkpoint_sha256': checkpoint, 'checkpoint_identity_scope': 'opaque_file_bytes_no_deserialization',
        'esmc_profile_hashes': status['esmc_profile_hashes'], 'validation_output': str(validation),
        'output_hashes': evaluation['output_hashes'], 'convergence_review_required': run['arm'] in ('S', 'B'),
        'cost': {'training_steps': trace['steps'], 'validation_evaluations': len(trace['validation_trace']),
                 'gpu_seconds': None, 'wall_seconds': None, 'source': 'Slurm accounting required; trace timing is not GPU cost'},
        '_evaluation': evaluation, '_prediction_ledger': ledger}


def mean_probabilities(arrays):
    core.require(len(arrays) == 3, 'exactly three fixed seeds required')
    shapes = {np.asarray(p).shape for p in arrays}
    core.require(len(shapes) == 1, 'ensemble prediction shapes differ')
    return np.mean(np.stack(arrays).astype(np.float64), axis=0)


def atomic_npz(path, pair_ids, probabilities):
    path = common.confined_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name, suffix='.npz')
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            np.savez_compressed(handle, pair_ids=pair_ids, probabilities=probabilities)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path, value):
    core.atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def write_rows(path, rows):
    core.atomic_text(path, ''.join(json.dumps(r, sort_keys=True, allow_nan=False) + '\n' for r in rows))


def ensemble_evaluation(context, arm, runs, labels):
    core.equal([r['seed'] for r in runs], list(SEEDS), 'ensemble fixed seed order')
    output = context['output'] / 'ensemble' / arm
    candidates = []
    ledger = []
    by_seed = [{r['candidate_id']: r for r in run['_prediction_ledger']} for run in runs]
    for row in context['valid']:
        cid = row['candidate_id']
        arrays = [load_probability(Path(entries[cid]['path']), labels[cid], context['output'], entries[cid]['sha256'])[0]
                  for entries in by_seed]
        p = mean_probabilities(arrays)
        ids = np.flatnonzero(labels[cid]['retained_mask'])
        prediction.validate_probabilities(p, ids, ids)
        record = candidate_metrics(row, context['summaries'][cid], labels[cid], p)
        candidates.append(record)
        path = output / (cid + '.npz')
        atomic_npz(path, ids, p)
        ledger.append({'arm': arm, 'candidate_id': cid, 'path': str(path), 'sha256': common.sha_file(path),
                       'source_prediction_sha256': [entries[cid]['sha256'] for entries in by_seed]})
    proteins = [{'protein_id': wt, 'ce': core.hierarchical([r for r in candidates if r['protein_id'] == wt])}
                for wt in context['roster']]
    decisions = core.expected_decisions(candidates)
    report = {'status': 'DEVELOPMENT_EVALUATED', 'arm': arm, 'candidates': len(candidates), 'proteins': len(proteins),
              'hierarchical_ce': core.hierarchical(candidates), 'budget_decisions': core.budget_decisions(decisions),
              'interpretation': INTERPRETATION, 'prediction_combination': 'equal arithmetic mean of aligned class probabilities',
              'seeds': list(SEEDS), 'seed_weights': [1 / 3] * 3}
    for name, rows in [('candidates.jsonl', candidates), ('proteins.jsonl', proteins), ('decisions.jsonl', decisions)]:
        write_rows(output / name, rows)
    write_json(output / 'report.json', report)
    evaluation = core.verify_evaluation(output, arm, report['hierarchical_ce'], context['valid'], context['summaries'], context['roster'])
    checkpoint_identity = {'kind': 'ordered_three_seed_opaque_checkpoint_sha256_tuple',
                           'seeds': list(SEEDS), 'sha256': [r['checkpoint_sha256'] for r in runs]}
    return {'arm': arm, 'lr': .001, 'checkpoint_sha256': checkpoint_identity,
            'checkpoint_identity': checkpoint_identity, 'hierarchical_ce': report['hierarchical_ce'],
            'validation_output': str(output), 'output_hashes': evaluation['output_hashes'],
            '_evaluation': evaluation, '_prediction_ledger': ledger}


def public_run(run):
    return {k: v for k, v in run.items() if not k.startswith('_')}


def verify_previous_audit(context, runs):
    """A repeat audit may add completed tasks, but cannot silently replace them."""
    path = context['output'] / 'report.json'
    if not path.exists():
        return
    previous = core.read_json(path)
    core.require(previous['status'] in (WAITING, READY), 'unrecognized previous replication audit')
    for key, value in context['identities'].items():
        if key != 'replication_review_sha256':
            core.equal(previous['identities'][key], value, 'previous audit input ' + key)
    current = {(r['arm'], r['seed']): r for r in runs}
    for old in previous['runs']:
        if old['state'] != 'COMPLETE':
            continue
        run = current[(old['arm'], old['seed'])]
        core.equal(run['state'], 'COMPLETE', 'previously completed run is now missing')
        for field in ('artifact_hashes', 'output_hashes', 'checkpoint_sha256'):
            core.equal(run[field], old[field], 'previous audit completed ' + field)
    ledger = context['output'] / 'artifact_ledger.jsonl'
    checked_hash(ledger, previous['artifact_ledger_sha256'], 'previous audit ledger')
    old_predictions = {(r['arm'], r['seed'], r['candidate_id']): r['sha256']
                       for r in common.read_jsonl(ledger) if 'seed' in r and 'arm' in r}
    for run in runs:
        for row in run.get('_prediction_ledger', []):
            key = (row['arm'], row['seed'], row['candidate_id'])
            if key in old_predictions:
                core.equal(row['sha256'], old_predictions[key], 'previous audit prediction hash')


def build_review(*, root, freeze, output):
    context = read_context(root, freeze, output)
    labels, _, label_ledger = prediction.load_labels(context)
    runs = [verify_run(context, run, labels) for run in inventory(context)]
    verify_previous_audit(context, runs)
    complete = all(r['state'] == 'COMPLETE' for r in runs)
    report = {'status': READY if complete else WAITING, 'interpretation': INTERPRETATION,
              'expected_runs': 15, 'completed_runs': sum(r['state'] == 'COMPLETE' for r in runs),
              'seeds': list(SEEDS), 'arms': list(ARMS), 'fixed_learning_rates': {a: .001 for a in ARMS},
              'seed_selection': 'none_all_three_prespecified_seeds_reported', 'runs': [public_run(r) for r in runs],
              'validation_candidates': len(context['valid']), 'validation_wt_components': context['roster'],
              'identities': context['identities'], 'output': str(context['output']),
              'scientific_pass_decision': None, 'confirmation_folding_authorized': False, 'paper': False,
              'GPU_used': False, 'gpu_calls': 0, 'training_calls': 0, 'structure_calls': 0,
              'PLM_forward_calls': 0, 'variant_structure_files_read': 0,
              'checkpoint_audit_limit': 'Original checkpoint hashes are sealed by the original core report. Prospective checkpoint bytes are hashed and rechecked during this audit; the frozen trainer did not independently seal their internal content.',
              'statistics': {'replicates': 10000, 'seed': SEEDS[0], 'confidence': context['protocol']['statistics']['confidence'],
                             'n_wts': len(context['roster']), 'n_components': len(set(context['roster'].values())),
                             'aggregation': 'within WT, then equal WT means within homology component, then equal component means',
                             'unknown_bounds': 'full WT roster retained; no complete-case filtering or zero imputation',
                             'seeds_are_independent_WT_units': False}}
    ledger = label_ledger + [entry for run in runs for entry in run.get('_prediction_ledger', [])]
    if complete:
        statistics = core.load_statistics(context['snapshot'])
        report['per_seed'] = {}
        for seed in SEEDS:
            selected = {r['arm']: r for r in runs if r['seed'] == seed}
            paired = core.paired_inputs(selected, context['roster'])
            path = context['output'] / 'per_seed' / str(seed) / 'paired_inputs.jsonl'
            write_rows(path, paired)
            report['per_seed'][str(seed)] = {'arms': {a: {'hierarchical_ce': r['best_ce'],
                'budget_decisions': core.budget_decisions(r['_evaluation']['decisions']),
                'checkpoint_sha256': r['checkpoint_sha256']} for a, r in selected.items()},
                'comparisons': core.compare_inputs(paired, context['roster'], statistics, context['protocol']['statistics']['confidence']),
                'paired_inputs_sha256': common.sha_file(path)}
        ensemble = {arm: ensemble_evaluation(context, arm, sorted([r for r in runs if r['arm'] == arm], key=lambda r: r['seed']), labels)
                    for arm in ARMS}
        paired = core.paired_inputs(ensemble, context['roster'])
        write_rows(context['output'] / 'ensemble/paired_inputs.jsonl', paired)
        report['ensemble'] = {'procedure': 'mean aligned five-class probabilities per retained pair before metrics and selection',
            'seed_weights': [1 / 3] * 3, 'seeds': list(SEEDS), 'arms': {a: public_run(r) for a, r in ensemble.items()},
            'comparisons': core.compare_inputs(paired, context['roster'], statistics, context['protocol']['statistics']['confidence']),
            'paired_inputs_sha256': common.sha_file(context['output'] / 'ensemble/paired_inputs.jsonl')}
        ledger += [entry for run in ensemble.values() for entry in run['_prediction_ledger']]
    else:
        report['waiting_reason'] = 'All fifteen validated completed arm/seed artifacts are required before efficacy comparisons or ensemble construction.'
    write_rows(context['output'] / 'artifact_ledger.jsonl', ledger)
    report['artifact_ledger_sha256'] = common.sha_file(context['output'] / 'artifact_ledger.jsonl')
    write_json(context['output'] / 'report.json', report)
    core.atomic_text(context['output'] / 'ENGINEERING.md', '# Three-seed replication review\n\n' + report['status'] + '\n\n' +
                     INTERPRETATION + '\n\n' + f"Completed validated runs: {report['completed_runs']}/15.\n\n" +
                     'Detailed run identities, per-seed results and ensemble results are in report.json. '
                     'No scientific pass decision or confirmation folding authorization is produced.\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'freeze', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    report = build_review(root=args.root, freeze=args.freeze, output=args.output)
    print(json.dumps({'status': report['status'], 'completed_runs': report['completed_runs'], 'output': report['output']}))


if __name__ == '__main__':
    main()
