"""CPU audit of all nine fixed-step D-ranking runs; final step 2000 only.

No inference, configuration selection, confirmation labels, or paper writes.
The three seeds are repeated models on the same ten development WTs.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import torch
import d_ranking_trial as trial

VALUES = np.array([1., .75, .5, .25, 0.])
BACKGROUND_REPORT_SHA256 = '7fa4662cd2b44069fedf6cd33d325811ad2e5673a40694a23ee526647fc0a868'
SOURCES = ('analysis/review_d_ranking_trial.py', 'analysis/tests/test_review_d_ranking_trial.py',
           'slurm/review_d_ranking_trial_cpu.sbatch', 'analysis/development_review.py',
           'analysis/cpu_prediction_diagnostics.py', 'analysis/review_scalar_baselines.py')
INTERPRETATION = ('Bounded development selection trial on ten reused validation WTs. '
    'Intervals are conditional on these fitted models and development data; seeds are not '
    'independent WT evidence. Independent confirmation is not established. No function '
    'claims, extra training, confirmation folding, or paper changes are authorized.')


def require(value, message):
    if not value: raise ValueError(message)


def helpers():
    # Preflight must import frozen runtime modules before this helper imports common.
    import development_review as core
    import cpu_prediction_diagnostics as diagnostics
    return core, diagnostics


class Ledger:
    """Hash before reads and reject any change before publishing the report."""
    def __init__(self, root): self.root = Path(root).resolve(); self.hashes = {}
    def add(self, path, expected=None):
        path = Path(path).absolute()
        require((self.root in path.resolve().parents or expected is not None) and path.resolve() == path, 'linked/unpinned external input')
        checksum = trial.sha_file(path)
        require(expected is None or checksum == expected, 'input hash differs: '+str(path))
        require(path not in self.hashes or self.hashes[path] == checksum, 'input changed during read')
        self.hashes[path] = checksum
        return checksum
    def check(self):
        for path, checksum in self.hashes.items():
            require(trial.sha_file(path) == checksum, 'input changed during audit: '+str(path))
    def rows(self):
        return [dict(path=str(p), sha256=h) for p, h in sorted(self.hashes.items())]


def protected_output(root, output):
    root = Path(root).resolve(); output = Path(output).absolute()
    require(output == root/'reports/d_ranking_trial/review', 'dedicated review output required')
    require(output.resolve() == output, 'linked output forbidden')
    require(not output.exists(), 'review output already exists; refuse stale overwrite')
    return output


def embedding_profile_identities(root, audit, ledger):
    """Bridge canonical audit hashes to frozen Inputs' default-spaced JSON hashes."""
    pins = audit['asset_hashes']; mapping = {}
    require(trial.identity(pins) == audit['assets_sha256'], 'embedding asset pin inventory differs')
    for relative, checksum in pins.items():
        if Path(relative).suffix != '.json': continue
        path = trial.confined(root, relative); ledger.add(path, checksum); meta = trial.read_json(path)
        require(meta['output_sha256'] == pins.get(Path(relative).with_suffix('.npy').as_posix()), 'sidecar embedding output pin differs')
        profile = {k: v for k, v in meta.items() if k not in ('sequence_sha256', 'length', 'output_sha256', 'gpu', 'batch_seconds')}
        canonical = trial.identity(profile)
        require(canonical in audit['profiles'], 'sidecar canonical embedding profile differs')
        legacy = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
        require(canonical not in mapping or mapping[canonical] == legacy, 'embedding profile serialization mapping differs')
        mapping[canonical] = legacy
    require(sorted(mapping) == sorted(audit['profiles']) and bool(mapping), 'embedding profile inventory incomplete')
    return dict(canonical_to_legacy=mapping, legacy_profiles=sorted(set(mapping.values())),
                canonical_serialization='sorted compact JSON', legacy_serialization='json.dumps(profile, sort_keys=True)')


def read_context(root, freeze_hash, ledger):
    root = trial.authorized_root(root)
    ledger.add(root/trial.NEW_FREEZE, freeze_hash)
    freeze = trial.checked_freeze(root, freeze_hash)
    for relative in SOURCES: ledger.add(root/relative)
    bundle = trial.preflight(root, sealed_embeddings=freeze['preflight']['embeddings'])
    require(bundle['audit'] == freeze['preflight'], 'preflight identity differs')
    profiles = embedding_profile_identities(root, freeze['preflight']['embeddings'], ledger)
    for relative, checksum in freeze['input_hashes'].items(): ledger.add(root/relative, checksum)
    for row in bundle['rows']:
        ledger.add(root/'labels/pairs'/(row['candidate_id']+'.npz'), bundle['summaries'][row['candidate_id']]['labels_sha256'])
        for field, hash_field in (('wt_native_json', 'wt_native_json_sha256'), ('reference_path', 'reference_sha256'), ('source_path', 'source_sha256')):
            ledger.add(row[field], row[hash_field])
        ledger.add(root/f"caches/wt_references/{row['protein_id']}.mapping.json")
    roster = {r['protein_id']: bundle['split']['component'][r['protein_id']] for r in bundle['valid']}
    require(len(roster) == 10 and len(set(roster.values())) == 10, 'ten WT/components required')
    return dict(bundle=bundle, root=root, freeze=freeze, freeze_hash=freeze_hash, roster=roster, ledger=ledger,
                embedding_profile_identities=profiles)


def verify_trace(directory, status, ledger):
    for name in ('train_trace.jsonl', 'validation_trace.json', 'sampling_trace.jsonl'): ledger.add(directory/name)
    train = trial.read_jsonl(directory/'train_trace.jsonl')
    require([r['step'] for r in train] == list(range(1, 2001)), 'train trace must contain 2000 sequential steps')
    for row in train:
        require(row['relation_updates'] == 1024, 'relation update count differs')
        require(all(np.isfinite(row[k]) and row[k] >= 0 for k in ('ce', 'ranking', 'loss', 'gradient_norm')), 'nonfinite/negative trace')
    validations = trial.read_json(directory/'validation_trace.json')
    require(validations == status['validations'], 'status/validation trace differs')
    require([r['step'] for r in validations] == [500, 1000, 1500, 2000], 'validation cadence differs')
    require([r['primary'] for r in validations] == [False, False, False, True], 'primary must be final step only')
    require(all(np.isfinite(r['hierarchical_ce']) and r['hierarchical_ce'] >= 0 for r in validations), 'invalid validation CE')
    require(ledger.add(directory/'sampling_trace.jsonl') == status['sampling_trace_sha256'], 'sampling trace hash differs')
    return validations


def probability(path, label, ledger):
    ledger.add(path)
    with np.load(path, allow_pickle=False) as z:
        require(set(z.files) == {'pair_ids', 'probabilities'}, 'prediction NPZ members differ')
        ids, p = z['pair_ids'], z['probabilities']
    require(np.issubdtype(ids.dtype, np.integer) and np.array_equal(ids, label['pair_ids']), 'prediction pair IDs differ')
    require(p.shape == (len(ids), 5) and np.issubdtype(p.dtype, np.floating), 'probability shape/type differs')
    require(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(), 'invalid probabilities')
    require(np.allclose(p.sum(1), 1., rtol=0, atol=1e-5), 'probability row sum differs')
    return p


def candidate(row, label, p):
    _, diagnostics = helpers(); y = label['target']; g = float((p@VALUES).mean())
    truth_g = float(VALUES[y].mean()); truth_t = label['C']*truth_g
    return {k: row[k] for k in ('candidate_id', 'protein_id', 'budget', 'generator')} | dict(
        C=label['C'], Ghat=g, That=label['C']*g, G_lower=truth_g, G_upper=truth_g,
        T_lower=truth_t, T_upper=truth_t, valid_pairs=len(y),
        ce=float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-30)).mean()),
        G_MAE=abs(g-truth_g), T_MAE=abs(label['C']*g-truth_t),
        statistics=diagnostics.sufficient_statistics(p, y))


def verify_pooled(path, expected, ledger):
    """Check the evaluator's duplicate pooled arrays by streaming NPY payloads."""
    ledger.add(path)
    with zipfile.ZipFile(path) as archive:
        require(set(archive.namelist()) == {'probabilities.npy', 'targets.npy'}, 'pooled NPZ members differ')
        for name, (shape, dtype, digest) in expected.items():
            with archive.open(name+'.npy') as handle:
                version = np.lib.format.read_magic(handle)
                observed_shape, fortran, observed_dtype = np.lib.format._read_array_header(handle, version)
                require(observed_shape == shape and not fortran and observed_dtype == dtype, 'pooled shape/dtype differs')
                h = hashlib.sha256()
                for chunk in iter(lambda: handle.read(1024*1024), b''): h.update(chunk)
                require(h.hexdigest() == digest, 'pooled probabilities/targets differ')


def verify_evaluation(context, directory, arm, ce, final):
    core, diagnostics = helpers(); bundle = context['bundle']; ledger = context['ledger']
    for name in ('candidates.jsonl', 'proteins.jsonl', 'decisions.jsonl', 'report.json'): ledger.add(directory/name)
    evaluation = core.verify_evaluation(directory, arm, ce, bundle['valid'], bundle['summaries'], context['roster'])
    expected_ids = {r['candidate_id']+'.npz' for r in bundle['valid']}
    require({p.name for p in directory.glob('*.npz')} == expected_ids | {'pooled_probabilities.npz'}, 'evaluation NPZ roster differs')
    if not final: return evaluation
    saved = {r['candidate_id']: r for r in evaluation['candidates']}; records = []
    ph, yh = hashlib.sha256(), hashlib.sha256(); count = 0; confusion = np.zeros((5, 5), dtype=np.int64)
    reliability = np.zeros((10, 3)); pdtype = None
    for row in bundle['valid']:
        cid = row['candidate_id']; label = bundle['labels'][cid]
        p = probability(directory/(cid+'.npz'), label, ledger); record = candidate(row, label, p)
        for key in ('ce', 'Ghat', 'That', 'C', 'G_lower', 'G_upper', 'T_lower', 'T_upper'):
            diagnostics.close(saved[cid][key], record[key], cid+' recomputed '+key)
        records.append(record); ph.update(p.tobytes()); y = label['target']; yh.update(y.tobytes()); count += len(y)
        require(pdtype is None or pdtype == p.dtype, 'mixed probability dtype'); pdtype = p.dtype
        np.add.at(confusion, (y, p.argmax(1)), 1)
        confidence, correct = p.max(1), p.argmax(1) == y
        for i in range(10):
            mask = (confidence >= i/10) & (confidence < (i+1)/10 if i < 9 else confidence <= 1)
            reliability[i] += [mask.sum(), confidence[mask].astype(float).sum(), correct[mask].sum()]
    verify_pooled(directory/'pooled_probabilities.npz', {'probabilities': ((count, 5), pdtype, ph.hexdigest()),
                  'targets': ((count,), bundle['labels'][bundle['valid'][0]['candidate_id']]['target'].dtype, yh.hexdigest())}, ledger)
    report = trial.read_json(directory/'report.json')
    require(report['confusion'] == confusion.tolist(), 'saved confusion differs')
    for i, (n, predicted, correct) in enumerate(reliability):
        saved_bin = report['calibration'][i]; require(saved_bin['count'] == round(n), 'saved calibration count differs')
        for field, value in (('confidence', predicted/n if n else None), ('accuracy', correct/n if n else None)):
            require(saved_bin[field] is None if value is None else np.isclose(saved_bin[field], value, atol=1e-6, rtol=0), 'saved calibration differs')
    decisions = core.expected_decisions(records)
    require([r['selected'] for r in decisions] == [r['selected'] for r in core.expected_decisions(evaluation['candidates'])], 'recomputed winner differs')
    return dict(candidates=records, decisions=decisions)


def verify_run(context, task, index):
    root = context['root']; directory = root/task['output']; ledger = context['ledger']
    for name in ('run.json', 'status.json', 'final.pt'): ledger.add(directory/name)
    require(not (directory/'failure.json').exists(), 'failed run cannot be reviewed')
    run = trial.read_json(directory/'run.json'); status = trial.read_json(directory/'status.json')
    expected = dict(technical=False, mode=task['mode'], seed=task['seed'], parameter_count=133701,
                    freeze_sha256=context['freeze_hash'], primary_step=2000)
    for key, value in expected.items():
        require(run.get(key) == value and status.get(key) == value, 'run/status '+key+' differs')
    require(status['status'] == 'D_RANKING_TRIAL_COMPLETE' and status['steps'] == 2000 and
            status['relation_updates'] == 2000*1024 and status['candidate_forwards'] == 16000 and
            status['checkpoint_selection'] is False and status['early_stopping'] is False, 'incomplete/nonfinal run')
    require(run['status'] == 'D_RANKING_TRIAL_RUNNING' and run['contract'] == trial.CONTRACT and
            run['input_audit'] == context['freeze']['preflight'] and
            run['new_source_hashes'] == context['freeze']['source_hashes'], 'run provenance differs')
    require(run['arm'] == task['arm'] and run['ranking_lambda'] == task['lambda'], 'mode configuration differs')
    for name, rows in (('training_wt', context['bundle']['train']), ('validation_wt', context['bundle']['valid'])):
        require(run[name] == sorted({r['protein_id'] for r in rows}), 'run WT roster differs')
    require(run['initialization_sha256'] == status['initialization_sha256'], 'initialization differs')
    profiles = context['embedding_profile_identities']['legacy_profiles']
    require(status['esmc_profile_hashes'] == profiles, 'embedding profiles differ')
    trial.validate_slurm_resources(run['slurm']['job_fields']); env = run['slurm']['environment']
    require(env['SLURM_ARRAY_TASK_ID'] == str(index) and env['SLURM_ARRAY_TASK_COUNT'] == '9' and
            env.get('SLURM_RESTART_COUNT', '0') == '0' and 'A40' in run['slurm']['gpu_name'], 'Slurm task identity differs')
    directory_ledger = root/'ledger/d_ranking_trial'/context['freeze_hash']
    for name in ('submission.claim', 'submission.json', 'array_owner.json', f'{index}.claim'): ledger.add(directory_ledger/name)
    receipt = trial.read_json(directory_ledger/'submission.json'); claim = trial.read_json(directory_ledger/f'{index}.claim')
    require(receipt['status'] == 'D_RANKING_TRIAL_SUBMITTED' and receipt['freeze_sha256'] == context['freeze_hash'] and
            receipt['job_id'] == env['SLURM_ARRAY_JOB_ID'] == trial.read_json(directory_ledger/'array_owner.json')['slurm_array_job_id'] and
            claim == dict(task_index=index, slurm_job_id=env['SLURM_JOB_ID'], slurm_array_job_id=env['SLURM_ARRAY_JOB_ID']), 'Slurm receipt/claim differs')
    require(all(run[k] == 0 for k in ('PLM_forward_calls', 'structure_calls', 'variant_structure_files_read')), 'unfrozen compute calls')
    validations = verify_trace(directory, status, ledger)
    checkpoint = torch.load(directory/'final.pt', map_location='cpu', weights_only=True)
    require(checkpoint['step'] == 2000 and checkpoint['mode'] == task['mode'] and
            checkpoint['arm'] == task['arm'] and checkpoint['provenance'] == run and
            checkpoint['esmc_profile_hashes'] == profiles and
            checkpoint['validation_ce'] == validations[-1]['hierarchical_ce'], 'final checkpoint metadata differs')
    model = context['bundle']['runtime']['model'].PairResponseHead(); model.load_state_dict(checkpoint['model'], strict=True)
    require(sum(p.numel() for p in model.parameters()) == 133701 and
            all(torch.isfinite(p).all() for p in model.parameters()) and
            trial.state_sha256(model) == status['final_state_sha256'], 'final checkpoint state differs')
    del checkpoint, model
    curves = []
    for validation in validations:
        step = validation['step']; evaluation = verify_evaluation(context, directory/f'validation_{step}', task['arm'], validation['hierarchical_ce'], step == 2000)
        curves.append(dict(step=step, hierarchical_ce=validation['hierarchical_ce'], primary=step == 2000))
    for value in [status['total_wall_seconds'], status['actual_gpu_seconds'], *status['elapsed'].values()]:
        require(np.isfinite(value) and value >= 0, 'invalid resource seconds')
    require(np.isclose(status['actual_gpu_seconds'], status['elapsed']['training_device_seconds']+status['elapsed']['validation_device_seconds']), 'device seconds differ')
    chunks = [(r, (len(context['bundle']['labels'][r['candidate_id']]['pair_ids'])+4095)//4096) for r in context['bundle']['valid']]
    multiplier = 2 if task['arm'] == 'C' else 1; counts = status['actual_projection_counts']
    require(counts['training']['project_calls'] == 16000*multiplier and
            counts['validation']['project_calls'] == 4*sum(n for _, n in chunks)*multiplier and
            counts['validation']['project_residues'] == 4*sum(n*(len(r['sequence'])+(len(r['wt_sequence']) if task['arm'] == 'C' else 0)) for r, n in chunks), 'projection counts differ')
    return dict(mode=task['mode'], seed=task['seed'], directory=str(directory), status=status,
        checkpoint_sha256=ledger.hashes[directory/'final.pt'], curves=curves, evaluation=evaluation,
        slurm=run['slurm'], initialization_sha256=run['initialization_sha256'])


def verify_sampling(context, runs):
    bundle = context['bundle']
    for seed in trial.SEEDS:
        group = [r for r in runs if r['seed'] == seed]
        require(len({r['status']['sampling_trace_sha256'] for r in group}) == 1 and
                len({r['initialization_sha256'] for r in group}) == 1, 'same-seed sampling/initialization differs')
        sampler = trial.PairedSampler(bundle['train'], bundle['split']['assignment'], seed)
        trace = trial.read_jsonl(Path(group[0]['directory'])/'sampling_trace.jsonl')
        require(len(trace) == 2000, 'sampling trace length differs')
        residues = {'D': 0, 'C': 0}; rows = {r['candidate_id']: r for r in bundle['train']}
        for step, observed in enumerate(trace, 1):
            _, expected = sampler.step(bundle['labels'], step)
            require(observed == expected, 'replayed sampling ledger differs')
            for condition in observed['conditions']:
                for item in condition['candidates']:
                    row = rows[item['candidate_id']]; residues['D'] += len(row['sequence'])
                    residues['C'] += len(row['sequence'])+len(row['wt_sequence'])
        for run in group:
            arm = trial.MODES[run['mode']]['arm']
            require(run['status']['actual_projection_counts']['training']['project_residues'] == residues[arm], 'training projected residues differ')


def summaries(evaluation):
    core, diagnostics = helpers(); rows = evaluation['candidates']; decisions = evaluation['decisions']
    selections = []
    for budget in ('all', *map(str, trial.BUDGETS)):
        for source in (*trial.SOURCES, 'mixed'):
            part = [r for r in decisions if r['generator'] == source and (budget == 'all' or r['budget'] == budget)]
            selections.append(dict(budget=budget, source=source, selected_T=core.hierarchical(part, 'T_lower', ('protein_id', 'budget')),
                regret=core.hierarchical(part, 'raw_regret_lower', ('protein_id', 'budget')),
                oracle_hit=core.hierarchical([r | {'hit': float(np.isclose(r['T_lower'], r['oracle_T_lower'], atol=1e-12, rtol=0))} for r in part], 'hit', ('protein_id', 'budget'))))
    if not rows: return dict(selection=selections)
    metrics = []
    for budget in ('all', *map(str, trial.BUDGETS)):
        for source in (*trial.SOURCES, 'mixed'):
            part = [r for r in rows if (source == 'mixed' or r['generator'] == source) and (budget == 'all' or str(r['budget']) == budget)]
            metrics.append(dict(budget=budget, source=source, G_MAE=core.hierarchical(part, 'G_MAE'), T_MAE=core.hierarchical(part, 'T_MAE'),
                **diagnostics.finish_statistics(diagnostics.hierarchy_statistics(part))))
    return dict(selection=selections, metrics=metrics)


def compare(left, right, roster, statistics):
    core, _ = helpers(); result = []
    key = lambda r: (r.get('seed'), *core.decision_key(r))
    x, y = core.keyed(left, key, 'left decision'), core.keyed(right, key, 'right decision')
    require(set(x) == set(y), 'paired decision roster differs')
    for budget in ('all', *map(str, trial.BUDGETS)):
        for source in (*trial.SOURCES, 'mixed'):
            keys = [k for k in x if x[k]['generator'] == source and (budget == 'all' or x[k]['budget'] == budget)]
            part = []; changes = dict(changed=0, good=0, bad=0, same_truth=0, unchanged=0)
            for k in keys:
                a, b = x[k], y[k]; delta = a['T_lower']-b['T_lower']; changed = a['selected'] != b['selected']
                changes['changed' if changed else 'unchanged'] += 1
                if changed: changes['good' if delta > 1e-12 else 'bad' if delta < -1e-12 else 'same_truth'] += 1
                for method, row in (('x', a), ('y', b)):
                    part.append(dict(wt_id=row['protein_id'], group_id=roster[row['protein_id']], method=method, lower=row['T_lower'], upper=row['T_lower']))
            boot = statistics.paired_bootstrap(part, 'x', 'y', roster, replicates=10000, seed=trial.SEEDS[0], confidence=.95)
            result.append(dict(budget=budget, source=source, selected_T=boot,
                regret_difference=-boot['difference_bounds'][0], regret_CI=[-boot['confidence_interval'][1], -boot['confidence_interval'][0]], top1_changes=changes))
    return result


def continuation(comparisons, seed_comparisons):
    lookup = {(r['budget'], r['source']): r['selected_T'] for r in comparisons}
    primary = lookup['all', 'mixed']; seed_deltas = [next(r for r in part if r['budget'] == 'all' and r['source'] == 'mixed')['selected_T']['difference_bounds'][0] for part in seed_comparisons]
    checks = [('mixed_mean_at_least_0.005', primary['difference_bounds'][0], .005, False),
              ('mixed_CI_lower_strictly_positive', primary['confidence_interval'][0], 0., True),
              ('positive_seed_count_at_least_2', sum(v > 0 for v in seed_deltas), 2, False)]
    checks += [('source_'+s+'_nonnegative', lookup['all', s]['difference_bounds'][0], 0., False) for s in trial.SOURCES]
    checks += [('mixed_budget_'+str(b)+'_at_least_minus_0.005', lookup[str(b), 'mixed']['difference_bounds'][0], -.005, False) for b in trial.BUDGETS]
    criteria = [dict(name=n, observed=v, threshold=t, strict=strict, passed=v > t if strict else v >= t) for n, v, t, strict in checks]
    supported = all(r['passed'] for r in criteria)
    return dict(status='CONTINUE_SUPPORTED' if supported else 'LIMITED_DIRECTIONAL' if primary['difference_bounds'][0] > 0 else 'UNSUPPORTED',
                criteria=criteria, seed_deltas=seed_deltas, failed=[r['name'] for r in criteria if not r['passed']], automatic_extra_work=False)


def background(context):
    core, _ = helpers(); root = context['root']; ledger = context['ledger']
    directory = root/'reports/development_review/scalar_baselines'
    try:
        ledger.add(root/'reports/first_seed_evidence_decision.json')
        seal = trial.read_json(root/'reports/first_seed_evidence_decision.json')
        require(seal['artifact_sha256']['reports/development_review/scalar_baselines/report.json'] == BACKGROUND_REPORT_SHA256, 'background evidence seal differs')
        ledger.add(directory/'report.json', BACKGROUND_REPORT_SHA256); old = trial.read_json(directory/'report.json')
        for name in ('report.json', 'candidates.jsonl'):
            ledger.add(root/'reports/utility_baseline'/name, old['identities']['baseline_'+('report' if name == 'report.json' else 'candidates')+'_sha256'])
        scores = core.keyed(trial.read_jsonl(root/'reports/utility_baseline/candidates.jsonl'), lambda r: r['candidate_id'], 'background candidate')
        valid = context['bundle']['valid']; summaries_ = context['bundle']['summaries']; kept = {r['candidate_id']: scores[r['candidate_id']] for r in valid}
        require(len(kept) == 420, 'background roster differs')
        for row in valid:
            score = kept[row['candidate_id']]
            require(all(score[k] == row[k] for k in ('protein_id', 'budget', 'generator')) and np.isclose(score['static_relationship_coverage'], summaries_[row['candidate_id']]['C']) and
                    all(np.isfinite(score[k]) for k in ('utility', 'static_relationship_coverage')), 'background identity differs')
        import review_scalar_baselines as scalar
        ledger.add(directory/'decisions.jsonl'); saved = trial.read_jsonl(directory/'decisions.jsonl'); result = {}
        for method in ('U', 'V'):
            decisions = scalar.scalar_decisions(valid, kept, summaries_, method)
            require([(core.decision_key(r), r['selected']) for r in saved if r['method'] == method] == [(core.decision_key(r), r['selected']) for r in decisions], 'background selection differs')
            for r in decisions:
                pool = [v for v in valid if v['protein_id'] == r['protein_id'] and str(v['budget']) == r['budget'] and (r['generator'] == 'mixed' or v['generator'] == r['generator'])]
                oracle = max(context['bundle']['labels'][v['candidate_id']]['T'] for v in pool)
                r.update(oracle_T_lower=oracle, raw_regret_lower=oracle-r['T_lower'])
            result[method] = summaries(dict(candidates=[], decisions=decisions))['selection']
        return dict(status='AVAILABLE_BACKGROUND_ONLY', selectors=result, matched_training_control=False)
    except (OSError, ValueError, KeyError) as error:
        return dict(status='UNAVAILABLE', reason=str(error), matched_training_control=False)


def build_review(root, freeze_sha256, output):
    root = trial.authorized_root(root); output = protected_output(root, output); ledger = Ledger(root)
    context = read_context(root, freeze_sha256, ledger); core, _ = helpers()
    runs = [verify_run(context, task, i) for i, task in enumerate(context['freeze']['tasks'])]
    verify_sampling(context, runs); statistics = core.load_statistics(root/trial.SNAPSHOT)
    comparisons = {}; per_seed = {}; per_wt = []; summary_rows = []
    pairs = (('D_rank', 'D0'), ('D_rank', 'C0'), ('D0', 'C0'))
    evaluations = {(r['mode'], r['seed']): r['evaluation'] for r in runs}
    for seed in trial.SEEDS:
        per_seed[str(seed)] = dict(modes={m: summaries(evaluations[m, seed]) for m in trial.MODES}, comparisons={})
        for a, b in pairs: per_seed[str(seed)]['comparisons'][a+'-'+b] = compare(evaluations[a, seed]['decisions'], evaluations[b, seed]['decisions'], context['roster'], statistics)
    for mode in trial.MODES:
        for seed in trial.SEEDS:
            for row in evaluations[mode, seed]['decisions']: per_wt.append(dict(mode=mode, seed=seed, **row))
            summary_rows += [dict(mode=mode, seed=seed, **r) for r in per_seed[str(seed)]['modes'][mode]['selection']]
    for a, b in pairs: comparisons[a+'-'+b] = compare([r for r in per_wt if r['mode'] == a], [r for r in per_wt if r['mode'] == b], context['roster'], statistics)
    mode_averages = {mode: summaries({kind: [r for seed in trial.SEEDS for r in evaluations[mode, seed][kind]]
                     for kind in ('candidates', 'decisions')}) for mode in trial.MODES}
    summary_rows += [dict(mode=mode, seed='all_seeds', **r) for mode in trial.MODES for r in mode_averages[mode]['selection']]
    ensemble = {}
    for mode in trial.MODES:
        records = []
        for row in context['bundle']['valid']:
            label = context['bundle']['labels'][row['candidate_id']]
            arrays = [probability(Path(next(r for r in runs if r['mode'] == mode and r['seed'] == seed)['directory'])/'validation_2000'/(row['candidate_id']+'.npz'), label, ledger) for seed in trial.SEEDS]
            records.append(candidate(row, label, np.mean(np.stack(arrays).astype(np.float64), axis=0)))
        ensemble[mode] = dict(candidates=records, decisions=core.expected_decisions(records))
        per_wt += [dict(mode=mode, seed='probability_ensemble', **r) for r in ensemble[mode]['decisions']]
        summary_rows += [dict(mode=mode, seed='probability_ensemble', **r) for r in summaries(ensemble[mode])['selection']]
    report = dict(status='D_RANKING_REVIEW_COMPLETE', primary_step=2000, freeze_sha256=freeze_sha256, interpretation=INTERPRETATION,
        validation_wt_components=context['roster'], embedding_profile_identities=context['embedding_profile_identities'],
        per_seed=per_seed, mode_averages=mode_averages, comparisons=comparisons,
        decision=continuation(comparisons['D_rank-D0'], [per_seed[str(s)]['comparisons']['D_rank-D0'] for s in trial.SEEDS]),
        ensemble=dict(procedure='equal three-seed five-class probabilities before metrics and selection', modes={m: summaries(e) for m, e in ensemble.items()},
            comparisons={a+'-'+b: compare(ensemble[a]['decisions'], ensemble[b]['decisions'], context['roster'], statistics) for a, b in pairs}),
        background=background(context), runs=[{k: v for k, v in r.items() if k != 'evaluation'} for r in runs],
        statistics=dict(replicates=10000, seed=trial.SEEDS[0], confidence=.95, averaging='budgets then seeds within WT, equal WT/component bootstrap',
            diagnostic_weighting='equal WT then budget then source then candidate; relations uniform within candidate', seeds_are_independent_WTs=False),
        source_hashes={p: ledger.hashes[root/p] for p in SOURCES}, GPU_used=False, inference_calls=0, structure_calls=0, PLM_forward_calls=0, confirmation_access=False, paper_changes=False,
        audit_limit='Final checkpoint bytes/state and predictions are verified and sealed by this audit; the trainer did not preseal validation output byte hashes.',
        resources=dict(actual_gpu_seconds=sum(r['status']['actual_gpu_seconds'] for r in runs), allocation_gpu_hours=None,
            allocation_note='Slurm sacct allocation hours may be reported separately; device seconds are not allocation seconds.'))
    ledger.check(); output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.review-') as staging:
        staging = Path(staging)
        for name, rows in (('per_wt.jsonl', per_wt), ('artifact_ledger.jsonl', ledger.rows())):
            (staging/name).write_text(''.join(trial.canonical(r)+'\n' for r in rows))
        with (staging/'summary.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0])); writer.writeheader(); writer.writerows(summary_rows)
        report['artifact_sha256'] = {name: trial.sha_file(staging/name) for name in ('per_wt.jsonl', 'artifact_ledger.jsonl', 'summary.csv')}
        trial.write_json(staging/'report.json', report); ledger.check()
        os.rename(staging, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--freeze-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    torch.set_num_threads(4)
    report = build_review(args.root, args.freeze_sha256, args.output)
    print(trial.canonical(dict(status=report['status'], decision=report['decision'], output=str(args.output))))


if __name__ == '__main__': main()
