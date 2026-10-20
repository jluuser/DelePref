"""CPU-only exploratory selection diagnosis of the sealed three-seed ensemble.

No selectors, predictions, checkpoints or frozen inputs are rewritten. Point
diagnostics require observed truth; fixed-denominator bounds retain every WT.
"""
import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
import replication_review as review

core, pred, common = review.core, review.prediction, review.common
ARMS = review.ARMS
COMPARISONS = (('C', 'B'), ('D', 'C'))


def ranks(values):
    """Ascending average ranks, including exact ties; no SciPy dependency."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind='stable'); result = np.empty(len(values))
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + 1 + end) / 2
        start = end
    return result


def correlation(x, y, rank=False):
    if any(v is None for v in [*x, *y]) or len(x) < 2:
        return None
    x, y = np.asarray(x, float), np.asarray(y, float)
    if rank:
        x, y = ranks(x), ranks(y)
    x, y = x - x.mean(), y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.clip(x @ y / denominator, -1, 1)) if denominator else None


def point(row, endpoint):
    lo, hi = row[endpoint + '_lower'], row[endpoint + '_upper']
    return lo if lo is not None and lo == hi else None


def hierarchy(rows, key):
    return None if not rows or any(r[key] is None for r in rows) else core.hierarchical(rows, key)


def descriptive(rows, key):
    """Undefined correlations remain counted; defined-pool mean is descriptive."""
    defined = [r for r in rows if r[key] is not None]
    unknown = sum(r.get('point_status') == 'UNDEFINED_POINT_TRUTH_BOUNDS' for r in rows)
    return dict(mean_defined=None if unknown else hierarchy(defined, key), defined=len(defined),
                undefined=len(rows) - len(defined), total=len(rows), unknown_truth_pools=unknown)


def error_metrics(rows, endpoint):
    x = [r[endpoint + 'hat'] for r in rows]; y = [point(r, endpoint) for r in rows]
    result = dict(candidates=len(rows), undefined_truth=sum(v is None for v in y),
                  pooled_pearson=correlation(x, y), pooled_spearman=correlation(x, y, True))
    if any(v is None for v in y):
        return result | dict(status='UNDEFINED_POINT_TRUTH_BOUNDS', mae=None, rmse=None, bias=None)
    errors = [dict(r, absolute=abs(a-b), square=(a-b)**2, error=a-b) for r, a, b in zip(rows, x, y)]
    return result | dict(status='OBSERVED', mae=hierarchy(errors, 'absolute'),
                         rmse=hierarchy(errors, 'square')**.5, bias=hierarchy(errors, 'error'))


def sealed_ensemble(context, labels, label_ledger):
    """Read only: validate original audit/ledger, full roster, paths and bytes."""
    root, output = context['root'], context['output']
    path = root / 'reports/replication_review/report.json'
    report_hash = common.sha_file(path); audit = core.read_json(path)
    core.equal(audit['status'], review.READY, 'replication readiness')
    core.require(audit['expected_runs'] == audit['completed_runs'] == len(audit['runs']) == 15,
                 'all fifteen completed runs required')
    core.equal({(r['arm'], r['seed']) for r in audit['runs']},
               {(a, s) for a in ARMS for s in review.SEEDS}, 'completed run roster')
    core.require(all(r['state'] == 'COMPLETE' for r in audit['runs']), 'incomplete run')
    core.equal(audit['seeds'], list(review.SEEDS), 'ensemble seeds')
    core.equal(audit['arms'], list(ARMS), 'ensemble arms')
    core.equal(audit['ensemble']['seeds'], list(review.SEEDS), 'ensemble fixed seed roster')
    core.equal(audit['ensemble']['seed_weights'], [1/3]*3, 'ensemble fixed seed weights')
    core.equal(audit['validation_wt_components'], context['roster'], 'ensemble WT roster')
    core.equal(audit['validation_candidates'], len(context['valid']), 'ensemble candidate count')
    for key, value in context['identities'].items():
        core.equal(audit['identities'][key], value, 'audit identity ' + key)
    ledger_path = path.parent / 'artifact_ledger.jsonl'
    ledger_hash = review.checked_hash(ledger_path, audit['artifact_ledger_sha256'], 'sealed replication ledger')
    ledger = common.read_jsonl(ledger_path)
    original_labels = [r for r in ledger if r.get('kind') in ('labels', 'wt_native_json')]
    core.equal(sorted(original_labels, key=lambda r:r['path']), sorted(label_ledger, key=lambda r:r['path']),
               'sealed label/native WT ledger')
    entries = core.keyed([r for r in ledger if 'arm' in r and 'seed' not in r],
                         lambda r:(r['arm'], r['candidate_id']), 'ensemble prediction')
    seeds = core.keyed([r for r in ledger if 'arm' in r and 'seed' in r],
                       lambda r:(r['arm'], r['seed'], r['candidate_id']), 'source prediction')
    expected = {(a, r['candidate_id']) for a in ARMS for r in context['valid']}
    core.equal(set(entries), expected, 'full ensemble prediction ledger roster')
    core.equal(set(seeds), {(a, s, cid) for a, cid in expected for s in review.SEEDS}, 'source ledger roster')
    runs = {(r['arm'],r['seed']):r for r in audit['runs']}
    for (arm,seed,cid), entry in seeds.items():
        core.equal(entry['path'],str(Path(runs[(arm,seed)]['validation_output'])/(cid+'.npz')), 'source prediction path')
        core.require(len(entry['sha256'])==64 and all(c in '0123456789abcdef' for c in entry['sha256']), 'source prediction digest')
    core.equal(set(audit['ensemble']['arms']), set(ARMS), 'ensemble evaluation roster')
    evaluations, candidates, seals = {}, {}, {str(path): report_hash, str(ledger_path): ledger_hash}
    for arm in ARMS:
        run = audit['ensemble']['arms'][arm]; directory = root / 'reports/replication_review/ensemble' / arm
        core.equal(Path(run['validation_output']), directory, 'ensemble output path')
        core.equal(set(run['output_hashes']), set(review.EVALUATION_FILES), 'ensemble output seals')
        review.scalar.reject_input_overlap(output, [directory / n for n in review.EVALUATION_FILES])
        for name, digest in run['output_hashes'].items():
            seals[str(directory/name)] = review.checked_hash(directory/name, digest, 'ensemble output ' + name)
        core.equal({p.name for p in directory.glob('*.npz')},
                   {r['candidate_id']+'.npz' for r in context['valid']}, 'ensemble NPZ file roster')
        evaluation = core.verify_evaluation(directory, arm, run['hierarchical_ce'], context['valid'],
                                            context['summaries'], context['roster'])
        evaluations[arm] = run | {'_evaluation': evaluation}
        observed = core.keyed(evaluation['candidates'], lambda r:r['candidate_id'], 'ensemble candidate')
        candidates[arm] = {}
        for row in context['valid']:
            cid = row['candidate_id']; entry = entries[(arm, cid)]; lab = labels[cid]
            probability_path = directory / (cid + '.npz')
            core.equal(entry['path'], str(probability_path), 'sealed ensemble prediction path')
            core.require(root in probability_path.resolve().parents, 'prediction escapes experiment root')
            core.equal(entry['source_prediction_sha256'], [seeds[(arm, s, cid)]['sha256'] for s in review.SEEDS],
                       'ensemble ordered source identities')
            p, _ = review.load_probability(probability_path, lab, output, entry['sha256'])
            recomputed = review.candidate_metrics(row, context['summaries'][cid], lab, p)
            for key in ('ce', 'Ghat', 'That', 'C'):
                pred.close(recomputed[key], observed[cid][key], cid + ' recomputed ' + key)
            ids = np.flatnonzero(lab['retained_mask']); positions = np.searchsorted(ids, lab['pair_ids'])
            known = np.asarray(p, float)[positions]; truth = pred.VALUES[lab['target']]
            residual = known @ pred.VALUES - truth
            candidates[arm][cid] = observed[cid] | dict(arm=arm, known_pairs=len(truth),
                unknown_pairs=len(ids)-len(truth), pair_preservation_mae=float(np.abs(residual).mean()),
                pair_preservation_brier=float(np.square(residual).mean()),
                multiclass_brier=float(np.square(known-np.eye(5)[lab['target']]).sum(1).mean()))
    for name, digest in seals.items():
        review.checked_hash(Path(name), digest, 'input after read')
    return evaluations, candidates, seals


def pools(rows):
    result = {}
    for row in rows:
        for source in (row['generator'], 'mixed'):
            result.setdefault((row['protein_id'], str(row['budget']), source), []).append(row)
    return result


def pool_metrics(rows, selected):
    ordered = sorted(rows, key=lambda r:(r['That'], r['candidate_id']), reverse=True)
    core.equal(selected, ordered[0]['candidate_id'], 'unchanged frozen Top1 choice')
    coverage = max(rows, key=lambda r:(r['C'], r['candidate_id']))
    preservation = max(rows, key=lambda r:(r['Ghat'], r['candidate_id']))
    truth = [point(r, 'T') for r in ordered]
    result = dict(selected=selected, winner_generator=ordered[0]['generator'], pool_size=len(rows),
                  candidate_ids=[r['candidate_id'] for r in ordered],
                  coverage_diagnostic_selected=coverage['candidate_id'], G_diagnostic_selected=preservation['candidate_id'],
                  coverage_winner_generator=coverage['generator'], G_diagnostic_winner_generator=preservation['generator'],
                  predicted_top2_gap=ordered[0]['That']-ordered[1]['That'] if len(rows)>1 else None)
    for key in ('ce','pair_preservation_mae','pair_preservation_brier','multiclass_brier'):
        result['selected_'+key] = ordered[0][key]
    for ep in ('G', 'T'):
        x = np.array([r[ep+'hat'] for r in ordered]); y = [point(r, ep) for r in ordered]
        result[ep+'_predicted_mean'] = float(x.mean()); result[ep+'_predicted_sd'] = float(x.std())
        result[ep+'_spearman'] = correlation(x, y, True); result[ep+'_pearson'] = correlation(x, y)
        result[ep+'_true_mean'] = float(np.mean(y)) if all(v is not None for v in y) else None
        result[ep+'_true_sd'] = float(np.std(y)) if all(v is not None for v in y) else None
        result[ep+'_demeaned_mae'] = float(np.abs((x-x.mean())-(np.array(y)-np.mean(y))).mean()) if all(v is not None for v in y) else None
        result[ep+'_error_sd'] = float(np.std(x-np.array(y))) if all(v is not None for v in y) else None
        result['selected_'+ep] = y[0]
        result['selected_abs_'+ep+'_error'] = abs(x[0]-y[0]) if y[0] is not None else None
    result['C_sd'] = float(np.std([r['C'] for r in rows]))
    result['C_vs_predicted_T_spearman'] = correlation([r['C'] for r in rows], [r['That'] for r in rows], True)
    result['C_vs_true_T_spearman'] = correlation([r['C'] for r in ordered], truth, True)
    result['C_vs_true_G_spearman'] = correlation([r['C'] for r in ordered], [point(r, 'G') for r in ordered], True)
    result['Ghat_vs_That_spearman'] = correlation([r['Ghat'] for r in rows], [r['That'] for r in rows], True)
    result['C_vs_Ghat_spearman'] = correlation([r['C'] for r in rows], [r['Ghat'] for r in rows], True)
    result['Gtrue_vs_Ttrue_spearman'] = correlation([point(r, 'G') for r in ordered], truth, True)
    fields = ('oracle_hit', 'regret', 'headroom', 'realized_T_gap_between_predicted_top2', 'oracle_sorted_true_top2_gap', 'coverage_selected_T', 'G_selected_T',
              'T_minus_coverage', 'oracle_minus_coverage', 'fraction_coverage_headroom_captured',
              'truth_best_predicted_rank', 'top1_recall', 'top2_recall', 'top3_recall', 'oracle_T')
    result.update({k:None for k in fields})
    result.update(oracle_candidate_ids=None, oracle_generators=None, oracle_tie_selected_generator=None,
                  point_status='UNDEFINED_POINT_TRUTH_BOUNDS')
    if all(v is not None for v in truth):
        oracle = max(truth); oracle_rows = [r for r, t in zip(ordered, truth) if t == oracle]
        rank = min(i+1 for i, t in enumerate(truth) if t == oracle)
        sorted_truth = sorted(truth, reverse=True)
        baseline = point(coverage, 'T'); headroom = oracle-baseline
        result.update(point_status='OBSERVED', oracle_T=oracle, oracle_hit=int(truth[0]==oracle),
            regret=oracle-truth[0], headroom=oracle-min(truth),
            realized_T_gap_between_predicted_top2=truth[0]-truth[1] if len(rows)>1 else None,
            oracle_sorted_true_top2_gap=sorted_truth[0]-sorted_truth[1] if len(rows)>1 else None,
            coverage_selected_T=baseline, G_selected_T=point(preservation, 'T'), T_minus_coverage=truth[0]-baseline,
            oracle_minus_coverage=headroom, fraction_coverage_headroom_captured=(truth[0]-baseline)/headroom if headroom else None,
            truth_best_predicted_rank=rank, **{f'top{k}_recall':int(rank<=k) for k in (1,2,3)},
            oracle_candidate_ids=[r['candidate_id'] for r in oracle_rows], oracle_generators=sorted({r['generator'] for r in oracle_rows}),
            oracle_tie_selected_generator=max(oracle_rows,key=lambda r:r['candidate_id'])['generator'])
    scale = result['T_error_sd']; gap = result['predicted_top2_gap']
    result['gap_over_within_pool_T_error_sd'] = gap/scale if scale and gap is not None else None
    return result


def delta_metrics(left, right):
    rows = []
    for cid, a in left.items():
        b = right[cid]; value = {k:a[k] for k in ('candidate_id','protein_id','budget','generator')}
        value.update(delta_ce=a['ce']-b['ce'], delta_Ghat=a['Ghat']-b['Ghat'], delta_That=a['That']-b['That'])
        for ep in ('G', 'T'):
            t = point(a, ep)
            value['delta_abs_'+ep+'_error'] = abs(a[ep+'hat']-t)-abs(b[ep+'hat']-t) if t is not None else None
        rows.append(value)
    return rows


def pool_delta(left, right):
    a = sorted(left,key=lambda r:(r['That'],r['candidate_id']),reverse=True)
    b = sorted(right,key=lambda r:(r['That'],r['candidate_id']),reverse=True)
    relevant = {r['candidate_id'] for r in a[:2]+b[:2]}; other = {r['candidate_id']:r for r in right}
    delta = [dict(r, d=r['ce']-other[r['candidate_id']]['ce']) for r in left]
    def part(ids):
        rows = [r for r in delta if (r['candidate_id'] in relevant)==ids]
        return dict(candidates=len(rows), candidate_mean_delta_ce=float(np.mean([r['d'] for r in rows])) if rows else None,
                    known_pair_weighted_delta_ce=float(np.average([r['d'] for r in rows],weights=[r['known_pairs'] for r in rows])) if rows else None)
    improve = core.hierarchical(delta,'d') < 0
    return dict(same_top1=int(a[0]['candidate_id']==b[0]['candidate_id']), ce_improves=int(improve),
                ce_improves_top1_unchanged=int(improve and a[0]['candidate_id']==b[0]['candidate_id']),
                delta_ce=core.hierarchical(delta,'d'), decision_relevant=part(True), other_candidates=part(False),
                decision_relevant_definition='union of predicted top two candidates from both frozen models')


def summarize(candidates, pool_rows):
    summaries = []
    strata = [('all','all')] + [(b,s) for b,s in sorted({(r['budget'],r['generator']) for r in pool_rows})]
    for arm in ARMS:
        for budget, source in strata:
            rows = [r for r in candidates[arm].values() if (budget=='all' or str(r['budget'])==budget) and (source in ('all','mixed') or r['generator']==source)]
            ps = [r for r in pool_rows if r['arm']==arm and (budget=='all' and r['generator']!='mixed' or r['budget']==budget and r['generator']==source)]
            value = dict(arm=arm, budget=budget, generator=source, WTs=len({r['protein_id'] for r in rows}),
                         ce=hierarchy(rows,'ce'), G=error_metrics(rows,'G'), T=error_metrics(rows,'T'))
            for key in ('pair_preservation_mae','pair_preservation_brier','multiclass_brier'):
                value[key] = hierarchy(rows,key)
            keys = ('G_spearman','T_spearman','G_pearson','T_pearson','G_demeaned_mae','T_demeaned_mae',
                    'G_predicted_sd','T_predicted_sd','G_true_sd','T_true_sd','G_error_sd','T_error_sd',
                    'oracle_hit','regret','headroom','selected_T','selected_G','coverage_selected_T','G_selected_T',
                    'T_minus_coverage','oracle_minus_coverage','fraction_coverage_headroom_captured',
                    'truth_best_predicted_rank','top1_recall','top2_recall','top3_recall','predicted_top2_gap',
                    'realized_T_gap_between_predicted_top2','oracle_sorted_true_top2_gap',
                    'gap_over_within_pool_T_error_sd','C_sd','C_vs_predicted_T_spearman','C_vs_true_T_spearman',
                    'C_vs_true_G_spearman','Ghat_vs_That_spearman','Gtrue_vs_Ttrue_spearman','C_vs_Ghat_spearman',
                    'selected_ce','selected_pair_preservation_mae','selected_pair_preservation_brier',
                    'selected_multiclass_brier','selected_abs_G_error','selected_abs_T_error')
            value['within_pool'] = {k:descriptive(ps,k) for k in keys}
            value['center_scales'] = {}
            for ep in ('G','T'):
                for kind in ('predicted','true'):
                    key = f'{ep}_{kind}_mean'; centers = [r[key] for r in ps]
                    wt_centers = [hierarchy([r for r in ps if r['protein_id']==wt],key) for wt in sorted({r['protein_id'] for r in ps})]
                    value['center_scales'][ep+'_'+kind] = dict(pool_center_sd=float(np.std(centers)) if all(v is not None for v in centers) else None,
                        WT_center_sd=float(np.std(wt_centers)) if all(v is not None for v in wt_centers) else None)
                    offsets = []
                    for wt,b in sorted({(r['protein_id'],r['budget']) for r in ps}):
                        group = [r[key] for r in ps if r['protein_id']==wt and r['budget']==b]
                        offsets.append(float(np.std(group)) if all(v is not None for v in group) else None)
                    value['center_scales'][ep+'_'+kind]['mean_within_WT_budget_source_center_sd'] = float(np.mean(offsets)) if all(v is not None for v in offsets) else None
            if source=='mixed':
                value['mixed_winner_counts'] = {name:{g:sum(r[name]==g for r in ps) for g in sorted({r['generator'] for r in rows})}
                    for name in ('winner_generator','oracle_tie_selected_generator','coverage_winner_generator','G_diagnostic_winner_generator')}
            summaries.append(value)
    return summaries


def paired_comparisons(evaluations, context):
    rows = core.paired_inputs(evaluations, context['roster']); stats = core.load_statistics(context['snapshot']); result = {}
    for left, right in (('C','B'),('D','C'),('D','B')):
        result[left+'-'+right] = {}
        for endpoint in sorted({r['endpoint'] for r in rows}):
            part = [r for r in rows if r['endpoint']==endpoint and r['method'] in (left,right)]
            absent = [wt for wt in context['roster'] if any(not [r for r in part if r['wt_id']==wt and r['method']==a] or
                any(r['lower'] is None or r['upper'] is None for r in part if r['wt_id']==wt and r['method']==a) for a in (left,right))]
            result[left+'-'+right][endpoint] = (dict(status='UNDEFINED_ENDPOINT_BOUNDS',undefined_wts=absent,expected_wts=context['roster']) if absent else
                stats.paired_bootstrap(part,left,right,context['roster'],replicates=10000,seed=core.SEED,confidence=context['protocol']['statistics']['confidence'])) | dict(complete_case_filtering=False)
    return result


def build_analysis(*, root, freeze, output):
    root = Path(root).resolve(); output = Path(output).absolute()
    core.equal(output, root/'reports/replication_review/selection_bottleneck', 'dedicated bottleneck output')
    context = review.read_context(root, freeze, output)
    labels, _, label_ledger = pred.load_labels(context)
    evaluations, candidates, seals = sealed_ensemble(context, labels, label_ledger)
    grouped = {a:pools(list(candidates[a].values())) for a in ARMS}; pool_rows, deltas, comparison_pools = [], [], []
    for arm in ARMS:
        decisions = core.keyed(evaluations[arm]['_evaluation']['decisions'], core.decision_key, 'frozen decision')
        core.equal(set(grouped[arm]), set(decisions), 'complete pool roster')
        for key, rows in sorted(grouped[arm].items()):
            wt, budget, source = key
            core.equal(len(rows), 14 if source=='mixed' else 7, 'frozen seven/fourteen candidate pool')
            frozen = {k:decisions[key][k] for k in ('G_lower','G_upper','T_lower','T_upper','oracle_T_lower','oracle_T_upper','raw_regret_lower','raw_regret_upper')}
            pool_rows.append(dict(arm=arm,protein_id=wt,budget=budget,generator=source,**frozen,**pool_metrics(rows,decisions[key]['selected'])))
    comparisons = {}
    for left, right in COMPARISONS:
        name = left+'-'+right; ds = delta_metrics(candidates[left],candidates[right]); deltas += [dict(r,comparison=name) for r in ds]
        ps = []
        for key in sorted(grouped[left]):
            wt,budget,source = key
            ps.append(dict(comparison=name,protein_id=wt,budget=budget,generator=source,**pool_delta(grouped[left][key],grouped[right][key])))
        comparison_pools += ps
        comparisons[name] = []
        for budget,source in [('all','all')]+sorted({(r['budget'],r['generator']) for r in ps}):
            sub = [r for r in ps if (budget=='all' and r['generator']!='mixed' or r['budget']==budget and r['generator']==source)]
            cs = [r for r in ds if (budget=='all' or str(r['budget'])==budget) and (source in ('all','mixed') or r['generator']==source)]
            comparisons[name].append(dict(budget=budget,generator=source,pools=len(sub),
                **{k:sum(r[k] for r in sub) for k in ('same_top1','ce_improves','ce_improves_top1_unchanged')},
                **{k:hierarchy(cs,k) for k in ('delta_ce','delta_abs_G_error','delta_abs_T_error','delta_Ghat','delta_That')},
                delta_ce_vs_delta_abs_T_error_pearson=correlation([r['delta_ce'] for r in cs],[r['delta_abs_T_error'] for r in cs]),
                decision_relevant_delta_ce=hierarchy([dict(r,v=r['decision_relevant']['candidate_mean_delta_ce']) for r in sub],'v'),
                other_candidates_delta_ce=hierarchy([dict(r,v=r['other_candidates']['candidate_mean_delta_ce']) for r in sub],'v')))
    agreement = {name:[] for name in ('C-B','C-D','D-B')}
    for name in agreement:
        left,right = name.split('-')
        for key in grouped[left]:
            wt,budget,source = key
            choice = lambda a:max(grouped[a][key],key=lambda r:(r['That'],r['candidate_id']))['candidate_id']
            agreement[name].append(dict(protein_id=wt,budget=budget,generator=source,same_top1=int(choice(left)==choice(right))))
    summary = summarize(candidates,pool_rows)
    overall = {r['arm']:r for r in summary if r['budget']=='all'}
    report = dict(status='CPU_SELECTION_BOTTLENECK_COMPLETE', output=str(output), validation_candidates=len(context['valid']),
        validation_wt_components=context['roster'], input_sha256=seals, implementation_sha256=common.sha_file(__file__),
        interpretation='Exploratory reused primary validation; all five sealed equal-probability three-seed ensembles. No independent confirmation, candidate retuning, or production selector changes.',
        aggregation='Candidate errors: equal WT -> budget -> source -> candidate. Global pools exclude duplicate mixed rows. Pooled correlations are descriptive candidate-level correlations; within-pool correlations use average tied ranks, with undefined constant pools counted. Bootstrap: frozen equal homology-component means, all WTs, 10000 replicates, seed 20261013.',
        uncertainty_margin='Predicted top-two gap divided by within-pool realized T-error SD is descriptive, not a calibrated uncertainty probability.',
        pair_metric_scope='CE, expected-preservation MAE, expected-preservation squared error (pair_preservation_brier), and categorical multiclass Brier use observed retained pairs within each candidate; unknown labels remain bounds in G/T and are never imputed.',
        diagnostic_selectors='coverage-only maximizes sealed C (the static relationship coverage); G-only maximizes frozen Ghat; both are posthoc diagnostics, never new production choices. T=C*G and truth-best T/G need not agree.',
        optional_U='Not loaded; prior first-seed C-versus-U evidence is not treated as an ensemble comparison.',
        CE_interpretation='Proper categorical CE followed by expected utility is Bayes consistent when probabilities are correct. Finite-sample aggregate error, covariance, coverage and small winner margins can prevent CE gains from changing within-pool T ordering.',
        D_scope='D uses edited-sequence embeddings but retains native WT structure and deletion features; it is not a pure sequence model.',
        D_evidence={other:dict(delta_ce=overall['D']['ce']-overall[other]['ce'],
            D=overall['D']['within_pool'], comparator=overall[other]['within_pool']) for other in 'SBC'},
        future_ranking_auxiliary='A future D-based within-WT/budget candidate-ranking auxiliary objective may be investigated if the reported within-pool ordering is weak; no objective/model training, thresholds or hyperparameter selection is performed here.',
        summary=summary, CE_translation=comparisons, choice_agreement={n:dict(pools=len(rs),same_top1=sum(r['same_top1'] for r in rs),
            by_stratum=[dict(budget=b,generator=s,pools=sum(r['budget']==b and r['generator']==s for r in rs),same_top1=sum(r['same_top1'] for r in rs if r['budget']==b and r['generator']==s)) for b,s in sorted({(r['budget'],r['generator']) for r in rs})]) for n,rs in agreement.items()},
        paired_comparisons=paired_comparisons(evaluations,context), GPU_used=False, training_calls=0, structure_calls=0,
        PLM_forward_calls=0, variant_structure_files_read=0, confirmation_folding_authorized=False, scientific_pass_decision=None)
    for path,digest in seals.items():
        review.checked_hash(Path(path),digest,'sealed input before output')
    artifacts = {'per_wt_pools.jsonl':pool_rows,'candidate_deltas.jsonl':deltas,'pool_comparisons.jsonl':comparison_pools}
    for name,rows in artifacts.items():
        review.write_rows(output/name,rows)
    table=io.StringIO(newline=''); columns=('arm','budget','generator','ce','G_mae','G_rmse','T_mae','T_rmse','T_within_spearman','oracle_hit','regret','selected_T','T_minus_coverage')
    writer=csv.DictWriter(table,fieldnames=columns); writer.writeheader()
    for r in summary:
        writer.writerow({**{k:r[k] for k in ('arm','budget','generator','ce')},
            **{ep+'_'+k:r[ep][k] for ep in ('G','T') for k in ('mae','rmse')},
            'T_within_spearman':r['within_pool']['T_spearman']['mean_defined'],
            **{k:r['within_pool'][k]['mean_defined'] for k in ('oracle_hit','regret','selected_T','T_minus_coverage')}})
    core.atomic_text(output/'summary.csv',table.getvalue())
    report['artifact_sha256'] = {name:common.sha_file(output/name) for name in [*artifacts,'summary.csv']}
    review.write_json(output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','freeze','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args(); report=build_analysis(**vars(args))
    print(json.dumps({k:report[k] for k in ('status','output','validation_candidates')}))


if __name__ == '__main__':
    main()
