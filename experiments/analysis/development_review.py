"""CPU-only historical development artifact review; never authorizes evidence gates.

No model/checkpoint deserialization, inference, GPU access, training, or folding.
The computational snapshot is verified before loading its paired bootstrap helper.
"""
import argparse
import csv
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import common

SEED = 20261013
ARMS = 'SABCD'
WAITING = 'DEVELOPMENT_REVIEW_WAITING'
READY = 'DEVELOPMENT_REVIEW_READY_FOR_HUMAN_OR_AGENT_EVIDENCE_REVIEW'
INTERPRETATION = (
    'Historical development-only diagnostics. Learning-rate selection and effect '
    'estimation share validation data; these are not external evidence or independent '
    'confirmation. G and T are realized bounds for historical T-selected pools, not '
    'matched G-selected results. Every C-B, C-S, C-A and C-D comparison, including '
    'unfavorable results, is retained. No automatic scientific pass decision.')


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name, lower=0., upper=None):
    require(type(value) in (float, int) and math.isfinite(value), name + ' must be finite numeric')
    require(value >= lower and (upper is None or value <= upper), name + ' out of range')
    return float(value)


def equal(actual, expected, name):
    if expected is None:
        require(actual is None, name + ' differs')
    elif type(expected) in (float, int) and type(actual) in (float, int):
        require(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10), name + ' differs')
    else:
        require(actual == expected, name + ' differs')


def keyed(rows, key, name):
    result = {}
    for row in rows:
        k = key(row)
        require(k not in result, 'duplicate ' + name + ': ' + str(k))
        result[k] = row
    return result


def hierarchical(rows, value='ce', keys=('protein_id', 'budget', 'generator')):
    require(bool(rows), 'empty hierarchy')
    if not keys:
        return math.fsum(r[value] for r in rows) / len(rows)
    groups = {}
    for row in rows:
        groups.setdefault(str(row[keys[0]]), []).append(row)
    return math.fsum(hierarchical(part, value, keys[1:]) for part in groups.values()) / len(groups)


def decision_key(row):
    return row['protein_id'], str(row['budget']), row['generator']


def expected_decisions(candidates):
    """Recompute frozen That selection/tie rule without importing GPU evaluation code."""
    decisions = []
    for mixed in (False, True):
        groups = {}
        for row in candidates:
            key = (row['protein_id'], str(row['budget']), 'mixed' if mixed else row['generator'])
            groups.setdefault(key, []).append(row)
        for (wt, budget, generator), pool in sorted(groups.items()):
            selected = max(pool, key=lambda r: (r['That'], r['candidate_id']))
            lo, hi = max(r['T_lower'] for r in pool), max(r['T_upper'] for r in pool)
            decisions.append({'protein_id': wt, 'budget': budget, 'generator': generator,
                'pool_size': len(pool), 'selected': selected['candidate_id'],
                **{k: selected[k] for k in ('Ghat','That','C','G_lower','G_upper','T_lower','T_upper')},
                'oracle_T_lower': lo, 'oracle_T_upper': hi,
                'raw_regret_lower': max(0., lo-selected['T_upper']),
                'raw_regret_upper': max(0., hi-selected['T_lower'])})
    return decisions


def budget_decisions(decisions):
    rows = []
    for budget in sorted({r['budget'] for r in decisions}):
        for source in sorted({r['generator'] for r in decisions if r['budget'] == budget}):
            part = [r for r in decisions if r['budget'] == budget and r['generator'] == source]
            rows.append({'budget': budget, 'generator': source,
                **{k: None if any(r[k] is None for r in part) else hierarchical(part, k, ('protein_id',))
                   for k in ('G_lower','G_upper','T_lower','T_upper','That','C','raw_regret_lower','raw_regret_upper')}})
    return rows


def verify_evaluation(directory, arm, best_ce, manifest, summaries, roster):
    candidates = common.read_jsonl(directory/'candidates.jsonl')
    observed = keyed(candidates, lambda r: r['candidate_id'], 'candidate')
    expected = keyed(manifest, lambda r: r['candidate_id'], 'frozen candidate')
    require(set(observed) == set(expected), str(directory) + ': candidate roster differs from frozen primary validation manifest')
    for cid, row in observed.items():
        for key in ('protein_id', 'budget', 'generator'):
            require(type(row[key]) is type(expected[cid][key]) and row[key] == expected[cid][key],
                    cid + ' candidate metadata ' + key + ' differs')
        summary = summaries[cid]
        equal(summary['protein_id'], expected[cid]['protein_id'], cid + ' label WT')
        equal(summary['budget'], expected[cid]['budget'], cid + ' label budget')
        if 'source' in expected[cid]:
            equal(summary['source'], expected[cid]['source'], cid + ' label source')
        finite(row['ce'], cid + ' candidate CE')
        finite(row['C'], cid + ' coverage', upper=1.)
        finite(row['That'], cid + ' That', upper=1.)
        # Positive known-pair counts imply retained pairs, so the frozen evaluator
        # cannot produce undefined Ghat for any completed validation candidate.
        finite(row['Ghat'], cid + ' Ghat', upper=1.)
        equal(row['That'], row['C']*row['Ghat'], cid + ' That=C*Ghat')
        require(type(row['valid_pairs']) is int and row['valid_pairs'] > 0, cid + ' invalid pair count')
        equal(row['valid_pairs'], summary['known_pairs'], cid + ' known pair count')
        equal(row['C'], summary['C'], cid + ' frozen coverage')
        for endpoint in ('G', 'T'):
            lo, hi = row[endpoint+'_lower'], row[endpoint+'_upper']
            for suffix in ('lower','upper'):
                equal(row[endpoint+'_'+suffix], summary[endpoint+'_'+suffix], cid + ' frozen endpoint '+endpoint)
            if lo is None or hi is None:
                require(endpoint == 'G' and lo is None and hi is None, cid + ' partial/undefined endpoint')
            else:
                finite(lo, cid+' lower '+endpoint, upper=1.); finite(hi, cid+' upper '+endpoint, upper=1.)
                require(lo <= hi, cid + ' inverted endpoint bounds')
    protein_rows = common.read_jsonl(directory/'proteins.jsonl')
    proteins = keyed(protein_rows, lambda r: r['protein_id'], 'protein')
    require(set(proteins) == set(roster), 'protein roster differs from complete validation WT roster')
    for wt, row in proteins.items():
        finite(row['ce'], wt + ' protein CE')
        equal(row['ce'], hierarchical([r for r in candidates if r['protein_id']==wt]), wt+' hierarchical CE')
    decisions = common.read_jsonl(directory/'decisions.jsonl')
    observed_decisions = keyed(decisions, decision_key, 'decision')
    expected_rows = expected_decisions(candidates)
    expected_map = keyed(expected_rows, decision_key, 'expected decision')
    require(set(observed_decisions) == set(expected_map), 'decision roster differs')
    for key, row in observed_decisions.items():
        for field, value in expected_map[key].items():
            equal(row[field], value, 'decision '+str(key)+' '+field)
    report = read_json(directory/'report.json')
    equal(report['status'], 'DEVELOPMENT_EVALUATED', 'evaluation status')
    equal(report['arm'], arm, 'evaluation arm')
    equal(report['candidates'], len(candidates), 'report candidate count')
    equal(report['proteins'], len(roster), 'report protein count')
    equal(report['hierarchical_ce'], hierarchical(candidates), 'report hierarchical CE')
    equal(report['hierarchical_ce'], best_ce, 'best trace/report CE')
    equal(report['hierarchical_ce'], math.fsum(r['ce'] for r in protein_rows)/len(roster), 'protein/report CE')
    observed_budget = keyed(report['budget_decisions'], lambda r:(r['budget'],r['generator']), 'report budget decision')
    expected_budget = keyed(budget_decisions(expected_rows), lambda r:(r['budget'],r['generator']), 'expected budget')
    require(set(observed_budget) == set(expected_budget), 'report source/budget roster differs')
    for key in expected_budget:
        for field, value in expected_budget[key].items():
            equal(observed_budget[key][field], value, 'budget report '+str(key)+' '+field)
    identities = {name: common.sha_file(directory/name) for name in
                  ('candidates.jsonl','proteins.jsonl','decisions.jsonl','report.json')}
    return {'candidates': candidates, 'proteins': protein_rows, 'decisions': decisions,
            'output_hashes': identities, 'validation_output': str(directory)}


def verify_trace(directory, status, train_config):
    train = common.read_jsonl(directory/'train_trace.jsonl')
    validation = common.read_jsonl(directory/'validation_trace.jsonl')
    steps = status['steps']; max_steps = train_config['max_steps']
    require(type(steps) is int and 0 < steps <= max_steps, 'invalid completed step count')
    require([r['step'] for r in train] == list(range(1,steps+1)), 'train trace steps are not contiguous/ordered')
    for row in train:
        finite(row['ce'], 'training CE')
    expected_steps = [i for i in range(1,steps+1) if i % train_config['eval_every']==0 or i==max_steps]
    require(bool(validation) and [r['step'] for r in validation] == expected_steps, 'validation trace steps differ from frozen evaluation cadence')
    best = math.inf; stale = 0
    for index, row in enumerate(validation):
        finite(row['ce'], 'validation CE')
        improved = row['ce'] < best
        require(type(row['improved']) is bool and row['improved'] == improved, 'validation improvement flag differs')
        stale = 0 if improved else stale+1
        best = min(best, row['ce'])
        require(index==len(validation)-1 or stale < train_config['patience'], 'trace continued beyond frozen patience')
    stopped = stale >= train_config['patience']
    require(type(status['stopped_by_patience']) is bool and status['stopped_by_patience']==stopped, 'patience stop status differs')
    require(validation[-1]['step']==steps and (stopped or steps==max_steps), 'completed run stopped before max steps/patience')
    finite(status['best_ce'], 'status best CE')
    equal(status['best_ce'], best, 'status best CE')
    selected = min(validation, key=lambda r:(r['ce'],r['step']))
    return {'best_step': selected['step'], 'best_ce': best, 'steps': steps,
            'stopped_by_patience': stopped, 'trailing_nonimproving_evaluations': stale,
            'train_trace': train, 'validation_trace': validation}


def load_statistics(snapshot):
    spec = importlib.util.spec_from_file_location('review_frozen_paired_statistics', Path(snapshot)/'paired_statistics.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def compare_inputs(rows, roster, statistics, confidence):
    comparisons = {}
    endpoints = sorted({r['endpoint'] for r in rows})
    for other in 'BSAD':
        comparisons['C-'+other] = {}
        for endpoint in endpoints:
            part = [r for r in rows if r['endpoint']==endpoint and r['method'] in ('C',other)]
            # Frozen bootstrap rejects unknown bounds. Preserve the complete roster and
            # explicitly mark the whole stratum undefined rather than complete-case filter.
            unknown = sorted(wt for wt in roster if any(
                not [r for r in part if r['wt_id']==wt and r['method']==arm] or
                any(r['lower'] is None or r['upper'] is None for r in part
                    if r['wt_id']==wt and r['method']==arm) for arm in ('C',other)))
            if unknown:
                value = {'status':'UNDEFINED_ENDPOINT_BOUNDS', 'undefined_wts':unknown,
                         'n_wts':len(roster), 'n_groups':len(set(roster.values())),
                         'expected_wts':roster, 'complete_case_filtering':False}
            else:
                value = statistics.paired_bootstrap(part,'C',other,roster,
                    replicates=10000,seed=SEED,confidence=confidence)
                value['status'] = 'DEVELOPMENT_DIAGNOSTIC'
            comparisons['C-'+other][endpoint] = value
    return comparisons


def paired_inputs(selected, roster):
    rows = []
    strata = sorted({(str(r['budget']),r['generator'])
                    for run in selected.values() for r in run['_evaluation']['decisions']})
    candidate_strata = sorted({(str(r['budget']),r['generator'])
                    for run in selected.values() for r in run['_evaluation']['candidates']})
    for arm, run in selected.items():
        evaluation = run['_evaluation']
        proteins = {r['protein_id']:r for r in evaluation['proteins']}
        decisions = {decision_key(r):r for r in evaluation['decisions']}
        for wt, component in sorted(roster.items()):
            base = {'wt_id':wt,'group_id':component,'method':arm,
                    'checkpoint_sha256':run['checkpoint_sha256'], 'lr':run['lr']}
            ce = proteins[wt]['ce']
            rows.append(base|{'endpoint':'CE_overall','lower':ce,'upper':ce,'status':'OBSERVED'})
            for budget, generator in candidate_strata:
                part = [r for r in evaluation['candidates'] if r['protein_id']==wt
                        and str(r['budget'])==budget and r['generator']==generator]
                value = hierarchical(part) if part else None
                rows.append(base|{'endpoint':f'CE|{budget}|{generator}', 'budget':budget,
                    'generator':generator, 'lower':value,'upper':value,
                    'status':'OBSERVED' if part else 'UNDEFINED_NO_POOL', 'candidate_count':len(part)})
            for budget, generator in strata:
                decision = decisions.get((wt,budget,generator))
                for endpoint in ('T','G'):
                    lo = decision[endpoint+'_lower'] if decision else None
                    hi = decision[endpoint+'_upper'] if decision else None
                    rows.append(base|{'endpoint':f'{endpoint}|{budget}|{generator}',
                        'budget':budget,'generator':generator,'lower':lo,'upper':hi,
                        'status':'OBSERVED_BOUNDS' if lo is not None and hi is not None else
                                 'UNDEFINED_ENDPOINT' if decision else 'UNDEFINED_NO_POOL',
                        'selected_candidate':decision['selected'] if decision else None,
                        'pool_size':decision['pool_size'] if decision else 0,
                        'selection':'historical_That_max_candidate_id_tie',
                        'pool_kind':'historical_mixed' if generator=='mixed' else 'historical_source_internal'})
    return rows


def atomic_text(path, text):
    """Replace the output entry, never truncate a linked existing destination."""
    path = common.confined_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix='.'+path.name+'.', suffix='.tmp')
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor,'w',encoding='utf-8',newline='') as handle:
            handle.write(text)
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def write_outputs(output, report, rows):
    output = common.confined_path(output)
    common.write_jsonl(output/'paired_inputs.jsonl', rows)
    report['paired_inputs_sha256'] = common.sha_file(output/'paired_inputs.jsonl')
    common.write_json(output/'report.json', report)
    table = io.StringIO(newline='')
    writer = csv.writer(table)
    writer.writerow(['arm','lr','state','best_step','best_ce','steps','convergence_review_required','checkpoint_sha256'])
    for run in report['runs']:
        writer.writerow([run.get(k,'') for k in ('arm','lr','state','best_step','best_ce','steps','convergence_review_required','checkpoint_sha256')])
    atomic_text(output/'engineering.csv',table.getvalue())
    lines = ['# ENGINEERING development artifact review', '', report['status'], '', INTERPRETATION, '',
             'No scientific pass decision or confirmation folding authorization is produced.', '',
             '| Arm | LR | State | Best step | Validation CE | Convergence review |',
             '| --- | --- | --- | --- | --- | --- |']
    for run in report['runs']:
        lines.append('| '+ ' | '.join(str(run.get(k,'—')) for k in
            ('arm','lr','state','best_step','best_ce','convergence_review_required'))+' |')
    if 'comparisons' in report:
        lines += ['', 'Differences are C minus comparator; negative CE favors C, positive G/T favors C.', '',
                  '| Comparison | Endpoint / budget / historical pool | Difference bounds | Confidence interval | Status |',
                  '| --- | --- | --- | --- | --- |']
        for comparison, endpoints in report['comparisons'].items():
            for endpoint, value in endpoints.items():
                lines.append(f"| {comparison} | {endpoint.replace('|',' / ')} | {value.get('difference_bounds','undefined')} | {value.get('confidence_interval','undefined')} | {value['status']} |")
    else:
        lines += ['', 'Waiting for all ten validated completed runs. No efficacy comparisons or LR selections were produced.']
    atomic_text(output/'ENGINEERING.md','\n'.join(lines)+'\n')


def build_review(*, root, config, manifest, splits, label_summary, cache_indices, output,
                 dispatch=None, freeze=None, run_root=None):
    root = Path(root).resolve()
    require(root == common.AUTHORIZED_ROOT.resolve(), 'unexpected experiment root')
    output = common.confined_path(output).resolve()
    dedicated = root/'reports/development_review'
    require(output==dedicated or dedicated in output.parents,
            'output must stay within the dedicated development review subtree')
    require(root/'paper' != output and root/'paper' not in output.parents, 'paper output is paused')
    seal_path = Path(freeze or root/'configs/execution_freeze.json').resolve()
    dispatch_path = Path(dispatch or root/'reports/dispatch_after_pilot.json').resolve()
    base = Path(run_root or root/'checkpoints').resolve()
    require(output!=base and base not in output.parents and output not in base.parents,
            'output overlaps run root')
    # All inputs are rooted explicitly and read-only. Output cannot overwrite inputs.
    for path in (config,manifest,splits,label_summary,seal_path,dispatch_path,*cache_indices):
        require(Path(path).resolve()!=output and output not in Path(path).resolve().parents, 'output contains an input')
    protocol = read_json(config); split = read_json(splits)
    require(Path(protocol['root']).resolve()==root, 'protocol root differs')
    require(protocol['seed']==SEED and protocol['model']['arms']==list(ARMS), 'unfrozen seed/arms')
    cfg = protocol['train']
    require(cfg['learning_rates']==[.0003,.001] and SEED in cfg['seeds'], 'unfrozen learning rates/seed')
    require(cfg['selection']=='hierarchical_protein_mean_validation_ce', 'unexpected LR selection objective')
    require(protocol['statistics']['bootstrap_replicates']==10000, 'unfrozen bootstrap replicates')
    confidence = protocol['statistics']['confidence']; require(0<confidence<1, 'invalid confidence')
    identity_paths = {'protocol':Path(config),'manifest':Path(manifest),'splits':Path(splits),
                      'label_summary_sha256':Path(label_summary)}
    identities = {k:common.sha_file(v) for k,v in identity_paths.items()}
    cache_hashes = {str(Path(p)):common.sha_file(p) for p in cache_indices}
    seal = read_json(seal_path)
    for name, path in identity_paths.items():
        require(root in path.resolve().parents, 'input outside root')
        relative = path.resolve().relative_to(root).as_posix()
        require(seal['input_hashes'].get(relative)==identities[name], 'execution freeze '+name+' identity differs')
    equal(split['status'],'SPLIT_FROZEN','split status')
    equal(split['protocol_sha256'],identities['protocol'],'split protocol')
    all_manifest = common.read_jsonl(manifest)
    keyed(all_manifest,lambda r:r['candidate_id'],'primary manifest candidate')
    for row in all_manifest:
        require(row['protein_id'] in split['assignment'], 'manifest WT has no split')
        require(row['generator']!='mixed', 'reserved mixed generator name in manifest')
    valid = [r for r in all_manifest if split['assignment'][r['protein_id']]=='validation']
    if 'primary_source' in protocol:
        require(all(r['source']==protocol['primary_source'] for r in all_manifest),
                'manifest includes a source outside the frozen primary cohort')
    # Use the manifest WT intersection, not the broader sensitivity-manifest cohort.
    roster = {wt:split['component'][wt] for wt in sorted({r['protein_id'] for r in valid})}
    require(bool(roster), 'empty primary validation WT roster')
    summaries = keyed(common.read_jsonl(label_summary),lambda r:r['candidate_id'],'label summary candidate')
    require(all(r['candidate_id'] in summaries for r in valid), 'missing validation label summary')
    dispatch_data = read_json(dispatch_path); snapshot = Path(dispatch_data['code_snapshot']).resolve()
    require(output!=snapshot and snapshot not in output.parents and output not in snapshot.parents,
            'output overlaps frozen code snapshot')
    code_hashes = dispatch_data['code_hashes']
    actual_snapshot = {p.relative_to(snapshot).as_posix():common.sha_file(p) for p in snapshot.rglob('*.py')}
    require(actual_snapshot==code_hashes, 'frozen code snapshot hashes differ from dispatch')
    computational = {k:v for k,v in code_hashes.items() if 'tests' not in Path(k).parts}
    require('paired_statistics.py' in computational, 'missing frozen paired statistics')
    statistics = load_statistics(snapshot)
    runs = []; selected = {}
    for arm in ARMS:
        arm_runs = []
        for lr in cfg['learning_rates']:
            directory = base/f'{arm}_s{SEED}_lr{lr:g}'
            run = {'arm':arm,'lr':lr,'seed':SEED,'directory':str(directory),
                   'state':'PENDING','convergence_review_required':arm in ('S','B')}
            if not (directory/'status.json').exists():
                run['state'] = 'INCOMPLETE' if directory.exists() else 'PENDING'
                run['reason'] = 'completed status.json is absent'
                runs.append(run); continue
            status = read_json(directory/'status.json')
            if status.get('status')!='DEVELOPMENT_RUN_COMPLETE':
                run['state']='INCOMPLETE'; run['observed_status']=status.get('status')
                runs.append(run); continue
            provenance = read_json(directory/'run.json')
            for key, value in identities.items():
                equal(provenance[key],value, str(directory)+' '+key+' hash')
            equal(provenance['cache_index_sha256'],cache_hashes,'cache index paths/hashes')
            equal(provenance['code'],computational,'run frozen computational code hashes')
            run_cfg = provenance['config']
            for key, value in {'arm':arm,'lr':lr,'seed':SEED,'technical':False,'steps':None}.items():
                equal(run_cfg[key],value,'run config '+key)
            for key, path in [('protocol',config),('manifest',manifest),('splits',splits),('label_summary',label_summary),('output',directory)]:
                require(Path(run_cfg[key]).resolve()==Path(path).resolve(), 'run config '+key+' path differs')
            require([str(Path(p)) for p in run_cfg['cache_index']]==list(cache_hashes), 'run config cache paths differ')
            equal(status['arm'],arm,'status arm')
            equal(status['convergence_review_required'],arm in ('S','B'),'status convergence review')
            require(bool(status['esmc_profile_hashes']), 'missing ESMC profile identities')
            run.update(verify_trace(directory,status,cfg))
            evaluation = verify_evaluation(directory/f"validation_{run['best_step']}",arm,run['best_ce'],valid,summaries,roster)
            run.update(state='COMPLETE',checkpoint_path=str(directory/'best.pt'),
                       checkpoint_sha256=common.sha_file(directory/'best.pt'),
                       esmc_profile_hashes=status['esmc_profile_hashes'],
                       artifact_hashes={name:common.sha_file(directory/name) for name in
                           ('run.json','status.json','train_trace.jsonl','validation_trace.jsonl')},
                       validation_output=evaluation['validation_output'],output_hashes=evaluation['output_hashes'])
            run['_evaluation'] = evaluation
            runs.append(run); arm_runs.append(run)
        if len(arm_runs)==2:
            selected[arm] = min(arm_runs,key=lambda r:(r['best_ce'],r['lr']))
    complete = all(r['state']=='COMPLETE' for r in runs)
    report = {'status':READY if complete else WAITING,'interpretation':INTERPRETATION,
              'seed':SEED,'expected_runs':10,'completed_runs':sum(r['state']=='COMPLETE' for r in runs),
              'runs':[{k:v for k,v in r.items() if k!='_evaluation'} for r in runs],
              'identities':identities|{'cache_index_sha256':cache_hashes,'code_snapshot':str(snapshot),
                  'code_hashes':code_hashes,'dispatch_sha256':common.sha_file(dispatch_path),
                  'execution_freeze_sha256':common.sha_file(seal_path),
                  'review_code_sha256':common.sha_file(__file__)},
              'validation_wt_components':roster,'frozen_validation_candidates':len(valid),
              'validation_sources':sorted({r.get('source','unknown') for r in valid}),
              'lr_tie_rule':'minimum hierarchical validation CE, then smaller numerical learning rate',
              'best_step_tie_rule':'earliest minimum validation CE',
              'statistics':{'replicates':10000,'seed':SEED,'confidence':confidence,
                  'aggregation':'within WT, then equal WT means within homology component, then equal component means',
                  'unknown_bounds':'full WT roster retained; undefined stratum receives no complete-case bootstrap'},
              'scientific_pass_decision':None,'confirmation_folding_authorized':False,
              'gpu_calls':0,'training_calls':0,'structure_calls':0,'output':str(output)}
    rows = []
    if complete:
        profiles = {tuple(sorted(r['esmc_profile_hashes'])) for r in runs}
        require(len(profiles)==1,'mixed run ESMC profiles')
        rows = paired_inputs(selected,roster)
        report['selected']={arm:{k:v for k,v in r.items() if k!='_evaluation'} for arm,r in selected.items()}
        report['comparisons']=compare_inputs(rows,roster,statistics,confidence)
    write_outputs(output,report,rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','config','manifest','splits','label-summary','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--cache-index',nargs='+',required=True)
    parser.add_argument('--dispatch'); parser.add_argument('--freeze'); parser.add_argument('--run-root')
    args = parser.parse_args()
    result = build_review(root=args.root,config=args.config,manifest=args.manifest,splits=args.splits,
        label_summary=args.label_summary,cache_indices=args.cache_index,output=args.output,
        dispatch=args.dispatch,freeze=args.freeze,run_root=args.run_root)
    print(json.dumps({'status':result['status'],'completed_runs':result['completed_runs'],'output':result['output']}))


if __name__=='__main__':
    main()
