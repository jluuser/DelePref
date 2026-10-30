"""Review fixed scalar selectors on historical primary validation pools, CPU only.

Example (absolute input paths are recommended)::

    python analysis/review_scalar_baselines.py --root ROOT --config PROTOCOL \
      --full-manifest FULL --primary-manifest PRIMARY --splits SPLITS \
      --label-summary LABELS --cache-index INDEX0 INDEX1 \
      --baseline-output ROOT/reports/utility_baseline \
      --core-report ROOT/reports/development_review/report.json \
      --output ROOT/reports/development_review/scalar_baselines

No model imports, checkpoint deserialization, forward passes, GPU operations,
training, folding, tuning, publication, or scientific pass decisions.
"""
import argparse
import json
from pathlib import Path

import development_review as core

common = core.common
WAITING = 'SCALAR_BASELINE_DEVELOPMENT_REVIEW_WAITING'
READY = 'SCALAR_BASELINE_DEVELOPMENT_REVIEW_READY_FOR_HUMAN_OR_AGENT_EVIDENCE_REVIEW'
SELECTORS = {'U':'utility','L':'local_cosine_similarity','V':'static_relationship_coverage'}
INTERPRETATION = (
    'Historical primary-validation development diagnostics. C learning-rate selection '
    'and effect estimation share validation data. U is the original fixed utility '
    'scalar, L is fixed local cosine similarity, V is fixed static relationship '
    'coverage; none is retuned. C maximizes predicted T; each scalar baseline '
    'maximizes its own score. G is secondary at those selected candidates, not '
    'matched G-selected confirmation. Historical source-internal and mixed pools '
    'stay separate by budget. All favorable and unfavorable comparisons are '
    'reported. No external evidence, independent confirmation, scientific pass '
    'decision, or automatic confirmation folding authorization is produced.')


def disjoint_output(output, directory, description):
    directory = Path(directory).resolve()
    core.require(output!=directory and directory not in output.parents and output not in directory.parents,
                 'output overlaps '+description)


def reject_input_overlap(output, paths):
    for path in paths:
        path = Path(path).resolve()
        core.require(path!=output and output not in path.parents, 'output contains an input: '+str(path))


def frozen_hash(root, path, seal, name):
    path = Path(path).resolve()
    core.require(root in path.parents, name+' outside experiment root')
    checksum = common.sha_file(path)
    core.equal(checksum,seal['input_hashes'].get(path.relative_to(root).as_posix()),name+' execution freeze hash')
    return checksum


def verify_baseline(root, output, full, identities, cache_hashes, profiles, review_output):
    report = core.read_json(output/'report.json')
    core.equal(report['status'],'CPU_UTILITY_BASELINE_COMPLETE','baseline status')
    core.equal(report['protocol_sha256'],identities['protocol'],'baseline protocol')
    core.equal(report['manifest_sha256'],identities['full_manifest'],'baseline manifest')
    core.equal(report['cache_index_sha256'],cache_hashes,'baseline cache index identities')
    core.equal(sorted(report['embedding_profile_hashes']),sorted(profiles),'baseline embedding profiles')
    core.equal(report['output_sha256'],common.sha_file(output/'candidates.jsonl'),'baseline output hash')
    core.equal(report['weights'],{'global_similarity':.35,'retained_similarity':.45,'junction_plausibility':.20},'original utility weights')
    core.require(report['GPU_used'] is False and report['PLM_forward_calls']==0 and report['structure_calls']==0,
                 'baseline is not the frozen CPU-only scalar computation')
    implementation = root/'analysis/prepare_utility_baseline.py'
    core.equal(report['implementation_sha256'],common.sha_file(implementation),'baseline implementation hash')
    source_seal = root/'provenance/utility_baseline_sources.json'
    reject_input_overlap(review_output,[implementation,source_seal])
    core.equal(report['sources'],core.read_json(source_seal),'baseline frozen source provenance')
    core.require(set(report['sources'])=={'teacher_scoring','proposal_validation'},'unexpected baseline source functions')
    for name, source in report['sources'].items():
        reject_input_overlap(review_output,[source['original_path'],source['snapshot']])
        for field in ('original_path','snapshot'):
            core.equal(common.sha_file(source[field]),source['sha256'],'baseline source '+name+' '+field)
    rows = common.read_jsonl(output/'candidates.jsonl')
    observed = core.keyed(rows,lambda r:r['candidate_id'],'baseline candidate')
    expected = core.keyed(full,lambda r:r['candidate_id'],'full manifest candidate')
    core.require(set(observed)==set(expected),'baseline full candidate roster differs from frozen full manifest')
    core.equal(report['candidates'],len(expected),'baseline candidate count')
    core.equal(report['WT'],len({r['protein_id'] for r in full}),'baseline WT count')
    for cid,row in observed.items():
        for field in ('protein_id','budget','generator'):
            core.require(type(row[field]) is type(expected[cid][field]) and row[field]==expected[cid][field],
                         'baseline candidate metadata '+cid+' '+field+' differs')
        for field in ('utility','static_relationship_coverage','direct_contact_loss'):
            core.finite(row[field],cid+' '+field,upper=1.)
        core.finite(row['local_cosine_similarity'],cid+' local cosine similarity',lower=-1.,upper=1.)
    return observed, report


def verify_c(selected, learned, protocol, manifest, summaries, roster, identities, cache_hashes):
    core.equal(selected['arm'],'C','selected arm')
    core.equal(selected['seed'],core.SEED,'selected seed')
    core.require(selected['lr'] in protocol['train']['learning_rates'],'selected LR outside frozen set')
    c_runs = [r for r in learned['runs'] if r['arm']=='C']
    core.require(len(c_runs)==2 and all(r['state']=='COMPLETE' for r in c_runs),'incomplete C LR inventory')
    expected = min(c_runs,key=lambda r:(r['best_ce'],r['lr']))
    for field in ('lr','best_ce','best_step','checkpoint_path','checkpoint_sha256','validation_output'):
        core.equal(selected[field],expected[field],'selected C LR decision '+field)
    directory = Path(selected['directory'])
    core.require(Path(selected['checkpoint_path']).resolve()==(directory/'best.pt').resolve(),'selected C checkpoint path differs')
    core.equal(common.sha_file(selected['checkpoint_path']),selected['checkpoint_sha256'],'selected C checkpoint hash')
    core.require(set(selected['artifact_hashes'])=={'run.json','status.json','train_trace.jsonl','validation_trace.jsonl'},
                 'selected C run artifact identities incomplete')
    for name, checksum in selected['artifact_hashes'].items():
        core.equal(common.sha_file(directory/name),checksum,'selected C run artifact hash '+name)
    provenance = core.read_json(directory/'run.json')
    for name in ('protocol','manifest','splits','label_summary_sha256'):
        core.equal(provenance[name],identities[name],'selected C input identity '+name)
    core.equal(provenance['cache_index_sha256'],cache_hashes,'selected C cache identities')
    computational = {k:v for k,v in learned['identities']['code_hashes'].items() if 'tests' not in Path(k).parts}
    core.equal(provenance['code'],computational,'selected C frozen code identities')
    for name,value in {'arm':'C','seed':core.SEED,'lr':selected['lr'],'technical':False,'steps':None}.items():
        core.equal(provenance['config'][name],value,'selected C config '+name)
    status = core.read_json(directory/'status.json')
    core.equal(status['status'],'DEVELOPMENT_RUN_COMPLETE','selected C status')
    core.equal(status['arm'],'C','selected C status arm')
    core.equal(sorted(status['esmc_profile_hashes']),sorted(selected['esmc_profile_hashes']),
               'selected C status embedding profiles')
    trace = core.verify_trace(directory,status,protocol['train'])
    for name in ('best_step','best_ce','steps'):
        core.equal(selected[name],trace[name],'selected C trace '+name)
    validation = Path(selected['validation_output'])
    core.require(validation.resolve()==(directory/f"validation_{trace['best_step']}").resolve(),'selected C validation path differs')
    core.require(set(selected['output_hashes'])=={'candidates.jsonl','proteins.jsonl','decisions.jsonl','report.json'},
                 'selected C output identities incomplete')
    for name,checksum in selected['output_hashes'].items():
        core.equal(common.sha_file(validation/name),checksum,'selected C output hash '+name)
    return core.verify_evaluation(validation,'C',selected['best_ce'],manifest,summaries,roster)


def scalar_decisions(manifest, scores, summaries, selector):
    score_name = SELECTORS[selector]
    groups = {}
    for mixed in (False,True):
        for row in manifest:
            key = (row['protein_id'],str(row['budget']),'mixed' if mixed else row['generator'])
            groups.setdefault(key,[]).append(row)
    result = []
    for (wt,budget,generator),pool in sorted(groups.items()):
        selected = max(pool,key=lambda r:(scores[r['candidate_id']][score_name],r['candidate_id']))
        cid = selected['candidate_id']; summary = summaries[cid]
        result.append({'protein_id':wt,'budget':budget,'generator':generator,
            'pool_size':len(pool),'selected':cid,'score_name':score_name,
            'score':scores[cid][score_name],
            **{field:summary[field] for field in ('G_lower','G_upper','T_lower','T_upper')}})
    return result


def paired_inputs(decisions, roster, checkpoint_sha256):
    rows = []
    strata = sorted({(str(r['budget']),r['generator']) for part in decisions.values() for r in part})
    for method,part in decisions.items():
        lookup = core.keyed(part,core.decision_key,'scalar review decision')
        for wt,group in sorted(roster.items()):
            for budget,generator in strata:
                row = lookup.get((wt,budget,generator))
                for endpoint in ('T','G'):
                    lo,hi = (row[endpoint+'_lower'],row[endpoint+'_upper']) if row else (None,None)
                    rows.append({'wt_id':wt,'group_id':group,'method':method,
                        'endpoint':f'{endpoint}|{budget}|{generator}', 'budget':budget,'generator':generator,
                        'lower':lo,'upper':hi,'selected_candidate':row['selected'] if row else None,
                        'pool_size':row['pool_size'] if row else 0,
                        'selection_score':SELECTORS.get(method,'predicted_T'),
                        'selection_tie_rule':'maximum candidate_id',
                        'pool_kind':'historical_mixed' if generator=='mixed' else 'historical_source_internal',
                        'checkpoint_sha256':checkpoint_sha256 if method=='C' else None,
                        'status':'OBSERVED_BOUNDS' if lo is not None and hi is not None else
                                 'UNDEFINED_ENDPOINT' if row else 'UNDEFINED_NO_POOL'})
    return rows


def compare_inputs(rows,roster,statistics,confidence):
    result = {}
    for method in SELECTORS:
        result['C-'+method] = {}
        for endpoint in sorted({r['endpoint'] for r in rows}):
            part = [r for r in rows if r['endpoint']==endpoint and r['method'] in ('C',method)]
            missing = []
            for wt in roster:
                if any(not [r for r in part if r['wt_id']==wt and r['method']==arm] or
                       any(r['lower'] is None or r['upper'] is None for r in part if r['wt_id']==wt and r['method']==arm)
                       for arm in ('C',method)):
                    missing.append(wt)
            if missing:
                value = {'status':'UNDEFINED_ENDPOINT_BOUNDS','undefined_wts':sorted(missing),
                         'n_wts':len(roster),'n_groups':len(set(roster.values())),
                         'complete_case_filtering':False,'expected_wts':roster}
            else:
                value = statistics.paired_bootstrap(part,'C',method,roster,
                    replicates=10000,seed=core.SEED,confidence=confidence)
                value['status']='DEVELOPMENT_DIAGNOSTIC'
            result['C-'+method][endpoint] = value
    return result


def write_outputs(output,report,rows,decisions):
    # Fresh temporary files also replace existing hardlinks/symlinks safely.
    for filename,values in [('paired_inputs.jsonl',rows),('decisions.jsonl',decisions)]:
        core.atomic_text(output/filename,''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in values))
        report[filename+'_sha256']=common.sha_file(output/filename)
    core.atomic_text(output/'report.json',json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    text = ['# ENGINEERING scalar baseline development review','',report['status'],'',INTERPRETATION,'']
    if 'comparisons' in report:
        text += ['Differences are C minus scalar baseline. Positive T/G favors C.','',
                 '| Comparison | Endpoint / budget / historical pool | Difference bounds | Confidence interval | Status |',
                 '| --- | --- | --- | --- | --- |']
        for comparison,endpoints in report['comparisons'].items():
            for endpoint,value in endpoints.items():
                text.append(f"| {comparison} | {endpoint.replace('|',' / ')} | {value.get('difference_bounds','undefined')} | {value.get('confidence_interval','undefined')} | {value['status']} |")
    else:
        text += ['Waiting for the core trained-arm review to be ready; no scalar effect comparisons produced.']
    core.atomic_text(output/'ENGINEERING.md','\n'.join(text)+'\n')


def build_review(*,root,config,full_manifest,primary_manifest,splits,label_summary,
                 cache_indices,baseline_output,core_report,output,freeze=None,dispatch=None):
    root = Path(root).resolve()
    core.require(root==common.AUTHORIZED_ROOT.resolve(),'unexpected experiment root')
    output = common.confined_path(output).resolve()
    dedicated = root/'reports/development_review/scalar_baselines'
    core.require(output==dedicated or dedicated in output.parents,'output must stay within dedicated scalar baseline subtree')
    baseline_output = Path(baseline_output).resolve(); core_report = Path(core_report).resolve()
    freeze = Path(freeze or root/'configs/execution_freeze.json').resolve()
    dispatch = Path(dispatch or root/'reports/dispatch_after_pilot.json').resolve()
    reject_input_overlap(output,[config,full_manifest,primary_manifest,splits,label_summary,
                                core_report,freeze,dispatch,*cache_indices])
    disjoint_output(output,baseline_output,'baseline output')
    disjoint_output(output,root/'checkpoints','run root')
    disjoint_output(output,root/'provenance','provenance')
    report = {'status':WAITING,'interpretation':INTERPRETATION,'selectors':SELECTORS,
              'selection_tie_rule':'maximum candidate_id','seed':core.SEED,'output':str(output),
              'scientific_pass_decision':None,'confirmation_folding_authorized':False,
              'GPU_used':False,'PLM_forward_calls':0,'training_calls':0,'structure_calls':0}
    if not core_report.exists():
        report['reason']='core review report absent'; write_outputs(output,report,[],[]); return report
    learned = core.read_json(core_report)
    if learned.get('status')!=core.READY:
        report['reason']='core review not ready'; report['core_status']=learned.get('status')
        write_outputs(output,report,[],[]); return report
    expected_runs = {(arm,lr,core.SEED) for arm in core.ARMS for lr in (.0003,.001)}
    inventory = learned.get('runs',[])
    observed_runs = {(r.get('arm'),r.get('lr'),r.get('seed')) for r in inventory}
    core.require(learned.get('expected_runs')==10 and learned.get('completed_runs')==10
                 and len(inventory)==10 and observed_runs==expected_runs
                 and all(r.get('state')=='COMPLETE' for r in inventory),
                 'core run inventory must contain all ten frozen completed runs')
    selected = learned['selected']['C']
    disjoint_output(output,selected['directory'],'selected C run')
    disjoint_output(output,selected['validation_output'],'selected C validation')
    protocol = core.read_json(config); seal = core.read_json(freeze)
    core.require(Path(protocol['root']).resolve()==root and protocol['seed']==core.SEED,'frozen root/seed differs')
    identities = {name:frozen_hash(root,path,seal,name) for name,path in
        [('protocol',config),('full_manifest',full_manifest),('manifest',primary_manifest),
         ('splits',splits),('label_summary_sha256',label_summary)]}
    cache_hashes = {str(Path(p)):common.sha_file(p) for p in cache_indices}
    for name in ('protocol','manifest','splits','label_summary_sha256'):
        core.equal(learned['identities'][name],identities[name],'core review input identity '+name)
    core.equal(learned['identities']['cache_index_sha256'],cache_hashes,'core cache identities')
    core.equal(learned['identities']['execution_freeze_sha256'],common.sha_file(freeze),'core execution freeze identity')
    core.equal(learned['identities']['review_code_sha256'],common.sha_file(core.__file__),'core review implementation identity')
    dispatched = core.read_json(dispatch); snapshot = Path(dispatched['code_snapshot']).resolve()
    disjoint_output(output,snapshot,'frozen code snapshot')
    hashes = {p.relative_to(snapshot).as_posix():common.sha_file(p) for p in snapshot.rglob('*.py')}
    core.equal(hashes,dispatched['code_hashes'],'frozen code snapshot identities')
    core.equal(hashes,learned['identities']['code_hashes'],'core snapshot identities')
    core.equal(str(snapshot),learned['identities']['code_snapshot'],'core snapshot path')
    core.equal(common.sha_file(dispatch),learned['identities']['dispatch_sha256'],'core dispatch identity')
    statistics = core.load_statistics(snapshot)
    split = core.read_json(splits)
    core.equal(split['status'],'SPLIT_FROZEN','split status')
    core.equal(split['protocol_sha256'],identities['protocol'],'split protocol identity')
    full = common.read_jsonl(full_manifest); primary = common.read_jsonl(primary_manifest)
    full_ids = core.keyed(full,lambda r:r['candidate_id'],'full manifest candidate')
    primary_ids = core.keyed(primary,lambda r:r['candidate_id'],'primary manifest candidate')
    for cid,row in primary_ids.items():
        core.require(cid in full_ids and row==full_ids[cid],'primary candidate differs from full manifest')
        core.require(row['protein_id'] in split['assignment'],'primary WT absent from split')
        core.require(row['generator']!='mixed','reserved mixed generator name')
        if 'primary_source' in protocol:
            core.equal(row['source'],protocol['primary_source'],'primary manifest source')
    valid = [r for r in primary if split['assignment'][r['protein_id']]=='validation']
    roster = {wt:split['component'][wt] for wt in sorted({r['protein_id'] for r in valid})}
    core.require(bool(roster),'empty primary validation roster')
    core.equal(roster,learned['validation_wt_components'],'core primary validation WT roster')
    core.equal(len(valid),learned['frozen_validation_candidates'],'core primary validation candidate count')
    summaries = core.keyed(common.read_jsonl(label_summary),lambda r:r['candidate_id'],'label summary')
    core.require(all(r['candidate_id'] in summaries for r in valid),'missing frozen validation label')
    c_evaluation = verify_c(selected,learned,protocol,valid,summaries,roster,identities,cache_hashes)
    baseline,baseline_report = verify_baseline(root,baseline_output,full,identities,cache_hashes,
                                               selected['esmc_profile_hashes'],output)
    intersection = {r['candidate_id'] for r in valid}&set(baseline)
    core.require(intersection=={r['candidate_id'] for r in valid},'baseline primary validation intersection differs')
    for row in valid:
        core.equal(baseline[row['candidate_id']]['static_relationship_coverage'],summaries[row['candidate_id']]['C'],
                   'baseline/frozen relationship coverage')
    decisions = {'C':c_evaluation['decisions']}
    for method in SELECTORS:
        decisions[method] = scalar_decisions(valid,baseline,summaries,method)
    rows = paired_inputs(decisions,roster,selected['checkpoint_sha256'])
    confidence = protocol['statistics']['confidence']
    core.require(protocol['statistics']['bootstrap_replicates']==10000 and 0<confidence<1,'unfrozen statistics')
    report.update(status=READY,validation_wt_components=roster,validation_candidates=len(valid),
        full_baseline_candidates=len(full),identities=identities|{'cache_index_sha256':cache_hashes,
            'core_review_sha256':common.sha_file(core_report),'baseline_report_sha256':common.sha_file(baseline_output/'report.json'),
            'baseline_candidates_sha256':common.sha_file(baseline_output/'candidates.jsonl'),
            'review_code_sha256':common.sha_file(__file__),'frozen_code_hashes':hashes,
            'source_provenance':baseline_report['sources'],'baseline_implementation_sha256':baseline_report['implementation_sha256'],
            'selected_C_checkpoint_sha256':selected['checkpoint_sha256'],'selected_C_output_hashes':selected['output_hashes']},
        statistics={'replicates':10000,'seed':core.SEED,'confidence':confidence,
            'aggregation':'within WT, then equal WT means within homology component, then equal component means',
            'unknown_bounds':'all expected WTs retained; undefined stratum receives no complete-case bootstrap'},
        comparisons=compare_inputs(rows,roster,statistics,confidence))
    decision_rows = [{'method':method,**r} for method,part in decisions.items() for r in part]
    write_outputs(output,report,rows,decision_rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','config','full-manifest','primary-manifest','splits','label-summary',
                 'baseline-output','core-report','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--cache-index',nargs='+',required=True)
    parser.add_argument('--freeze'); parser.add_argument('--dispatch')
    args = vars(parser.parse_args()); args['cache_indices']=args.pop('cache_index')
    result = build_review(**args)
    print(json.dumps({'status':result['status'],'output':result['output']}))


if __name__=='__main__':
    main()
