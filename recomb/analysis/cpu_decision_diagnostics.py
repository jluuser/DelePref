"""CPU-only historical choice diagnostics on shared retained WT relationships.

Frozen primary selections remain unchanged. Common relationship sets include
unknown-coordinate outcomes, whose contribution is bounded by [0,1].
"""
import argparse
from decimal import Decimal, ROUND_FLOOR
import json
from pathlib import Path
import numpy as np
import cpu_prediction_diagnostics as pred

core=pred.core
common=pred.common
scalar=pred.scalar
METHODS='SABCDULV'
COMPARATORS='BSADULV'
READY='CPU_DECISION_DIAGNOSTICS_COMPLETE'


def common_retained(candidate_ids,labels):
    core.require(bool(candidate_ids),'empty candidate pool')
    masks=[labels[cid]['retained_mask'] for cid in candidate_ids]
    core.require(all(m.shape==masks[0].shape for m in masks),'reference E sizes differ within pool')
    return np.flatnonzero(np.logical_and.reduce(masks))


def common_bounds(label,ids):
    if not len(ids): return None,None,0,0
    known,indices,_=np.intersect1d(label['pair_ids'],ids,assume_unique=True,return_indices=True)
    n=len(known); unknown=len(ids)-n; contribution=float(pred.VALUES[label['target'][indices]].sum())
    return contribution/len(ids),(contribution+unknown)/len(ids),n,unknown


def pools(manifest):
    groups={}
    for row in manifest:
        for source in (row['generator'],'mixed'):
            groups.setdefault((row['protein_id'],str(row['budget']),source),[]).append(row)
    return groups


def common_decision_rows(manifest,labels,decisions,roster):
    grouped=pools(manifest); choices={method:core.keyed(rows,core.decision_key,'frozen method decision') for method,rows in decisions.items()}
    expected=set(grouped)
    for method,mapping in choices.items(): core.equal(set(mapping),expected,'frozen decision pool roster '+method)
    result=[]
    for (wt,budget,source),pool in sorted(grouped.items()):
        ids=common_retained([r['candidate_id'] for r in pool],labels)
        original=len(labels[pool[0]['candidate_id']]['retained_mask'])
        retained=[int(labels[r['candidate_id']]['retained_mask'].sum()) for r in pool]
        common_fraction=[len(ids)/n if n else None for n in retained]
        for method,mapping in choices.items():
            choice=mapping[(wt,budget,source)]; cid=choice['selected']
            core.require(cid in {r['candidate_id'] for r in pool},'frozen selected candidate outside pool')
            core.equal(choice['pool_size'],len(pool),'frozen choice pool size')
            lo,hi,n,u=common_bounds(labels[cid],ids)
            result.append(dict(wt_id=wt,group_id=roster[wt],method=method,endpoint=f'G_common|{budget}|{source}',
                budget=budget,generator=source,selected_candidate=cid,selection='unchanged frozen primary choice',
                pool_size=len(pool),common_retained_pairs=len(ids),original_reference_pairs=original,
                common_reference_fraction=len(ids)/original if original else None,
                candidate_retained_reference_fractions=[n/original if original else None for n in retained],
                common_fraction_of_each_candidate_retained=common_fraction,
                selected_known_common_pairs=n,selected_unknown_common_pairs=u,lower=lo,upper=hi,
                status='OBSERVED_BOUNDS' if len(ids) else 'UNDEFINED_EMPTY_COMMON_RETAINED',
                pool_kind='historical_mixed' if source=='mixed' else 'historical_source_internal'))
    return result


def compare_common(rows,roster,statistics,confidence,comparators=COMPARATORS):
    result={}
    for other in comparators:
        result['C-'+other]={}
        for endpoint in sorted({r['endpoint'] for r in rows}):
            part=[r for r in rows if r['endpoint']==endpoint and r['method'] in ('C',other)]
            absent=[]
            for wt in roster:
                if any(not [r for r in part if r['wt_id']==wt and r['method']==m] or
                       any(r['lower'] is None or r['upper'] is None for r in part if r['wt_id']==wt and r['method']==m)
                       for m in ('C',other)): absent.append(wt)
            if absent:
                value=dict(status='UNDEFINED_EMPTY_COMMON_RETAINED_OR_MISSING_POOL',undefined_wts=sorted(absent),
                           expected_wts=roster,n_wts=len(roster),n_groups=len(set(roster.values())),complete_case_filtering=False)
            else:
                value=statistics.paired_bootstrap(part,'C',other,roster,replicates=10000,seed=core.SEED,confidence=confidence)
                value.update(status='HISTORICAL_DEVELOPMENT_DIAGNOSTIC',complete_case_filtering=False)
            result['C-'+other][endpoint]=value
    return result


def fixed_bin(value):
    # Decimal(str(...)) fixes decimal edge behavior (0.15 belongs to bin 3).
    return int((Decimal(str(value))/Decimal('0.05')).to_integral_value(rounding=ROUND_FLOOR))


def matching_key(row,score):
    mask=np.asarray(row['mask'],dtype=bool)
    changes=np.diff(np.r_[False,mask,False].astype(int)); starts=np.flatnonzero(changes==1); ends=np.flatnonzero(changes==-1)
    lengths=tuple(sorted((ends-starts).tolist()))
    terminal=('both' if mask[0] and mask[-1] else 'N' if mask[0] else 'C' if mask[-1] else 'none')
    return (row['protein_id'],str(row['budget']),row['generator'],len(lengths),lengths,terminal,
            fixed_bin(score['static_relationship_coverage']),fixed_bin(score['direct_contact_loss']))


def matching_feasibility(manifest,scores,roster):
    groups={}
    for row in manifest: groups.setdefault(matching_key(row,scores[row['candidate_id']]),[]).append(row)
    group_rows=[]
    for key,part in sorted(groups.items()):
        wt,budget,generator,n,lengths,terminal,cbin,lbin=key
        c=[scores[r['candidate_id']]['static_relationship_coverage'] for r in part]
        loss=[scores[r['candidate_id']]['direct_contact_loss'] for r in part]
        feasible=len(part)>=2
        core.require(max(c)-min(c)<=.05+1e-12 and max(loss)-min(loss)<=.05+1e-12,'fixed-bin range exceeded')
        group_rows.append(dict(protein_id=wt,budget=budget,generator=generator,num_spans=n,
            sorted_span_lengths=list(lengths),terminal_type=terminal,coverage_bin=cbin,contact_loss_bin=lbin,
            candidates=[r['candidate_id'] for r in part],candidate_count=len(part),feasible=feasible,
            coverage_range=[min(c),max(c)],contact_loss_range=[min(loss),max(loss)]))
    strata=[]
    for budget,source in sorted({(str(r['budget']),r['generator']) for r in manifest}):
        per_wt=[]
        for wt in roster:
            part=[r for r in group_rows if r['protein_id']==wt and r['budget']==budget and r['generator']==source]
            feasible=[r for r in part if r['feasible']]
            per_wt.append(dict(protein_id=wt,total_groups=len(part),feasible_groups=len(feasible),
                candidates_in_feasible_groups=sum(r['candidate_count'] for r in feasible),
                eligible_candidates=sum(r['candidate_count'] for r in part)))
        strata.append(dict(budget=budget,generator=source,per_wt=per_wt,
            feasible_groups=sum(r['feasible_groups'] for r in per_wt),
            WTs_with_feasible_group=[r['protein_id'] for r in per_wt if r['feasible_groups']],
            expected_wts=sorted(roster)))
    feasible=[r for r in group_rows if r['feasible']]
    return dict(status='DESCRIPTIVE_FEASIBILITY_ONLY',bin_width=.05,minimum_candidates=2,
                total_groups=len(group_rows),feasible_groups=len(feasible),
                candidates_in_feasible_groups=sum(r['candidate_count'] for r in feasible),
                WTs_with_feasible_group=sorted({r['protein_id'] for r in feasible}),
                expected_wts=sorted(roster),by_budget_generator=strata),group_rows


def component_mean(values,roster):
    grouped={}
    for wt,value in values.items(): grouped.setdefault(roster[wt],[]).append(value)
    return float(np.mean([np.mean(part) for part in grouped.values()]))


def ce_influence(rows,roster,comparators='BSAD'):
    result={}
    for other in comparators:
        result['C-'+other]={}
        for endpoint in sorted({r['endpoint'] for r in rows if r['endpoint'].startswith('CE')}):
            differences={}
            for wt in roster:
                methods={m:[r for r in rows if r['endpoint']==endpoint and r['wt_id']==wt and r['method']==m] for m in ('C',other)}
                core.require(all(len(v)==1 for v in methods.values()),'CE influence missing/duplicate frozen WT')
                for part in methods.values():
                    core.equal(part[0]['lower'],part[0]['upper'],'CE point bound')
                    core.equal(part[0]['group_id'],roster[wt],'CE homology component')
                differences[wt]=methods['C'][0]['lower']-methods[other][0]['lower']
            loo={wt:component_mean({w:v for w,v in differences.items() if w!=wt},roster)
                 if len(roster)>1 else None for wt in roster}
            result['C-'+other][endpoint]=dict(mean=component_mean(differences,roster),per_WT=differences,
                leave_one_WT_out=loo,unfavorable_WTs=sorted(wt for wt,v in differences.items() if v>0),
                definition='C minus comparator CE; positive is unfavorable to C',new_bootstrap=False)
    return result


def source_bias(manifest,labels,summaries,scores,roster):
    rows=[]; per_wt=[]; result=[]
    for row in manifest:
        cid=row['candidate_id']; summary=summaries[cid]; target=labels[cid]['target']; n=len(target)
        result_row={k:row[k] for k in ('protein_id','budget','generator','candidate_id')}
        result_row.update(true_class_fraction=(np.bincount(target,minlength=5)/n).tolist() if n else None,
                          known_true_G=float(pred.VALUES[target].mean()) if n else None,
                          **{k:summary[k] for k in ('G_lower','G_upper','T_lower','T_upper')},
                          static_relationship_coverage=scores[cid]['static_relationship_coverage'],
                          direct_contact_loss=scores[cid]['direct_contact_loss'])
        rows.append(result_row)
    fields=('true_class_fraction','known_true_G','G_lower','G_upper','T_lower','T_upper',
            'static_relationship_coverage','direct_contact_loss')
    for budget,generator in sorted({(str(r['budget']),r['generator']) for r in rows}):
        part=[r for r in rows if str(r['budget'])==budget and r['generator']==generator]
        for wt in roster:
            wt_rows=[r for r in part if r['protein_id']==wt]
            core.require(bool(wt_rows),'source bias missing WT pool')
            per_wt.append(dict(protein_id=wt,budget=budget,generator=generator,candidate_count=len(wt_rows),
                              **{f:np.mean([r[f] for r in wt_rows],axis=0).tolist() if all(r[f] is not None for r in wt_rows) else None for f in fields}))
        wt_rows=[r for r in per_wt if r['budget']==budget and r['generator']==generator]
        result.append(dict(budget=budget,generator=generator,n_wts=len(wt_rows),candidate_count=len(part),
                           **{f:np.mean([r[f] for r in wt_rows],axis=0).tolist() if all(r[f] is not None for r in wt_rows) else None for f in fields}))
    return dict(interpretation='Descriptive equal-candidate means within each WT, then equal WT means; source and mixed pools are not independent observations',
                by_budget_generator=result),per_wt


def load_frozen_choices(context):
    paths=context['paths']; report=core.read_json(paths['scalar_report'])
    core.equal(report['status'],scalar.READY,'scalar review readiness')
    core.equal(report['validation_wt_components'],context['roster'],'scalar validation WT roster')
    core.equal(report['validation_candidates'],len(context['valid']),'scalar validation candidates')
    identities=report['identities']
    for key in ('protocol','manifest','splits','label_summary_sha256','cache_index_sha256'):
        core.equal(identities[key],context['identities'][key],'scalar input '+key)
    core.equal(identities['core_review_sha256'],context['identities']['core_review_sha256'],'scalar/core report identity')
    core.equal(identities['frozen_code_hashes'],context['identities']['frozen_code_hashes'],'scalar snapshot identity')
    pred.checked_hash(scalar.__file__,identities['review_code_sha256'],'scalar implementation')
    pred.checked_hash(paths['utility_report'],identities['baseline_report_sha256'],'scalar utility report')
    pred.checked_hash(paths['utility_candidates'],identities['baseline_candidates_sha256'],'scalar utility candidates')
    pred.checked_hash(paths['scalar_decisions'],report['decisions.jsonl_sha256'],'scalar decisions')
    utility=core.read_json(paths['utility_report'])
    core.equal(utility['status'],'CPU_UTILITY_BASELINE_COMPLETE','utility readiness')
    core.equal(utility['output_sha256'],common.sha_file(paths['utility_candidates']),'utility output identity')
    core.equal(utility['protocol_sha256'],context['identities']['protocol'],'utility protocol identity')
    core.equal(utility['manifest_sha256'],identities['full_manifest'],'utility full manifest identity')
    seal=core.read_json(paths['freeze'])
    full_path=context['root']/'manifests/development.jsonl'
    scalar.frozen_hash(context['root'],full_path,seal,'full manifest')
    pred.checked_hash(full_path,identities['full_manifest'],'scalar full manifest')
    scalar.reject_input_overlap(context['output'],[full_path])
    core.equal(utility['cache_index_sha256'],context['identities']['cache_index_sha256'],'utility caches')
    core.equal(utility['GPU_used'],False,'utility GPU use')
    core.equal(utility['PLM_forward_calls'],0,'utility forward calls')
    core.equal(utility['structure_calls'],0,'utility structure calls')
    core.equal(utility['weights'],{'global_similarity':.35,'retained_similarity':.45,'junction_plausibility':.20},'utility fixed weights')
    pred.checked_hash(context['root']/'analysis/prepare_utility_baseline.py',utility['implementation_sha256'],'utility implementation')
    core.equal(utility['sources'],identities['source_provenance'],'utility source provenance')
    for source in utility['sources'].values():
        for key in ('original_path','snapshot'):
            scalar.reject_input_overlap(context['output'],[source[key]])
            pred.checked_hash(source[key],source['sha256'],'utility original source')
    scores=core.keyed(common.read_jsonl(paths['utility_candidates']),lambda r:r['candidate_id'],'utility candidate')
    core.equal(len(scores),utility['candidates'],'utility full candidate count')
    for row in context['valid']:
        cid=row['candidate_id']; core.require(cid in scores,'missing utility candidate')
        for key in ('protein_id','budget','generator'): core.equal(scores[cid][key],row[key],'utility metadata')
        for key in ('utility','static_relationship_coverage','direct_contact_loss'): core.finite(scores[cid][key],cid+' '+key,upper=1.)
        core.finite(scores[cid]['local_cosine_similarity'],cid+' local cosine',lower=-1.,upper=1.)
        core.equal(scores[cid]['static_relationship_coverage'],context['summaries'][cid]['C'],cid+' static label coverage')
    frozen=common.read_jsonl(paths['scalar_decisions'])
    choices={a:context['evaluations'][a]['decisions'] for a in core.ARMS}
    core.equal({r['method'] for r in frozen},set('CULV'),'scalar method roster')
    for method in 'CULV':
        actual=[{k:v for k,v in r.items() if k!='method'} for r in frozen if r['method']==method]
        verified=choices['C'] if method=='C' else scalar.scalar_decisions(context['valid'],scores,context['summaries'],method)
        core.equal(core.keyed(actual,core.decision_key,'scalar choice'),core.keyed(verified,core.decision_key,'verified frozen choice'),method+' frozen choice verification')
        if method!='C': choices[method]=actual
    context['identities'].update(scalar_report_sha256=common.sha_file(paths['scalar_report']),
                                scalar_decisions_sha256=common.sha_file(paths['scalar_decisions']),
                                utility_report_sha256=common.sha_file(paths['utility_report']),
                                utility_candidates_sha256=common.sha_file(paths['utility_candidates']))
    return choices,scores


def build_diagnostics(config):
    context=pred.load_context(config,'decision'); labels,references,ledger=pred.load_labels(context)
    choices,scores=load_frozen_choices(context)
    rows=common_decision_rows(context['valid'],labels,choices,context['roster'])
    statistics=core.load_statistics(context['snapshot']); cfg=context['config']['statistics']
    core.equal(cfg['bootstrap_replicates'],10000,'bootstrap replicates'); core.equal(cfg['seed'],core.SEED,'bootstrap seed')
    core.equal(cfg['comparisons'],['C-'+m for m in COMPARATORS],'prespecified comparison roster')
    core.equal(context['config']['common_retained']['selectors'],list(METHODS),'prespecified method roster')
    core.equal(cfg['confidence'],context['protocol']['statistics']['confidence'],'bootstrap confidence')
    paired_path=context['paths']['core_report'].parent/'paired_inputs.jsonl'
    scalar.reject_input_overlap(context['output'],[paired_path])
    pred.checked_hash(paired_path,context['learned']['paired_inputs_sha256'],'core paired inputs')
    paired=common.read_jsonl(paired_path)
    expected_paired=core.paired_inputs({a:{**context['learned']['selected'][a],'_evaluation':context['evaluations'][a]} for a in core.ARMS},context['roster'])
    core.equal(paired,expected_paired,'frozen paired CE/effect inputs')
    matching,matching_rows=matching_feasibility(context['valid'],scores,context['roster'])
    bias,bias_wt=source_bias(context['valid'],labels,context['summaries'],scores,context['roster'])
    coverage=[]
    for endpoint in sorted({r['endpoint'] for r in rows}):
        part=[r for r in rows if r['endpoint']==endpoint and r['method']=='C']
        core.equal(sorted(r['wt_id'] for r in part),sorted(context['roster']),'common coverage WT roster')
        coverage.append(dict(endpoint=endpoint,n_wts=len(part),empty_common_wts=[r['wt_id'] for r in part if not r['common_retained_pairs']],
                             mean_common_reference_fraction=float(np.mean([r['common_reference_fraction'] for r in part])),
                             minimum_common_retained_pairs=min(r['common_retained_pairs'] for r in part),
                             maximum_common_retained_pairs=max(r['common_retained_pairs'] for r in part)))
    report=pred.base_report(context)
    report.update(status=READY,implementation_sha256=common.sha_file(__file__),
        shared_input_audit_implementation_sha256=common.sha_file(pred.__file__),
        common_retained_definition=context['config']['common_retained'],statistics=cfg,
        comparisons=compare_common(rows,context['roster'],statistics,cfg['confidence']),common_coverage=coverage,
        static_matching=matching,source_bias=bias,ce_influence=ce_influence(paired,context['roster']),
        primary_choice_changes=0,decision_methods=list(METHODS),
        unknown_policy='Common sets intersect retained masks, independently of known label status. Unknown selected outcomes contribute [0,1]; empty sets remain undefined for the full WT roster.')
    pred.write_artifacts(context['output'],report,{'common_bounds.jsonl':rows,'matching_groups.jsonl':matching_rows,
        'source_bias_per_wt.jsonl':bias_wt,'artifact_ledger.jsonl':ledger})
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--config',required=True)
    result=build_diagnostics(parser.parse_args().config)
    print(json.dumps({'status':result['status'],'output':result['output']}))


if __name__=='__main__': main()
