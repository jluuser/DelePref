"""Protein/family-balanced fixed-variant metrics, explicit coverage and undefined cases."""
from collections import defaultdict,Counter
import json
import math
import numpy as np
from recomb_expanded_external_prepare_20261017 import ROOT,BASE,write,sha


def ranks(x):
    _,inverse,counts=np.unique(np.asarray(x,dtype=float),return_inverse=True,return_counts=True)
    ends=np.cumsum(counts);average=(ends-counts+1+ends)/2
    return average[inverse]


def spearman(x,y):
    if len(x)<2:return None
    a=ranks(x);b=ranks(y);a-=a.mean();b-=b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a.dot(b)/den) if den>0 else None


def rank_metrics(y,p,binary):
    y=np.asarray(y);p=np.asarray(p);N=len(y);rho=spearman(y,p)
    if N<2:return dict(n=N,Spearman=rho,NDCG=None,AUROC=None,top10pct_recall=None,top10pct_enrichment=None)
    rel=(ranks(y)-1)/(N-1);gain=np.power(2.,rel)-1.;discount=1/np.log2(np.arange(N)+2.)
    ideal=float(np.sort(gain)[::-1].dot(discount));actual=0.;offset=0
    for value in sorted(set(p),reverse=True):
        idx=np.flatnonzero(p==value);actual+=float(gain[idx].mean()*discount[offset:offset+len(idx)].sum());offset+=len(idx)
    auc=None;recall=None;enrichment=None
    if all(v in (0,1) for v in binary) and len(set(binary))==2:
        b=np.asarray(binary,dtype=int);n1=int(b.sum());n0=N-n1
        auc=float((ranks(p)[b==1].sum()-n1*(n1+1)/2)/(n1*n0))
        k=max(1,math.ceil(.1*N));remaining=k;hits=0.
        for value in sorted(set(p),reverse=True):
            idx=np.flatnonzero(p==value);take=min(remaining,len(idx));hits+=take*float(b[idx].mean());remaining-=take
            if remaining==0:break
        recall=hits/n1;enrichment=(hits/k)/(n1/N)
    return dict(n=N,Spearman=rho,NDCG=actual/ideal if ideal>0 else None,AUROC=auc,
                top10pct_recall=recall,top10pct_enrichment=enrichment)


def macro_bootstrap(values,component,replicates=10000):
    # The denominator remains proteins; sample family components with replacement.
    if not values or any(v is None for v in values.values()):return dict(value=None,CI95=None,status='UNDEFINED_REQUIRED_TARGET_RETAINED')
    groups=defaultdict(list)
    for target,v in values.items():groups[component[target]].append(v)
    sums=np.array([sum(v) for v in groups.values()]);counts=np.array([len(v) for v in groups.values()])
    rng=np.random.default_rng(20261017);samples=rng.integers(0,len(sums),(replicates,len(sums)))
    estimates=sums[samples].sum(1)/counts[samples].sum(1)
    return dict(value=float(np.mean(list(values.values()))),CI95=np.quantile(estimates,[.025,.975]).tolist(),
                status='DEFINED_ON_LABEL_INFORMATIVE_FIXED_ROSTER',proteins=len(values),families=len(groups))


def main():
    out=ROOT/BASE
    cohort=json.loads((out/'cohort_verified_manifest.json').read_text())
    for n,h in cohort['files'].items():assert sha(out/n)==h
    rows=[json.loads(l) for l in (out/'rows.jsonl').read_text().splitlines()]
    public={r['row_id']:r['scores'] for r in map(json.loads,(out/'public_score_matrix.jsonl').read_text().splitlines())}
    completion=json.loads((out/'scoring_completion.json').read_text());assert sha(out/'model_mapping_scores.json')==completion['scores_sha256']
    mapping=json.loads((out/'model_mapping_scores.json').read_text())
    exposure={x['target']:x for x in json.loads((out/'training_exposure.json').read_text())}
    component={k:v['family_component'] for k,v in exposure.items()}
    methods=set(k for v in public.values() for k in v)|set(k for v in mapping.values() for k in v)
    observations=[];coverage=Counter()
    for row in rows:
        scores=dict(public[row['row_id']]);maps=[mapping[k] for k in row['edit_keys']]
        for method in methods:
            if maps and all(method in m for m in maps):scores[method]=float(np.mean([m[method] for m in maps]))
        for method in scores:coverage[method]+=1
        observations.append(dict(row,scores=scores,training_exposed=exposure[row['target']]['threshold40_both80'],
                                 family_component=component[row['target']]))
    groups={
      'primary_unique_maps':[r for r in observations if r['primary_unique_mapping']],
      'all_compatible_with_equal_map_means':observations,
      'single_deletion':[r for r in observations if r['deleted_length']==1],
      'multiple_deletion':[r for r in observations if r['deleted_length']>1],
      'primary_no_threshold_training_hit':[r for r in observations if r['primary_unique_mapping'] and not r['training_exposed']],
      'primary_threshold_training_hit':[r for r in observations if r['primary_unique_mapping'] and r['training_exposed']]}
    assay_results=[];matrix={};comparisons=[];length_results=[]
    core=['original_Scorer','original_U','Tranception_L','PoET']
    for group,cohort_rows in groups.items():
        assays=defaultdict(list)
        for r in cohort_rows:assays[r['assay_id']].append(r)
        informative={a for a,rs in assays.items() if len(rs)>=2 and len({r['measurement'] for r in rs})>=2}
        methods_here=sorted(methods);matrix[group]={}
        by_model={}
        for method in methods_here:
            targets=defaultdict(list);phenotypes=defaultdict(lambda:defaultdict(list));undefined=[];missing=[]
            for assay,rs in sorted(assays.items()):
                covered=[r for r in rs if method in r['scores']]
                if len(covered)!=len(rs):
                    missing.append(assay)
                    result=dict(n=len(rs),covered=len(covered),Spearman=None,NDCG=None,AUROC=None,status='INCOMPLETE_FIXED_ASSAY_COVERAGE')
                else:result=dict(rank_metrics([r['measurement'] for r in rs],[r['scores'][method] for r in rs],
                                             [r['official_binary'] for r in rs]),covered=len(rs),status='FULL_FIXED_ASSAY_COVERAGE')
                if assay in informative:
                    targets[rs[0]['target']].append(result['Spearman']);phenotypes[rs[0]['selection_type']][rs[0]['target']].append(result['Spearman'])
                if result['Spearman'] is None:undefined.append(assay)
                if method in core and group=='primary_unique_maps' and len(covered)==len(rs) and assay in informative:
                    y=np.asarray([r['measurement'] for r in rs]);p=np.asarray([r['scores'][method] for r in rs]);rng=np.random.default_rng(20261017)
                    boot=[]
                    for _ in range(1000):
                        ix=rng.integers(0,len(rs),len(rs));v=spearman(y[ix],p[ix])
                        if v is not None:boot.append(v)
                    result['conditional_assay_row_bootstrap_CI95']=np.quantile(boot,[.025,.975]).tolist() if boot else None
                    result['conditional_assay_bootstrap_defined_replicates']=len(boot)
                assay_results.append(dict(group=group,method=method,assay_id=assay,target=rs[0]['target'],
                                          phenotype=rs[0]['selection_type'],**result))
            target_values={k:float(np.mean(v)) if all(x is not None for x in v) else None for k,v in targets.items()}
            by_model[method]=target_values
            phenotype_values={p:{k:float(np.mean(v)) if all(x is not None for x in v) else None for k,v in ts.items()}
                              for p,ts in phenotypes.items()}
            matrix[group][method]=dict(macro=macro_bootstrap(target_values,component),
                phenotype_macros={p:macro_bootstrap(v,component) for p,v in phenotype_values.items()},
                informative_assays=len(informative),all_assays=len(assays),undefined_assays=undefined,missing_coverage_assays=missing,
                target_mean_Spearman=target_values)
        for method in ['original_U','Tranception_L','Tranception_L_no_retrieval','PoET','SCISOR_U90_S_M1_single_deletion_only']:
            x=by_model['original_Scorer'];y=by_model.get(method,{})
            delta={k: x[k]-y[k] if x[k] is not None and y.get(k) is not None else None for k in x}
            comparisons.append(dict(group=group,comparison='original_Scorer minus '+method,result=macro_bootstrap(delta,component)))
        # Do not let deletion amount substitute for placement prediction.
        for assay,rs in assays.items():
            lengths=defaultdict(list)
            for r in rs:lengths[r['deleted_length']].append(r)
            if len(lengths)>1:
                for D,lr in sorted(lengths.items()):
                    for method in core+['Policy_additive_reference','deletion_length_only']:
                        v=spearman([r['measurement'] for r in lr],[r['scores'][method] for r in lr]) if all(method in r['scores'] for r in lr) else None
                        length_results.append(dict(group=group,assay_id=assay,deleted_length=D,method=method,n=len(lr),Spearman=v))
    write(BASE+'/assay_metrics.json',assay_results);write(BASE+'/baseline_matrix.json',matrix)
    write(BASE+'/paired_model_comparisons.json',comparisons);write(BASE+'/same_deletion_length_metrics.json',length_results)
    write(BASE+'/metrics_completion.json',dict(status='EXPANDED_FIXED_VARIANT_METRICS_COMPLETE',observations=len(rows),WT=len(exposure),
        methods=sorted(methods),model_coverage=dict(coverage),primary_unique_rows=len(groups['primary_unique_maps']),
        primary_unique_model_macros={m:v['macro'] for m,v in matrix['primary_unique_maps'].items()},
        missing_and_constant_targets_retained=True,public_scores_already_direction_adjusted=True,
        RAG_generation_not_evaluated_by_fixed_variant_scoring=True,phenotype_ranked_edit_recommendations=False))
    print(json.dumps(dict(status='EXPANDED_FIXED_VARIANT_METRICS_COMPLETE',WT=len(exposure),observations=len(rows))),flush=True)


if __name__=='__main__':main()
