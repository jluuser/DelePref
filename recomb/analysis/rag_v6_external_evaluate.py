"""Evaluate complete precomputed scores; aggregate statistics only.

No model, retrieval, sequence generation, or phenotype-ranked edit output.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import rag_v6_retrieval as r


def average_ranks(values):
    a=np.asarray(values,dtype=float);order=np.argsort(a,kind='mergesort');result=np.empty(len(a))
    start=0
    while start<len(a):
        end=start+1
        while end<len(a) and a[order[end]]==a[order[start]]:end+=1
        result[order[start:end]]=(start+1+end)/2
        start=end
    return result


def spearman(a,b):
    if len(a)<2:return None
    x,y=average_ranks(a),average_ranks(b);x-=x.mean();y-=y.mean()
    denom=float(np.sqrt(np.dot(x,x)*np.dot(y,y)))
    return float(np.dot(x,y)/denom) if denom>0 else None


def predictions(rows,entries,level):
    expected={row['row_id']:row for row in rows};found={}
    if level=='row':
        for item in entries:
            key=item['row_id']
            if key in found:raise ValueError('duplicate prediction row')
            found[key]=float(item['score'])
        if set(found)!=set(expected):raise ValueError('all methods must cover exactly the same complete primary row set')
    else:
        mapped={}
        for item in entries:
            key=(item['row_id'],int(item['deleted_position']))
            if key in mapped:raise ValueError('duplicate mapping prediction')
            mapped[key]=float(item['score'])
        required={(key,p) for key,row in expected.items() for p in row['possible_deleted_positions']}
        if set(mapped)!=required:raise ValueError('mapping scores must cover every legal map, with no outcome-based filtering')
        found={key:float(np.mean([mapped[(key,p)] for p in row['possible_deleted_positions']])) for key,row in expected.items()}
    if not all(np.isfinite(x) for x in found.values()):raise ValueError('nonfinite predictions are not silently removed')
    return found


def evaluate(rows,scores):
    groups={}
    for row in rows:groups.setdefault((row['target'],row['assay_id']),[]).append(row)
    assays=[];per_target={};eligible_targets=set();prediction_undefined=set()
    for (target,assay),members in sorted(groups.items()):
        y=[x['measurement'] for x in members];p=[scores[x['row_id']] for x in members]
        truth_eligible=len(y)>1 and len(set(y))>1
        rho=spearman(y,p) if truth_eligible else None
        reason=None if rho is not None else ('singleton_or_constant_measurements' if not truth_eligible else 'constant_predictions')
        assays.append(dict(target=target,assay_id=assay,physical_observations=len(members),rho=rho,undefined_reason=reason))
        if truth_eligible:
            eligible_targets.add(target)
            if rho is None:prediction_undefined.add(target)
            else:per_target.setdefault(target,[]).append(rho)
    targets={target:(None if target in prediction_undefined else float(np.mean(per_target[target]))) for target in sorted(eligible_targets)}
    macro=float(np.mean(list(targets.values()))) if targets and all(v is not None for v in targets.values()) else None
    return dict(assays=assays,target_mean_Spearman=targets,macro_Spearman=macro,
                eligible_targets=len(eligible_targets),total_targets=len({x['target'] for x in rows}),
                constant_prediction_targets=sorted(prediction_undefined),
                aggregation='equal_conditions_within_target_then_equal_eligible_target; no method-specific target dropping')


def target_cluster_intervals(rows,scores,replicates=2000):
    """Same protein sequence is resampled jointly across its assay conditions."""
    targets={}
    for row in rows:targets.setdefault(row['target'],[]).append(row)
    result={}
    for target,members in sorted(targets.items()):
        conditions={}
        for row in members:conditions.setdefault(row['assay_id'],[]).append(row)
        conditions={a:rs for a,rs in conditions.items() if len(rs)>1 and len({x['measurement'] for x in rs})>1}
        if not conditions:continue
        clusters=sorted({row['sequence'] for rs in conditions.values() for row in rs})
        lookup={a:{row['sequence']:row for row in rs} for a,rs in conditions.items()}
        seed=20261015+int(hashlib.sha256(target.encode()).hexdigest()[:8],16)
        rng=np.random.default_rng(seed);draws=rng.integers(len(clusters),size=(replicates,len(clusters)))
        values=[];undefined=0
        for draw in draws:
            correlations=[]
            for assay,mapping in lookup.items():
                sampled=[mapping[clusters[i]] for i in draw if clusters[i] in mapping]
                rho=spearman([x['measurement'] for x in sampled],[scores[x['row_id']] for x in sampled])
                if rho is None:break
                correlations.append(rho)
            if len(correlations)!=len(lookup):undefined+=1
            else:values.append(float(np.mean(correlations)))
        result[target]=dict(sequence_clusters=len(clusters),fixed_conditions=len(lookup),
            requested_replicates=replicates,defined_replicates=len(values),undefined_replicates=undefined,
            conditional_CI95=np.quantile(values,[.025,.975]).tolist() if values else None,
            seed=seed,unit='physical_sequence_cluster_shared_across_conditions',
            interpretation='Interval conditional on defined draws; undefined draws are disclosed, no condition is silently dropped.')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--scores',type=Path,required=True);p.add_argument('--level',choices=['row','mapping'],required=True)
    p.add_argument('--method',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--entry-contract',type=Path,required=True);a=p.parse_args()
    protocol=json.loads(a.protocol.read_text(encoding='utf-8'))
    contract=json.loads(a.entry_contract.read_text(encoding='utf-8'))
    if r.sha(a.protocol)!=contract['data_protocol_sha256'] or r.sha(__file__)!=contract['entry_sha256']:
        raise ValueError('evaluation entry or protocol differs from its sealed contract')
    if r.sha(a.dataset)!=protocol['primary_rows_sha256']:raise ValueError('evaluation cohort differs from sealed protocol')
    rows=[json.loads(x) for x in a.dataset.read_text(encoding='utf-8').splitlines() if x.strip()]
    entries=[json.loads(x) for x in a.scores.read_text(encoding='utf-8').splitlines() if x.strip()]
    aggregated=predictions(rows,entries,a.level)
    result=evaluate(rows,aggregated)
    result['per_target_cluster_intervals']=target_cluster_intervals(rows,aggregated)
    if result['macro_Spearman'] is not None:
        means=np.asarray(list(result['target_mean_Spearman'].values()))
        rng=np.random.default_rng(20261015);boot=means[rng.integers(len(means),size=(10000,len(means)))].mean(1)
        result['conditional_protein_bootstrap_CI95']=np.quantile(boot,[.025,.975]).tolist()
        result['uncertainty_limit']='Only four eligible protein targets; this interval is not evidence for large multi-span shrinking or all protein families.'
    result.update(method=a.method,score_level=a.level,input_scores_sha256=r.sha(a.scores),
                  dataset_sha256=r.sha(a.dataset),protocol_sha256=r.sha(a.protocol),
                  entry_contract_sha256=r.sha(a.entry_contract),model_fits=0,
                  ranking_recommendations_emitted=False)
    root=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013').resolve()
    if root not in a.output.resolve().parents:raise ValueError('report output outside experiment workspace')
    r.write_json(a.output,result,atomic=True)


if __name__=='__main__':main()
