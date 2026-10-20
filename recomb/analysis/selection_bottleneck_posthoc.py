"""Reproduce sealed posthoc pool-MSE and checkpoint-curve diagnostics on CPU.

The supplied hashes pin the two existing inline reports, which seal the original
audit, pool rows and historical checkpoint reports. New outputs are isolated;
neither CE-selected production checkpoints nor candidate choices are changed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import selection_bottleneck as sb

core, review, common = sb.core, sb.review, sb.common
BASE = Path('reports/replication_review/selection_bottleneck')
POSTHOC = 'posthoc_reproduction'


def observed(value, name):
    core.require(value is not None and np.isfinite(value), 'missing/unknown ' + name)
    return float(value)


def require_point(row, endpoint):
    lo = observed(row[endpoint+'_lower'], endpoint + ' truth')
    hi = observed(row[endpoint+'_upper'], endpoint + ' truth')
    core.equal(lo, hi, 'unknown ' + endpoint + ' truth bounds')
    return lo


def reproduce_equal(actual, expected, name):
    """Numeric tolerance covers roundoff only, never a scientific threshold."""
    if isinstance(expected, dict):
        core.equal(set(actual), set(expected), name + ' keys')
        return max((reproduce_equal(actual[k], expected[k], name+'/'+str(k)) for k in expected), default=0.)
    if isinstance(expected, list):
        core.equal(len(actual), len(expected), name + ' length')
        return max((reproduce_equal(a,b,name+f'/{i}') for i,(a,b) in enumerate(zip(actual,expected))),default=0.)
    if isinstance(expected, (int,float)) and not isinstance(expected,bool):
        a,b = observed(actual,name),observed(expected,name); difference = abs(a-b)
        core.require(difference<=1e-12, 'reproduction differs: '+name)
        return difference
    core.equal(actual,expected,name)
    return 0.


def pool_decomposition(candidates, pool_rows, roster, budgets, sources):
    expected = {(arm,wt,budget,source) for arm in sb.ARMS for wt in roster for budget in budgets for source in (*sources,'mixed')}
    mapping = core.keyed(pool_rows,lambda r:(r['arm'],r['protein_id'],str(r['budget']),r['generator']),'posthoc pool')
    core.equal(set(mapping),expected,'complete posthoc pool roster')
    result = {}
    for arm in sb.ARMS:
        grouped = sb.pools(list(candidates[arm].values())); result[arm] = {}
        for ep in ('G','T'):
            parts = []
            for wt,budget,source in sorted((wt,b,s) for wt in roster for b in budgets for s in sources):
                rows = grouped[(wt,budget,source)]; saved = mapping[(arm,wt,budget,source)]
                core.equal(len(rows),7,'balanced seven-candidate source pool')
                core.equal(saved['pool_size'],7,'stored source pool size')
                core.equal(saved['point_status'],'OBSERVED','unknown stored pool truth')
                ordered = sorted(rows,key=lambda r:(r['That'],r['candidate_id']),reverse=True)
                core.equal(saved['candidate_ids'],[r['candidate_id'] for r in ordered],'stored pool candidates/order')
                truth = np.array([require_point(r,ep) for r in rows])
                predicted = np.array([observed(r[ep+'hat'],ep+' prediction') for r in rows])
                error = predicted-truth
                for key,value in ((ep+'_predicted_mean',predicted.mean()),(ep+'_true_mean',truth.mean()),(ep+'_error_sd',error.std(ddof=0))):
                    reproduce_equal(saved[key],float(value),'stored pool '+key)
                parts.append((float(np.mean(error**2)),float(error.mean()**2),float(np.var(error,ddof=0))))
            total,center,within = np.asarray(parts).mean(axis=0).tolist()
            result[arm][ep] = dict(total_MSE=total,pool_mean_error_MSE=center,within_pool_centered_error_MSE=within,
                                   identity_absolute_error=abs(total-center-within))
    return result


def summarize_curve(curve, selected_step):
    core.require(bool(curve),'empty checkpoint curve')
    core.equal(len({r['step'] for r in curve}),len(curve),'duplicate checkpoint curve step')
    selected = next((r for r in curve if r['step']==selected_step),None)
    core.require(selected is not None,'CE-selected checkpoint absent from curve')
    best = max(curve,key=lambda r:(observed(r['mixed_T_three_budget_mean'],'curve T'),-r['step']))
    return dict(CE_selected_step=selected_step,selected_T=selected['mixed_T_three_budget_mean'],
                hindsight_best_T_step=best['step'],hindsight_best_T=best['mixed_T_three_budget_mean'],
                hindsight_T_gap=best['mixed_T_three_budget_mean']-selected['mixed_T_three_budget_mean'],curve=curve)


def checkpoint_curves(context, audit, reference, seals):
    ledger = core.keyed(reference['report_ledger'],lambda r:r['path'],'checkpoint report seal')
    expected_paths, prepared = set(),[]
    for run in audit['runs']:
        directory = context['root']/f"checkpoints/{run['arm']}_s{run['seed']}_lr0.001"
        core.equal(Path(run['directory']),directory,'sealed curve run directory')
        core.equal(set(run['artifact_hashes']),set(review.ARTIFACTS),'sealed run artifacts')
        for name,digest in run['artifact_hashes'].items():
            path=directory/name; review.scalar.reject_input_overlap(context['output'],[path])
            seals[str(path)] = review.checked_hash(path,digest,'sealed training curve '+name)
        status = core.read_json(directory/'status.json')
        core.equal(status['status'],'DEVELOPMENT_RUN_COMPLETE','curve completed status')
        core.equal(status['arm'],run['arm'],'curve status arm')
        trace = core.verify_trace(directory,status,context['protocol']['train'])
        for key in ('best_step','best_ce','steps'):
            core.equal(trace[key],run[key],'sealed curve '+key)
        core.equal(Path(run['validation_output']),directory/f"validation_{run['best_step']}",'frozen CE-selected path')
        for entry in trace['validation_trace']:
            expected_paths.add(str(directory/f"validation_{entry['step']}"/'report.json'))
        prepared.append((run,directory,trace))
    core.equal(set(ledger),expected_paths,'complete historical checkpoint report roster')
    rows = []
    for run,directory,trace in prepared:
        curve=[]
        for entry in trace['validation_trace']:
            validation=directory/f"validation_{entry['step']}"; path=validation/'report.json'
            review.scalar.reject_input_overlap(context['output'],[path])
            digest=review.checked_hash(path,ledger[str(path)]['sha256'],'sealed historical checkpoint report')
            seals[str(path)]=digest
            if entry['step']==run['best_step']:
                core.equal(digest,run['output_hashes']['report.json'],'CE-selected report seal')
            # Validate the full WT/candidate/source roster and frozen label truth,
            # including all unknown bounds, rather than trust an aggregate alone.
            evaluation=core.verify_evaluation(validation,run['arm'],entry['ce'],context['valid'],context['summaries'],context['roster'])
            for name,digest in evaluation['output_hashes'].items():
                seals[str(validation/name)]=digest
            for candidate in evaluation['candidates']:
                require_point(candidate,'G'); require_point(candidate,'T')
            report=core.read_json(path)
            selected=core.keyed(report['budget_decisions'],lambda r:(r['budget'],r['generator']),'checkpoint budget/source')
            budgets=sorted({str(r['budget']) for r in context['valid']})
            per_budget={b:require_point(selected[(b,'mixed')],'T') for b in budgets}
            curve.append(dict(step=entry['step'],ce=entry['ce'],per_budget_T=per_budget,
                              mixed_T_three_budget_mean=float(np.mean(list(per_budget.values())))))
        rows.append(dict(arm=run['arm'],seed=run['seed'],**summarize_curve(curve,run['best_step'])))
    return rows


def build_posthoc(*, root, freeze, output, decomposition_sha256, curve_sha256):
    root=Path(root).resolve(); output=Path(output).absolute()
    core.equal(output,root/BASE/POSTHOC,'dedicated posthoc output')
    context=review.read_context(root,freeze,output); seals={}
    references={}
    for name,digest in (('pool_error_decomposition.json',decomposition_sha256),('checkpoint_curve_audit.json',curve_sha256)):
        path=root/BASE/name; review.scalar.reject_input_overlap(output,[path])
        seals[str(path)]=review.checked_hash(path,digest,'pinned posthoc reference '+name)
        references[name]=core.read_json(path)
    decomposition=references['pool_error_decomposition.json']; checkpoint=references['checkpoint_curve_audit.json']
    core.equal(decomposition['status'],'DESCRIPTIVE_BALANCED_POOL_MSE_DECOMPOSITION','decomposition reference status')
    core.equal(checkpoint['status'],'READ_ONLY_CHECKPOINT_DECISION_CURVE_AUDIT','curve reference status')
    for reference in references.values():
        for key in ('GPU_calls','structure_calls','training_calls'):
            core.equal(reference[key],0,'reference CPU-only '+key)
    bottleneck_path=root/BASE/'report.json'; audit_path=root/'reports/replication_review/report.json'
    seals[str(bottleneck_path)]=review.checked_hash(bottleneck_path,decomposition['input_sha256']['report.json'],'sealed bottleneck report')
    seals[str(audit_path)]=review.checked_hash(audit_path,checkpoint['source_audit_sha256'],'sealed source audit')
    bottleneck=core.read_json(bottleneck_path); audit=core.read_json(audit_path)
    core.equal(bottleneck['status'],'CPU_SELECTION_BOTTLENECK_COMPLETE','bottleneck status')
    core.equal(bottleneck['implementation_sha256'],common.sha_file(sb.__file__),'bottleneck implementation seal')
    for path,digest in bottleneck['input_sha256'].items():
        path=review.input_path(root,Path(path),output); seals[str(path)]=review.checked_hash(path,digest,'bottleneck sealed input')
    for name,digest in bottleneck['artifact_sha256'].items():
        core.require(Path(name).name==name,'invalid bottleneck artifact path')
        path=root/BASE/name; seals[str(path)]=review.checked_hash(path,digest,'sealed bottleneck artifact')
    pools_path=root/BASE/'per_wt_pools.jsonl'
    review.checked_hash(pools_path,decomposition['input_sha256']['per_wt_pools.jsonl'],'original pool decomposition source')
    labels,_,label_ledger=sb.pred.load_labels(context)
    for cid,lab in labels.items():
        core.require(len(lab['unknown_pair_ids'])==0,'unknown retained pair labels: '+cid)
        require_point(context['summaries'][cid],'G'); require_point(context['summaries'][cid],'T')
    _,candidates,ensemble_seals=sb.sealed_ensemble(context,labels,label_ledger); seals.update(ensemble_seals)
    budgets=sorted({str(r['budget']) for r in context['valid']}); sources=sorted({r['generator'] for r in context['valid']})
    core.equal(budgets,['0.2','0.3','0.4'],'three frozen budgets'); core.equal(sources,['evocompress','scisor'],'two frozen sources')
    core.equal(len(context['roster']),10,'ten frozen WTs')
    result=pool_decomposition(candidates,common.read_jsonl(pools_path),context['roster'],budgets,sources)
    rows=checkpoint_curves(context,audit,checkpoint,seals)
    differences=dict(pool_MSE=reproduce_equal(result,decomposition['result'],'pool MSE'),
                     checkpoint_curves=reproduce_equal(rows,checkpoint['rows'],'checkpoint curves'))
    metadata=dict(posthoc=True,production_checkpoint_selection='unchanged original CE-selected checkpoint',
        GPU_calls=0,training_calls=0,structure_calls=0,PLM_forward_calls=0,variant_structure_files_read=0,
        confirmation_folding_authorized=False,source_sha256=seals,implementation_sha256=common.sha_file(__file__))
    reports={'pool_error_decomposition.json':dict(decomposition,result=result,**metadata),
             'checkpoint_curve_audit.json':dict(checkpoint,rows=rows,**metadata)}
    for path,digest in seals.items():
        review.checked_hash(Path(path),digest,'posthoc source after read')
    for name,report in reports.items():
        review.write_json(output/name,report)
    report=dict(status='CPU_POSTHOC_REPRODUCTION_COMPLETE',output=str(output),source_pool_count_per_arm=60,
        curve_count=len(rows),checkpoint_report_count=len(checkpoint['report_ledger']),reproduction_max_absolute_difference=differences,
        interpretation='Posthoc reused-validation reproduction. MSE alone decomposes into squared pool-mean error plus centered error variance. Hindsight T is optimistic descriptive evidence; it does not select or replace any production checkpoint.',
        artifact_sha256={name:common.sha_file(output/name) for name in reports},**metadata)
    review.write_json(output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','freeze','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    for name in ('decomposition-sha256','curve-sha256'):
        parser.add_argument('--'+name,required=True)
    report=build_posthoc(**vars(parser.parse_args()))
    print(json.dumps({k:report[k] for k in ('status','source_pool_count_per_arm','curve_count','checkpoint_report_count','reproduction_max_absolute_difference','output')}))


if __name__=='__main__':
    main()
