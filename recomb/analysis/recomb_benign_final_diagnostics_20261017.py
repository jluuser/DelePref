"""Frozen-roster source, fragmentation, homology exposure and matched-K controls."""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import numpy as np

from recomb_scope_screen_20261017 import read,sha,write
from recomb_benign_structure_evaluate_20261017 import paired_summary
from delepref_v8_structure_metrics import common_retained_diagnostic


def build(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    out=root/'reports/recomb_benign_final_diagnostics_20261017'
    if out.exists():raise FileExistsError('diagnostics already saved')
    cat=root/'reports/recomb_benign_prediction_catalog_20261017';pdir=root/'reports/recomb_benign_structure_evaluation_20261017'
    done=read(pdir/'completion.json')
    if sha(pdir/'per_output_metrics.json')!=done['outputs']['per_output_metrics.json']:raise ValueError('full metric source changed')
    all_rows=read(pdir/'per_output_metrics.json');rows=[r for r in all_rows if r['profile_name']=='esmfold' and r['budget']==.2 and r['evaluation_role']=='comparison']
    panel={r['protein_id']:r for r in read(root/'reports/recomb_benign_scope_20261017/general_panel.json')}
    protocol=read(cat/'statistical_protocol.json');artifact={r['physical_id']:r for r in read(pdir/'prediction_identity_audit.json')}
    coverage={r['WT_sha256']:r for r in read(root/'reports/recomb_general100_input_coverage_20261017/per_WT.json') if r['protein_id'] in panel}
    train=read(root/'reports/delepref_v8_panel/training_homology_audit.json')
    nn={r['sequence_sha256']:r for r in train['queries'] if r['sequence_sha256'] in {p['sequence_sha256'] for p in panel.values()}}
    if len(nn)!=44 or len(coverage)!=44:raise ValueError('reviewed input metadata coverage differs')
    homology=[];fragment=[];common=[];paired_source=[]
    for pid,p in sorted(panel.items()):
        record=nn[p['sequence_sha256']];any_hit=record['nearest_retrieved_any'];covered=record['nearest_retrieved_both_covered']
        if record['threshold_hits']!=0:raise ValueError('a frozen query fails configured training exclusion')
        partial_domain_like=bool(any_hit and any_hit['identity_shorter_full']>=.4 and any_hit['query_coverage']>=.8 and any_hit['target_coverage']<.8)
        homology.append(dict(protein_id=pid,WT_sha256=p['sequence_sha256'],source=p['stratum'],training_WT=269915,
            retrieved_training_hits=record['retrieved_training_hits'],configured_threshold_hits=record['threshold_hits'],
            nearest_returned_identity_shorter_full=any_hit['identity_shorter_full'] if any_hit else None,
            nearest_returned_query_coverage=any_hit['query_coverage'] if any_hit else None,
            nearest_returned_target_coverage=any_hit['target_coverage'] if any_hit else None,
            nearest_both80covered_identity_shorter_full=covered['identity_shorter_full'] if covered else None,
            high_query_low_target_coverage_homology=partial_domain_like,search_hit_cap_reached=record['at_configured_hit_cap']))
        methods={r['method']:r for r in rows if r['protein_id']==pid}
        for method,r in methods.items():
            states=r['correspondence']['position_states'];deleted=np.array([s=='deleted' for s in states])
            span_count=int(np.sum(np.diff(np.r_[False,deleted].astype(int))==1)) if 'ambiguous' not in states else None
            fragment.append(dict(protein_id=pid,method=method,source=p['stratum'],span_count=span_count,
                greater_than_original3span_training_range=span_count>3 if span_count is not None else None,
                static_relationship_loss_lower=r['geometry']['static_relationship_loss_lower'],
                static_relationship_loss_upper=r['geometry']['static_relationship_loss_upper'],
                static_contact_loss=r['geometry']['static_contact_loss'],retained_relationship_G_lower=r['geometry']['G_lower'],
                retained_relationship_G_upper=r['geometry']['G_upper']))
        a=methods['rag_unlimited'];b=methods['delepref_unlimited']
        xyz=read(Path(p['wt_native_json']));ref=np.full((len(p['sequence']),3),np.nan)
        for i,v in xyz['coordinates'].items():ref[int(i)]=v
        def mapped(r):
            kept=np.array([i for i,s in enumerate(r['correspondence']['position_states']) if s=='stable'],dtype=int)
            with np.load(artifact[r['physical_id']]['artifact_path'],allow_pickle=False) as data:
                pred=data['coordinates'][[r['correspondence']['wt_to_edit'][i] for i in kept]]
            return pred,kept
        pa,ka=mapped(a);pb,kb=mapped(b);c=common_retained_diagnostic(ref,pa,ka,pb,kb)
        if a['physical_id']==b['physical_id'] and a['correspondence']==b['correspondence']:c['difference_lower']=c['difference_upper']=0.
        common.append(dict(protein_id=pid,source=p['stratum'],**c))
        for endpoint in ['retained_ca_lddt','contact_recovery']:
            paired_source.append(dict(protein_id=pid,source=p['stratum'],endpoint=endpoint,
                difference_lower=a['geometry'][endpoint+'_lower']-b['geometry'][endpoint+'_upper'],
                difference_upper=a['geometry'][endpoint+'_upper']-b['geometry'][endpoint+'_lower']))
    summaries=[]
    for source in ['cath_domain','pdb_chain']:
        for endpoint in ['retained_ca_lddt','contact_recovery']:
            selected=[r for r in paired_source if r['source']==source and r['endpoint']==endpoint]
            summaries.append(dict(source=source,endpoint=endpoint,**paired_summary(selected,protocol['component_by_protein_id'],protocol['bootstrap_seed'])))
    common_pairs=[dict(protein_id=r['protein_id'],difference_lower=r.get('difference_lower'),difference_upper=r.get('difference_upper')) for r in common]
    common_summary=paired_summary(common_pairs,protocol['component_by_protein_id'],protocol['bootstrap_seed'])
    sub=read(root/'reports/recomb_benign_final_subsets_20261017/subsets.json');control=[]
    for method in ['delepref_unlimited','rag_unlimited','scisor_native','scisor_K8_same_scorer']:
        selected=[r for r in all_rows if r['profile_name']=='esmfold' and r['budget']==.2 and r['protein_id'] in sub['completion_ablation'] and r['method']==method]
        if len(selected)!=20:raise ValueError('K8 control cohort mismatch')
        control.append(dict(method=method,N=20,CA_lddt=float(np.mean([r['geometry']['retained_ca_lddt_lower'] for r in selected])),
            native_contact_recovery=float(np.mean([r['geometry']['contact_recovery_lower'] for r in selected])),
            full_WT_TM_lower=float(np.mean([r['geometry']['full_WT_TM']['lower'] for r in selected])),
            full_WT_TM_upper=float(np.mean([r['geometry']['full_WT_TM']['upper'] for r in selected]))))
    out.mkdir()
    for name,data in [('training_homology44.json',homology),('fragmentation_and_static_loss.json',fragment),('common_retained_pairs.json',common),('source_stratified_effects.json',summaries),('SCISOR_K8_same_scorer20.json',control)]:write(out/name,data)
    with (out/'training_homology44.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(homology[0]));writer.writeheader();writer.writerows(homology)
    report=dict(status='FIXED_ROSTER_INPUT_AND_COMMON_RELATIONSHIP_DIAGNOSTICS_COMPLETE',WT=44,
        source_counts=dict(Counter(p['stratum'] for p in panel.values())),superkingdom=dict(Counter(p['superkingdom'] for p in coverage.values())),
        length_min=min(p['sequence_length'] for p in panel.values()),length_max=max(p['sequence_length'] for p in panel.values()),
        unique_exact_functional_maps=sum(p['functional_mapping_status']=='UNIQUE_EXACT_SUBSEQUENCE_MAP' for p in coverage.values()),
        configured_training_threshold_hits=0,high_query_low_target_coverage_homology=sum(r['high_query_low_target_coverage_homology'] for r in homology),
        homology_contract=train['contract'],homology_search_scope=train['scope'],
        not_strict_homology_family_disjoint=True,no_reselection_of_cohort=True,
        span_greater3_by_method={m:sum(r['greater_than_original3span_training_range'] is True for r in fragment if r['method']==m) for m in ['delepref_unlimited','rag_unlimited','terminal','scisor_native']},
        common_retained_relationship_effect=common_summary,new_GPU_calls=0,
        source_sha256=sha(Path(__file__)),outputs={p.name:sha(p) for p in out.glob('*') if p.is_file()},job_id=os.environ['SLURM_JOB_ID'])
    write(out/'completion.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['outputs','homology_contract']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
    build(p.parse_args().root)
