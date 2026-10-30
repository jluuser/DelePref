"""Direct paired contrasts for the already fixed20/10 mechanism cohorts."""
import argparse
import json
import os
from pathlib import Path

from recomb_scope_screen_20261017 import read,sha,write
from recomb_benign_structure_evaluate_20261017 import paired_summary


def build(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    base=root/'reports/recomb_benign_general_readout_20261017';done=read(base/'completion.json')
    for n,h in done['outputs'].items():
        if sha(base/n)!=h:raise ValueError('complete general readout changed')
    out=root/'reports/recomb_benign_mechanism_contrasts_20261017'
    if out.exists():raise FileExistsError('mechanism contrasts already saved')
    values={(r['protein_id'],r['method']):r for r in read(base/'per_output_metrics.json') if r['budget']==.2}
    sub=read(root/'reports/recomb_benign_final_subsets_20261017/subsets.json')
    protocol=read(root/'reports/recomb_benign_prediction_catalog_20261017/statistical_protocol.json')
    comparisons=[('rag_unlimited',m,sub['completion_ablation']) for m in
        ['rag_position_gap_frequency','rag_marginal_gap','rag_cooccurrence_null','rag_direct_only','rag_no_scorer']]
    comparisons += [('rag_evidence_zero','delepref_unlimited',sub['scarcity']),('rag_evidence_min8','rag_unlimited',sub['scarcity'])]
    def interval(r,endpoint):
        return (r['geometry']['full_WT_TM']['lower'],r['geometry']['full_WT_TM']['upper']) if endpoint=='full_WT_TM' else (r['geometry'][endpoint+'_lower'],r['geometry'][endpoint+'_upper'])
    pairs=[];summaries=[]
    for left,right,ids in comparisons:
        for endpoint in ['retained_ca_lddt','contact_recovery','full_WT_TM']:
            group=[];identical_count=0
            for pid in ids:
                a=values[(pid,left)];b=values[(pid,right)]
                identical=(a['physical_id']==b['physical_id'] and a['correspondence']['wt_to_edit']==b['correspondence']['wt_to_edit']
                    and a['correspondence']['position_states']==b['correspondence']['position_states'])
                if identical:lo=hi=0.;identical_count+=1
                else:
                    al,au=interval(a,endpoint);bl,bu=interval(b,endpoint)
                    lo=al-bu if al is not None and bu is not None else None;hi=au-bl if au is not None and bl is not None else None
                group.append(dict(protein_id=pid,left_method=left,right_method=right,endpoint=endpoint,
                    difference_lower=lo,difference_upper=hi,exact_same_output_and_mapping=identical))
            pairs+=group
            summaries.append(dict(left_method=left,right_method=right,endpoint=endpoint,
                exact_identical_output_and_mapping=identical_count,
                **paired_summary(group,protocol['component_by_protein_id'],protocol['bootstrap_seed'])))
    out.mkdir();write(out/'per_protein_contrasts.json',pairs);write(out/'paired_contrasts.json',summaries)
    report=dict(status='PRESPECIFIED_COMPONENT_CONTRASTS_COMPLETED',mechanism_WT=20,scarcity_WT=10,
        no_new_selection_or_metric_definition=True,no_new_model_forwards=True,
        identity_handling='same physical prediction and same fixed WT correspondence imply exact difference0; marginal unknown bounds used otherwise',
        main_primary_analysis_unchanged=True,not_proof_of_joint_evolutionary_mechanism=True,
        general_readout_completion_sha256=sha(base/'completion.json'),source_script_sha256=sha(Path(__file__)),
        outputs={p.name:sha(p) for p in out.glob('*.json')},job_id=os.environ['SLURM_JOB_ID'])
    write(out/'completion.json',report)
    print(json.dumps([r for r in summaries if r['endpoint']!='full_WT_TM']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
    build(p.parse_args().root)
