"""CPU-only preparation of controls, subsets and statistical protocol before predictions."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import delepref_v8_controls as controls
import delepref_unlimited_planning as planning
import delepref_v8_runtime as rt
from delepref_v8_baseline import ROOT,write

FREEZE='configs/delepref_v8_evaluation_preparation_freeze.json'
BASE='reports/delepref_v8_evaluation_preparation'


def prepare(root):
    sources=['analysis/delepref_v8_controls.py','analysis/delepref_v8_evaluation_prepare.py',
        'analysis/tests/test_delepref_v8_controls.py','slurm/delepref_v8_evaluation_prepare.sbatch',
        'analysis/delepref_unlimited_planning.py','analysis/delepref_unlimited_scorer.py',
        'code/build_pairs.py','code/paired_statistics.py','code/adapters/structure_adapter.py','code/budget_ledger.py']
    inputs=['reports/delepref_v8_native_panel/main_panel.json','reports/delepref_v8_native_panel/completion.json',
        'retrieval/delepref_v8_main/profiles.json','retrieval/delepref_v8_main/completion.json',
        'configs/delepref_v8_rag_unlimited_freeze.json','configs/delepref_v8_main_retrieval_freeze.json',
        'configs/delepref_v8_baseline_freeze.json']
    (root/BASE).mkdir(exist_ok=False)
    write(root/FREEZE,dict(status='V8_EVALUATION_INPUT_PREPARATION_FROZEN',contract=controls.CONTRACT,
        sources={p:rt.sha(root/p) for p in sources},inputs={p:rt.sha(root/p) for p in inputs},
        original_checkpoint_sha256=rt.CHECKPOINT_SHA,CPU_only=True,method_outputs_not_yet_frozen=True,
        structural_prediction_allowed=False,new_PLM_forwards=0,new_structure_predictions=0,paper_changes=False))
    return dict(freeze_sha256=rt.sha(root/FREEZE))


def execute(root,checksum):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU Slurm required')
    if rt.sha(root/FREEZE)!=checksum:raise ValueError('preparation freeze changed')
    freeze=json.loads((root/FREEZE).read_text())
    if freeze['contract']!=controls.CONTRACT:raise ValueError('control specification changed')
    for p,h in {**freeze['sources'],**freeze['inputs']}.items():
        if rt.sha(root/p)!=h:raise ValueError('frozen source/input changed: '+p)
    out=root/BASE;start=time.time()
    write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID'],freeze_sha256=checksum))
    panel=json.loads((root/'reports/delepref_v8_native_panel/main_panel.json').read_text())
    profiles=json.loads((root/'retrieval/delepref_v8_main/profiles.json').read_text())
    native_complete=json.loads((root/'reports/delepref_v8_native_panel/completion.json').read_text())
    retrieval_complete=json.loads((root/'retrieval/delepref_v8_main/completion.json').read_text())
    if rt.sha(root/'reports/delepref_v8_native_panel/main_panel.json')!=native_complete['panel_sha256']:
        raise ValueError('native panel identity mismatch')
    if rt.sha(root/'retrieval/delepref_v8_main/profiles.json')!=retrieval_complete['profiles_sha256']:
        raise ValueError('retrieval profile identity mismatch')
    if len(panel)!=100 or set(profiles)!={r['protein_id'] for r in panel}:raise ValueError('main roster mismatch')
    subsets=controls.fixed_subsets(panel);write(out/'subsets.json',subsets)
    inputs=[];candidates=[];evidence=[];conservation=[]
    import numpy as np
    for row in panel:
        pid=row['protein_id'];sequence=row['sequence'];profile=profiles[pid]
        if profile['query']['sequence_sha256']!=row['sequence_sha256'] or controls.sequence_sha(sequence)!=row['sequence_sha256']:
            raise ValueError('WT/retrieval identity mismatch')
        logits,audit=controls.conservation_logits(sequence,profile)
        conservation.append(dict(protein_id=pid,sequence_sha256=row['sequence_sha256'],logits=logits.tolist(),audit=audit))
        for budget in (.1,.2):
            ev=controls.input_evidence(sequence,profile,budget)
            evidence.append(dict(protein_id=pid,budget=budget,stratum=row['stratum'],**ev))
            for seed in (13,29):
                cseed=controls.condition_seed(seed,pid,sequence,budget)
                terminal=controls.terminal_proposals(len(sequence),budget)
                if audit['all_unknown']:msa=[];cost=dict(status='needs_exact_DelePref_fallback_after_original_Policy')
                else:
                    msa,cost=planning.policy_proposals(logits,budget,cseed)
                    if len(msa)!=8:raise RuntimeError('incomplete MSA K: '+pid)
                for method,items in [('terminal',terminal),('msa_conservation',msa)]:
                    for rank,item in enumerate(items,1):
                        mask=planning.validate_plan(len(sequence),budget,item['spans'])
                        candidates.append(dict(protein_id=pid,sequence_sha256=row['sequence_sha256'],budget=budget,
                            generation_seed=seed,condition_seed=cseed,method=method,raw_candidate_rank=rank,
                            deleted_intervals=item['spans'],deleted_length=int(mask.sum()),span_count=len(item['spans']),
                            deletion_mask_sha256=hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest(),
                            qmax=None,generation_cost=cost if method=='msa_conservation' else dict(attempts=8,duplicates=0),
                            requires_original_Scorer=True,selected=False))
        inputs.append(dict(protein_id=pid,wildtype_sequence=sequence,budget=.2))
    write(out/'main_encoder_inputs.json',inputs)
    write(out/'control_candidates.json',candidates);write(out/'conservation_evidence.json',conservation)
    write(out/'retrieval_budget_evidence.json',evidence)
    protocol=dict(status='STATISTICAL_AND_INPUT_SUBSET_PROTOCOL_FIXED_BEFORE_METHOD_PREDICTIONS',
        WT=100,primary_budget=.2,primary_generation_seed=13,K=8,qmax=None,
        methods=['delepref_unlimited','rag_unlimited','msa_conservation','terminal'],
        selector='same original step4000 trained Scorer, unlimited adapter; deterministic descending score then interval tuple',
        original_native_3span='historical reference only, excluded from matched primary quality comparison',
        secondary_budget=.1,secondary_WT=40,secondary_methods=['delepref_unlimited','rag_unlimited'],
        secondary_generation_seed=13,CPU_stability_seed=29,CPU_stability_scope='primary budget; masks and scores only, no extra folding',
        second_predictor_WT=8,second_predictor_methods=['delepref_unlimited','rag_unlimited','terminal','WT'],
        completion_ablation_WT=20,completion_ablation='same 4+4 pool but accept only exact-budget intact natural events; otherwise original fallback; fixed subset includes unsupported WTs',
        primary_endpoint=dict(name='full_native_relationship_preservation_T',
            reference='fixed WT native CA pairs, distance<15A, WT sequence separation>=3',
            contribution='1/.75/.5/.25/0 for absolute distance error<.5/1/2/4A/otherwise; deleted endpoints contribute zero',
            denominator='all valid fixed WT native pairs, identical for every method on a WT',
            missing_variant_coordinates='retained unknown pair contribution bounded [0,1]; not dropped',
            failure='all retained pairs unknown; lower=0, upper=static retained-pair fraction',
            implementation='code/build_pairs.py reference_pairs, reference_preservation_labels, preservation_bounds'),
        primary_comparison=['rag_unlimited','delepref_unlimited'],
        secondary_endpoints=['per-residue retained CA lDDT with declared retained-neighbor definition',
            'fixed correspondence full-WT-normalized TM once implementation is frozen',
            'WT native contact retention at8A with sequence separation>=3, fixed full WT contact denominator',
            'static pair/contact coverage lost through deletion','generation success/unique K/segment count/runtime/cost'],
        shared_retained_analysis='RAG versus original on exactly the same native relationships retained by both plans; report coverage and missing bounds',
        statistics=dict(unit='WT/homology component, not residues, candidates or seeds',
            paired=True,weighting='equal WT; one WT per frozen panel component',
            bootstrap_replicates=10000,bootstrap_seed=20261016,confidence=.95,
            missing='retain all 100 WTs with endpoint bounds; paired difference [X_lower-Y_upper,X_upper-Y_lower]',
            primary_inference='single prespecified contrast on T; secondary results descriptive with CIs, no selective significance claims',
            implementation='code/paired_statistics.py paired_bootstrap',complete_case='diagnostic only, always report N'),
        strata=dict(source=['pdb_chain','cath_domain'],length='150-249,250-399,400-550',
            retrieval='complete/partial/none from input_evidence; no WT excluded by support or outcome'),
        case_selection='illustrations: first two input-hash WTs from highest and lowest quartiles of paired T difference; first technical failure by hash if present. Full cohort remains reported; no biological success label inferred',
        compute=dict(structure_call_limit=800,structure_retries_max=1,
            structure_categories=dict(main_outputs=400,WT_controls=100,SCISOR=100,secondary_10pct=80,
                second_predictor=32,completion_ablation=20,retry=68),
            GPU_only_Slurm=True,max_task_GPUs=2,predict_final_selected_only=True,
            cache='identical sequence/predictor weight/profile/seed/sample digest only',ledger='code/budget_ledger.py'),
        outstanding_before_folding=['real encoder engineering pass','100WT original/RAG/Scorer-selected outputs',
            'SCISOR implementation/weights/coverage or documented unavailable decision',
            'primary and second predictor weights/profile/seed identity',
            'TM/lDDT implementation audit and whole evaluation source freeze','all output identities and fold dispatch freeze'],
        structural_prediction_allowed=False,biological_function_not_validated=True,paper_changes=False)
    write(out/'statistical_protocol.json',protocol)
    stats={str(b):dict(Counter(x['support_class'] for x in evidence if x['budget']==b)) for b in (.1,.2)}
    report=dict(status='CPU_CONTROLS_SUBSETS_AND_STATISTICS_PREPARED_NOT_FULL_METHOD_FREEZE',
        job_id=os.environ['SLURM_JOB_ID'],seconds=time.time()-start,WT=100,control_candidates=len(candidates),
        subset_sizes={k:len(subsets[k]) for k in ('secondary_10pct','second_predictor','completion_ablation')},
        input_support_by_budget=stats,conservation_all_unknown_WT=sum(x['audit']['all_unknown'] for x in conservation),
        source_freeze_sha256=checksum,outputs={p.name:rt.sha(p) for p in out.glob('*.json') if p.name!='execution_claim.json'},
        actual_Scorer_run=False,PLM_encodings=0,new_structure_predictions=0,structural_prediction_allowed=False,paper_changes=False)
    write(out/'completion.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--prepare',action='store_true');p.add_argument('--freeze-sha256');a=p.parse_args()
    if a.prepare:print(json.dumps(prepare(a.root)))
    else:execute(a.root,a.freeze_sha256)
