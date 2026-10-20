"""Official-weight SCISOR native sampling at the panel's matched exact count."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import delepref_v8_runtime as rt
import delepref_v8_controls as controls
from delepref_v8_baseline import ROOT,write
from delepref_v8_scisor_compat import patch_tokenizer
from delepref_v8_scisor_mapping import mapped_plan
from delepref_v8_scisor_inference_import import load_native_class
import delepref_unlimited_planning as planning

FREEZE='configs/delepref_v8_scisor_native_freeze_v2.json'
BASE='reports/delepref_v8_scisor_native_v2'
SOURCE=Path('/public/home/zhangyangroup/chengshiz/keyuan.zhou/SurgEdit')


def prepare(root):
    inventory=json.loads((root/'reports/delepref_v8_scisor_inventory.json').read_text())
    upstream=json.loads((root/'reports/delepref_v8_scisor_upstream.json').read_text())
    checked=json.loads((root/'reports/delepref_v8_scisor_preflight_v2.json').read_text())
    attention=json.loads((root/'reports/delepref_v8_scisor_attention_check.json').read_text())
    if checked['status']!='SCISOR_OFFLINE_CPU_MODEL_AND_TOKENIZER_PASS' or attention['status']!='SDPA_COMPATIBILITY_LAYER_CHECK_PASS':
        raise ValueError('SCISOR compatibility not verified')
    asset=inventory['assets']['checkpoint']
    if upstream['HF']['file']['lfs']['sha256']!=asset['sha256']:raise ValueError('not official checkpoint object')
    core=['shortening_scud.py','shortening_diffusion.py','continuous_time_diffusion.py','alignments.py','utils.py']
    core_identity={}
    for name in core:
        local=SOURCE/'SCISOR'/name;official=root/'provenance/scisor_upstream_20261016/SCISOR'/name
        if rt.sha(local)!=rt.sha(official):raise ValueError('official sampling core differs: '+name)
        core_identity[name]=rt.sha(local)
    out=root/BASE;out.mkdir(exist_ok=False)
    inputs=['reports/delepref_v8_native_panel/main_panel.json','reports/delepref_v8_native_panel/completion.json',
        'reports/delepref_v8_scisor_inventory.json','reports/delepref_v8_scisor_upstream.json',
        'reports/delepref_v8_scisor_preflight_v2.json','reports/delepref_v8_scisor_attention_check.json',
        'provenance/scisor_upstream_20261016/manifest.json','provenance/scisor_upstream_20261016/LICENSE.md']
    sources=['analysis/delepref_v8_scisor_native_v2.py','analysis/delepref_v8_scisor_mapping.py','analysis/delepref_v8_scisor_inference_import.py',
        'analysis/delepref_v8_scisor_compat.py','analysis/delepref_v8_controls.py',
        'analysis/delepref_unlimited_planning.py','analysis/delepref_v8_runtime.py','analysis/delepref_v8_baseline.py',
        'analysis/tests/test_delepref_v8_scisor_mapping.py','slurm/delepref_v8_scisor_native_v2.sbatch']
    write(root/FREEZE,dict(status='SCISOR_NATIVE_MAIN_PANEL_GENERATION_FROZEN',WT=100,budget=.2,K=1,qmax=None,
        temperature=1.,generation_seed=13,condition_seed='same WT identity/budget hash convention as main methods',
        deletion_count='common exact round(beta*L) clamped to retain at least one residue; official CLI ceil adapted only to match count',
        batching='one WT to preserve explicit per-condition seed and native index reconstruction',
        alpha_rate='1/1.1 as official shrink_sequences.py',core_byte_identity_with_official_revision=core_identity,
        github_revision=upstream['github']['revision'],HF_revision=upstream['HF']['revision'],
        license='MIT; public source/license snapshot retained',
        checkpoint_sha256=asset['sha256'],assets=inventory['assets'],server_source_hashes=inventory['code_sha256'],
        inputs={p:rt.sha(root/p) for p in inputs},sources={p:rt.sha(root/p) for p in sources},
        attention_backend='existing server SDPA compatibility layer, numerically checked against standard ESM on synthetic hidden states',
        comparison='K1 native sampler with matched exact count, separate from K8 same-Scorer methods; not equal-K/compute',
        optional_training_plugin='DeepSpeed hidden only in this inference process; no CUDA availability or model changes',previous_failed_job='1116593',model_or_structure_outcomes_used_to_set_rules=False,new_training=False,new_structure_predictions=0,paper_changes=False))
    return dict(freeze_sha256=rt.sha(root/FREEZE),WT=100,K=1)


def execute(root,checksum):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Slurm GPU allocation required')
    if rt.sha(root/FREEZE)!=checksum:raise ValueError('native SCISOR freeze mismatch')
    freeze=json.loads((root/FREEZE).read_text())
    for p,h in {**freeze['inputs'],**freeze['sources'],**freeze['server_source_hashes']}.items():
        if rt.sha(root/p)!=h:raise ValueError('SCISOR source/input changed: '+p)
    for asset in freeze['assets'].values():
        if rt.sha(Path(asset['path']))!=asset['sha256']:raise ValueError('SCISOR checkpoint/p0 changed')
    panel=json.loads((root/'reports/delepref_v8_native_panel/main_panel.json').read_text())
    if len(panel)!=100 or len({x['protein_id'] for x in panel})!=100:raise ValueError('wrong SCISOR panel')
    out=root/BASE;write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID'],freeze_sha256=checksum))
    start=time.time();torch.set_num_threads(4);sys.path.insert(0,str(SOURCE))
    cls=load_native_class()
    model=cls.load_from_checkpoint(freeze['assets']['checkpoint']['path'],map_location='cpu').to('cuda').eval()
    patch_tokenizer(model.tokenizer)
    model.p0=torch.load(freeze['assets']['p0']['path'],map_location='cuda',weights_only=True)
    rate=1/1.1;model.alpha=lambda t:(1-t)**rate;model.beta=lambda t:rate/(1-t)
    model.log_alpha=lambda t:-torch.log(model.alpha(t))
    print(json.dumps(dict(event='SCISOR_NATIVE_MODEL_LOADED',checkpoint_sha256=freeze['checkpoint_sha256'],WT=100)),flush=True)
    rows=[]
    with (out/'plans_progress.jsonl').open('x') as progress,torch.inference_mode():
        for item in panel:
            sequence=item['sequence'];pid=item['protein_id'];budget=.2
            seed=controls.condition_seed(13,pid,sequence,budget)
            random.seed(seed);np.random.seed(seed%(2**32-1));torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            ids=model.tokenizer(sequence).input_ids
            if len(ids)!=len(sequence)+2:raise ValueError('unexpected WT tokenization: '+pid)
            x=torch.tensor(ids,device='cuda').unsqueeze(0)
            count=planning.delete_count(len(sequence),budget);torch.cuda.reset_peak_memory_stats()
            before=time.time()
            output,preserved=model.shrink_sequence(x,torch.tensor([count],device='cuda'),temperature=1.)
            result=mapped_plan(sequence,budget,preserved[0],output[0].detach().cpu().tolist(),model.tokenizer)
            elapsed=time.time()-before
            mask=planning.mask_for(len(sequence),result['deleted_intervals'])
            row=dict(protein_id=pid,wildtype_sequence=sequence,wildtype_sequence_sha256=item['sequence_sha256'],
                budget=budget,generation_seed=13,condition_seed=seed,method='SCISOR_U90_S_native',K=1,
                decoder='official_shrink_sequence_explicit_count',temperature=1.,
                deletion_mask_sha256=hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest(),
                edited_sequence_sha256=controls.sequence_sha(result['edited_sequence']),
                seconds=elapsed,max_GPU_memory_bytes=torch.cuda.max_memory_allocated(),**result)
            rows.append(row);progress.write(json.dumps(row,sort_keys=True)+'\n');progress.flush()
            print(json.dumps(dict(event='SCISOR_NATIVE_WT_COMPLETE',protein_id=pid,completed=len(rows),span_count=row['span_count'],seconds=elapsed)),flush=True)
    write(out/'native_plans.json',rows)
    report=dict(status='SCISOR_NATIVE_MATCHED_COUNT_MAIN_PANEL_COMPLETE',job_id=os.environ['SLURM_JOB_ID'],
        WT=len(rows),budget=.2,K=1,qmax=None,seconds=time.time()-start,freeze_sha256=checksum,
        checkpoint_sha256=freeze['checkpoint_sha256'],plans_sha256=rt.sha(out/'native_plans.json'),
        max_span_count=max(r['span_count'] for r in rows),all_exact_budgets_and_reconstruction=True,
        actual_hardware=torch.cuda.get_device_name(0),attention_backend=freeze['attention_backend'],
        no_candidates_selected_by_structure_or_function=True,not_equal_K_or_compute=True,
        external_measured_variant_scores=False,new_training=False,new_structure_predictions=0,paper_changes=False)
    write(out/'completion.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--prepare',action='store_true')
    p.add_argument('--freeze-sha256');a=p.parse_args()
    if a.prepare:print(json.dumps(prepare(a.root)),flush=True)
    else:execute(a.root,a.freeze_sha256)
