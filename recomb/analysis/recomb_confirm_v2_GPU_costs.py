"""Timing-only fixed20 replay on one A100; never replace scientific candidates or structures."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
from recomb_confirm_v2_diagnostics import read,sha,write

BASE=Path('reports/recomb_confirm_v2_General100_GPU_costs')
FZ=Path('configs/recomb_confirm_v2_General100_GPU_costs_freeze.json')
ROOT=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013')


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('E8 same-A100 timing protocol already frozen')
    cases=read(root/'reports/recomb_confirm_v2_General100_costs/cases.json')
    if len(cases)!=20:raise ValueError('The fixed input-only E8 roster differs')
    names=['analysis/recomb_confirm_v2_GPU_costs.py','analysis/recomb_confirm_v2_costs.py',
        'analysis/recomb_confirm_v2_diagnostics.py','analysis/recomb_confirm_v2_scisor_L_loader_v2.py',
        'slurm/recomb_confirm_v2_GPU_costs.sbatch',
        'reports/recomb_confirm_v2_General100_costs/cases.json','reports/recomb_confirm_v2_general_native_v2/General100_panel.json',
        'reports/recomb_confirm_v2_General100_model/encoding_completion.json',
        'retrieval/recomb_confirm_v2_General100/profiles.json',
        'reports/recomb_confirm_v2_General100_raygun_v2/raygun_checkpoint_load_contract.json',
        'reports/recomb_confirm_v2_General100_raygun_v2/raygun_tasks.json',
        'configs/recomb_confirm_v2_General100_costs_freeze.json',
        'configs/recomb_confirm_v2_General100_raygun_freeze.json',
        'configs/recomb_confirm_v2_General100_scisor_L_freeze.json',
        'reports/recomb_confirm_v2_scisor_L/preflight_v4.json',
        'configs/delepref_v8_baseline_freeze.json','provenance/original_delepref_assets_20261016.json']
    native=read(root/'configs/recomb_confirm_v2_General100_raygun_freeze.json')
    import yaml
    config=root/'configs/delepref_original_restored.yaml';names.append(str(config.relative_to(root)))
    snapshot=Path(yaml.safe_load(config.read_text())['model']['esmc_snapshot'])
    weights={str(p):sha(p) for p in snapshot.rglob('*') if p.is_file() and not {'.cache','__pycache__'}&set(p.parts) and p.suffix in ('.safetensors','.pt','.pth','.bin','.json','.txt','.model','.yaml')}
    if not weights:raise ValueError('Original ESMC local snapshot is absent')
    for asset in native['assets'].values():
        if isinstance(asset,dict) and 'path' in asset and 'sha256' in asset:
            if sha(asset['path'])!=asset['sha256']:raise ValueError('Existing Raygun asset changed')
            weights[asset['path']]=asset['sha256']
    L=root/'models/scisor_u90_l/SCISOR_U90_L.ckpt';weights[str(L)]=read(root/'reports/recomb_confirm_v2_scisor_L/preflight_v4.json')['checkpoint_sha256']
    p0=read(root/'configs/recomb_confirm_v2_General100_scisor_L_freeze.json')['p0'];weights[p0['path']]=p0['sha256']
    scisor_code=read(root/'reports/delepref_v8_scisor_inventory.json')['code_sha256']
    baseline=read(root/'configs/delepref_v8_baseline_freeze.json')
    write(root/FZ,dict(status='E8_SINGLE_A100_FIXED20_MODEL_SETUP_UNCACHED_WT_AND_NATIVE_TIMINGS_FROZEN',
        files={n:sha(root/n) for n in names},Raygun_external_code=native['external_code'],assets_sha256=weights,
        SCISOR_external_code=scisor_code,original_dependencies_sha256={**baseline['inputs'],**baseline['code']},
        WT=20,budget=.2,seed=13,hardware='A100_80GB',CPU_threads=4,
        process_setup='one process per model family; filesystem and shared model cache may already be warm',
        esmc='one explicit first-case warmup then20uncached_singleWT_calls; record actual encoder batch increments; each resulting real embedding used by frozen CPU heads',
        CPU_cached_repeats=3,SCISOR='same frozen officialL native K1 plus7 new samples per WT forK8; total160 sampler calls',
        Raygun='same existing8.8M/800M native worker, noise1, batch8;500 generated and500 scored, retain50/top1;20WT',
        timing_outputs_do_not_replace_scientific_sequences=True,no_benchmark_outputs_folded=True,
        measure_model_resident_peak_allocated_memory=True,
        uncached_retrieval_and_database_preparation='not included; observed original batch totals remain separate',
        no_profile_model_noise_or_score_retuning=True,new_training=False,new_structure_predictions=0,
        dispatch='not auto-submitted; live total taskGPU<=2 required; reserve only1A100'))
    (root/BASE).mkdir(parents=True,exist_ok=True);print(sha(root/FZ),flush=True)


def load_verified(root,stage):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('E8 GPU inference requires Slurm')
    f=read(root/FZ)
    for p,h in {**f['files'],**f['Raygun_external_code'],**f['SCISOR_external_code'],**f['original_dependencies_sha256'],**f['assets_sha256']}.items():
        path=Path(p) if Path(p).is_absolute() else root/p
        if sha(path)!=h:raise ValueError('Frozen E8 timing source/input changed: '+p)
    import torch
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name():raise RuntimeError('Same-device E8 requires A100')
    out=root/BASE/stage;done=out/'completion.json'
    if done.exists():
        saved=read(done)
        if saved['freeze_sha256']!=sha(root/FZ):raise ValueError('Existing cost stage has different source freeze')
        for p,h in saved['outputs'].items():
            if sha(out/p)!=h:raise ValueError('Existing successful timing artifact changed')
        print('E8_VERIFIED_STAGE_REUSE '+stage,flush=True);return None
    if (out/'execution_claim.json').exists():raise FileExistsError('E8 stage already claimed; inspect before repair')
    write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID'],freeze_sha256=sha(root/FZ)))
    cases=read(root/'reports/recomb_confirm_v2_General100_costs/cases.json')
    panel={r['protein_id']:r for r in read(root/'reports/recomb_confirm_v2_general_native_v2/General100_panel.json')}
    for c in cases:
        if panel[c['protein_id']]['WT_sha256']!=c['WT_sha256']:raise ValueError('Ordinary E8 WT roster identity changed')
    torch.set_num_threads(4);print('E8_GPU_STAGE_LOADED '+stage+' '+torch.cuda.get_device_name(),flush=True)
    return torch,f,out,cases,panel


def esmc_stage(root,torch,f,out,cases,panel):
    import delepref_v8_runtime as rt
    from delepref_v8_main import cached_chunk
    from recomb_confirm_v2_costs import cached_trial
    before=time.perf_counter();gpu,api=rt.load(root);torch.cuda.synchronize();GPU_load=time.perf_counter()-before
    before=time.perf_counter();cpu,cpuapi=rt.load(root,device='cpu',cached_heads_only=True);CPU_load=time.perf_counter()-before
    profile=read(root/'retrieval/recomb_confirm_v2_General100/profiles.json');rows=[]
    first=cases[0]['protein_id'];record=dict(protein_id=first,wildtype_sequence=panel[first]['sequence'],budget=.2)
    with torch.inference_mode():
        warmup_before=gpu.model.encoder.forward_batches
        api.encode_chunk(gpu,[record]);torch.cuda.synchronize()
        warmup_batches=gpu.model.encoder.forward_batches-warmup_before
        if warmup_batches!=1:raise ValueError('ESMC warmup encoder count differs')
        for case in cases:
            pid=case['protein_id'];seq=panel[pid]['sequence'];record=dict(protein_id=pid,wildtype_sequence=seq,budget=.2)
            torch.cuda.reset_peak_memory_stats();before_batches=gpu.model.encoder.forward_batches;start=time.perf_counter()
            chunk=api.encode_chunk(gpu,[record]);torch.cuda.synchronize();seconds=time.perf_counter()-start
            batches=gpu.model.encoder.forward_batches-before_batches
            if batches!=1:raise ValueError('Uncached WT benchmark did not execute exactly one real encoder batch')
            before=time.perf_counter()
            cache=dict(embeddings={k.protein_id:v.detach().cpu() for k,v in chunk.encoded.embeddings.items()},sequences={pid:seq})
            host=cached_chunk(cpuapi,cache,[record]);transfer=time.perf_counter()-before
            trials=[]
            for trial in range(3):
                before=time.perf_counter();r=cached_trial(cpu,cpuapi,host,record,profile[pid]);r.update(total_four_branch_CPU_seconds=time.perf_counter()-before,trial=trial);trials.append(r)
            rows.append(dict(protein_id=pid,WT_sha256=case['WT_sha256'],WT_length=len(seq),
                real_uncached_WT_GPU_encoding_seconds=seconds,real_encoder_batch_calls=batches,embedding_GPU_to_CPU_and_cache_adapter_seconds=transfer,
                peak_GPU_allocated_bytes=int(torch.cuda.max_memory_allocated()),CPU_branch_trials=trials,
                timing_only_new_singleWT_embedding_output_not_added_to_scientific_cache=True))
            print('E8_ACTUAL_WT_ENCODER_COMPLETE '+pid,flush=True);del chunk,host,cache;gc.collect();torch.cuda.empty_cache()
    return rows,dict(GPU_model_setup_seconds=GPU_load,CPU_heads_setup_seconds=CPU_load,
        actual_WT_encoder_batches=sum(r['real_encoder_batch_calls'] for r in rows)+warmup_batches,first_case_warmup_batches=warmup_batches,new_structure_predictions=0,
        latency_scope='resident singleWT encoder plus actual host transfer and CPU branch timings; retrieval cache hit; process setup separate')


def scisor_stage(root,torch,f,out,cases,panel):
    sys.path.insert(0,'/public/home/zhangyangroup/chengshiz/keyuan.zhou/SurgEdit')
    from recomb_confirm_v2_scisor_L_loader_v2 import load
    from delepref_v8_scisor_mapping import mapped_plan
    from delepref_v8_controls import condition_seed
    from delepref_unlimited_planning import delete_count,validate_plan
    source=read(root/'configs/recomb_confirm_v2_General100_scisor_L_freeze.json')
    before=time.perf_counter();model,payload,_=load(root);del payload
    model=model.to('cuda').eval();model.p0=torch.load(source['p0']['path'],map_location='cuda',weights_only=True)
    rate=source['alpha_rate'];model.alpha=lambda t:(1-t)**rate;model.beta=lambda t:rate/(1-t);model.log_alpha=lambda t:-torch.log(model.alpha(t))
    torch.cuda.synchronize();setup=time.perf_counter()-before;rows=[]
    with torch.inference_mode():
        for c in cases:
            pid=c['protein_id'];seq=panel[pid]['sequence'];samples=[]
            for rank in range(8):
                seed=condition_seed(13,pid,seq,.2)+rank*1_000_003
                random.seed(seed);np.random.seed(seed%(2**32-1));torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                torch.cuda.reset_peak_memory_stats();before=time.perf_counter()
                tokens=torch.tensor(model.tokenizer(seq).input_ids,device='cuda').unsqueeze(0)
                result,preserved=model.shrink_sequence(tokens,torch.tensor([delete_count(len(seq),.2)],device='cuda'),temperature=1.)
                item=mapped_plan(seq,.2,preserved[0],result[0].detach().cpu().tolist(),model.tokenizer)
                validate_plan(len(seq),.2,item['deleted_intervals']);torch.cuda.synchronize()
                samples.append(dict(candidate_index=rank,seconds=time.perf_counter()-before,
                    peak_GPU_allocated_bytes=int(torch.cuda.max_memory_allocated()),edited_sequence_sha256=hashlib.sha256(item['edited_sequence'].encode()).hexdigest()))
                del result,preserved,tokens
            rows.append(dict(protein_id=pid,sampler_calls=8,native_K1_seconds=samples[0]['seconds'],standalone_K8_sampling_seconds=sum(r['seconds'] for r in samples),samples=samples,
                benchmark_candidates_never_replace_frozen_pools=True))
            print('E8_L_SAMPLER_COMPLETE '+pid,flush=True)
    return rows,dict(model_setup_seconds=setup,new_sampler_calls=160,
        K8_Scorer_and_optional_WT_encoder_costs='measured separately, not hidden in sampling',new_structure_predictions=0)


def raygun_stage(root,torch,f,out,cases,panel):
    source='/public/nfs-data/keyuan.zhou/evocompress/results/formal_v9/external_raygun/tools/raygun_worker.py'
    spec=importlib.util.spec_from_file_location('E8_existing_native_raygun',source);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    before=time.perf_counter();torch,_,ray,encoder,alphabet,incompatible=native.load_raygun_models('cuda');torch.cuda.synchronize();setup=time.perf_counter()-before
    contract=read(root/'reports/recomb_confirm_v2_General100_raygun_v2/raygun_checkpoint_load_contract.json')
    if sorted(incompatible.missing_keys)!=sorted(contract['missing_keys']) or list(incompatible.unexpected_keys)!=contract['unexpected_keys']:raise ValueError('Native Raygun load contract differs')
    rows=[];tasks={r['protein_id']:r for r in read(root/'reports/recomb_confirm_v2_General100_raygun_v2/raygun_tasks.json') if r['budget']==.2}
    for c in cases:
        pid=c['protein_id'];seq=panel[pid]['sequence'];target=len(seq)-max(1,min(len(seq)-1,round(.2*len(seq))))
        seed=native.stable_seed(20260830,'official-native',tasks[pid]['dataset'],pid,.2)
        random.seed(seed);np.random.seed(seed&0xffffffff);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats();before=time.perf_counter()
        embedding,_=native._esm_embeddings(torch,encoder,alphabet,[seq],'cuda');torch.cuda.synchronize();WT_seconds=time.perf_counter()-before
        before=time.perf_counter();generated,_=native._generate_sequences(torch,encoder,alphabet,ray,embedding[:1],[target]*500,[1.]*500,'N/A','cuda',batch_size=8)
        torch.cuda.synchronize();generation=time.perf_counter()-before
        if len(generated)!=500 or any(len(s)!=target for s in generated):raise ValueError('Native Raygun500/target length differs')
        scores,score_seconds=native._pll_scores(torch,encoder,alphabet,generated,'cuda',batch_size=8);torch.cuda.synchronize()
        if len(scores)!=500 or not np.isfinite(scores).all():raise ValueError('Native all500 scoring failed')
        order=sorted(range(500),key=lambda i:(-scores[i],i));winner=generated[order[0]]
        rows.append(dict(protein_id=pid,WT_encoding_seconds=WT_seconds,raw500_generation_seconds=generation,
            all500_adjusted_ESM_scoring_seconds=score_seconds,retained_top50=50,selected=1,
            whole_resident_model_native_components_seconds=WT_seconds+generation+score_seconds,
            peak_GPU_allocated_bytes=int(torch.cuda.max_memory_allocated()),chosen_timing_only_sequence_sha256=hashlib.sha256(winner.encode()).hexdigest(),
            no_timing_sequence_or_selection_added_to_scientific_outputs=True))
        print('E8_RAYGUN_NATIVE_COMPLETE '+pid,flush=True);del generated,scores,embedding;gc.collect();torch.cuda.empty_cache()
    return rows,dict(model_setup_seconds=setup,raw_sequence_samples=10000,variant_sequences_scored=10000,
        true_score_definition='same upstream-named unmasked adjusted PLL; no masker or ranking change',new_structure_predictions=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--prepare',action='store_true');p.add_argument('--stage',choices=['esmc','scisor','raygun']);a=p.parse_args()
    if a.prepare:prepare(a.root);return
    if not a.stage:p.error('stage required')
    bundle=load_verified(a.root,a.stage)
    if bundle is None:return
    torch,f,out,cases,panel=bundle;before=time.perf_counter()
    if a.stage=='esmc':rows,extra=esmc_stage(a.root,torch,f,out,cases,panel)
    elif a.stage=='scisor':rows,extra=scisor_stage(a.root,torch,f,out,cases,panel)
    else:rows,extra=raygun_stage(a.root,torch,f,out,cases,panel)
    write(out/'per_WT_timings.json',rows)
    done=dict(status='FIXED20_SAME_A100_TIMING_ONLY_STAGE_COMPLETE',stage=a.stage,WT=20,
        actual_GPU=torch.cuda.get_device_name(),job_id=os.environ['SLURM_JOB_ID'],seconds=time.perf_counter()-before,
        freeze_sha256=sha(a.root/FZ),outputs={'per_WT_timings.json':sha(out/'per_WT_timings.json')},
        source_dtypes_preserved=True,new_training=False,no_structures_predicted=True,no_scientific_output_replacement=True,**extra)
    write(out/'completion.json',done);print(done,flush=True)


if __name__=='__main__':main()
