"""Reuse the verified official-native Raygun worker on qualified General100."""
import argparse
import hashlib
import os
from pathlib import Path
from recomb_confirm_v2_diagnostics import read,sha,write

BASE=Path('reports/recomb_confirm_v2_General100_raygun')
FZ=Path('configs/recomb_confirm_v2_General100_raygun_freeze.json')


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('Raygun General100 inputs already frozen')
    panelpath=root/'reports/recomb_confirm_v2_general_native_v2/General100_panel.json'
    panel=read(panelpath);subpath=root/'reports/recomb_confirm_v2_General100_subsets/subsets.json'
    subs=read(subpath);budget_ids={r['protein_id'] for r in subs['budget40']}
    if len(panel)!=100 or len(budget_ids)!=40:raise ValueError('Wrong final General100/subset')
    scope={r['candidate_id']:r for r in read(root/'reports/recomb_confirm_v2_design_function_review/per_candidate_function_review_v2.json')}
    tasks=[]
    for r in panel:
        if scope[r['candidate_id']]['status']!='FUNCTIONAL_SCOPE_PASS_NATIVE_AND_FAMILY_PENDING':
            raise ValueError('General target lacks current functional scope approval')
        if hashlib.sha256(r['sequence'].encode()).hexdigest()!=r['WT_sha256'] or r['native_CA_coverage']<.9:
            raise ValueError('Qualified input identity/native criterion differs')
        for beta in ([.2,.1,.3,.4] if r['protein_id'] in budget_ids else [.2]):
            target=len(r['sequence'])-max(1,min(len(r['sequence'])-1,round(len(r['sequence'])*beta)))
            tasks.append(dict(protein_id=r['protein_id'],sequence=r['sequence'],WT_sha256=r['WT_sha256'],
                dataset='General100_confirm_v2',budget=beta,target_length=target,mode='native',candidate_count=500))
    if len(tasks)!=220:raise ValueError('Incorrect native baseline condition count')
    old=read(root/'configs/recomb_general100_baselines_freeze_20261017.json')
    assets=old['raygun_assets']
    for key in ['raygun_checkpoint','input_encoder']:
        if sha(assets[key]['path'])!=assets[key]['sha256']:raise ValueError('Existing Raygun model asset changed')
    paths=['analysis/recomb_confirm_v2_general_raygun.py','analysis/recomb_confirm_v2_diagnostics.py',
        'analysis/recomb_general100_baselines_20261017.py','analysis/recomb_general100_common_20261017.py',
        'slurm/recomb_confirm_v2_general_raygun.sbatch',
        'reports/recomb_confirm_v2_general_native_v2/General100_panel.json',
        'reports/recomb_confirm_v2_General100_subsets/subsets.json',
        'reports/recomb_confirm_v2_design_function_review/per_candidate_function_review_v2.json']
    native='/public/nfs-data/keyuan.zhou/evocompress/results/formal_v9/external_raygun/tools/raygun_worker.py'
    if sha(native)!=old['files'][native]:raise ValueError('Previously verified native worker changed')
    repo=Path(assets['source_repo']);code={str(p):sha(p) for p in sorted((repo/'raygun').rglob('*.py'))}
    code[native]=sha(native)
    contract=root/'reports/recomb_benign_baselines_20261017/raygun_checkpoint_load_contract.json'
    load=read(contract)
    if load['checkpoint_sha256']!=assets['raygun_checkpoint']['sha256']:raise ValueError('Native load contract not exact')
    write(root/BASE/'raygun_tasks.json',tasks)
    write(root/BASE/'raygun_checkpoint_load_contract.json',load)
    paths += [str(BASE/'raygun_tasks.json'),str(BASE/'raygun_checkpoint_load_contract.json')]
    result=dict(status='GENERAL100_OFFICIAL_NATIVE_RAYGUN_GENERATION_FROZEN',
        files={p:sha(root/p) for p in paths},external_code=code,assets=assets,
        protocol=old['raygun'],native_loader_contract='unchanged completed official-native worker',
        model_name='raygun_8_8mil_800M',WT=100,conditions=220,main_20pct=100,additional_40WT_3budgets=120,
        raw_candidates=110000,primary_choice='native adjusted PLL top1 of 500, retain top50',
        qmax=None,new_training=False,structure_prediction_allowed=False,task_GPU_concurrency_limit=2,
        model_identity_evidence=dict(pretrained_source=str(repo/'raygun/pretrained.py'),
            pretrained_source_sha256=sha(repo/'raygun/pretrained.py'),
            README_sha256=sha(repo/'README.md'),
            official_factory='raygun_8_8mil_800M references species_function_aware_sep30_val_blosum_0.9856.ckpt'))
    write(root/FZ,result);print(sha(root/FZ),flush=True)


def run(root,shard,shards):
    f=read(root/FZ)
    for p,h in f['files'].items():
        if sha(root/p)!=h:raise ValueError('Frozen General100 Raygun input differs: '+p)
    for p,h in f['external_code'].items():
        if sha(p)!=h:raise ValueError('Frozen Raygun worker/repository source differs')
    for key in ['raygun_checkpoint','input_encoder']:
        if sha(f['assets'][key]['path'])!=f['assets'][key]['sha256']:raise ValueError('Raygun model changed')
    import recomb_general100_common_20261017 as common
    common.require_gpu()
    # Existing worker receives only this newly gated task roster, never old mixed tasks.
    import recomb_general100_baselines_20261017 as runner
    runner.BASE=BASE
    runner.run_raygun(root,f,shard,shards)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shards',type=int,default=4)
    a=p.parse_args();prepare(a.root) if a.prepare else run(a.root,a.shard_index,a.shards)
