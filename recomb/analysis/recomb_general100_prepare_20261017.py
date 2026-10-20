"""Freeze missing native baseline generation; existing results stay immutable."""
import argparse
import json
from pathlib import Path

import delepref_unlimited_planning as planning
from recomb_general100_common_20261017 import ROOT, BASE, FREEZE, sha, write

RAY = Path('/public/nfs-data/keyuan.zhou/evocompress/results/formal_v9/external_raygun')
OLD_RUN = RAY / 'runs/V9-S6-raygun_official-native_20260901T055959Z_attempt01'


def prepare(root):
    out = root / BASE
    out.mkdir(exist_ok=False)
    panel_path = root / 'reports/delepref_v8_native_panel/main_panel.json'
    if sha(panel_path) != 'bdd70e2b49069b63a7ee0021cb7a770e4bc093c9cacbdb1ef273328e86b2205d':
        raise ValueError('General100 roster identity changed')
    panel = json.loads(panel_path.read_text())
    subsets = json.loads((root / 'reports/delepref_v8_evaluation_preparation/subsets.json').read_text())
    ids = set(subsets['completion_ablation'])
    ray_tasks, scisor_tasks = [], []
    for record in panel:
        pid, sequence = record['protein_id'], record['sequence']
        common = dict(protein_id=pid, sequence=sequence, sequence_sha256=record['sequence_sha256'],
                      dataset='General100', stratum=record['stratum'])
        for beta in ([.2, .1, .3, .4] if pid in ids else [.2]):
            task = dict(common, budget=beta, target_length=len(sequence)-planning.delete_count(len(sequence), beta))
            ray_tasks.append(task)
            if beta != .2:
                scisor_tasks.append(dict(task, mode='native', candidate_count=1))
        if pid in ids:
            scisor_tasks.append(dict(common, budget=.2, target_length=len(sequence)-planning.delete_count(len(sequence), .2),
                                     mode='same_scorer_K8', candidate_count=8, reuse_original_native_as_candidate0=True))
    if len(ray_tasks) != 160 or len(scisor_tasks) != 80:
        raise ValueError('baseline task count mismatch')
    write(out / 'raygun_tasks.json', ray_tasks)
    write(out / 'scisor_tasks.json', scisor_tasks)
    missing_contract = json.loads((OLD_RUN / 'metrics/shards/task_0000.json').read_text())['checkpoint_load']
    write(out / 'raygun_checkpoint_load_contract.json', missing_contract)
    sources = ['analysis/recomb_general100_common_20261017.py',
               'analysis/recomb_general100_prepare_20261017.py',
               'analysis/recomb_general100_baselines_20261017.py',
               'analysis/delepref_v8_scisor_inference_import_v2.py',
               'analysis/delepref_v8_scisor_compat.py', 'analysis/delepref_v8_scisor_mapping.py',
               'analysis/delepref_unlimited_planning.py', 'analysis/delepref_v8_controls.py',
               'slurm/recomb_general100_raygun_20261017.sbatch',
               'slurm/recomb_general100_scisor_20261017.sbatch']
    inputs = ['reports/delepref_v8_native_panel/main_panel.json',
              'reports/delepref_v8_evaluation_preparation/subsets.json',
              'reports/delepref_v8_scisor_native_v3/native_plans.json',
              'reports/delepref_v8_scisor_inventory.json',
              str(BASE / 'raygun_tasks.json'), str(BASE / 'scisor_tasks.json'),
              str(BASE / 'raygun_checkpoint_load_contract.json')]
    files = {p: sha(root / p) for p in sources + inputs}
    files[str(RAY / 'tools/raygun_worker.py')] = sha(RAY / 'tools/raygun_worker.py')
    files[str(OLD_RUN / 'protocol_snapshot.json')] = sha(OLD_RUN / 'protocol_snapshot.json')
    assets = json.loads((OLD_RUN / 'checkpoint_snapshot.json').read_text())
    # Hash the two existing model objects once at freezing, not once per WT.
    for key in ('raygun_checkpoint', 'input_encoder'):
        asset = assets[key]
        if sha(asset['path']) != asset['sha256']:
            raise ValueError('Raygun model object changed: ' + key)
    scisor = json.loads((root / 'reports/delepref_v8_scisor_inventory.json').read_text())
    for asset in scisor['assets'].values():
        if sha(asset['path']) != asset['sha256']:
            raise ValueError('SCISOR model object changed')
    for path, digest in scisor['code_sha256'].items():
        if sha(path) != digest:
            raise ValueError('SCISOR source changed: ' + path)
        files[path] = digest
    freeze = dict(status='GENERAL100_MISSING_BASELINES_FROZEN_BEFORE_GENERATION', files=files,
                  raygun_assets=assets, scisor_assets=scisor['assets'],
                  raygun=dict(raw_candidates=500, PLL_keep=50, select='native_PLL_top1',
                              noise=1., cycles=1, temperature='N/A', batch_size=8,
                              seed_root=20260830, target_length='L-common_round_beta_L'),
                  scisor=dict(native_K=1, shared_scorer_control_K=8, temperature=1.,
                              seed_root=13, candidate0='reuse_existing_native', alpha_rate=1/1.1),
                  original_model_unchanged=True, new_training=False, qmax=None,
                  GPUs_per_job=1, task_GPU_concurrency_limit=2, allowed_GPU_types=['A40', 'A100'],
                  complete_structure_prediction_limit=None,
                  authorization='User 2026-10-17 approved full plan, then removed prediction count limit and allowed A40/A100',
                  structure_prediction_allowed=False, paper_paused=True)
    write(root / FREEZE, freeze)
    write(out / 'preparation_completion.json', dict(status='MISSING_BASELINE_INPUTS_FROZEN',
          freeze_sha256=sha(root / FREEZE), raygun_conditions=160, scisor_conditions=80,
          new_structure_predictions=0, model_objects_reused=True))
    print(json.dumps(dict(freeze_sha256=sha(root / FREEZE), raygun_conditions=160, scisor_conditions=80)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    prepare(parser.parse_args().root)
