"""Run only missing preplanned subset conditions on already reviewed proteins."""
import argparse
import json
from pathlib import Path
from recomb_scope_screen_20261017 import read,sha,write

BASE=Path('reports/recomb_benign_baseline_extension_20261017')
FREEZE=Path('configs/recomb_benign_baseline_extension_freeze_20261017.json')


def prepare(root):
    out=root/BASE;out.mkdir(exist_ok=False)
    sub=root/'reports/recomb_benign_final_subsets_20261017';done=read(sub/'completion.json')
    for n,h in done['files'].items():
        if sha(sub/n)!=h:raise ValueError('input-only subset changed')
    roster=read(root/'reports/recomb_benign_scope_20261017/general_panel.json')
    from recomb_benign_baselines_20261017 import validate_tasks
    for method in ['raygun','scisor']:
        tasks=read(sub/(method+'_delta_tasks.json'));validate_tasks(tasks,roster);write(out/(method+'_tasks.json'),tasks)
    write(out/'raygun_checkpoint_load_contract.json',read(root/'reports/recomb_benign_baselines_20261017/raygun_checkpoint_load_contract.json'))
    paths=[root/'reports/recomb_benign_scope_20261017/general_panel.json',sub/'subsets.json',sub/'completion.json',Path(__file__),
        root/'analysis/recomb_benign_baselines_20261017.py',out/'raygun_tasks.json',out/'scisor_tasks.json',out/'raygun_checkpoint_load_contract.json']
    write(root/FREEZE,dict(status='MISSING_INPUT_HASH_SUBSET_CONDITIONS_FROZEN',
        files={str(p.relative_to(root)):sha(p) for p in paths},parent_freeze_sha256=sha(root/'configs/recomb_general100_baselines_freeze_20261017.json'),
        allowed_WT=44,processed_WT=13,raygun_conditions=39,scisor_conditions=52,scisor_sampler_calls=130,
        new_target_proteins=0,method_parameters_unchanged=True,task_GPU_concurrency_limit=2,structure_prediction_allowed=False))
    print(json.dumps(dict(freeze_sha256=sha(root/FREEZE))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
    p.add_argument('--prepare',action='store_true');p.add_argument('--method',choices=['raygun','scisor']);p.add_argument('--freeze-sha256')
    p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    if a.prepare:prepare(a.root)
    else:
        import recomb_benign_baselines_20261017 as runner
        runner.BASE=BASE;runner.FREEZE=FREEZE;runner.main(a)
