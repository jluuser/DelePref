"""Native baselines only for the reviewed, corrected application constructs."""
import argparse
import json
from pathlib import Path
from recomb_scope_screen_20261017 import read,sha,write

BASE=Path('reports/recomb_benign_app_baselines_20261017')
FREEZE=Path('configs/recomb_benign_app_baselines_freeze_20261017.json')
PANEL=Path('reports/recomb_benign_app_model_20261017/panel.json')


def prepare(root):
    rows=read(root/PANEL);byid={r['protein_id']:r for r in rows}
    accepted={r['protein_id'] for r in read(root/'reports/recomb_benign_application_scope_20261017/audit.json') if r['status']=='RETAIN'}
    if len(rows)!=23 or set(byid)!=accepted:raise ValueError('reviewed application scope mismatch')
    source=root/'reports/recomb_general100_app_baselines_20261017/reused_baseline_outputs.json'
    reused=[]
    for old in read(source):
        p=byid.get(old['protein_id'])
        if not p or old['sequence_sha256']!=p['sequence_sha256'] or old['sequence']!=p['sequence']:continue
        if len(old['edited_sequence'])!=p['target_length']:raise ValueError('reused budget mismatch')
        updated=dict(old);updated.update(p)
        updated['reused_generation']=True
        updated['complex_cache_requires_current_partner_identity']=True
        reused.append(updated)
    seen={(r['protein_id'],r['method']) for r in reused}
    raygun=[r for r in rows if (r['protein_id'],'Raygun_500_native') not in seen]
    scisor=[dict(r,candidate_count=1,mode='native',reuse_original_native_as_candidate0=False) for r in rows if (r['protein_id'],'SCISOR_U90_S_native') not in seen]
    if len(reused)!=32 or len(raygun)!=12 or len(scisor)!=2:raise ValueError('unexpected reuse/missing matrix')
    out=root/BASE;out.mkdir(exist_ok=False)
    write(out/'reused_generation_outputs.json',reused);write(out/'raygun_tasks.json',raygun);write(out/'scisor_tasks.json',scisor)
    contract=root/'reports/recomb_benign_baselines_20261017/raygun_checkpoint_load_contract.json'
    write(out/'raygun_checkpoint_load_contract.json',read(contract))
    paths=[root/PANEL,source,root/'reports/recomb_benign_application_scope_20261017/audit.json',
           root/'configs/recomb_benign_app_inputs_freeze_20261017.json',Path(__file__),
           out/'reused_generation_outputs.json',out/'raygun_tasks.json',out/'scisor_tasks.json',out/'raygun_checkpoint_load_contract.json']
    parent=root/'configs/recomb_general100_baselines_freeze_20261017.json'
    write(root/FREEZE,dict(status='REVIEWED_APPLICATION_NATIVE_BASELINES_FROZEN',
        files={str(p.relative_to(root)):sha(p) for p in paths},parent_freeze_sha256=sha(parent),
        WT=23,reused_generation_outputs=32,new_Raygun_conditions=12,new_SCISOR_conditions=2,
        allowed_GPU_types=['A40','A100'],task_GPU_concurrency_limit=2,new_training=False,structure_prediction_allowed=False))
    write(out/'preparation_completion.json',dict(freeze_sha256=sha(root/FREEZE),reused=32,new_Raygun=12,new_SCISOR=2))
    print(json.dumps(read(out/'preparation_completion.json')))


def run(root,checksum,method):
    if sha(root/FREEZE)!=checksum:raise ValueError('freeze changed')
    frozen=read(root/FREEZE)
    for p,h in frozen['files'].items():
        if sha(root/p)!=h:raise ValueError('application source changed')
    from recomb_benign_baselines_20261017 import validate_tasks
    tasks=read(root/BASE/(method+'_tasks.json'));validate_tasks(tasks,read(root/PANEL))
    import recomb_general100_common_20261017 as common
    parent=common.verify(root,frozen['parent_freeze_sha256']);common.require_gpu()
    import recomb_general100_baselines_20261017 as native
    native.BASE=BASE
    print(json.dumps(dict(event='REVIEWED_APPLICATION_BASELINE_GATE_PASSED',method=method,conditions=len(tasks))),flush=True)
    if method=='raygun':native.run_raygun(root,parent,0,1)
    else:
        native.run_scisor(root,parent)
        p=root/BASE/'scisor/completion.json';result=read(p)
        result.update(actual_new_sampler_calls=2,native_conditions=2,K8_conditions=0,status='REVIEWED_APPLICATION_SCISOR_COMPLETE')
        write(p,result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
    p.add_argument('--prepare',action='store_true');p.add_argument('--freeze-sha256');p.add_argument('--method',choices=['raygun','scisor']);a=p.parse_args()
    prepare(a.root) if a.prepare else run(a.root,a.freeze_sha256,a.method)
