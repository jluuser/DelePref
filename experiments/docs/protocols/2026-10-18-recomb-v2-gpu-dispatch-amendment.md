# RECOMB A100/A40 dispatch amendment implementation plan

> For agentic workers: execute inline under the user's existing execution authorization. No new thread or subagent delegation. This implements the latest explicit removal of the GPU cap.

**Goal:** Use idle permitted A100 and A40 allocations without repeating completed predictions or changing the biological/scoring experiment.

**Architecture:** Preserve the existing confirmation and sources. Derive a versioned hardware catalog before the first application prediction. Route every WT's methods/budgets/seeds together by the maximum frozen input size (protein residues plus actual CCD heavy atoms); shared predictions promote the complete connected WT group to A100. Profiles differ only in device class; retain identical YAML/CSV/weights/seeds and prove an input bijection. A40 uses one predefined WT request first, then all missing requests on available cards. The actual WT result is reused and never re-predicted.

**Tech stack:** Python, existing Boltz environments and helpers, unittest, Slurm. Only gpu_gua/gpu_gub; no fixed task GPU cap, no other project jobs touched. Long inputs (>1024 routing tokens) remain A100. Parent parser/symmetry/metric QA is reused by exact input SHA; no repeated science or new training.

## Tasks and exact files

- [x] Increase existing General array 1119264 throttle to4 (two unfinished shards); keep its bound scientific profile and outputs.
- [ ] Create the following unit tests, execute them before the module exists and verify ImportError, then implement the module below. Run the five tests once in the Boltz CPU environment and require PASS before deriving the hardware confirmation.
- [ ] Submit the CPU amendment with afterok:1119471; do not modify that existing release or native parser. On success, validate the hardware routing release and all SHA mappings. Freeze all derived sources and request identities before calling a predictor.
- [ ] Read actual node CfgTRES minus AllocTRES in the two permitted partitions, excluding DRAIN/DOWN/FAIL. Submit A100 WT shards up to actual available eligible cards; submit exactly one A40 WT canary. Only after its valid output, submit remaining A40 WT shards. Keep the exact canary task in the catalog; the shared prediction journal skips its completed identity.
- [ ] After all140 WT seed17 predictions and the frozen WT metric readout, run main and extension with the same hardware assignments. At each phase choose the shard count from actual eligible idle cards and remaining requests. Run the derived evaluator for WT/main/all, then the unchanged repeat-report functions. Keep full failures/unknowns and physical accounting. Recompute ETA from actual WT throughput, not the old two-GPU assumption.

## Complete implementation: analysis/tests/test_recomb_confirm_v2_application_hardware.py

```python
import unittest
from recomb_confirm_v2_application_hardware import assign_hardware,remap_catalog,digest

class HardwareRoutingTests(unittest.TestCase):
    def test_whole_WT_is_routed_by_largest_input_not_method_outcome(self):
        rows=[dict(protein_id='small',physical_id='a'),dict(protein_id='large',physical_id='b'),dict(protein_id='large',physical_id='c')]
        route,size=assign_hardware(rows,dict(a=400,b=700,c=1800))
        self.assertEqual(route,dict(small='A40',large='A100'))
        self.assertEqual(size['large'],1800)

    def test_shared_inputs_close_the_WT_hardware_group(self):
        rows=[dict(protein_id='a',physical_id='shared'),dict(protein_id='b',physical_id='shared'),dict(protein_id='b',physical_id='long')]
        route,_=assign_hardware(rows,dict(shared=400,long=2000))
        self.assertEqual(route,dict(a='A100',b='A100'))

    def test_threshold_is_inclusive_and_input_only(self):
        rows=[dict(protein_id='a',physical_id='x'),dict(protein_id='b',physical_id='y')]
        route,_=assign_hardware(rows,dict(x=1024,y=1025))
        self.assertEqual(route,dict(a='A40',b='A100'))

    def test_hardware_rekeys_identity_without_changing_model_input_or_seed(self):
        profile=dict(device_class='A100_80GB',recycling_steps=3,sampling_steps=200,MSA_sha='real')
        physical=[dict(task_id='old',profile_name='p',profile_sha256=digest(profile),sequence_sha256='seq',weights_sha256='weights',seed=17,sample=0,model_input_sha256='actual',chains=[dict(sequence='AAA')])]
        logical=[dict(request_id='logical',physical_id='old',protein_id='a',method='rag_unlimited',budget=.2,evaluation_role='primary20pct',seed=17)]
        request,rows,profiles,mapping=remap_catalog(physical,logical,dict(p=profile),dict(a='A40'))
        self.assertNotEqual(request[0]['profile_sha256'],physical[0]['profile_sha256'])
        self.assertEqual(request[0]['model_input_sha256'],'actual')
        self.assertEqual(request[0]['chains'],physical[0]['chains'])
        self.assertEqual((request[0]['seed'],request[0]['sample']),(17,0))
        self.assertEqual(rows[0]['physical_id'],mapping['old'])
        self.assertEqual(profiles[request[0]['profile_name']]['sampling_steps'],200)

    def test_mixed_assignment_for_same_shared_prediction_is_rejected(self):
        physical=[dict(task_id='x')]
        rows=[dict(protein_id='a',physical_id='x'),dict(protein_id='b',physical_id='x')]
        with self.assertRaises(ValueError):remap_catalog(physical,rows,{},dict(a='A40',b='A100'))

if __name__=='__main__':unittest.main()
```

Test command: Boltz Python -m unittest discover -s analysis/tests -p test_recomb_confirm_v2_application_hardware.py -v. Expected before implementation: ModuleNotFoundError. Expected after implementation: 5 tests OK.

## Complete implementation: analysis/recomb_confirm_v2_application_hardware.py

```python
"""Versioned A100/A40 routing amendment; identical frozen application inputs."""
import argparse
import ast
from collections import Counter, defaultdict
import copy
import os
from pathlib import Path
from recomb_confirm_v2_diagnostics import read, sha, write
from recomb_confirm_v2_application_catalog_v2 import digest

PARENT_CAT=Path('reports/recomb_confirm_v2_application_catalog_v2')
CAT=Path('reports/recomb_confirm_v2_application_catalog_hardware_v3')
PARENT_CONFIRM=Path('configs/recomb_confirm_v2_application_confirmation_freeze.json')
CONFIRM=Path('configs/recomb_confirm_v2_application_hardware_v3_confirmation_freeze.json')
STRUCT=Path('reports/recomb_confirm_v2_application_structures_hardware_v3')
EVAL=Path('reports/recomb_confirm_v2_application_evaluation_hardware_v3')
REPORT=Path('reports/recomb_confirm_v2_application_final_report_hardware_v3')
SOFTWARE=Path('configs/recomb_confirm_v2_application_repeat_report_hardware_v3_freeze.json')
THRESHOLD=1024


def assign_hardware(logical, sizes, threshold=THRESHOLD):
    """Input-only routing, closed over deduplicated shared requests."""
    by=defaultdict(set)
    for row in logical:by[row['protein_id']].add(row['physical_id'])
    max_size={pid:max(sizes[t] for t in tasks) for pid,tasks in by.items()}
    route={pid:'A40' if n<=threshold else 'A100' for pid,n in max_size.items()}
    while True:
        task_types=defaultdict(set)
        for pid,tasks in by.items():
            for task in tasks:task_types[task].add(route[pid])
        promote={pid for pid,tasks in by.items() if route[pid]=='A40'
                 and any('A100' in task_types[t] for t in tasks)}
        if not promote:break
        for pid in promote:route[pid]='A100'
    return route,max_size


def remap_catalog(physical,logical,profiles,route):
    owners=defaultdict(set)
    for row in logical:owners[row['physical_id']].add(row['protein_id'])
    requests=[];new_profiles={};crosswalk={}
    for old in physical:
        types={route[pid] for pid in owners[old['task_id']]}
        if len(types)!=1:raise ValueError('Shared full input assigned two hardware classes')
        hardware=next(iter(types))
        profile=copy.deepcopy(profiles[old['profile_name']])
        profile['device_class']='A40_48GB' if hardware=='A40' else 'A100_80GB'
        name='application_'+digest(profile)[:24]
        new_profiles[name]=profile
        identity={k:old[k] for k in ['sequence_sha256','weights_sha256','seed','sample']}
        identity['profile_sha256']=digest(profile)
        task='appboltz_'+digest(identity)
        row=dict(old,**identity)
        row.update(task_id=task,profile_name=name,device_class=hardware,parent_task_id=old['task_id'])
        requests.append(row);crosswalk[old['task_id']]=task
    if len({r['task_id'] for r in requests})!=len(physical):raise ValueError('Hardware identity collision')
    indexed={r['task_id']:r for r in requests};result=[]
    for old in logical:
        physical_id=crosswalk[old['physical_id']];request=indexed[physical_id]
        row=dict(old,physical_id=physical_id,profile_name=request['profile_name'],
                 device_class=request['device_class'],parent_request_id=old['request_id'])
        row['request_id']=digest({k:row[k] for k in ['protein_id','method','budget','evaluation_role','seed','physical_id']})
        result.append(row)
    return requests,result,new_profiles,crosswalk


def replace_once(text,before,after):
    if text.count(before)!=1:raise ValueError('Parent source differs at controlled adapter: '+before[:75])
    return text.replace(before,after,1)


def generate_sources(root):
    worker=(root/'analysis/recomb_confirm_v2_application_predict.py').read_text()
    for old,new in [(str(PARENT_CAT),str(CAT)),('reports/recomb_confirm_v2_application_structures',str(STRUCT)),
                    (str(PARENT_CONFIRM),str(CONFIRM))]:
        worker=worker.replace(old,new)
    worker=replace_once(worker,";a=p.parse_args()",";p.add_argument('--device-class',choices=['A100','A40'],required=True);p.add_argument('--request-id');a=p.parse_args()")
    worker=replace_once(worker,"    if not 0<=a.shard_index<a.shards:",
        "    requests=[r for r in requests if r['device_class']==a.device_class]\n"
        "    if a.request_id:\n"
        "        requests=[r for r in requests if r['task_id']==a.request_id]\n"
        "        if len(requests)!=1:raise ValueError('Canary must select exactly one frozen request')\n"
        "    if not 0<=a.shard_index<a.shards:")
    worker=replace_once(worker,
        "    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name():raise RuntimeError('Frozen application profile requires A100')",
        "    if not torch.cuda.is_available() or a.device_class not in torch.cuda.get_device_name():raise RuntimeError('Assigned hardware differs from actual GPU')\n"
        "    if not torch.cuda.is_bf16_supported():raise RuntimeError('Frozen bf16 predictor requires hardware support')")
    worker=replace_once(worker,"'boltz_'+a.phase+'_shard_'","'boltz_'+a.phase+'_'+a.device_class+'_shard_'")
    evaluator=(root/'analysis/recomb_confirm_v2_application_evaluate.py').read_text()
    for old,new in [(str(PARENT_CAT),str(CAT)),('reports/recomb_confirm_v2_application_structures',str(STRUCT)),
                    ('reports/recomb_confirm_v2_application_evaluation',str(EVAL)),(str(PARENT_CONFIRM),str(CONFIRM))]:
        evaluator=evaluator.replace(old,new)
    evaluator=replace_once(evaluator,"        if artifact:\n",
        "        if artifact and r['device_class'] not in saved.get('actual_GPU',''):raise ValueError('Actual GPU differs from input-frozen hardware assignment')\n        if artifact:\n")
    report=(root/'analysis/recomb_confirm_v2_application_repeat_report.py').read_text()
    for old,new in [(str(PARENT_CAT),str(CAT)),('reports/recomb_confirm_v2_application_evaluation/all',str(EVAL/'all')),
                    ('reports/recomb_confirm_v2_application_final_report',str(REPORT)),
                    ('configs/recomb_confirm_v2_application_repeat_report_freeze.json',str(SOFTWARE)),
                    (str(PARENT_CONFIRM),str(CONFIRM))]:
        report=report.replace(old,new)
    paths=['analysis/recomb_confirm_v2_application_predict_hardware_v3.py',
           'analysis/recomb_confirm_v2_application_evaluate_hardware_v3.py',
           'analysis/recomb_confirm_v2_application_repeat_report_hardware_v3.py']
    for path,source in zip(paths,[worker,evaluator,report]):
        p=root/path
        if p.exists():raise FileExistsError('Versioned source already exists: '+path)
        ast.parse(source);p.write_text(source,encoding='utf-8')
    def functions(source):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    old=functions((root/'analysis/recomb_confirm_v2_application_evaluate.py').read_text());new=functions(evaluator)
    for name in set(old)-{'evaluate'}:
        if old[name]!=new[name]:raise ValueError('An endpoint/statistical function changed')
    old=functions((root/'analysis/recomb_confirm_v2_application_repeat_report.py').read_text());new=functions(report)
    if old!=new:raise ValueError('Repeat comparison/report functions changed')
    return paths


def prepare(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm amendment required')
    if (root/CONFIRM).exists():raise FileExistsError('Hardware amendment already frozen')
    for folder in ['reports/recomb_confirm_v2_application_structures',str(STRUCT)]:
        p=root/folder/'prediction_journal.jsonl'
        if p.exists() and p.stat().st_size:raise ValueError('Application outcomes already started; do not change hardware after outcomes')
    parent=read(root/PARENT_CONFIRM)
    if not parent['fold_allowed']:raise ValueError('Parent native input/metric release is required')
    for item in parent['files']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Parent scientific input/source changed')
    physical=read(root/PARENT_CAT/'physical_requests.json');logical=read(root/PARENT_CAT/'logical_requests.json')
    profiles=read(root/PARENT_CAT/'predictor_profiles.json')
    inputs={r['model_input_relative_path']:read(root/r['model_input_relative_path']) for r in physical}
    from boltz.data.mol import load_molecules
    mol_dir=Path('/public/home/zhangyangroup/chengshiz/.boltz/mols')
    codes={code for inp in inputs.values() for x in inp['sequences'] if 'ligand' in x
           for code in x['ligand']['ccd']}
    molecules=load_molecules(mol_dir,sorted(codes))
    heavy={code:int(molecules[code].GetNumHeavyAtoms()) for code in codes}
    sizes={}
    for r in physical:
        inp=inputs[r['model_input_relative_path']]
        sizes[r['task_id']]=sum(len(x['protein']['sequence']) if 'protein' in x
                              else sum(heavy[c] for c in x['ligand']['ccd']) for x in inp['sequences'])
    route,max_size=assign_hardware(logical,sizes)
    requests,rows,new_profiles,crosswalk=remap_catalog(physical,logical,profiles,route)
    if len(rows)!=2160 or len(requests)!=1759:raise ValueError('Lost original logical or physical predictions')
    originals={r['task_id']:r for r in physical}
    bridge=[]
    for r in requests:
        old=originals[r['parent_task_id']]
        if any(r[k]!=old[k] for k in ['model_input_relative_path','model_input_sha256','chains','weights_sha256','seed','sample','sequence','canonical_nonspatial_input_sha256']):
            raise ValueError('Hardware routing changed a scientific input')
        a=dict(new_profiles[r['profile_name']]);b=dict(profiles[old['profile_name']])
        a.pop('device_class',None);b.pop('device_class',None)
        if a!=b:raise ValueError('Predictor settings other than hardware changed')
        bridge.append(dict(parent_task_id=old['task_id'],task_id=r['task_id'],device_class=r['device_class'],
                           model_input_sha256=r['model_input_sha256'],same_actual_input_MSA_chemical_graph=True))
    out=root/CAT;out.mkdir(parents=True,exist_ok=True)
    protocol=read(root/PARENT_CAT/'statistical_protocol.json')
    protocol['GPU_concurrency']='No fixed task cap; user authorized all idle allowed Slurm A100/A40'
    protocol['hardware_routing']='WT max over all frozen inputs: protein residues+actual CCD heavy atoms<=1024 ->A40; otherwiseA100; shared inputs promote wholeWT group toA100; no outcome use'
    for name,value in [('physical_requests.json',requests),('logical_requests.json',rows),('predictor_profiles.json',new_profiles),
                       ('statistical_protocol.json',protocol),('hardware_assignment.json',dict(by_WT=route,max_input_tokens_by_WT=max_size,threshold=THRESHOLD,CCD_heavy_atoms=heavy)),
                       ('parent_input_bijection.json',bridge)]:
        write(out/name,value)
    sources=generate_sources(root)
    all_files={p['path']:p['sha256'] for p in parent['files']}
    all_files[str(root/PARENT_CONFIRM)]=sha(root/PARENT_CONFIRM)
    names=sources+['analysis/recomb_confirm_v2_application_hardware.py','analysis/tests/test_recomb_confirm_v2_application_hardware.py',
                   'slurm/recomb_confirm_v2_application_hardware_prepare.sbatch','slurm/recomb_confirm_v2_application_hardware_gpu.sbatch',
                   'slurm/recomb_confirm_v2_application_hardware_evaluate.sbatch','slurm/recomb_confirm_v2_application_hardware_report.sbatch',
                   'reports/recomb_confirm_v2_execution/GPU_resource_policy_20261018.json']
    names += [str(CAT/p) for p in ['physical_requests.json','logical_requests.json','predictor_profiles.json','statistical_protocol.json','hardware_assignment.json','parent_input_bijection.json']]
    for p in names:all_files[str(root/p)]=sha(root/p)
    write(root/SOFTWARE,dict(status='READ_ONLY_REPEAT_REPORT_REUSED_UNCHANGED_FUNCTIONS_HARDWARE_NAMESPACE_ONLY',
          files={p:sha(root/p) for p in sources+['analysis/recomb_confirm_v2_application_hardware.py']},
          parent_report_software_freeze_sha256=sha(root/'configs/recomb_confirm_v2_application_repeat_report_freeze.json')))
    all_files[str(root/SOFTWARE)]=sha(root/SOFTWARE)
    write(root/CONFIRM,dict(status='APPLICATION_USER_AUTHORIZED_HARDWARE_AMENDMENT_BEFORE_FIRST_PREDICTION',
        fold_allowed=True,all_selected_outputs_frozen=True,total_logical_requests=2160,physical_requests=1759,
        parent_confirmation_sha256=sha(root/PARENT_CONFIRM),maximum_task_GPUs=None,
        all_scientific_inputs_endpoints_statistics_unchanged=True,files=[dict(path=p,sha256=h) for p,h in sorted(all_files.items())]))
    wt={r['physical_id'] for r in rows if r['method']=='WT' and r['seed']==20261017}
    canary=next((r['task_id'] for r in requests if r['task_id'] in wt and r['device_class']=='A40'),None)
    done=dict(status='HARDWARE_ROUTING_INPUT_BIJECTION_AND_UNCHANGED_METRICS_PROVED_GPU_RELEASE_READY',
              logical=2160,physical=1759,WT_counts=dict(Counter(route.values())),
              physical_counts=dict(Counter(r['device_class'] for r in requests)),WT_A40_canary_request=canary,
              WT_physical_counts=dict(Counter(r['device_class'] for r in requests if r['task_id'] in wt)),
              freeze_sha256=sha(root/CONFIRM),new_model_calls=0,new_structures=0,job_id=os.environ['SLURM_JOB_ID'])
    write(out/'release_completion.json',done)
    for folder in [STRUCT,EVAL,REPORT]:(root/folder).mkdir(parents=True,exist_ok=True)
    print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);prepare(p.parse_args().root)
```

## Complete launcher: slurm/recomb_confirm_v2_application_hardware_prepare.sbatch

```bash
#!/bin/bash
#SBATCH --job-name=recomb-v2-app-hardware-release
#SBATCH --partition=cpu_cu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=reports/recomb_confirm_v2_execution/hardware_prepare_%j.log
set -euo pipefail
cd /public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$PWD/analysis"
PY=/public/home/zhangyangroup/chengshiz/anaconda3/envs/boltz/bin/python
"$PY" -m unittest discover -s analysis/tests -p test_recomb_confirm_v2_application_hardware.py -v
"$PY" -u analysis/recomb_confirm_v2_application_hardware.py --root "$PWD"
```

## Complete launcher: slurm/recomb_confirm_v2_application_hardware_gpu.sbatch

```bash
#!/bin/bash
#SBATCH --job-name=recomb-v2-app-hardware-Boltz
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=18:00:00
#SBATCH --output=reports/recomb_confirm_v2_application_structures_hardware_v3/boltz_%A_%a.log
set -euo pipefail
cd /public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD/analysis:$PWD/code"
ARGS=(--freeze-sha256 "$1" --phase "$2" --device-class "$3" --shard-index "${SLURM_ARRAY_TASK_ID:-0}" --shards "$4")
if [ -n "${5:-}" ]; then ARGS+=(--request-id "$5"); fi
/public/home/zhangyangroup/chengshiz/anaconda3/envs/boltz/bin/python -u analysis/recomb_confirm_v2_application_predict_hardware_v3.py "${ARGS[@]}"
```

## Complete launcher: slurm/recomb_confirm_v2_application_hardware_evaluate.sbatch

```bash
#!/bin/bash
#SBATCH --job-name=recomb-v2-app-hardware-metrics
#SBATCH --partition=cpu_cu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=reports/recomb_confirm_v2_application_evaluation_hardware_v3/metrics_%j.log
set -euo pipefail
cd /public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/public/home/zhangyangroup/chengshiz/keyuan.zhou/evocompress/.venv-v7/bin/python -u analysis/recomb_confirm_v2_application_evaluate_hardware_v3.py --root "$PWD" --phase "$1"
```

## Complete launcher: slurm/recomb_confirm_v2_application_hardware_report.sbatch

```bash
#!/bin/bash
#SBATCH --job-name=recomb-v2-app-hardware-report
#SBATCH --partition=cpu_cu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=reports/recomb_confirm_v2_application_final_report_hardware_v3/report_%j.log
set -euo pipefail
cd /public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/public/home/zhangyangroup/chengshiz/keyuan.zhou/evocompress/.venv-v7/bin/python -u analysis/recomb_confirm_v2_application_repeat_report_hardware_v3.py --root "$PWD"
```

## Dispatch commands

CPU: sbatch --dependency=afterok:1119471 slurm/recomb_confirm_v2_application_hardware_prepare.sbatch. Expected release status HARDWARE_ROUTING_INPUT_BIJECTION_AND_UNCHANGED_METRICS_PROVED_GPU_RELEASE_READY and zero new model calls.

A100 WT: sbatch --partition=gpu_gua --gres=gpu:NVIDIAA10080GBPCIe:1 --array=0-$((N-1)) slurm/recomb_confirm_v2_application_hardware_gpu.sbatch FROZEN_SHA WT A100 N. Here N is the measured available A100 count capped only by remaining assigned WT inputs, never by an arbitrary task limit.

A40 canary: sbatch --partition=gpu_gub --gres=gpu:NVIDIAA40:1 slurm/recomb_confirm_v2_application_hardware_gpu.sbatch FROZEN_SHA WT A40 1 EXACT_FROZEN_CANARY_ID. Require actual GPU=A40, bf16 support, complete atom/input/chemical output audit, and SUCCEEDED ledger. It is a scheduled WT prediction, not an extra pilot or quality-selection gate.

After WT, submit derived evaluator with --dependency=afterok:WT_JOB_IDS and phase WT. Continue the fixed main then extension matrices. All job IDs and hardware manifests are persisted. No further approval step is needed: the user explicitly authorized execution and both GPU types.

Self-review: source/control definitions exist above; WT routing prevents method/hardware confounding within WT; input hashes retain source/chemistry/MSA identity; profile hash changes with device; old completed structures are untouched; original metrics and seed summaries retain identical function ASTs; actual allocation is distinguished from physical idle resources; no tensor, roster, selector, budget, native template, or endpoint is changed.



Execution note: one additional real-parent-source generation regression is included in the queued test suite (6 tests total). The local runtime lacks Bio; required tests use the existing Boltz CPU environment through job1119658. User authorization already selects inline execution; no approval or agent handoff is requested.
