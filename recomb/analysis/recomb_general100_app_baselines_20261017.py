"""Reuse exact application baselines; only missing Binder Raygun is generated."""
import argparse
import json
import os
from pathlib import Path
import numpy as np

import recomb_general100_baselines_20261017 as native
import recomb_general100_common_20261017 as common
from recomb_general100_common_20261017 import ROOT,sha,write,sequence_sha,require_gpu
from recomb_general100_app_prepare_20261017 import BASE as APP,LEGACY
import delepref_unlimited_planning as planning

BASE=Path('reports/recomb_general100_app_baselines_20261017')
FREEZE=Path('configs/recomb_general100_app_baselines_freeze_20261017.json')


def prepare(root):
    out=root/BASE;out.mkdir(exist_ok=False)
    panel=json.loads((root/APP/'panel.json').read_text())
    previous=json.loads((root/APP/'reused_baseline_outputs.json').read_text())
    files={str(root/APP/'panel.json'):sha(root/APP/'panel.json')}
    inventory=json.loads((root/APP/'cache_inventory.json').read_text())
    inventory=[r for r in inventory if r.get('source_method')!='evocompress']
    for p in panel:
        if p['dataset']!='Enzyme20':continue
        source=LEGACY/'Existing_core/groups'/('E2_'+p['original_protein_id']+'.json')
        g=json.loads(source.read_text());files[str(source)]=sha(source)
        if g['wt_sequence']!=p['sequence']:raise ValueError('legacy SCISOR WT construct differs')
        choices=[r for r in g['rows'] if r['method']=='scisor' and float(r['budget_requested'])==.2 and int(r['candidate_index'])==0]
        if len(choices)!=1:raise ValueError('expected one actual native SCISOR output')
        old=choices[0];mapping=old['wt_to_edit']
        if len(mapping)!=len(p['sequence']):raise ValueError('legacy SCISOR mapping length differs')
        kept=[i for i,x in enumerate(mapping) if x is not None]
        if [mapping[i] for i in kept]!=list(range(len(old['sequence']))) or ''.join(p['sequence'][i] for i in kept)!=old['sequence']:
            raise ValueError('legacy SCISOR explicit mapping does not reconstruct')
        mask=np.array([i is None for i in mapping]);intervals=planning.intervals(mask)
        planning.validate_plan(len(p['sequence']),.2,intervals)
        previous.append(dict(p,method='SCISOR_U90_S_native',edited_sequence=old['sequence'],
            edited_sequence_sha256=sequence_sha(old['sequence']),deleted_intervals=list(map(list,intervals)),
            reused_generation=True,source_path=str(source),native_candidate_record=old))
    wt_source=Path('/public/nfs-data/keyuan.zhou/evocompress/results/formal_v9/revision_20260906_abc/B/e2_predictive_v2/wt_structure_validation.json')
    files[str(wt_source)]=sha(wt_source)
    wtrows={r['protein_id']:r for r in json.loads(wt_source.read_text())['rows']}
    for p in panel:
        if p['dataset']!='Enzyme20':continue
        row=wtrows[p['original_protein_id']]
        if row.get('status')=='PASS' and row['wt_sequence_sha256']==p['sequence_sha256']:
            structure=Path(row['structure_path'])
            if sha(structure)!=row['structure_sha256']:raise ValueError('old enzyme WT structure changed')
            inventory.append(dict(protein_id=p['protein_id'],method='WT',sequence_sha256=p['sequence_sha256'],
                structure_path=str(structure),structure_sha256=sha(structure),
                profile='Enzyme_Boltz2_r3_s200_seed20260908',cache_status='SOURCE_FOUND_PROFILE_AUDIT_REQUIRED'))
    if len(previous)!=60:raise ValueError('expected60 reusable application baseline outputs')
    tasks=[dict(p,sequence=p['sequence'],budget=.2) for p in panel if p['dataset']=='Binder20']
    write(out/'raygun_tasks.json',tasks)
    write(out/'raygun_checkpoint_load_contract.json',json.loads((root/common.BASE/'raygun_checkpoint_load_contract.json').read_text()))
    write(out/'reused_baseline_outputs.json',previous);write(out/'cache_inventory.json',inventory)
    frozen=json.loads((root/common.FREEZE).read_text())
    files.update(frozen['files'])
    for p in [Path(__file__),root/BASE/'raygun_tasks.json',root/BASE/'reused_baseline_outputs.json',root/BASE/'cache_inventory.json']:
        files[str(p)]=sha(p)
    write(root/FREEZE,dict(frozen,status='APPLICATION_MISSING_RAYGUN_FROZEN',files=files,
        application_WT=40,missing_Raygun_conditions=20,reused_native_generations=60,
        complete_structure_prediction_limit=None))
    write(out/'preparation_completion.json',dict(status='APPLICATION_NATIVE_BASELINES_REUSED_AND_MISSING_RAYGUN_FROZEN',
        freeze_sha256=sha(root/FREEZE),missing_Raygun_conditions=20,reused_generation_outputs=60,
        candidate_structure_cache_records=len(inventory),new_structure_predictions=0))
    print(json.dumps(json.loads((out/'preparation_completion.json').read_text())),flush=True)


def execute(root,checksum):
    require_gpu()
    common.FREEZE=FREEZE
    frozen=common.verify(root,checksum)
    native.BASE=BASE
    native.run_raygun(root,frozen,0,1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--prepare',action='store_true');p.add_argument('--freeze-sha256')
    a=p.parse_args();prepare(a.root) if a.prepare else execute(a.root,a.freeze_sha256)
