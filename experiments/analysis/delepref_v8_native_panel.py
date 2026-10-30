"""Validate native references before selecting the final sequence-only v8 panel."""
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import sys
import tarfile
import time

import numpy as np
import delepref_v8_panel as panel

ROOT=panel.ROOT
BASE='reports/delepref_v8_native_panel'
FREEZE='configs/delepref_v8_native_panel_protocol.json'
CONTRACT=dict(minimum_native_CA_coverage=.90,
    mapping='existing_audited_exact_entity_segment_or_unique_exact_CATH_subsequence',
    altloc='first_CA_in_original_file_order',failed_reference='exclude_before_model_outputs_and_continue_frozen_hash_order',
    validation_scope='all_training-eligible_input_candidates',variant_structures_used=False,
    selection_order=panel.CONTRACT['order'],target_WT=100)

def prepare(root):
    out=root/BASE;out.mkdir(exist_ok=False)
    inputs=['reports/delepref_v8_panel/completion.json','reports/delepref_v8_panel/candidate_universe.json',
        'reports/delepref_v8_panel/queries.json','reports/delepref_v8_panel/training_homology_audit.json',
        'reports/delepref_v8_panel/self_hits.tsv','reports/delepref_v8_panel/provisional_panel.json','reports/asset_audit.json']
    sources=['analysis/delepref_v8_native_panel.py','analysis/delepref_v8_panel.py',
        'code/adapters/structure_adapter.py','code/common.py','code/tests/test_mapping.py','slurm/delepref_v8_native_panel.sbatch']
    write=panel.write
    write(root/FREEZE,dict(status='NATIVE_PANEL_PROTOCOL_FROZEN_BEFORE_NATIVE_MAPPING_RESULTS',
        contract=CONTRACT,inputs={p:panel.sha(root/p) for p in inputs},sources={p:panel.sha(root/p) for p in sources}))
    return dict(freeze_sha256=panel.sha(root/FREEZE))

def components_from_inputs(candidates,queries,self_path):
    components=panel.Components([x['candidate_id'] for x in candidates]);groups=defaultdict(list)
    for row in candidates:
        cid=row['candidate_id'];groups[('sha',row['sequence_sha256'])].append(cid)
        pid=panel.pdb_identity(row)
        if pid:groups[('pdb',pid)].append(cid)
        for family in panel.families(row):groups[('CATH_H',family)].append(cid)
        if row.get('sequence_homology_component'):
            groups[('source_component',row['source_file'],row['sequence_homology_component'])].append(cid)
    for ids in groups.values():
        for other in ids[1:]:components.union(ids[0],other)
    with self_path.open() as f:
        for line in f:
            h=panel.alignment(line.rstrip('\n').split('\t'),queries)
            if h['query']==h['target'] or not panel.passes(h,.3):continue
            for a in queries[h['query']]['candidate_ids']:
                for b in queries[h['target']]['candidate_ids']:components.union(a,b)
    return components

def execute(root,checksum):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    if panel.sha(root/FREEZE)!=checksum:raise ValueError('native protocol changed')
    protocol=json.loads((root/FREEZE).read_text())
    if protocol['contract']!=CONTRACT:raise ValueError('native selection contract changed')
    for p,h in {**protocol['inputs'],**protocol['sources']}.items():
        if panel.sha(root/p)!=h:raise ValueError('native source/input changed: '+p)
    out=root/BASE;panel.write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID']))
    start=time.time();previous=root/'reports/delepref_v8_panel'
    completion=json.loads((previous/'completion.json').read_text())
    if panel.sha(previous/'training_homology_audit.json')!=completion['audit_sha256']:raise ValueError('homology audit identity changed')
    candidates=json.loads((previous/'candidate_universe.json').read_text())
    queries={q['query_id']:q for q in json.loads((previous/'queries.json').read_text())}
    audits=json.loads((previous/'training_homology_audit.json').read_text())['queries']
    rejected={cid for q in audits if q['threshold_hits'] for cid in q['candidate_ids']}
    components=components_from_inputs(candidates,queries,previous/'self_hits.tsv')
    before=panel.select(candidates,rejected,components)
    expected=json.loads((previous/'provisional_panel.json').read_text())
    if [r['candidate_id'] for r in before]!=[r['candidate_id'] for r in expected]:
        raise ValueError('native stage does not reproduce provisional input selection')
    eligible=[r for r in candidates if r['candidate_id'] not in rejected]
    need={r['archive_member']:r for r in eligible if not (r.get('coordinate_path') or r.get('structure_path'))}
    references=out/'references';references.mkdir(exist_ok=False)
    extracted={}
    if need:
        record=json.loads((root/'reports/asset_audit.json').read_text())['cath_native_archive']
        archive=Path(record['path'])
        if panel.sha(archive)!=record['sha256']:raise ValueError('CATH reference archive hash changed')
        with tarfile.open(archive,'r|gz') as handle:
            for member in handle:
                name=member.name.removeprefix('./')
                if name not in need:continue
                if not member.isfile() or not 0<member.size<10*1024*1024:raise ValueError('invalid CATH PDB member')
                row=need[name];destination=references/(row['sequence_sha256']+'.pdb')
                data=handle.extractfile(member).read()
                with destination.open('xb') as f:f.write(data)
                extracted[name]=destination
                if len(extracted)==len(need):break
    sys.path.insert(0,str(root/'code'))
    from adapters.structure_adapter import read_coordinates,read_cath_reference
    results=[];valid=[];failure_ids=set()
    for row in sorted(eligible,key=lambda x:x['candidate_id']):
        cid=row['candidate_id'];sequence=row['sequence'];reference=None
        try:
            if row['stratum']=='cath_domain':
                reference=Path(row['coordinate_path']) if row.get('coordinate_path') else extracted.get(row['archive_member'])
                if reference is None:raise ValueError('CATH archive member missing')
                chain=row['domain_id'][4]
                xyz,mapping=read_cath_reference(reference,sequence,chain)
            elif row['stratum']=='pdb_chain':
                reference=Path(row.get('structure_path') or row['coordinate_path'])
                chain=row['chain_id']
                xyz,mapping=read_coordinates(reference,sequence,chain,native=True)
            else:raise ValueError('unsupported native stratum')
            coverage=float(np.isfinite(xyz).all(1).mean())
            if coverage<.9:raise ValueError('native CA coverage below predeclared 0.90')
            target=references/(row['sequence_sha256']+'.json')
            native=dict(protein_id=row['protein_id'],wt_sequence_sha256=row['sequence_sha256'],
                reference_path=str(reference),reference_sha256=panel.sha(reference),chain=chain,
                coordinates={str(i):p.tolist() for i,p in enumerate(xyz) if np.isfinite(p).all()},
                missing_positions=np.flatnonzero(~np.isfinite(xyz).all(1)).tolist(),
                CA_coverage=coverage,mapping=mapping)
            panel.write(target,native)
            validated=dict(row,native_reference_path=str(reference),native_reference_sha256=native['reference_sha256'],
                wt_native_json=str(target),wt_native_json_sha256=panel.sha(target),wt_CA_coverage=coverage,reference_chain=chain)
            valid.append(validated)
            results.append(dict(candidate_id=cid,status='PASS',CA_coverage=coverage,wt_native_json=str(target)))
        except (ValueError,KeyError,RuntimeError) as error:
            failure_ids.add(cid);results.append(dict(candidate_id=cid,status='REFERENCE_REJECTED',type=type(error).__name__,reason=str(error)))
    final=panel.select(valid,set(),components)
    panel.write(out/'reference_audit.json',results)
    panel.write(out/'main_panel.json',final)
    with (out/'main_panel.fasta').open('x') as f:
        for i,row in enumerate(final):f.write('>v8main_%04d\n%s\n'%(i,row['sequence']))
    report=dict(status='NATIVE_VALIDATED_INPUT_PANEL_READY_METHOD_OUTPUTS_NOT_FROZEN',job_id=os.environ['SLURM_JOB_ID'],
        seconds=time.time()-start,protocol_sha256=checksum,homology_eligible=len(eligible),
        native_valid=len(valid),native_rejected=len(failure_ids),extracted_CATH_members=len(extracted),
        final_WT=len(final),source_types=dict(Counter(x['stratum'] for x in final)),
        failure_reasons=dict(Counter(r['reason'] for r in results if r['status']!='PASS')),
        model_or_variant_outcomes_used=False,final_method_outputs_frozen=False,structure_prediction_allowed=False,
        native_audit_sha256=panel.sha(out/'reference_audit.json'),panel_sha256=panel.sha(out/'main_panel.json'),
        new_GPU_jobs=0,new_structure_predictions=0,paper_changes=False)
    panel.write(out/'completion.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=__import__('argparse').ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--prepare',action='store_true');p.add_argument('--freeze-sha256');a=p.parse_args()
    print(json.dumps(prepare(a.root)) if a.prepare else execute(a.root,a.freeze_sha256),flush=True)
