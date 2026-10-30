"""Input-only native validation and a balanced, outcome-blind General100 release."""
import argparse
from collections import Counter,defaultdict,deque
import hashlib
import os
from pathlib import Path
import sys
import tarfile
import numpy as np
from recomb_confirm_v2_diagnostics import read,sha,write

BASE=Path('reports/recomb_confirm_v2_general_native_v2')
FZ=Path('configs/recomb_confirm_v2_general_native_freeze_v2.json')
ARCHIVE_SHA='e47e8ddcabcb87f4bb061626cafc7da04b252f6202e59d9413d0fe764f8dd23e'


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('General native protocol already frozen')
    suffix=Path('data/par_contraction/cath_v4_3_0/cath-dataset-nonredundant-S40.pdb.tgz')
    archives=[Path('/public/home/zhangyangroup/chengshiz/keyuan.zhou')/name/suffix for name in ['SurgEdit','delepath','DelePref']]
    archive=next((p.resolve() for p in archives if p.exists()),None)
    if archive is None:raise FileNotFoundError('Known cached CATH archive path not found')
    names=['analysis/recomb_confirm_v2_general_native_v2.py','analysis/recomb_confirm_v2_general_encode.py',
        'analysis/recomb_confirm_v2_diagnostics.py','analysis/delepref_v8_runtime.py',
        'code/adapters/structure_adapter.py','code/common.py','configs/delepref_original_restored.yaml',
        'provenance/original_delepref_assets_20261016.json','slurm/recomb_confirm_v2_general_native_v2.sbatch',
        'slurm/recomb_confirm_v2_general_encode.sbatch',
        'reports/recomb_confirm_v2_general_input_extension/General_native_candidate_pool.json',
        'reports/recomb_confirm_v2_design_function_review/per_candidate_function_review_v2.json',
        'reports/recomb_confirm_v2_inventory/application_candidate_metadata.json']
    names += ['reports/recomb_confirm_v2_general_native/all_native_valid_candidates.json',
              'reports/recomb_confirm_v2_general_native/native_audit.json',
              'configs/recomb_confirm_v2_general_native_freeze.json',
              'reports/recomb_confirm_v2_general_input_extension/provisional_Enzyme80_reserve.json']
    write(root/FZ,dict(status='GENERAL100_INPUT_NATIVE_AND_SELECTION_RULES_FROZEN',files={p:sha(root/p) for p in names},
        archive_path=str(archive),archive_sha256=ARCHIVE_SHA,min_native_CA_coverage=.9,
        length_min=150,length_max=1022,target_N=100,target_whole_chains=50,minimum_whole_chains=40,
        chain_priority='previously qualified PPI chains first; MCSA topup excludes fixed provisional Enzyme80 components',
        chain_shortfall='if40to49qualified_use_all_and_fill_domains',
        one_per_sequence_component_and_PDB=True,domain_order='round_robin_CATHclasses1_2_3_hash_within_class',
        chain_order='round_robin_length_bins_150_249_250_399_400_1022_hash_within_bin',
        variant_or_predicted_structure_outcomes_used=False,no_retrieval_success_selection=True))
    (root/BASE).mkdir(parents=True,exist_ok=True);print(sha(root/FZ))


def execute(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    f=read(root/FZ)
    for p,h in f['files'].items():
        if sha(root/p)!=h:raise ValueError('Frozen General source/input differs: '+p)
    out=root/BASE
    if (out/'completion.json').exists():raise FileExistsError('General native stage completed')
    # Original CATH references already passed hash/mapping checks; reuse immutable outputs.
    reused={r['candidate_id']:r for r in read(root/'reports/recomb_confirm_v2_general_native/all_native_valid_candidates.json')}
    previous_rejected={r['candidate_id']:r for r in read(root/'reports/recomb_confirm_v2_general_native/native_audit.json') if r['status']!='PASS'}
    sys.path.insert(0,str(root/'code'))
    from adapters.structure_adapter import read_coordinates,read_cath_reference
    source=read(root/'reports/recomb_confirm_v2_general_input_extension/General_native_candidate_pool.json')
    candidates=[r for r in source if r['role'] in ('general_domain','general_chain')]
    scope={r['candidate_id']:r for r in read(root/'reports/recomb_confirm_v2_design_function_review/per_candidate_function_review_v2.json')}
    refs=out/'references';refs.mkdir(parents=True,exist_ok=True)
    needs={r['archive_member'].removeprefix('./'):r for r in candidates if r['role']=='general_domain' and r['candidate_id'] not in reused}
    if needs:raise RuntimeError('Unexpected missing old CATH reference; inspect before extraction')
    extracted={}
    valid=[];audit=[]
    for r in candidates:
        if scope[r['candidate_id']]['status']!='FUNCTIONAL_SCOPE_PASS_NATIVE_AND_FAMILY_PENDING':raise ValueError('Functional HOLD cannot enter native release')
        if r['candidate_id'] in reused:
            old=reused[r['candidate_id']]
            if old['WT_sha256']!=r['WT_sha256'] or sha(old['wt_native_json'])!=old['wt_native_json_sha256']:
                raise ValueError('Native reuse identity differs')
            keep={k:old[k] for k in ['PDB','wt_native_json','wt_native_json_sha256','native_CA_coverage','reference_chain','stratum']}
            valid.append(dict(r,**keep))
            audit.append(dict(candidate_id=r['candidate_id'],status='PASS_REUSED',native_CA_coverage=old['native_CA_coverage']))
            continue
        if r['candidate_id'] in previous_rejected:
            audit.append(dict(previous_rejected[r['candidate_id']],reused_negative_input_audit=True))
            continue
        try:
            seq=r['sequence']
            if not f['length_min']<=len(seq)<=f['length_max']:raise ValueError('Length outside General contract')
            if r['role']=='general_domain':
                path=extracted[r['archive_member'].removeprefix('./')];chain=r['domain_id'][4]
                xyz,mapping=read_cath_reference(path,seq,chain);pdb=r['domain_id'][:4].upper()
            else:
                path=Path(r['native_reference_path']);chain=r['native_reference_chain']
                xyz,mapping=read_coordinates(path,seq,chain,native=True);pdb=r['native_reference_pdb'].upper()
            coverage=float(np.isfinite(xyz).all(1).mean())
            if coverage<f['min_native_CA_coverage']:raise ValueError('Native CA coverage below input criterion')
            ref=refs/(r['WT_sha256']+'.json')
            write(ref,dict(wt_sequence_sha256=r['WT_sha256'],reference_path=str(path),reference_sha256=sha(path),
                chain=chain,coordinates={str(i):p.tolist() for i,p in enumerate(xyz) if np.isfinite(p).all()},
                missing_positions=np.flatnonzero(~np.isfinite(xyz).all(1)).tolist(),CA_coverage=coverage,mapping=mapping))
            valid.append(dict(r,PDB=pdb,wt_native_json=str(ref),wt_native_json_sha256=sha(ref),native_CA_coverage=coverage,
                reference_chain=chain,stratum='cath_domain' if r['role']=='general_domain' else 'pdb_chain'))
            audit.append(dict(candidate_id=r['candidate_id'],status='PASS',native_CA_coverage=coverage))
        except (ValueError,RuntimeError,KeyError,FileNotFoundError) as error:
            audit.append(dict(candidate_id=r['candidate_id'],status='INPUT_NATIVE_REJECTED',reason=str(error)))
    used_components=set();used_pdb=set();selected=[]
    def choose(items,target,category):
        groups=defaultdict(list)
        for r in items:groups[category(r)].append(r)
        queues={key:deque(sorted(rows,key=lambda r:hashlib.sha256(('recomb-confirm-v2-general100|'+r['WT_sha256']+'|'+r['candidate_id']).encode()).hexdigest())) for key,rows in groups.items()}
        picked=[]
        while len(picked)<target:
            added=False
            for key in sorted(queues):
                queue=queues[key]
                while queue:
                    r=queue.popleft();component=r['evaluation_sequence_component']
                    if component in used_components or r['PDB'] in used_pdb:continue
                    used_components.add(component);used_pdb.add(r['PDB']);picked.append(r);added=True;break
                if len(picked)==target:break
            if not added:break
        return picked
    bins=lambda r:0 if len(r['sequence'])<250 else 1 if len(r['sequence'])<400 else 2
    chains=choose([r for r in valid if r['role']=='general_chain' and r.get('origin_role')!='enzyme'],50,bins)
    chains += choose([r for r in valid if r['role']=='general_chain' and r.get('origin_role')=='enzyme'],50-len(chains),bins)
    write(out/'native_audit.json',audit);write(out/'all_native_valid_candidates.json',valid)
    if len(chains)<40:raise RuntimeError('Qualified whole-chain input shortfall; retain audit and extend ordinary inputs, do not model partial cohort')
    selected=chains+choose([r for r in valid if r['role']=='general_domain'],100-len(chains),lambda r:r['cath_class'])
    if len(selected)!=100:raise RuntimeError('General100 input shortfall before models')
    for r in selected:
        r.update(protein_id='General100v2|'+r['candidate_id'],WT_encoding_allowed=True,
                 deletion_generation_allowed=False,structure_prediction_allowed=False)
    write(out/'General100_panel.json',selected)
    inputs=[dict(protein_id=r['protein_id'],wildtype_sequence=r['sequence'],budget=.2) for r in selected]
    write(out/'encoder_inputs.json',inputs)
    canonical=root/'reports/recomb_confirm_v2_general_native'
    for name in ['General100_panel.json','encoder_inputs.json']:
        target=canonical/name
        if target.exists():raise FileExistsError('Do not overwrite a released canonical panel')
        target.write_bytes((out/name).read_bytes())
    write(canonical/'release_alias_v2.json',dict(source=str(out),reason='v1 quota shortfall; first successful panel is v2',old_audit_preserved=True))
    names=['analysis/recomb_confirm_v2_general_encode.py','analysis/delepref_v8_runtime.py',
        'configs/delepref_original_restored.yaml','provenance/original_delepref_assets_20261016.json',
        str(BASE/'General100_panel.json'),str(BASE/'encoder_inputs.json'),str(FZ),
        'reports/recomb_confirm_v2_general_native/General100_panel.json',
        'reports/recomb_confirm_v2_general_native/encoder_inputs.json']
    write(root/'configs/recomb_confirm_v2_General100_encoder_freeze.json',dict(status='GENERAL100_REAL_WT_ENCODING_RELEASED_ONLY',
        files={p:sha(root/p) for p in names},WT=100,trained_checkpoint_step=4000,model_weights_fixed=True,
        allowed_GPU_types=['A40','A100'],new_training=False,deletion_generation_allowed=False,fold_allowed=False))
    result=dict(status='NATIVE_VALIDATED_GENERAL100_READY_FOR_REAL_WT_ENCODING',WT=100,
        native_candidates=len(candidates),native_valid=len(valid),native_rejected=len(audit)-len(valid),
        sources=dict(Counter(r['stratum'] for r in selected)),
        reused_PPI_and_CATH_reference_count=sum(r['candidate_id'] in reused for r in selected),
        MCSA_topup_whole_chains=sum(r.get('origin_role')=='enzyme' for r in selected),
        length_min=min(len(r['sequence']) for r in selected),
        length_max=max(len(r['sequence']) for r in selected),
        training_domain_like_exposures=sum(r['training_domain_like_exposure'] for r in selected),
        not_strict_training_family_disjoint=True,new_model_calls=0,job_id=os.environ['SLURM_JOB_ID'],
        encoder_freeze_sha256=sha(root/'configs/recomb_confirm_v2_General100_encoder_freeze.json'))
    write(out/'completion.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();(prepare if a.prepare else execute)(a.root)
