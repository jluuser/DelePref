"""Freeze Enzyme80 from qualified ordinary protein assemblies using input strata only."""
import argparse
from collections import Counter,defaultdict,deque
import hashlib
import os
from pathlib import Path
from recomb_confirm_v2_diagnostics import read,sha,write

BASE=Path('reports/recomb_confirm_v2_Enzyme80_constructs')
FZ=Path('configs/recomb_confirm_v2_Enzyme80_selection_freeze.json')


def key(row,salt):
    return hashlib.sha256((salt+'|'+row['candidate_id']+'|'+row['WT_sha256']).encode()).hexdigest()


def ec(row):return sorted(row['ec_numbers'])[0].split('.')[0]


def length_bin(row):
    n=len(row['sequence'])
    return 'short' if n<250 else ('medium' if n<500 else 'long')


def ordered_by_strata(rows,salt):
    groups=defaultdict(list)
    for row in rows:groups[(length_bin(row),row['protein_instances']>1)].append(row)
    queues=[deque(sorted(groups[g],key=lambda r:key(r,salt))) for g in sorted(groups)]
    result=[]
    while any(queues):
        for queue in queues:
            if queue:result.append(queue.popleft())
    return result


def select(rows,n,salt):
    groups=defaultdict(list)
    for row in rows:groups[ec(row)].append(row)
    queues=[deque(ordered_by_strata(groups[g],salt)) for g in sorted(groups)]
    result=[];used=set();hashes=set();pdbs=set()
    while any(queues) and len(result)<n:
        for queue in queues:
            while queue:
                row=queue.popleft()
                if row['evaluation_sequence_component'] in used or row['WT_sha256'] in hashes or row['native_reference_pdb'].upper() in pdbs:continue
                result.append(row);used.add(row['evaluation_sequence_component']);hashes.add(row['WT_sha256']);pdbs.add(row['native_reference_pdb'].upper());break
            if len(result)==n:break
    if len(result)!=n:raise ValueError('Qualified unique components/PDB do not reach target; do not lower standards or backfill by performance')
    return result


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('Enzyme80 selection already frozen')
    source=root/'reports/recomb_confirm_v2_enzyme_assemblies/qualified_protein_constructs.json'
    done=read(source.parent/'completion.json')
    if sha(source)!=done['outputs'][source.name] or done['protein_constructs_pass']<80:raise ValueError('Enzyme assembly input audit incomplete')
    names=['analysis/recomb_confirm_v2_enzyme_select.py','analysis/recomb_confirm_v2_diagnostics.py',
        str(source.relative_to(root)),str((source.parent/'completion.json').relative_to(root)),
        'slurm/recomb_confirm_v2_enzyme_select.sbatch']
    write(root/FZ,dict(status='ENZYME80_INPUT_ONLY_COMPONENT_AND_COVERAGE_SELECTION_FROZEN',
        files={p:sha(root/p) for p in names},selection='EC round-robin; within EC length/assembly round-robin then salted input hash; one representative per existing sequence component, WT SHA and PDB',
        N=80,minimum_components=60,subsets_N=30,
        annotation_tiers='MCSA catalytic network and reviewed UniProt Active site kept separate; EC7 supplement has one precise label, not a complete MCSA network',
        inputs_or_model_outcomes_used='input/source/native identity and predefined coverage only; no RAG support or variant quality',
        WT_encoding_stage_only=True,chemical_context_and_full_predictor_profile_pending=True,
        no_native_coordinates_as_templates=True,structure_prediction_allowed=False))
    (root/BASE).mkdir(parents=True,exist_ok=True);print(sha(root/FZ),flush=True)


def run(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm only')
    f=read(root/FZ)
    for p,h in f['files'].items():
        if sha(root/p)!=h:raise ValueError('Frozen enzyme selection source/input changed')
    out=root/BASE
    if (out/'execution_claim.json').exists():raise FileExistsError('Enzyme selection claimed; inspect existing job')
    write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID']))
    source=root/'reports/recomb_confirm_v2_enzyme_assemblies/qualified_protein_constructs.json'
    selected=select(read(source),80,'Enzyme80_v2_input_selection_20261018');panel=[]
    for r in selected:
        if sha(r['native_assembly_reference'])!=r['native_assembly_reference_sha256']:raise ValueError('Enzyme protein assembly reference changed')
        native=read(r['native_assembly_reference'])
        pid='Enzyme80v2|'+key(r,'enzyme80_id')[:16]
        panel.append(dict(r,protein_id=pid,sequence_sha256=r['WT_sha256'],design_chain=native['primary_target_chain'],
            predictor_chains=[{k:p[k] for k in ['predictor_chain_id','source_native_chain','native_label_chain','role','sequence','sequence_sha256','operation_name','target_gene_copy']} for p in native['chains']],
            protein_assembly_complete=True,WT_encoding_allowed=True,PLM_encoding_allowed=True,
            deletion_generation_allowed=False,structure_prediction_allowed=False,
            chemical_context_and_predictor_inputs_pending=True))
    sub=select(panel,30,'Enzyme80_v2_repetition_subset_20261018')
    subset=[{k:r[k] for k in ['protein_id','WT_sha256','evaluation_sequence_component','annotation_tier']} for r in sub]
    sp=root/'reports/recomb_confirm_v2_Enzyme80_subsets/subsets.json'
    write(sp,dict(budget10pct_and_three_structure_seeds=subset,generation_seed=13,
        structure_seeds=[20261017,20261018,20261019],selection='input-only EC/length/assembly round robin and hash, unique components',
        no_RAG_or_variant_quality_selection=True,seed_repeats_not_independent_proteins=True))
    write(out/'Enzyme80_panel.json',panel)
    write(out/'encoder_inputs.json',[dict(protein_id=r['protein_id'],wildtype_sequence=r['sequence'],budget=.2) for r in panel])
    done=dict(status='ENZYME80_DOCUMENTED_PROTEIN_CONSTRUCTS_READY_FOR_WT_ENCODING_CHEMICAL_PROFILE_PENDING',
        cases=80,sequence_components=len({r['evaluation_sequence_component'] for r in panel}),
        unique_WT=len({r['WT_sha256'] for r in panel}),unique_PDB=len({r['native_reference_pdb'].upper() for r in panel}),
        EC_first_digit=dict(Counter(ec(r) for r in panel)),length_strata=dict(Counter(length_bin(r) for r in panel)),
        protein_instance_counts=dict(Counter(str(r['protein_instances']) for r in panel)),
        annotation_tiers=dict(Counter(r['annotation_tier'] for r in panel)),
        training_domain_like_exposure=sum(r['training_domain_like_exposure'] for r in panel),
        full_context_AA_min=min(r['full_protein_context_AA'] for r in panel),full_context_AA_max=max(r['full_protein_context_AA'] for r in panel),
        subsets_N=30,subset_sha256=sha(sp),WT_encoding_allowed=True,fold_allowed=False,
        outputs={p:sha(out/p) for p in ['Enzyme80_panel.json','encoder_inputs.json']},
        job_id=os.environ['SLURM_JOB_ID'],new_model_calls=0,new_structure_predictions=0)
    write(out/'completion.json',done);print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();prepare(a.root) if a.prepare else run(a.root)
