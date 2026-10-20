"""Release selected ordinary Binder WT chains with complete documented protein assemblies."""
import argparse
from collections import Counter
import os
from pathlib import Path
import time
from recomb_confirm_v2_diagnostics import read,sha,write
from recomb_confirm_v2_assembly import expand_instances
from recomb_benign_binding_native_20261017 import contact_pairs

BASE=Path('reports/recomb_confirm_v2_Binder60_constructs')
FZ=Path('configs/recomb_confirm_v2_Binder60_constructs_freeze.json')


def author_rank(assembly):
    return 0 if 'author' in str(assembly['annotation'].get('details','')).lower() else 1


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('Selected Binder constructs already frozen')
    selected=root/'reports/recomb_confirm_v2_binder_components/Binder60_roster_pending_constructs.json'
    summary=read(selected.parent/'component_completion.json')
    if summary['selected_cases']!=60 or summary['selected_combined_components']<40 or sha(selected)!=summary['selected_roster_sha256']:
        raise ValueError('Final input-only Binder60 selection not ready')
    paths=['analysis/recomb_confirm_v2_binder_constructs.py','analysis/recomb_confirm_v2_assembly.py',
        'analysis/recomb_confirm_v2_diagnostics.py','analysis/recomb_benign_binding_native_20261017.py',
        'analysis/tests/test_recomb_confirm_v2_assembly.py','slurm/recomb_confirm_v2_binder_constructs.sbatch',
        str(selected.relative_to(root)),str((selected.parent/'component_completion.json').relative_to(root)),
        'configs/recomb_confirm_v2_binder_component_merge_freeze.json']
    refs={r['native_reference']:r['native_reference_sha256'] for r in read(selected)}
    native={read(p)['native_file']:read(p)['native_file_sha256'] for p in refs}
    write(root/FZ,dict(status='SELECTED_BINDER60_DOCUMENTED_PROTEIN_ASSEMBLIES_FROZEN',
        files={p:sha(root/p) for p in paths},native_references=refs,native_coordinate_files=native,
        assembly_choice='documented assembly with all listed protein labels and no extra protein labels; author supported first, then fewest generated protein instances, then lexical ID',
        target_copies='all operation copies of design H are edited identically; fixed light/antigen instances unchanged',
        chemical_context='record native nonprotein context separately; not released as fold inputs here',
        protein_context_required_before_WT_only_encoding=True,fold_and_generation_release=False))
    (root/BASE).mkdir(parents=True,exist_ok=True);print(sha(root/FZ),flush=True)


def run(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm only')
    f=read(root/FZ)
    for p,h in f['files'].items():
        if sha(root/p)!=h:raise ValueError('Frozen construct source/input changed')
    for p,h in {**f['native_references'],**f['native_coordinate_files']}.items():
        if sha(p)!=h:raise ValueError('Native construct reference changed')
    out=root/BASE
    if (out/'execution_claim.json').exists():raise FileExistsError('Construct stage claimed; inspect job')
    write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID']))
    selected=read(root/'reports/recomb_confirm_v2_binder_components/Binder60_roster_pending_constructs.json')
    panel=[];audit=[];start=time.monotonic()
    for row in selected:
        reference=read(row['native_reference'])
        options=[]
        for a in reference['compatible_documented_assemblies']:
            if a['additional_polymer_labels']:continue
            parts,primary=expand_instances(reference['native_file'],a['assembly_id'],reference['chains'])
            options.append((author_rank(a),len(parts),a['assembly_id'],parts,primary,a))
        if not options:raise ValueError('No complete documented protein assembly; preserve selected list, do not substitute by quality')
        _,_,aid,parts,primary,choice=min(options,key=lambda x:x[:3])
        declared=choice['annotation'].get('oligomeric_count')
        if declared and declared.isdigit() and int(declared)!=len(parts):raise ValueError('Generated protein instances differ from declared assembly count')
        target=next(p for p in parts if p['predictor_chain_id']==primary)
        contacts=[];light=[]
        for p in parts:
            if p['role'] not in ('fixed_antigen','fixed_light_chain'):continue
            dst=contacts if p['role']=='fixed_antigen' else light
            dst.extend(dict(design_index=i,partner_chain=p['predictor_chain_id'],partner_index=j)
                for i,j in contact_pairs(target['heavy_atoms'],p['heavy_atoms']))
        if not contacts:raise ValueError('Documented assembly lacks observed primary H-antigen contact')
        native=dict(protein_id=row['protein_id'],native_case_key=row['native_case_key'],PDB=row['PDB'],
            assembly_id=aid,assembly_annotation=choice['annotation'],primary_target_chain=primary,chains=parts,
            observed_H_antigen_contacts=contacts,observed_H_light_contacts=light,
            target_gene_copy_count=sum(p['target_gene_copy'] for p in parts),
            native_file=reference['native_file'],native_file_sha256=reference['native_file_sha256'],
            native_coordinates_for_metrics_only_not_prediction_templates=True,
            chemical_context_prediction_release_pending=True)
        path=out/'references'/(row['native_case_key']+'.json');write(path,native)
        item=dict(row,sequence=row['sequence'],sequence_sha256=row['WT_sha256'],design_chain=primary,
            target_gene_copy_count=native['target_gene_copy_count'],assembly_id=aid,
            predictor_chains=[{k:p[k] for k in ['predictor_chain_id','source_native_chain','native_label_chain','role','sequence','sequence_sha256','operation_name','target_gene_copy']} for p in parts],
            full_protein_context_AA=sum(len(p['sequence']) for p in parts),
            native_assembly_reference=str(path),native_assembly_reference_sha256=sha(path),
            WT_encoding_allowed=True,deletion_generation_allowed=False,structure_prediction_allowed=False,
            protein_assembly_complete=True,chemical_context_and_predictor_inputs_pending=True)
        panel.append(item);audit.append(dict(protein_id=row['protein_id'],PDB=row['PDB'],assembly_id=aid,
            protein_instances=len(parts),target_gene_copies=native['target_gene_copy_count'],
            total_AA=item['full_protein_context_AA'],H_antigen_contacts=len(contacts),H_light_contacts=len(light)))
    if len(panel)!=60 or len({r['WT_sha256'] for r in panel})!=60:raise ValueError('Binder case/WT roster changed')
    write(out/'Binder60_panel.json',panel);write(out/'assembly_audit.json',audit)
    write(out/'encoder_inputs.json',[dict(protein_id=r['protein_id'],wildtype_sequence=r['sequence'],budget=.2) for r in panel])
    done=dict(status='BINDER60_WT_AND_DOCUMENTED_PROTEIN_ASSEMBLIES_READY_FOR_WT_ENCODING_ONLY',
        cases=60,unique_H=60,antigen_components=len({r['antigen_component'] for r in panel}),
        combined_components=len({r['combined_component'] for r in panel}),scaffold=dict(Counter(r['scaffold'] for r in panel)),
        protein_instance_counts=dict(Counter(str(r['protein_instances']) for r in audit)),
        symmetry_expanded_cases=sum(r['target_gene_copies']>1 for r in audit),
        total_context_AA_min=min(r['total_AA'] for r in audit),total_context_AA_max=max(r['total_AA'] for r in audit),
        new_model_calls=0,new_structure_predictions=0,seconds=time.monotonic()-start,job_id=os.environ['SLURM_JOB_ID'],
        outputs={n:sha(out/n) for n in ['Binder60_panel.json','assembly_audit.json','encoder_inputs.json']},fold_allowed=False)
    write(out/'completion.json',done);print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();prepare(a.root) if a.prepare else run(a.root)
