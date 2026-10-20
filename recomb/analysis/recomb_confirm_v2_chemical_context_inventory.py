"""Inventory native chemical/branched context for frozen ordinary application inputs."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import os
from pathlib import Path
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from recomb_confirm_v2_diagnostics import read,sha,write
from recomb_confirm_v2_enzyme_context import rows,values

BASE=Path('reports/recomb_confirm_v2_chemical_context')
FZ=Path('configs/recomb_confirm_v2_chemical_context_inventory_freeze.json')
MOLS=Path('/public/home/zhangyangroup/chengshiz/.boltz/mols')


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('Application chemical inventory already frozen')
    files=['analysis/recomb_confirm_v2_chemical_context_inventory.py','analysis/recomb_confirm_v2_enzyme_context.py',
        'analysis/recomb_confirm_v2_diagnostics.py','slurm/recomb_confirm_v2_chemical_context.sbatch']
    inputs=[]
    for cohort in ['Binder60','Enzyme80']:
        pp=root/('reports/recomb_confirm_v2_'+cohort+'_constructs/'+cohort+'_panel.json')
        for row in read(pp):
            p=row['native_assembly_reference']
            if sha(p)!=row['native_assembly_reference_sha256']:raise ValueError('Application assembly reference changed')
            ref=read(p)
            inputs.append(dict(cohort=cohort,protein_id=row['protein_id'],assembly_reference=p,
                assembly_reference_sha256=sha(p),native_file=ref['native_file'],native_file_sha256=ref['native_file_sha256']))
        files.append(str(pp.relative_to(root)))
    write(root/BASE/'inputs.json',inputs);files.append(str(BASE/'inputs.json'))
    write(root/FZ,dict(status='APPLICATION_NATIVE_CHEMICAL_AND_BRANCH_CONTEXT_INVENTORY_ONLY_FROZEN',
        files={p:sha(root/p) for p in files},native_files={r['native_file']:r['native_file_sha256'] for r in inputs},
        all_native_nonpolymers_not_assumed_required=True,native_coordinates_not_exported_as_prediction_templates=True,
        chemical_prediction_release=False,new_model_calls=0,new_structure_predictions=0))
    print(sha(root/FZ),flush=True)


def inspect_one(row):
    path=Path(row['native_file']);reference=read(row['assembly_reference'])
    if sha(path)!=row['native_file_sha256'] or sha(row['assembly_reference'])!=row['assembly_reference_sha256']:
        raise ValueError('Frozen native chemical reference changed')
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as h:d=MMCIF2Dict(h)
    asym={r['id']:r['entity_id'] for r in rows(d,'_struct_asym.')}
    nonpoly=rows(d,'_pdbx_entity_nonpoly.');branched=rows(d,'_pdbx_entity_branch.')
    nonprotein_entities={r['entity_id'] for r in nonpoly+branched}
    nonprotein_labels={label for label,eid in asym.items() if eid in nonprotein_entities}
    gens=[r for r in rows(d,'_pdbx_struct_assembly_gen.') if r['assembly_id']==reference['assembly_id']]
    membership={a.strip() for g in gens for a in g['asym_id_list'].split(',')}
    chemical={r['id']:r for r in rows(d,'_chem_comp.')}
    branch_units=rows(d,'_pdbx_entity_branch_list.');branch_links=rows(d,'_pdbx_entity_branch_link.')
    branch_scheme=rows(d,'_pdbx_branch_scheme.')
    codes={r['comp_id'] for r in nonpoly}|{r['comp_id'] for r in branch_units}
    contexts=[]
    for code in sorted(codes):
        p=MOLS/(code+'.pkl')
        contexts.append(dict(CCD=code,native_chemical_definition=chemical.get(code),
            cached_Boltz_molecule_available=p.exists(),cached_Boltz_molecule_path=str(p) if p.exists() else None,
            cached_Boltz_molecule_sha256=sha(p) if p.exists() else None))
    conn=[]
    for c in rows(d,'_struct_conn.'):
        if c.get('ptnr1_label_asym_id') in nonprotein_labels or c.get('ptnr2_label_asym_id') in nonprotein_labels:
            conn.append(c)
    cofactor=reference.get('canonical_cofactor_annotations',[])
    return dict(protein_id=row['protein_id'],cohort=row['cohort'],assembly_id=reference['assembly_id'],
        native_nonpolymer_entities=nonpoly,native_branched_entities=branched,branch_units=branch_units,
        branch_linkages=branch_links,branch_scheme=branch_scheme,native_chemical_connections=conn,
        documented_assembly_generation=gens,nonprotein_asym_labels_in_assembly=sorted(nonprotein_labels&membership),
        chemical_components=contexts,canonical_cofactor_annotations=cofactor,
        native_file_sha256=row['native_file_sha256'],
        interpretation='identity/context inventory only; buffers, substrates, cofactors and covalent glycans require separate explicit roles',
        full_chemical_prediction_protocol_pending=True)


def run(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    f=read(root/FZ)
    for p,h in f['files'].items():
        if sha(root/p)!=h:raise ValueError('Frozen chemical context inventory source/input changed')
    out=root/BASE
    if (out/'execution_claim.json').exists():raise FileExistsError('Chemical inventory claimed; inspect existing job')
    write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID']))
    with ProcessPoolExecutor(max_workers=8) as executor:result=list(executor.map(inspect_one,read(out/'inputs.json')))
    write(out/'per_case_chemical_inventory.json',result)
    summary={}
    for cohort in ['Binder60','Enzyme80']:
        r=[x for x in result if x['cohort']==cohort]
        summary[cohort]=dict(cases=len(r),with_nonwater_nonpolymer=sum(any(c['comp_id']!='HOH' for c in x['native_nonpolymer_entities']) for x in r),
            with_branched_context=sum(bool(x['native_branched_entities']) for x in r),
            with_nonprotein_connections=sum(bool(x['native_chemical_connections']) for x in r),
            with_canonical_cofactor_annotations=sum(bool(x['canonical_cofactor_annotations']) for x in r),
            native_CCD_case_counts=dict(Counter(c['CCD'] for x in r for c in x['chemical_components'])),
            unavailable_CCD=sorted({c['CCD'] for x in r for c in x['chemical_components'] if not c['cached_Boltz_molecule_available']}))
    done=dict(status='APPLICATION_CHEMICAL_CONTEXT_INVENTORY_COMPLETE_ROLE_AND_INPUT_PROTOCOL_PENDING',
        cohorts=summary,inventory_sha256=sha(out/'per_case_chemical_inventory.json'),
        new_model_calls=0,new_structure_predictions=0,chemical_prediction_release=False,job_id=os.environ['SLURM_JOB_ID'])
    write(out/'completion.json',done);print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();prepare(a.root) if a.prepare else run(a.root)
