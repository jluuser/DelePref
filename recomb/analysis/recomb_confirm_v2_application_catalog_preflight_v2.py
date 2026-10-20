"""Validate materialized complex/chemical/MSA inputs without any neural forward."""
import argparse
from collections import Counter
import csv
import hashlib
import os
from pathlib import Path
from recomb_confirm_v2_diagnostics import read,sha,write
from recomb_confirm_v2_application_catalog_v2 import BASE,digest
from recomb_confirm_v2_native_bond_audit import verify_native_covalent_bonds

FZ=Path('configs/recomb_confirm_v2_application_catalog_parser_v2_freeze.json')


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('Application variant parser audit already frozen')
    done=read(root/BASE/'completion.json')
    if done['logical']!=2160 or done['fold_allowed']:raise ValueError('Wrong application input matrix or premature release')
    names=['analysis/recomb_confirm_v2_application_catalog_preflight_v2.py','analysis/recomb_confirm_v2_application_catalog_v2.py',
        'analysis/recomb_confirm_v2_native_bond_audit.py','analysis/recomb_confirm_v2_diagnostics.py',
        'slurm/recomb_confirm_v2_application_catalog_preflight_v2.sbatch',
        'configs/recomb_confirm_v2_application_catalog_inputs_v2_freeze.json',str(BASE/'completion.json')]
    names += [str(BASE/p) for p in done['outputs']]
    write(root/FZ,dict(status='ALL_APPLICATION_MATERIALIZED_VARIANT_INPUTS_NATIVE_PARSER_AUDIT_FROZEN',
        files={n:sha(root/n) for n in names},input_identity_is_full_protein_chemical_graph_and_actual_MSA=True,
        no_native_XYZ_template_pocket_or_affinity_inputs=True,new_structure_predictions=0,fold_allowed=False))
    print(sha(root/FZ),flush=True)


def audit(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    for p,h in read(root/FZ)['files'].items():
        if sha(root/p)!=h:raise ValueError('Frozen variant parser input changed: '+p)
    out=root/BASE
    if (out/'native_parser_completion.json').exists():raise FileExistsError('Variant native parser audit already completed')
    from boltz.data import const
    from boltz.data.mol import load_canonicals,load_molecules
    from boltz.data.parse.schema import parse_boltz_schema
    from boltz.data.parse.csv import parse_csv
    msas=read(out/'MSA_manifest.json');rows_by_path={}
    for m in msas:
        p=root/m['CSV_relative_path']
        if sha(p)!=m['CSV_sha256']:raise ValueError('Actual variant MSA CSV changed')
        with p.open(newline='') as handle:rows=list(csv.DictReader(handle))
        if hashlib.sha256(rows[0]['sequence'].encode()).hexdigest()!=m['query_sha256']:raise ValueError('Query SHA mismatch')
        if [hashlib.sha256(r['sequence'].replace('-','').upper().encode()).hexdigest() for r in rows[1:]]!=m['donor_fragment_hashes']:
            raise ValueError('Real natural donor amino-acid conservation failed')
        parsed=parse_csv(p);width=len(rows[0]['sequence'])
        if any(int(r['res_end'])-int(r['res_start'])!=width or int(r['taxonomy'])!=-1 for r in parsed.sequences):
            raise ValueError('Native parsed MSA width/pairing differs')
        rows_by_path[str(p)]=rows[0]['sequence']
    components=load_canonicals(Path('/public/home/zhangyangroup/chengshiz/.boltz/mols'))
    mol_dir=Path('/public/home/zhangyangroup/chengshiz/.boltz/mols')
    profiles=read(out/'predictor_profiles.json');checks=[];seen={}
    for request in read(out/'physical_requests.json'):
        p=root/request['model_input_relative_path']
        if sha(p)!=request['model_input_sha256']:raise ValueError('Frozen actual complex input changed')
        profile=profiles[request['profile_name']]
        if digest(profile)!=request['profile_sha256'] or profile['actual_nonspatial_input_sha256']!=request['canonical_nonspatial_input_sha256']:
            raise ValueError('Full context not bound to predictor profile')
        if request['model_input_sha256'] in seen:continue
        inp=read(p)
        if set(inp)-{'version','sequences','constraints'}:raise ValueError('Unapproved template/affinity field')
        if any(set(x)!={'bond'} for x in inp.get('constraints',[])):raise ValueError('Noncovalent native geometry constraint added')
        proteins={x['protein']['id']:x['protein'] for x in inp['sequences'] if 'protein' in x}
        ligands={x['ligand']['id']:x['ligand'] for x in inp['sequences'] if 'ligand' in x}
        for chain in request['chains']:
            expected=proteins[chain['id']];msa=root/chain['msa_relative_path']
            if expected['sequence']!=chain['sequence'] or expected['msa']!=str(msa) or rows_by_path[str(msa)]!=chain['sequence']:
                raise ValueError('Model input/query/physical chain differs')
            if sha(msa)!=chain['msa_csv_sha256']:raise ValueError('Chain MSA digest differs')
        if set(proteins)!={c['id'] for c in request['chains']}:raise ValueError('Physical protein chain copies differ')
        needed={ccd for l in ligands.values() for ccd in l['ccd']}
        for ccd in needed-set(components):components.update(load_molecules(mol_dir,[ccd]))
        try:
            parsed=parse_boltz_schema('APPctx_'+request['model_input_sha256'][:16],inp,components,mol_dir,boltz_2=True)
            if {str(c['name']) for c in parsed.structure.chains}!=set(proteins)|set(ligands):raise ValueError('Native parsed chain identities differ')
            n=verify_native_covalent_bonds(parsed.structure,inp.get('constraints',[]),const.bond_type_ids['COVALENT'])
            for chain in parsed.record.chains:
                if chain.chain_name in proteins and chain.msa_id!=proteins[chain.chain_name]['msa']:
                    raise ValueError('Native grouped entity MSA differs from explicit chain MSA')
            result=dict(model_input_sha256=request['model_input_sha256'],status='PASS_WITH_DECLARED_CCD_REPRESENTATION_LIMITS',
                protein_chains=len(proteins),ligand_chains=len(ligands),covalent_bonds=n,parsed_atoms=len(parsed.structure.atoms))
        except Exception as error:
            result=dict(model_input_sha256=request['model_input_sha256'],status='NATIVE_SCHEMA_UNRESOLVED_NO_FOLD_RELEASE',error=repr(error))
        seen[request['model_input_sha256']]=result;checks.append(result)
        print('APPLICATION_VARIANT_NATIVE_CHECK '+str(len(checks))+' '+result['status'],flush=True)
    write(out/'native_parser_audit.json',checks)
    done=dict(status='APPLICATION_ALL_MATERIALIZED_INPUTS_NATIVE_AUDIT_COMPLETE_FINAL_METRICS_AND_RELEASE_PENDING',
        unique_inputs=len(checks),real_MSA_queries=len(msas),disposition=dict(Counter(r['status'] for r in checks)),
        unresolved=sum(r['status']!='PASS_WITH_DECLARED_CCD_REPRESENTATION_LIMITS' for r in checks),
        all_native_coordinates_used_for_metrics_only=True,native_parser_audit_sha256=sha(out/'native_parser_audit.json'),
        new_model_calls=0,new_structure_predictions=0,fold_allowed=False,job_id=os.environ['SLURM_JOB_ID'])
    write(out/'native_parser_completion.json',done);print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();prepare(a.root) if a.prepare else audit(a.root)
