"""Real WT MSA and native Boltz chemistry-interface checks; no neural model calls."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import copy
import csv
import hashlib
import os
from pathlib import Path
from recomb_confirm_v2_diagnostics import read,sha,write

BASE=Path('reports/recomb_confirm_v2_application_context_validation')
FZ=Path('configs/recomb_confirm_v2_application_context_validation_freeze.json')


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('Application WT context validation already frozen')
    source=root/'reports/recomb_confirm_v2_application_contexts_v2/WT_contexts.json'
    if sha(source)!=read(source.parent/'completion.json')['output_sha256']:raise ValueError('Completed WT chemistry templates changed')
    names=['analysis/recomb_confirm_v2_application_context_preflight.py','analysis/recomb_confirm_v2_application_contexts_v2.py',
        'analysis/recomb_confirm_v2_uniform_msa.py','analysis/recomb_confirm_v2_audit_msa.py',
        'analysis/recomb_confirm_v2_diagnostics.py','slurm/recomb_confirm_v2_application_context_MSA.sbatch',
        'slurm/recomb_confirm_v2_application_context_native.sbatch',str(source.relative_to(root)),
        'reports/recomb_confirm_v2_application_contexts_v2/completion.json',
        'reports/recomb_confirm_v2_uniform_WT_MSA/csv_manifest.json','configs/recomb_confirm_v2_uniform_WT_MSA_freeze.json',
        'external/recomb_confirm_v2_application_WT_MSA_reviewed/completion.json',
        'external/recomb_confirm_v2_Enzyme80_WT_MSA_reviewed/completion.json']
    sources={}
    for folder in ['recomb_confirm_v2_application_WT_MSA_reviewed','recomb_confirm_v2_Enzyme80_WT_MSA_reviewed']:
        for r in read(root/'external'/folder/'completion.json')['files']:
            key=r['query_sha256'];sources.setdefault(key,dict(path=str(Path('external')/folder/'a3m'/(key+'.a3m')),sha256=r['sha256']))
    needed={p['sequence_sha256'] for r in read(source) for p in r['proteins']}
    if needed-set(sources):raise ValueError('A released protein copy lacks a reviewed real WT MSA')
    write(root/BASE/'source_MSAs.json',{k:sources[k] for k in sorted(needed)})
    names.append(str(BASE/'source_MSAs.json'))
    write(root/FZ,dict(status='APPLICATION_SHARED_WT_MSA_AND_NATIVE_CHEMISTRY_VALIDATION_FROZEN',
        files={n:sha(root/n) for n in names},query_only_sources_retained=True,
        donor_policy='same fixed uniform-query alignment preserving every reviewed donor AA fragment',
        nonspatial_chemical_graph_only=True,strict_chemical_valence_not_guaranteed_by_native_CCD_interface=True,
        structure_prediction_allowed=False,new_structure_predictions=0))
    print(sha(root/FZ),flush=True)


def msa_one(item):
    root_s,key,source=item;root=Path(root_s)
    from recomb_confirm_v2_audit_msa import records
    from recomb_confirm_v2_uniform_msa import aligned_rows
    p=root/source['path']
    if sha(p)!=source['sha256']:raise ValueError('Reviewed MSA input identity changed')
    rows=records(p);seq=rows[0][1]
    if hashlib.sha256(seq.encode()).hexdigest()!=key:raise ValueError('Actual WT query does not match the full chain')
    aligned=aligned_rows(seq,rows);dest=root/BASE/'msa_csv'/(key+'.csv');dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('w',encoding='utf-8',newline='') as h:
        w=csv.writer(h);w.writerow(['key','sequence'])
        for _,s in aligned:w.writerow([-1,s])
    donors=[hashlib.sha256(s.replace('-','').upper().encode()).hexdigest() for _,s in aligned[1:]]
    return dict(WT_sha256=key,CSV_relative_path=str(dest.relative_to(root)),CSV_sha256=sha(dest),
        rows=len(aligned),donor_fragment_hashes=donors,all_donor_AA_preserved=True)


def verify(root):
    f=read(root/FZ)
    for p,h in f['files'].items():
        if sha(root/p)!=h:raise ValueError('Frozen WT context validation input/source changed: '+p)
    return f


def msa_stage(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    verify(root);out=root/BASE
    if (out/'MSA_completion.json').exists():raise FileExistsError('WT context MSAs already completed')
    source=read(out/'source_MSAs.json');old={r['query_sha256']:r for r in read(root/'reports/recomb_confirm_v2_uniform_WT_MSA/csv_manifest.json')}
    result=[];pending=[]
    for key,item in source.items():
        if key in old and old[key]['source_MSA_sha256']==item['sha256']:
            r=old[key]
            if sha(root/r['CSV_relative_path'])!=r['CSV_sha256']:raise ValueError('Reusable verified uniform WT CSV changed')
            result.append(dict(r,WT_sha256=key,reused_verified_uniform_WT_CSV=True))
        else:pending.append((str(root),key,item))
    with ProcessPoolExecutor(max_workers=8) as executor:result.extend(executor.map(msa_one,pending))
    write(out/'WT_MSA_manifest.json',result)
    done=dict(status='APPLICATION_REAL_WT_MSA_ALIGNMENT_COMPLETE_NATIVE_SCHEMA_PENDING',unique_sequences=len(result),
        reused=sum(r.get('reused_verified_uniform_WT_CSV',False) for r in result),new_uniform_alignments=len(pending),
        manifest_sha256=sha(out/'WT_MSA_manifest.json'),new_model_calls=0,new_structure_predictions=0,job_id=os.environ['SLURM_JOB_ID'])
    write(out/'MSA_completion.json',done);print(done,flush=True)


def native_stage(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    verify(root);out=root/BASE
    if (out/'native_completion.json').exists():raise FileExistsError('Native WT context schema checks already completed')
    from boltz.data import const
    from boltz.data.mol import load_canonicals,load_molecules
    from boltz.data.parse.schema import parse_boltz_schema
    from boltz.data.parse.csv import parse_csv
    from Bio.SeqUtils import seq3
    mol_dir=Path('/public/home/zhangyangroup/chengshiz/.boltz/mols');components=load_canonicals(mol_dir)
    msas={r['WT_sha256']:r for r in read(out/'WT_MSA_manifest.json')}
    if sha(out/'WT_MSA_manifest.json')!=read(out/'MSA_completion.json')['manifest_sha256']:raise ValueError('WT MSA manifest changed')
    for m in msas.values():
        p=root/m['CSV_relative_path']
        if sha(p)!=m['CSV_sha256']:raise ValueError('WT CSV changed')
        with p.open(newline='') as h:records=list(csv.DictReader(h))
        if hashlib.sha256(records[0]['sequence'].encode()).hexdigest()!=m['WT_sha256']:raise ValueError('MSA query differs')
        if [hashlib.sha256(r['sequence'].replace('-','').upper().encode()).hexdigest() for r in records[1:]]!=m['donor_fragment_hashes']:raise ValueError('Donor AA content changed')
        parsed=parse_csv(p)
        if any(int(s['res_end'])-int(s['res_start'])!=len(records[0]['sequence']) or int(s['taxonomy'])!=-1 for s in parsed.sequences):raise ValueError('Native parsed MSA width/pairing differs')
    contexts=read(root/'reports/recomb_confirm_v2_application_contexts_v2/WT_contexts.json');results=[];normalized=[]
    for context in contexts:
        c=copy.deepcopy(context);proteins={p['predictor_chain_id']:p for p in c['proteins']};ligands={p['id']:p for p in c['ligands']}
        needed={code for lig in ligands.values() for code in lig['ccd']}
        for code in needed-set(components):components.update(load_molecules(mol_dir,[code]))
        valid=[];unsupported=[];leaving=[]
        for bond in c['covalent_bonds']:
            supported=True
            for a in [bond['atom1'],bond['atom2']]:
                if a['kind']=='protein':
                    seq=proteins[a['chain']]['sequence'];aa=seq[a['residue']-1];names=set(const.ref_atoms[seq3(aa).upper()])
                else:
                    code=ligands[a['chain']]['ccd'][a['residue']-1];mol=components[code];names={atom.GetProp('name') for atom in mol.GetAtoms()}
                    center=next((atom for atom in mol.GetAtoms() if atom.GetProp('name')==a['atom']),None)
                    if center is not None:
                        neighbors=[atom.GetProp('name') for atom in center.GetNeighbors() if atom.HasProp('leaving_atom') and str(atom.GetProp('leaving_atom')).lower() in ('1','true')]
                        if neighbors:leaving.append(dict(chain=a['chain'],CCD=code,atom=a['atom'],native_schema_leaving_atoms=neighbors))
                if a['atom'] not in names:supported=False
            (valid if supported else unsupported).append(bond)
        c['covalent_bonds']=valid;c['native_interface_unsupported_bonds']=unsupported;c['native_CCD_leaving_atom_limitations']=leaving
        seqs=[dict(protein=dict(id=p['predictor_chain_id'],sequence=p['sequence'],msa=str(root/msas[p['sequence_sha256']]['CSV_relative_path']))) for p in c['proteins']]
        seqs += [dict(ligand=dict(id=p['id'],ccd=p['ccd'])) for p in c['ligands']]
        bonds=[dict(bond={k:[b[k]['chain'],b[k]['residue'],b[k]['atom']] for k in ('atom1','atom2')}) for b in valid]
        model_input=dict(version=1,sequences=seqs,constraints=bonds)
        if any(k in model_input for k in ('templates','properties')):raise ValueError('Unexpected template/affinity input')
        try:
            target=parse_boltz_schema('WTctx_'+hashlib.sha256(c['protein_id'].encode()).hexdigest()[:16],model_input,components,mol_dir,boltz_2=True)
            chain_names={str(x['name']) for x in target.structure.chains}
            if chain_names!=set(proteins)|set(ligands):raise ValueError('Native schema changed chain identities/copy counts')
            if len(target.structure.connections)!=len(valid):raise ValueError('Native covalent-connection count differs')
            status='NATIVE_SCHEMA_PASS_WITH_EXPLICIT_REPRESENTATION_LIMITS'
            row=dict(protein_id=c['protein_id'],status=status,protein_chains=len(proteins),ligand_chains=len(ligands),
                native_parsed_atoms=len(target.structure.atoms),native_parsed_connections=len(target.structure.connections),
                unsupported_bonds=len(unsupported),leaving_atom_interface_limits=len(leaving))
            path=out/'WT_inputs'/('WTctx_'+hashlib.sha256(c['protein_id'].encode()).hexdigest()[:16]+'.yaml');write(path,model_input)
            c.update(WT_model_input_path=str(path),WT_model_input_sha256=sha(path),native_schema_status=status)
        except Exception as error:
            row=dict(protein_id=c['protein_id'],status='NATIVE_SCHEMA_INPUT_UNSUPPORTED_NO_MODEL_CALL',error=repr(error));c['native_schema_status']=row['status']
        results.append(row);normalized.append(c);print('WT_CONTEXT_NATIVE_CHECK '+str(len(results))+'/'+str(len(contexts))+' '+row['status'],flush=True)
    write(out/'native_schema_audit.json',results);write(out/'validated_WT_contexts.json',normalized)
    done=dict(status='APPLICATION_WT_CONTEXT_NATIVE_SCHEMA_AUDIT_COMPLETE_NOT_FOLD_RELEASE',cases=len(results),
        disposition=dict(Counter(r['status'] for r in results)),
        unsupported_bonds=sum(r.get('unsupported_bonds',0) for r in results),
        leaving_atom_interface_limits=sum(r.get('leaving_atom_interface_limits',0) for r in results),
        parsed_real_MSA_queries=len(msas),new_model_calls=0,new_structure_predictions=0,fold_allowed=False,
        outputs={p:sha(out/p) for p in ['native_schema_audit.json','validated_WT_contexts.json']},job_id=os.environ['SLURM_JOB_ID'])
    write(out/'native_completion.json',done);print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--stage',choices=['msa','native'])
    a=p.parse_args()
    if a.prepare:prepare(a.root)
    elif a.stage=='msa':msa_stage(a.root)
    elif a.stage=='native':native_stage(a.root)
    else:p.error('stage or prepare required')
