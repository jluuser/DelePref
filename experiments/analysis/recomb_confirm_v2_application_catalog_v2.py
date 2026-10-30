"""Build all predeclared application inputs; this stage does not release folding."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import os
from pathlib import Path
import numpy as np
from recomb_confirm_v2_diagnostics import read,sha,write
from recomb_confirm_v2_application_contexts_v2 import apply_variant

BASE=Path('reports/recomb_confirm_v2_application_catalog_v2')
FZ=Path('configs/recomb_confirm_v2_application_catalog_inputs_v2_freeze.json')
CONTEXT=Path('reports/recomb_confirm_v2_application_context_validation_v2')
OLDMSA=Path('reports/recomb_confirm_v2_application_context_validation')
OUTPUT=Path('reports/recomb_confirm_v2_application_outputs')


def digest(value):
    import json
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('Application input catalog already frozen')
    done=read(root/OUTPUT/'completion.json');native=read(root/CONTEXT/'native_completion.json')
    if done['logical_outputs']!=1460 or native['disposition']!={'NATIVE_SCHEMA_PASS_WITH_EXPLICIT_REPRESENTATION_LIMITS':140}:
        raise ValueError('Application six-method outputs or WT parser audit incomplete')
    names=['analysis/recomb_confirm_v2_application_catalog_v2.py','analysis/recomb_confirm_v2_application_contexts_v2.py',
        'analysis/recomb_confirm_v2_diagnostics.py','analysis/recomb_confirm_v2_uniform_msa.py',
        'analysis/recomb_confirm_v2_audit_msa.py','slurm/recomb_confirm_v2_application_catalog_v2.sbatch',
        'docs/superpowers/plans/2026-10-18-recomb-v2-application-chemical-input-contract.md',
        str(OUTPUT/'completion.json'),str(OUTPUT/'all_logical_outputs.json'),
        str(CONTEXT/'native_completion.json'),str(CONTEXT/'validated_WT_contexts.json'),
        str(OLDMSA/'WT_MSA_manifest.json'),str(OLDMSA/'source_MSAs.json'),
        'configs/recomb_confirm_v2_uniform_WT_MSA_freeze.json',
        'reports/recomb_confirm_v2_WT_calibration/profiles.json',
        'reports/delepref_v8_predictors/boltz2_preflight.json']
    for cohort in ['Binder60','Enzyme80']:
        names += ['reports/recomb_confirm_v2_'+cohort+'_constructs/'+cohort+'_panel.json',
            'reports/recomb_confirm_v2_'+cohort+'_subsets/subsets.json']
    for p,h in done['outputs'].items():
        if sha(root/OUTPUT/p)!=h:raise ValueError('Completed application outputs changed')
    if sha(root/CONTEXT/'validated_WT_contexts.json')!=native['outputs']['validated_WT_contexts.json']:
        raise ValueError('Verified WT chemical input changed')
    write(root/FZ,dict(status='APPLICATION_INPUT_CATALOG_RULES_FROZEN_NO_FOLD_RELEASE',files={p:sha(root/p) for p in names},
        base_logical_outputs=1460,main_or_budget_seed=20261017,repeated20pct_subset=dict(Binder60=20,Enzyme80=30),
        additional_structure_seeds=[20261018,20261019],expected_logical=2160,
        model='unchanged Boltz2 source-filtered-MSA r3/200 core, actual context and MSA fingerprints',
        samegene_all_copies_edited=True,partners_fixed=True,extra_budget_repeated_seeds=False,
        no_native_XYZ_template_pocket_or_contact_constraints=True,fold_allowed=False))
    (root/BASE).mkdir(parents=True,exist_ok=True);print(sha(root/FZ),flush=True)


def align_one(item):
    root_s,key,query,wt,source=item;root=Path(root_s)
    from recomb_confirm_v2_uniform_msa import aligned_rows
    from recomb_confirm_v2_audit_msa import records
    path=root/source['path']
    if sha(path)!=source['sha256']:raise ValueError('Reviewed source MSA changed')
    rows=records(path)
    if hashlib.sha256(rows[0][1].encode()).hexdigest()!=wt:raise ValueError('Natural donor WT differs')
    aligned=aligned_rows(query,rows)
    donor_hashes=[hashlib.sha256(s.replace('-','').upper().encode()).hexdigest() for _,s in rows[1:]]
    if [hashlib.sha256(s.replace('-','').upper().encode()).hexdigest() for _,s in aligned[1:]]!=donor_hashes:
        raise ValueError('Variant alignment changed actual natural donor amino acids')
    dest=root/BASE/'msa_csv'/(key+'.csv');dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('w',encoding='utf-8',newline='') as h:
        writer=csv.writer(h);writer.writerow(['key','sequence'])
        for _,seq in aligned:writer.writerow([-1,seq])
    return dict(key=key,query_sha256=hashlib.sha256(query.encode()).hexdigest(),WT_sha256=wt,
        CSV_relative_path=str(dest.relative_to(root)),CSV_sha256=sha(dest),rows=len(aligned),
        donor_fragment_hashes=donor_hashes,reused_WT_CSV=False)


def materialize_input(context,msas,root):
    proteins=[]
    for p in context['proteins']:
        m=msas[p['predictor_chain_id']]
        proteins.append(dict(protein=dict(id=p['predictor_chain_id'],sequence=p['sequence'],msa=str(root/m['CSV_relative_path']))))
    ligands=[dict(ligand=dict(id=p['id'],ccd=p['ccd'])) for p in context['ligands']]
    bonds=[dict(bond={k:[b[k]['chain'],b[k]['residue'],b[k]['atom']] for k in ['atom1','atom2']}) for b in context['covalent_bonds']]
    return dict(version=1,sequences=proteins+ligands,constraints=bonds)


def smoke(root):
    """Audit actual finished outputs; do not create mismatched placeholder MSAs."""
    from recomb_confirm_v2_application_outputs import panel_and_completed
    _,rows,_=panel_and_completed(root)
    contexts={r['protein_id']:r for r in read(root/CONTEXT/'validated_WT_contexts.json')}
    copies=lost=0
    for row in rows:
        c=contexts[row['protein_id']]
        transformed=apply_variant(c,row['edited_sequence'],row['correspondence'])
        if len(transformed['proteins'])!=len(c['proteins']):raise ValueError('A protein copy disappeared')
        for p,old in zip(transformed['proteins'],c['proteins']):
            expected=row['edited_sequence'] if old['target_gene_copy'] else old['sequence']
            if p['sequence']!=expected or p['predictor_chain_id']!=old['predictor_chain_id']:
                raise ValueError('Target/partner copy transformation differs')
            copies+=1
        lost+=len(transformed['glycan_chains_removed_with_lost_anchor'])
    write(root/BASE/'preassembly_transform_smoke.json',dict(status='ALL1100_COMPLETED_OUTPUTS_COPY_AND_CONTEXT_TRANSFORM_PASS',
        outputs=len(rows),protein_copies=copies,removed_glycan_chains=lost,
        no_placeholder_MSAs_or_model_calls=True,new_structure_predictions=0,fold_allowed=False))
    print('APPLICATION_CONTEXT_TRANSFORM_SMOKE_PASS '+str(len(rows)),flush=True)


def build(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm only')
    f=read(root/FZ)
    for p,h in f['files'].items():
        if sha(root/p)!=h:raise ValueError('Frozen application catalog input changed: '+p)
    out=root/BASE
    if (out/'execution_claim.json').exists():raise FileExistsError('Application input build already claimed')
    write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID']))
    rows=read(root/OUTPUT/'all_logical_outputs.json')
    contexts={r['protein_id']:r for r in read(root/CONTEXT/'validated_WT_contexts.json')}
    sources=read(root/OLDMSA/'source_MSAs.json');old={r['WT_sha256']:r for r in read(root/OLDMSA/'WT_MSA_manifest.json')}
    protocol=read(root/'configs/recomb_confirm_v2_uniform_WT_MSA_freeze.json')
    import Bio
    from Bio.Align import substitution_matrices
    if Bio.__version__!=protocol['Biopython_version'] or hashlib.sha256(np.asarray(substitution_matrices.load('BLOSUM62'),dtype=np.float64).tobytes()).hexdigest()!=protocol['matrix_sha256']:
        raise ValueError('Uniform donor alignment runtime changed')
    processed={};jobs={};reuse={};msa_keys={};native_refs={}
    for row in rows:
        pid=row['protein_id'];c=contexts[pid]
        if c['WT_sha256']!=row['WT_sha256']:raise ValueError('Variant/WT chemical context identity differs')
        if pid not in native_refs:
            ref_path=Path(row['native_assembly_reference'])
            if sha(ref_path)!=row['native_assembly_reference_sha256']:raise ValueError('Application native assembly changed')
            raw=read(ref_path)
            native_refs[pid]=dict(chains=[{k:p[k] for k in ['chain_id','predictor_chain_id','role','target_gene_copy']} for p in raw['chains']])
        transformed=apply_variant(c,row['edited_sequence'],row['correspondence'])
        label=(pid,row['method'],row['budget']);processed[label]=transformed
        for p,wtp in zip(transformed['proteins'],c['proteins']):
            if p['predictor_chain_id']!=wtp['predictor_chain_id']:raise ValueError('Physical protein copy identities changed')
            if not p['target_gene_copy'] and p['sequence']!=wtp['sequence']:raise ValueError('Independent partner was edited')
            wt=wtp['sequence_sha256'];source=sources[wt]
            key=digest(dict(WT=wt,query=p['sequence_sha256'],source=source['sha256'],uniform_protocol=sha(root/'configs/recomb_confirm_v2_uniform_WT_MSA_freeze.json')))
            msa_keys[label,p['predictor_chain_id']]=key
            if p['sequence_sha256']==wt:
                m=old[wt]
                if sha(root/m['CSV_relative_path'])!=m['CSV_sha256']:raise ValueError('Reusable WT MSA CSV changed')
                reuse[key]=dict(m,key=key,query_sha256=wt,WT_sha256=wt,reused_WT_CSV=True)
            else:jobs.setdefault(key,(str(root),key,p['sequence'],wt,source))
    with ProcessPoolExecutor(max_workers=8) as executor:msas=list(reuse.values())+list(executor.map(align_one,jobs.values()))
    msaby={m['key']:m for m in msas};write(out/'MSA_manifest.json',msas)
    calibration=read(root/'reports/recomb_confirm_v2_WT_calibration/profiles.json')['source_filtered_MSA_r3_s200']
    core=dict(calibration,recycling_steps=3,sampling_steps=200,templates='none',affinity=False,
        msa='uniform_query_realignment_of_unchanged_reviewed_natural_donor_fragments',
        chemical_context='same fixed pre-structure evidence and covalent anchor rules for all methods',
        chemical_contract_sha256=sha(root/'docs/superpowers/plans/2026-10-18-recomb-v2-application-chemical-input-contract.md'),
        uniform_alignment_protocol_sha256=sha(root/'configs/recomb_confirm_v2_uniform_WT_MSA_freeze.json'))
    for k in ['additional_ligands','source_review_completion_sha256','source_policy_sha256','existing_verified_constructs_only']:core.pop(k,None)
    weights=read(root/'reports/delepref_v8_predictors/boltz2_preflight.json')['files'][0]['sha256']
    physical={};profiles={};logical=[];context_diagnostics=[]
    repeated={}
    for cohort in ['Binder60','Enzyme80']:
        repeated[cohort]={r['protein_id'] for r in read(root/('reports/recomb_confirm_v2_'+cohort+'_subsets/subsets.json'))['budget10pct_and_three_structure_seeds']}
    for row in rows:
        pid=row['protein_id'];label=(pid,row['method'],row['budget']);c=processed[label]
        ms={p['predictor_chain_id']:msaby[msa_keys[label,p['predictor_chain_id']]] for p in c['proteins']}
        inp=materialize_input(c,ms,root)
        semantic=dict(proteins=[dict(id=p['predictor_chain_id'],sequence=p['sequence'],msa_csv_sha256=ms[p['predictor_chain_id']]['CSV_sha256']) for p in c['proteins']],
            ligands=[dict(id=l['id'],ccd=l['ccd']) for l in c['ligands']],constraints=inp['constraints'])
        signature=digest(semantic);input_path=out/'model_inputs'/(signature+'.yaml')
        if not input_path.exists():write(input_path,inp)
        profile=dict(core,actual_nonspatial_input_sha256=signature)
        pname='application_'+digest(profile)[:24];profiles.setdefault(pname,profile)
        seqsha=digest(dict(proteins=[dict(id=p['predictor_chain_id'],sequence=p['sequence']) for p in c['proteins']],
            ligands=semantic['ligands'],constraints=semantic['constraints']))
        chains=[dict(id=p['predictor_chain_id'],sequence=p['sequence'],target_gene_copy=p['target_gene_copy'],
            msa_relative_path=ms[p['predictor_chain_id']]['CSV_relative_path'],msa_csv_sha256=ms[p['predictor_chain_id']]['CSV_sha256']) for p in c['proteins']]
        design=next(p['predictor_chain_id'] for p in c['proteins'] if p['target_gene_copy'])
        seeds=[20261017]
        if pid in repeated[row['cohort']] and row['evaluation_role'] in ['primary20pct','WT_reference']:seeds += [20261018,20261019]
        for seed in seeds:
            identity=dict(sequence_sha256=seqsha,weights_sha256=weights,profile_sha256=digest(profile),seed=seed,sample=0)
            task='appboltz_'+digest(identity)
            request=dict(identity,task_id=task,predictor='boltz2',profile_name=pname,chains=chains,
                model_input_relative_path=str(input_path.relative_to(root)),model_input_sha256=sha(input_path),
                canonical_nonspatial_input_sha256=signature,primary_design_chain_id=design,
                sequence=row['edited_sequence'],category='application_'+row['cohort'],
                wt_variant='WT' if row['method']=='WT' else 'variant',requires_all_atom_output=True)
            if task in physical and physical[task]!=request:raise ValueError('Complex chemical/MSA identity collision')
            physical.setdefault(task,request)
            role=row['evaluation_role'] if seed==20261017 else 'structure_seed_repeat20pct'
            item=dict(row,predictor='boltz2',profile_name=pname,seed=seed,sample=0,physical_id=task,evaluation_role=role,
                native_to_prediction_chains=[dict(native_chain_id=p['predictor_chain_id'],native_instance_id=p['predictor_chain_id'],source_native_chain_id=p['chain_id'],prediction_chain_id=p['predictor_chain_id'],role=p['role'],target_gene_copy=p['target_gene_copy']) for p in native_refs[pid]['chains']])
            item['request_id']=digest({k:item[k] for k in ['protein_id','method','budget','evaluation_role','seed','physical_id']});logical.append(item)
        context_diagnostics.append(dict(protein_id=pid,method=row['method'],budget=row['budget'],input_identity=signature,
            target_gene_copies=sum(p['target_gene_copy'] for p in c['proteins']),ligand_instances=len(c['ligands']),
            covalent_bonds=len(c['covalent_bonds']),lost_anchor_glycan_chains=c['glycan_chains_removed_with_lost_anchor'],
            variant_context_flags=c['variant_context_flags']))
    if len(logical)!=2160 or len({r['request_id'] for r in logical})!=2160:raise ValueError('Application full matrix lost logical records')
    priority={p:2 for p in physical}
    for r in logical:priority[r['physical_id']]=min(priority[r['physical_id']],0 if r['evaluation_role'] in ['primary20pct','WT_reference'] else 1)
    requests=sorted(physical.values(),key=lambda r:(priority[r['task_id']],r['task_id']))
    for name,value in [('physical_requests.json',requests),('logical_requests.json',logical),('predictor_profiles.json',profiles),('context_transform_audit.json',context_diagnostics)]:write(out/name,value)
    write(out/'completion.json',dict(status='APPLICATION_FULL_INPUT_CATALOG_BUILT_NATIVE_VARIANT_PREFLIGHT_AND_FINAL_RELEASE_PENDING',
        logical=len(logical),physical=len(physical),seed_repeats=700,main_and_budget_and_WT=1460,
        MSA_queries=len(msas),reused_WT_CSV=len(reuse),new_uniform_variant_alignments=len(jobs),
        role_counts=dict(Counter(r['evaluation_role'] for r in logical)),
        new_model_calls=0,new_structure_predictions=0,fold_allowed=False,
        outputs={n:sha(out/n) for n in ['physical_requests.json','logical_requests.json','predictor_profiles.json','context_transform_audit.json','MSA_manifest.json']},
        job_id=os.environ['SLURM_JOB_ID']))
    print(read(out/'completion.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.smoke:smoke(a.root)
    elif a.prepare:prepare(a.root)
    else:build(a.root)
