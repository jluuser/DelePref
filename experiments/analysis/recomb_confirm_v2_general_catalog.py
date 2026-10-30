"""Freeze General100 prediction inputs, exact MSA-aware identities and statistics."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from recomb_confirm_v2_diagnostics import read,sha,write

BASE=Path('reports/recomb_confirm_v2_General100_catalog')
INPUT=Path('configs/recomb_confirm_v2_General100_catalog_inputs_freeze.json')
CONFIRM=Path('configs/recomb_confirm_v2_General100_confirmation_freeze.json')


def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def prepare(root):
    if (root/INPUT).exists():raise FileExistsError('General catalog inputs already frozen')
    names=['analysis/recomb_confirm_v2_general_catalog.py','analysis/recomb_confirm_v2_general_predict.py',
        'analysis/recomb_confirm_v2_general_evaluate.py','analysis/recomb_confirm_v2_diagnostics.py',
        'analysis/recomb_benign_fold_worker_20261017.py','analysis/recomb_benign_metrics_20261017.py',
        'analysis/recomb_confirm_v2_structure_alignment.py','analysis/delepref_v8_structure_metrics.py',
        'analysis/recomb_confirm_v2_uniform_msa.py','analysis/recomb_confirm_v2_audit_msa.py',
        'code/prediction_journal.py','code/budget_ledger.py',
        'analysis/tests/test_recomb_confirm_v2_alignment.py','analysis/tests/test_recomb_benign_metrics.py',
        'analysis/tests/test_recomb_confirm_v2_general_statistics.py',
        'slurm/recomb_confirm_v2_general_catalog.sbatch','slurm/recomb_confirm_v2_general_catalog_preflight.sbatch',
        'slurm/recomb_confirm_v2_General100_ESMFold.sbatch','slurm/recomb_confirm_v2_General100_Boltz.sbatch',
        'slurm/recomb_confirm_v2_General100_evaluate.sbatch',
        'reports/recomb_confirm_v2_General100_outputs/completion.json',
        'reports/recomb_confirm_v2_General100_outputs/all_logical_outputs.json',
        'reports/recomb_confirm_v2_General100_subsets/subsets.json',
        'reports/recomb_confirm_v2_general_native_v2/General100_panel.json',
        'reports/delepref_v8_structure_preparation/predictor_profiles.json',
        'reports/delepref_v8_predictors/esmfold_preflight.json','reports/delepref_v8_predictors/boltz2_preflight.json',
        'reports/recomb_confirm_v2_uniform_WT_MSA/csv_manifest.json',
        'reports/recomb_confirm_v2_uniform_WT_MSA/parser_preflight.json',
        'configs/recomb_confirm_v2_uniform_WT_MSA_freeze.json',
        'configs/recomb_confirm_v2_uniform_WT_MSA_parser_freeze.json',
        'external/recomb_confirm_v2_application_WT_MSA_reviewed/completion.json',
        'external/recomb_confirm_v2_application_WT_MSA_reviewed/review_policy.json']
    completed=read(root/'reports/recomb_confirm_v2_General100_outputs/completion.json')
    if sha(root/'reports/recomb_confirm_v2_General100_outputs/all_logical_outputs.json')!=completed['outputs']['all_logical_outputs.json'] or completed['logical_outputs']!=1864:
        raise ValueError('Full General output matrix is not complete and immutable')
    native={r['wt_native_json']:r['wt_native_json_sha256'] for r in read(root/'reports/recomb_confirm_v2_general_native_v2/General100_panel.json')}
    write(root/INPUT,dict(status='GENERAL100_FULL_PREDICTION_CATALOG_INPUTS_FROZEN',
        files={n:sha(root/n) for n in names},native_reference_sha256=native,
        binary_sha256=sha(root/'provenance/usalign_20261017/USalign'),
        primary_common_shape_endpoints=['structure_mapped_CA_lddt','structure_mapped_native_contact_recovery'],
        fixed_original_identity_metrics_reported_in_parallel=True,new_structure_predictions=0))
    (root/BASE).mkdir(parents=True,exist_ok=True);print(sha(root/INPUT),flush=True)


def verify(root):
    f=read(root/INPUT)
    for p,h in f['files'].items():
        if sha(root/p)!=h:raise ValueError('Frozen catalog source/input changed: '+p)
    for p,h in f['native_reference_sha256'].items():
        if sha(p)!=h:raise ValueError('Native reference changed')
    return f


def align_variant(item):
    root_s,key,query,wt_sha,source,source_sha=item;root=Path(root_s)
    from recomb_confirm_v2_uniform_msa import aligned_rows
    from recomb_confirm_v2_audit_msa import records
    path=root/source
    if sha(path)!=source_sha:raise ValueError('Reviewed donor MSA changed')
    rows=records(path)
    if hashlib.sha256(rows[0][1].encode()).hexdigest()!=wt_sha:raise ValueError('Reviewed donor query WT differs')
    aligned=aligned_rows(query,rows);dest=root/BASE/'msa_csv'/(key+'.csv');dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['key','sequence'])
        for _,seq in aligned:writer.writerow([-1,seq])
    donor_hashes=[hashlib.sha256(s.replace('-','').upper().encode()).hexdigest() for _,s in aligned[1:]]
    if donor_hashes!=[hashlib.sha256(s.replace('-','').upper().encode()).hexdigest() for _,s in rows[1:]]:
        raise ValueError('Variant alignment changed natural donor AA')
    return dict(key=key,query_sha256=hashlib.sha256(query.encode()).hexdigest(),WT_sha256=wt_sha,
        CSV_relative_path=str(dest.relative_to(root)),CSV_sha256=sha(dest),rows=len(aligned),donor_fragment_hashes=donor_hashes)


def build(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm only')
    frozen=verify(root);out=root/BASE
    if (out/'execution_claim.json').exists():raise FileExistsError('Catalog build claimed; inspect existing job')
    write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID']))
    panel={r['protein_id']:r for r in read(root/'reports/recomb_confirm_v2_general_native_v2/General100_panel.json')}
    rows=read(root/'reports/recomb_confirm_v2_General100_outputs/all_logical_outputs.json')
    ids={r['protein_id'] for r in read(root/'reports/recomb_confirm_v2_General100_subsets/subsets.json')['second_predictor30']}
    second=[r for r in rows if r['protein_id'] in ids and (r['evaluation_role']=='primary20pct' or r['method']=='WT')]
    if len(second)!=210:raise ValueError('Second predictor six-method/WT cohort changed')
    donor_contract=read(root/'configs/recomb_confirm_v2_uniform_WT_MSA_freeze.json')
    import Bio
    from Bio.Align import substitution_matrices
    if Bio.__version__!=donor_contract['Biopython_version'] or hashlib.sha256(np.asarray(substitution_matrices.load('BLOSUM62'),dtype=np.float64).tobytes()).hexdigest()!=donor_contract['matrix_sha256']:
        raise ValueError('Uniform alignment runtime/matrix changed')
    jobs={};msa_key={}
    for r in second:
        wt=panel[r['protein_id']]['WT_sha256'];source='external/recomb_confirm_v2_application_WT_MSA_reviewed/a3m/'+wt+'.a3m'
        expected=donor_contract['reviewed_sources'][source]
        key=digest(dict(query=r['edited_sequence_sha256'],WT=wt,source=expected,protocol=sha(root/'configs/recomb_confirm_v2_uniform_WT_MSA_freeze.json')))
        jobs.setdefault(key,(str(root),key,r['edited_sequence'],wt,source,expected));msa_key[r['protein_id'],r['edited_sequence_sha256']]=key
    with ProcessPoolExecutor(max_workers=8) as executor:msas=list(executor.map(align_variant,jobs.values()))
    write(out/'variant_MSA_manifest.json',msas);bykey={r['key']:r for r in msas}
    old=read(root/'reports/delepref_v8_structure_preparation/predictor_profiles.json')
    profiles={'esmfold':old['esmfold']};weights={}
    for name,suffix in [('esmfold','pytorch_model.bin'),('boltz2','boltz2_conf.ckpt')]:
        report=read(root/('reports/delepref_v8_predictors/'+name+'_preflight.json'))
        weights[name]=next(f['sha256'] for f in report['files'] if f['path'].endswith(suffix))
    boltz=dict(old['boltz2'],recycling_steps=3,sampling_steps=200,
        msa='uniform_query_realignment_of_unchanged_reviewed_natural_donor_fragments',
        uniform_alignment_protocol_sha256=sha(root/'configs/recomb_confirm_v2_uniform_WT_MSA_freeze.json'),
        source_review_completion_sha256=sha(root/'external/recomb_confirm_v2_application_WT_MSA_reviewed/completion.json'),
        MSA_pairing='unpaired_key_minus1',seed_policy='request_seed_20261017_18_19',no_variant_specific_retrieval=True)
    boltz.pop('seed',None)
    physical={};logical=[]
    def add(row,predictor,seed,role):
        chain=dict(id='A',sequence=row['edited_sequence']);pname='esmfold'
        if predictor=='boltz2':
            msa=bykey[msa_key[row['protein_id'],row['edited_sequence_sha256']]]
            chain.update(msa_relative_path=msa['CSV_relative_path'],msa_csv_sha256=msa['CSV_sha256'])
            profile=dict(boltz,input_MSA_CSV_sha256=msa['CSV_sha256'])
            pname='boltz_general_'+digest(profile)[:20];profiles.setdefault(pname,profile)
        profile=profiles[pname]
        signature=dict(sequence_sha256=row['edited_sequence_sha256'],weights_sha256=weights[predictor],
            profile_sha256=digest(profile),seed=seed,sample=0)
        task=predictor+'_'+digest(signature)
        if task not in physical:physical[task]=dict(signature,task_id=task,predictor=predictor,profile_name=pname,
            sequence=row['edited_sequence'],chains=[chain],canonical_chains_sha256=digest([chain]),
            category='general_esmfold' if predictor=='esmfold' else 'general_second_predictor',
            wt_variant='WT' if row['method']=='WT' else 'variant',requires_all_atom_output=False)
        elif [(c['id'],c['sequence'],c.get('msa_csv_sha256')) for c in physical[task]['chains']]!=[(chain['id'],chain['sequence'],chain.get('msa_csv_sha256'))]:raise ValueError('MSA-aware input identity collision')
        item=dict(row,cohort='general',predictor=predictor,profile_name=pname,profile_family='esmfold' if predictor=='esmfold' else 'boltz_general',
            seed=seed,sample=0,physical_id=task,evaluation_role=role,
            native_to_prediction_chains=[dict(native_chain_id=panel[row['protein_id']]['reference_chain'],prediction_chain_id='A',role='design')])
        item['request_id']=digest({k:item[k] for k in ['protein_id','method','budget','evaluation_role','predictor','seed','physical_id']});logical.append(item)
    for r in rows:add(r,'esmfold',int(profiles['esmfold']['seed']),r['evaluation_role'])
    for r in second:
        for seed in [20261017,20261018,20261019]:add(r,'boltz2',seed,'second_predictor')
    if len(logical)!=2494 or len({r['request_id'] for r in logical})!=2494:raise ValueError('Full General prediction roster lost')
    # Only query files named by the positively reviewed output sequence identities.
    reuse={};journal_indices={}
    folders=['reports/recomb_benign_structures_20261017','reports/delepref_v8_structures','reports/recomb_confirm_v2_span_structures']
    for p in physical.values():
        if p['predictor']!='esmfold':continue
        for folder in folders:
            journal=next((root/folder/n for n in ['prediction_journal.jsonl','budget_ledger.jsonl'] if (root/folder/n).exists()),None)
            if journal is None:continue
            if str(journal) not in journal_indices:
                raw=journal.read_bytes()
                if raw and not raw.endswith(b'\n'):raise ValueError('Historical journal is incomplete')
                journal_indices[str(journal)]={}
                for line in raw.splitlines():
                    j=json.loads(line);journal_indices[str(journal)][j['task_id']]=j
            for name in [p['task_id'],'esmfold_'+p['sequence_sha256']]:
                rp=root/folder/'records'/(name+'.json')
                if not rp.exists():continue
                r=read(rp);entry=journal_indices[str(journal)].get(r.get('task_id',name),{})
                keys=['sequence_sha256','weights_sha256','profile_sha256','seed','sample']
                if r.get('status')!='SUCCEEDED' or any(r.get(k)!=p[k] for k in keys) or r.get('sequence')!=p['sequence']:continue
                artifact=Path(r['output_path'])
                if entry.get('state') not in ('succeeded','cache_hit') or entry.get('output_sha')!=r['output_sha256']:continue
                if any(entry.get(a)!=p[b] for a,b in [('sequence_sha','sequence_sha256'),('predictor_weights_sha','weights_sha256'),('profile_sha','profile_sha256'),('seed','seed'),('sample','sample')]):continue
                if sha(artifact)!=r['output_sha256']:raise ValueError('Successful matching cache artifact changed')
                with np.load(artifact,allow_pickle=False) as z:
                    if str(z['sequence'])!=p['sequence'] or z['coordinates'].shape!=(len(p['sequence']),3):raise ValueError('Cached sequence/shape differs')
                reuse[p['task_id']]=dict(record_path=str(rp),record_sha256=sha(rp),artifact_path=str(artifact),artifact_sha256=r['output_sha256'],journal_path=str(journal),journal_sha256=sha(journal));break
            if p['task_id'] in reuse:break
    # Main20pct and WT physical calls first; no order is based on predicted quality.
    priority={r['physical_id']:9 for r in logical}
    for r in logical:priority[r['physical_id']]=min(priority[r['physical_id']],0 if r['evaluation_role'] in ['primary20pct','WT_reference'] else 1)
    requests=sorted(physical.values(),key=lambda p:(priority[p['task_id']],p['task_id']))
    protocol=dict(status='PROSPECTIVE_GENERAL100_MULTIMETRIC_AND_COMPONENT_STATISTICS',
        WT=100,methods=['delepref_unlimited','rag_unlimited','gap_frequency','terminal','scisor_L_native','raygun_native'],
        co_primary_endpoints=['structure_mapped_CA_lddt','structure_mapped_native_contact_recovery'],
        primary_comparison='rag_unlimited minus delepref_unlimited at20pct',
        ordinary_CI=.95,Bonferroni_joint_CI_per_endpoint=.975,
        bootstrap_replicates=10000,bootstrap_seed=20261016,
        component_by_protein_id={pid:r['evaluation_sequence_component'] for pid,r in panel.items()},
        no_prediction_or_method_reselection=True,all_failures_and_unknowns_preserved=True,
        second_predictor_seeds=[20261017,20261018,20261019],seed_aggregation='within WT and method before component inference, all predefined seeds',
        common_shape_not_original_residue_identity=True,structural_proxies_not_functional_validation=True,
        compute=dict(structure_call_limit=None,structure_categories=dict(general_esmfold=None,general_second_predictor=None,retry=None),structure_retries_max=1))
    for name,value in [('logical_requests.json',logical),('physical_requests.json',requests),('compatible_cache.json',reuse),('predictor_profiles.json',profiles),('statistical_protocol.json',protocol)]:write(out/name,value)
    done=dict(status='GENERAL100_FULL_PREDICTION_CATALOG_COMPLETE_NATIVE_PARSER_AND_RELEASE_PENDING',
        logical=2494,ESMFold_logical=1864,Boltz_logical=630,physical=len(physical),
        physical_by_predictor=dict(Counter(p['predictor'] for p in requests)),verified_cache=len(reuse),
        missing_by_predictor=dict(Counter(p['predictor'] for p in requests if p['task_id'] not in reuse)),
        variant_MSA_profiles=len(msas),new_structure_predictions=0,
        outputs={p:sha(out/p) for p in ['logical_requests.json','physical_requests.json','compatible_cache.json','predictor_profiles.json','statistical_protocol.json','variant_MSA_manifest.json']})
    write(out/'completion.json',done);print(done,flush=True)


def preflight(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm only')
    frozen=verify(root);out=root/BASE;done=read(out/'completion.json')
    for p,h in done['outputs'].items():
        if sha(out/p)!=h:raise ValueError('Frozen catalog result changed')
    from boltz.data.parse.csv import parse_csv
    profiles=read(out/'predictor_profiles.json')
    for request in read(out/'physical_requests.json'):
        profile=profiles[request['profile_name']]
        if digest(profile)!=request['profile_sha256']:raise ValueError('Physical request profile mismatch')
        if request['predictor']=='boltz2' and profile['input_MSA_CSV_sha256']!=request['chains'][0]['msa_csv_sha256']:
            raise ValueError('Journal fingerprint does not bind the exact MSA')
    checks=[]
    for r in read(out/'variant_MSA_manifest.json'):
        p=root/r['CSV_relative_path']
        if sha(p)!=r['CSV_sha256']:raise ValueError('Variant MSA CSV changed')
        with p.open(newline='') as h:rows=list(csv.DictReader(h))
        if hashlib.sha256(rows[0]['sequence'].encode()).hexdigest()!=r['query_sha256']:raise ValueError('Variant MSA query differs')
        if [hashlib.sha256(row['sequence'].replace('-','').upper().encode()).hexdigest() for row in rows[1:]]!=r['donor_fragment_hashes']:
            raise ValueError('Variant natural donor conservation failed')
        parsed=parse_csv(p);width=len(rows[0]['sequence'])
        if any(int(s['res_end'])-int(s['res_start'])!=width or int(s['taxonomy'])!=-1 for s in parsed.sequences):raise ValueError('Native Boltz MSA parsing changed query width/pairing')
        checks.append(dict(query_sha256=r['query_sha256'],parsed_rows=len(parsed.sequences),input_rows=len(rows)))
    write(out/'parser_preflight.json',dict(status='ALL_GENERAL_SECOND_PREDICTOR_MSAS_NATIVE_PARSER_AND_DONOR_PASS',queries=len(checks),checks=checks,new_model_calls=0))
    files={str((out/p).relative_to(root)):h for p,h in done['outputs'].items()}
    files.update(frozen['files']);files[str((out/'parser_preflight.json').relative_to(root))]=sha(out/'parser_preflight.json')
    for r in read(out/'variant_MSA_manifest.json'):files[r['CSV_relative_path']]=r['CSV_sha256']
    native_files=frozen['native_reference_sha256']
    files.update(native_files);files['provenance/usalign_20261017/USalign']=frozen['binary_sha256']
    write(root/CONFIRM,dict(status='GENERAL100_COMPLETE_INPUT_OUTPUT_PREDICTOR_METRIC_STATISTICS_RELEASE',
        fold_allowed=True,all_selected_outputs_frozen=True,structure_prediction_limit=None,
        files=[dict(path=str(Path(p) if Path(p).is_absolute() else root/p),sha256=h) for p,h in files.items()],
        total_logical_requests=2494,physical_requests=done['physical'],no_new_training=True,
        exact_cache_and_MSA_fingerprints_required=True,General100_CI_joint_correction_frozen=True))
    write(out/'release_completion.json',dict(status='GENERAL100_ALL_FREEZES_AND_NATIVE_PREFLIGHT_PASS',confirmation_freeze_sha256=sha(root/CONFIRM),new_structure_predictions=0))
    print(read(out/'release_completion.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--stage',choices=['build','preflight'])
    a=p.parse_args()
    if a.prepare:prepare(a.root)
    elif a.stage=='build':build(a.root)
    elif a.stage=='preflight':preflight(a.root)
    else:p.error('stage or prepare required')
