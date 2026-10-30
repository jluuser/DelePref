"""Stage WT-only MSAs for qualified Binder60 and fixed General second-predictor inputs."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import importlib.util
from pathlib import Path
import sys
import time
from recomb_confirm_v2_diagnostics import read,sha,write

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'external/recomb_confirm_v2_application_WT_MSA_staging'


def prepare():
    if (BASE/'request_manifest.json').exists():return read(BASE/'request_manifest.json')
    panelpath=ROOT/'reports/recomb_confirm_v2_Binder60_constructs/Binder60_panel.json'
    panel=read(panelpath);done=read(panelpath.parent/'completion.json')
    if len(panel)!=60 or sha(panelpath)!=done['outputs']['Binder60_panel.json']:raise ValueError('Qualified Binder60 panel changed')
    if not all(r['WT_encoding_allowed'] and r['protein_assembly_complete'] for r in panel):raise ValueError('Binder protein input scope not released')
    groups={s:sorted([r for r in panel if r['scaffold']==s],key=lambda r:hashlib.sha256(
        ('recomb-confirm-v2-binder-subsets|'+r['WT_sha256']+'|'+r['native_case_key']).encode()).hexdigest())
        for s in ['explicitly_annotated_VHH_or_nanobody','curated_antibody_heavy_chain_with_fixed_light']}
    subset=[dict(protein_id=r['protein_id'],WT_sha256=r['WT_sha256'],scaffold=r['scaffold'],
        combined_component=r['combined_component']) for s in groups for r in groups[s][:10]]
    if len(subset)!=20:raise ValueError('Insufficient input strata for fixed Binder20 subset')
    sub=ROOT/'reports/recomb_confirm_v2_Binder60_subsets'
    write(sub/'subsets.json',dict(budget10pct_and_three_structure_seeds=subset,
        selection='source-balanced salted input SHA prefixes, no model or retrieval outcomes',
        generation_seed=13,structure_seeds=[20261017,20261018,20261019],three_seeds_do_not_multiply_N=True))
    queries={}
    def add(seq,use):
        h=hashlib.sha256(seq.encode()).hexdigest()
        if set(seq)-set('ACDEFGHIKLMNPQRSTVWY'):raise ValueError('Nonstandard released sequence')
        queries.setdefault(h,dict(sequence=seq,sequence_sha256=h,uses=[]))['uses'].append(use)
    for r in panel:
        for c in r['predictor_chains']:
            add(c['sequence'],dict(cohort='Binder60',protein_id=r['protein_id'],chain_id=c['predictor_chain_id'],
                role=c['role'],target_gene_copy=c['target_gene_copy']))
    gen=read(ROOT/'reports/recomb_confirm_v2_general_native_v2/General100_panel.json')
    ids={r['protein_id'] for r in read(ROOT/'reports/recomb_confirm_v2_General100_subsets/subsets.json')['second_predictor30']}
    for r in gen:
        if r['protein_id'] in ids:add(r['sequence'],dict(cohort='General30_second_predictor',protein_id=r['protein_id'],chain_id='A',role='design'))
    ordered=[queries[h] for h in sorted(queries)]
    vendor=ROOT/'caches/boltz_mmseqs2_official.py'
    manifest=dict(status='REVIEWED_WT_SEQUENCES_ONLY_MSA_STAGING_NOT_PREDICTION_RELEASE',
        queries=ordered,unique_queries=len(ordered),Binder_cases=60,General_second_predictor_cases=30,
        batch_size=32,helper_sha256=sha(vendor),provider='https://api.colabfold.com',
        use_env=True,use_filter=True,use_pairing=False,use_templates=False,
        Binder_panel_sha256=sha(panelpath),Binder_subsets_sha256=sha(sub/'subsets.json'),
        source_sha256=sha(Path(__file__)),new_structure_predictions=0,
        native_coordinates_not_uploaded=True,all_returned_donors_require_source_identity_review=True)
    write(BASE/'request_manifest.json',manifest);return manifest


def main():
    if (BASE/'completion.json').exists():raise FileExistsError('MSA stage already completed; reuse outputs')
    manifest=prepare()
    if manifest['source_sha256']!=sha(Path(__file__)):raise ValueError('Frozen MSA staging source changed')
    sys.path.insert(0,str(ROOT/'caches/msa_runtime'))
    vendor=ROOT/'caches/boltz_mmseqs2_official.py'
    if sha(vendor)!=manifest['helper_sha256']:raise ValueError('Official existing MSA helper changed')
    spec=importlib.util.spec_from_file_location('recomb_existing_official_mmseqs',vendor)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    old=ROOT/'external/recomb_confirm_v2_WT_MSA_staging';prior=read(old/'completion.json')
    reuse={r['sequence_sha256']:r for r in prior['files']}
    files=[];pending=[];start=time.monotonic()
    for q in manifest['queries']:
        h=q['sequence_sha256'];p=BASE/'a3m'/(h+'.a3m')
        if h in reuse:
            original=old/'a3m'/(h+'.a3m')
            if sha(original)!=reuse[h]['sha256']:raise ValueError('Previously completed MSA identity differs')
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(original.read_bytes())
            files.append(dict(sequence_sha256=h,path=str(p),sha256=sha(p),reused_prior_stage=True))
        else:pending.append(q)
    for index in range(0,len(pending),32):
        batch=pending[index:index+32];number=index//32;out=BASE/('batch_%02d'%number)
        if (out/'completion.json').exists():
            d=read(out/'completion.json')
            if d['query_hashes']!=[q['sequence_sha256'] for q in batch]:raise ValueError('Batch identity changed')
            for record in d['files']:
                if sha(record['path'])!=record['sha256']:raise ValueError('Saved batch MSA changed')
            files.extend(d['files']);continue
        if (out/'execution_claim.json').exists():raise FileExistsError('Existing MSA batch may be active; inspect job/cache rather than resubmitting')
        write(out/'execution_claim.json',dict(query_hashes=[q['sequence_sha256'] for q in batch],started_at=datetime.now(timezone.utc).isoformat()))
        print('APPLICATION_WT_MSA_BATCH_START '+str(number)+' '+str(len(batch)),flush=True)
        lines=module.run_mmseqs2([q['sequence'] for q in batch],prefix=str(out/'official_unpaired'),
            use_env=True,use_filter=True,use_pairing=False,host_url=manifest['provider'])
        if not isinstance(lines,list) or len(lines)!=len(batch):raise ValueError('Official MSA batch output count differs')
        saved=[]
        for q,text in zip(batch,lines):
            p=BASE/'a3m'/(q['sequence_sha256']+'.a3m');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8')
            saved.append(dict(sequence_sha256=q['sequence_sha256'],path=str(p),sha256=sha(p),
                records=sum(s.startswith('>') for s in text.splitlines()),reused_prior_stage=False))
        write(out/'completion.json',dict(query_hashes=[q['sequence_sha256'] for q in batch],files=saved));files.extend(saved)
        write(BASE/'progress.json',dict(completed_queries=len(files),total_queries=manifest['unique_queries'],completed_batch=number))
    if len(files)!=manifest['unique_queries']:raise ValueError('MSA roster incomplete')
    report=dict(status='APPLICATION_AND_GENERAL_SECOND_WT_MSAS_STAGED_SOURCE_REVIEW_PENDING',
        unique_queries=len(files),Binder_cases=60,General_second_predictor_cases=30,
        prior_MSA_reused=sum(r['reused_prior_stage'] for r in files),files=files,seconds=time.monotonic()-start,
        new_model_calls=0,new_structure_predictions=0,prediction_release_allowed=False,
        manifest_sha256=sha(BASE/'request_manifest.json'))
    write(BASE/'completion.json',report);print({k:v for k,v in report.items() if k!='files'},flush=True)


if __name__=='__main__':main()
