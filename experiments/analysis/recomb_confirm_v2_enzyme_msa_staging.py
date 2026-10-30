"""WT-only standard MSA staging for the frozen ordinary Enzyme80 assemblies."""
from datetime import datetime,timezone
import hashlib
import importlib.util
from pathlib import Path
import sys
import time
from recomb_confirm_v2_diagnostics import read,sha,write

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'external/recomb_confirm_v2_Enzyme80_WT_MSA_staging'


def prepare():
    if (BASE/'request_manifest.json').exists():return read(BASE/'request_manifest.json')
    pp=ROOT/'reports/recomb_confirm_v2_Enzyme80_constructs/Enzyme80_panel.json'
    panel=read(pp);done=read(pp.parent/'completion.json')
    if len(panel)!=80 or sha(pp)!=done['outputs'][pp.name]:raise ValueError('Enzyme80 frozen roster changed')
    queries={}
    for row in panel:
        if not row['WT_encoding_allowed'] or not row['protein_assembly_complete']:raise ValueError('Ordinary WT input not released')
        for chain in row['predictor_chains']:
            seq=chain['sequence'];h=hashlib.sha256(seq.encode()).hexdigest()
            if h!=chain['sequence_sha256'] or set(seq)-set('ACDEFGHIKLMNPQRSTVWY'):raise ValueError('Protein input sequence identity differs')
            q=queries.setdefault(h,dict(sequence=seq,sequence_sha256=h,uses=[]))
            q['uses'].append(dict(cohort='Enzyme80',protein_id=row['protein_id'],chain_id=chain['predictor_chain_id'],
                role=chain['role'],target_gene_copy=chain['target_gene_copy']))
    manifest=dict(status='QUALIFIED_ORDINARY_ENZYME_WT_AND_PARTNERS_ONLY_MSA_STAGING',
        queries=[queries[h] for h in sorted(queries)],unique_queries=len(queries),enzyme_cases=80,
        provider='https://api.colabfold.com',batch_size=32,use_env=True,use_filter=True,use_pairing=False,use_templates=False,
        panel_sha256=sha(pp),subsets_sha256=done['subset_sha256'],source_sha256=sha(Path(__file__)),
        helper_sha256=sha(ROOT/'caches/boltz_mmseqs2_official.py'),
        no_variant_sequences_or_native_coordinates_uploaded=True,all_donors_require_source_identity_review=True)
    write(BASE/'request_manifest.json',manifest);return manifest


def main():
    if (BASE/'completion.json').exists():raise FileExistsError('Enzyme WT MSAs already complete; reuse')
    manifest=prepare()
    if manifest['source_sha256']!=sha(Path(__file__)):raise ValueError('MSA staging code changed')
    sys.path.insert(0,str(ROOT/'caches/msa_runtime'))
    vendor=ROOT/'caches/boltz_mmseqs2_official.py'
    if sha(vendor)!=manifest['helper_sha256']:raise ValueError('Official MSA helper changed')
    spec=importlib.util.spec_from_file_location('enzyme_official_mmseqs',vendor)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    prior={}
    for prefix in ['recomb_confirm_v2_WT_MSA_staging','recomb_confirm_v2_application_WT_MSA_staging']:
        old=ROOT/'external'/prefix;done=old/'completion.json'
        if done.exists():
            for row in read(done)['files']:prior[row['sequence_sha256']]=(old/'a3m'/(row['sequence_sha256']+'.a3m'),row['sha256'])
    pending=[];files=[];start=time.monotonic()
    for q in manifest['queries']:
        h=q['sequence_sha256'];p=BASE/'a3m'/(h+'.a3m')
        if h not in prior:pending.append(q);continue
        original,expected=prior[h]
        if sha(original)!=expected:raise ValueError('Reusable completed WT MSA changed')
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(original.read_bytes())
        files.append(dict(sequence_sha256=h,path=str(p),sha256=sha(p),reused_prior_stage=True))
    for index in range(0,len(pending),32):
        batch=pending[index:index+32];number=index//32;out=BASE/('batch_%02d'%number)
        if (out/'completion.json').exists():
            d=read(out/'completion.json')
            if d['query_hashes']!=[q['sequence_sha256'] for q in batch]:raise ValueError('Existing batch query identities differ')
            for r in d['files']:
                if sha(r['path'])!=r['sha256']:raise ValueError('Existing MSA artifact changed')
            files.extend(d['files']);continue
        if (out/'execution_claim.json').exists():raise FileExistsError('MSA batch claimed; inspect provider cache/status before any retry')
        write(out/'execution_claim.json',dict(query_hashes=[q['sequence_sha256'] for q in batch],started_at=datetime.now(timezone.utc).isoformat()))
        print('ENZYME_WT_MSA_BATCH_START '+str(number)+' '+str(len(batch)),flush=True)
        lines=module.run_mmseqs2([q['sequence'] for q in batch],prefix=str(out/'official_unpaired'),
            use_env=True,use_filter=True,use_pairing=False,host_url=manifest['provider'])
        if not isinstance(lines,list) or len(lines)!=len(batch):raise ValueError('Provider MSA query count differs')
        saved=[]
        for q,text in zip(batch,lines):
            p=BASE/'a3m'/(q['sequence_sha256']+'.a3m');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8')
            saved.append(dict(sequence_sha256=q['sequence_sha256'],path=str(p),sha256=sha(p),reused_prior_stage=False))
        write(out/'completion.json',dict(query_hashes=[q['sequence_sha256'] for q in batch],files=saved));files.extend(saved)
        write(BASE/'progress.json',dict(completed_queries=len(files),total_queries=manifest['unique_queries']))
    if len(files)!=manifest['unique_queries']:raise ValueError('Enzyme WT MSA roster incomplete')
    done=dict(status='ORDINARY_ENZYME_WT_MSA_STAGED_SOURCE_AND_IDENTITY_REVIEW_PENDING',
        enzyme_cases=80,unique_queries=len(files),files=files,prior_MSA_reused=sum(r['reused_prior_stage'] for r in files),
        seconds=time.monotonic()-start,manifest_sha256=sha(BASE/'request_manifest.json'),
        new_model_calls=0,new_structure_predictions=0,prediction_release_allowed=False)
    write(BASE/'completion.json',done);print({k:v for k,v in done.items() if k!='files'},flush=True)


if __name__=='__main__':main()
