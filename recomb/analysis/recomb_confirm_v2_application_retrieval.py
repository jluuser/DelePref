"""Reuse the frozen natural UniRef search protocol for qualified application WTs."""
import argparse
from collections import defaultdict
import os
from pathlib import Path
import subprocess
import time
from recomb_confirm_v2_diagnostics import read,sha,write
import delepref_v8_retrieval as native
import rag_v6_retrieval as original

SPECS={'Binder60':('reports/recomb_confirm_v2_Binder60_constructs','Binder60_panel.json',60),
       'Enzyme80':('reports/recomb_confirm_v2_Enzyme80_constructs','Enzyme80_panel.json',80)}


def paths(cohort):return Path('retrieval/recomb_confirm_v2_'+cohort),Path('configs/recomb_confirm_v2_'+cohort+'_retrieval_freeze.json')


def prepare(root,cohort):
    base,FZ=paths(cohort);source,panelname,N=SPECS[cohort]
    if (root/FZ).exists():raise FileExistsError('Application natural search already frozen')
    panel=read(root/source/panelname)
    if len(panel)!=N or not all(r['WT_encoding_allowed'] for r in panel):raise ValueError('Application WT scope not released')
    queries=[dict(query_id=cohort+'q_%04d'%i,protein_id=r['protein_id'],sequence=r['sequence'],
        sequence_sha256=r['WT_sha256']) for i,r in enumerate(panel)]
    excluded=set(read(root/'retrieval/delepref_v8_main/excluded_sequence_hashes.json'))
    excluded.update(q['sequence_sha256'] for q in queries)
    write(root/base/'queries.json',queries);write(root/base/'excluded_sequence_hashes.json',sorted(excluded))
    with (root/base/'queries.fasta').open('w',encoding='ascii',newline='\n') as f:
        for q in queries:f.write('>'+q['query_id']+'\n'+q['sequence']+'\n')
    inherited=read(root/'configs/delepref_v8_main_retrieval_freeze.json')
    for p,m in inherited['database'].items():
        s=Path(p).stat()
        if (s.st_size,s.st_mtime_ns)!=(m['size'],m['mtime_ns']):raise ValueError('Existing natural database changed')
        if 'sha256' in m and sha(p)!=m['sha256']:raise ValueError('Existing natural index changed')
    names=['analysis/recomb_confirm_v2_application_retrieval.py','analysis/recomb_confirm_v2_diagnostics.py',
        'analysis/delepref_v8_retrieval.py','analysis/rag_v6_features.py','analysis/rag_v6_retrieval.py',
        'slurm/recomb_confirm_v2_application_retrieval.sbatch',str(Path(source)/panelname),
        str(base/'queries.json'),str(base/'queries.fasta'),str(base/'excluded_sequence_hashes.json')]
    quality={k:v for k,v in native.CONTRACT.items() if k!='main_WT'}
    write(root/FZ,dict(status='QUALIFIED_APPLICATION_WT_NATURAL_SEARCH_ONLY_FROZEN',cohort=cohort,WT=N,
        files={p:sha(root/p) for p in names},database=inherited['database'],target=inherited['target'],
        quality_contract=quality,mmseqs_sha256=sha(original.MMSEQS),source_evidence_review_required=True,
        RAG_generation_allowed=False,new_GPU_calls=0,new_structure_calls=0))
    print(sha(root/FZ),flush=True)


def run(root,cohort):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    base,FZ=paths(cohort);f=read(root/FZ);out=root/base
    for p,h in f['files'].items():
        if sha(root/p)!=h:raise ValueError('Application retrieval input/source changed')
    for p,m in f['database'].items():
        s=Path(p).stat()
        if (s.st_size,s.st_mtime_ns)!=(m['size'],m['mtime_ns']):raise ValueError('Natural database changed')
        if 'sha256' in m and sha(p)!=m['sha256']:raise ValueError('Natural small index changed')
    if f['quality_contract']!={k:v for k,v in native.CONTRACT.items() if k!='main_WT'} or sha(original.MMSEQS)!=f['mmseqs_sha256']:
        raise ValueError('Natural search implementation or numeric quality rules changed')
    if (out/'execution_claim.json').exists():raise FileExistsError('Search claimed; inspect existing job')
    write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID']));start=time.monotonic()
    qdb=str(out/'query_db');rdb=str(out/'result_db')
    commands=[[original.MMSEQS,'createdb',str(out/'queries.fasta'),qdb,'--threads','32'],
        [original.MMSEQS,'search',qdb,f['target'],rdb,str(out/'tmp'),'-a','1','--alignment-mode','3',
         '-e','0.001','--min-seq-id','0.20','-c','0.60','--cov-mode','2','--threads','32','-s','7.5',
         '--max-seqs','512','--split-mode','0','--split-memory-limit','160G','--remove-tmp-files','1'],
        [original.MMSEQS,'convertalis',qdb,f['target'],rdb,str(out/'hits.tsv'),'--threads','32','--format-output',native.FORMAT]]
    for i,command in enumerate(commands):
        with (out/('command_'+str(i)+'.log')).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    queries={q['query_id']:q for q in read(out/'queries.json')};grouped=defaultdict(list)
    with (out/'hits.tsv').open() as handle:
        for line in handle:
            a=line.rstrip('\n').split('\t');grouped[a[0]].append(native.project(a,queries[a[0]]['sequence']))
    excluded=set(read(out/'excluded_sequence_hashes.json'));profiles={}
    for qid,q in queries.items():
        representatives,audit=native.representatives(grouped[qid],excluded)
        profiles[q['protein_id']]=dict(query=q,representatives=representatives,audit=audit,
            contract=dict(f['quality_contract'],cohort=cohort,WT=f['WT']),evidence_status='UNREVIEWED_NOT_FOR_GENERATION')
    write(out/'profiles_unreviewed_evidence.json',profiles)
    done=dict(status='APPLICATION_NATURAL_RETRIEVAL_COMPLETE_SOURCE_IDENTITY_REVIEW_PENDING',cohort=cohort,WT=f['WT'],
        raw_hits=sum(map(len,grouped.values())),representative_rows=sum(len(p['representatives']) for p in profiles.values()),
        profiles_sha256=sha(out/'profiles_unreviewed_evidence.json'),seconds=time.monotonic()-start,
        job_id=os.environ['SLURM_JOB_ID'],new_model_calls=0,new_structure_predictions=0,RAG_generation_allowed=False)
    write(out/'completion.json',done);print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--cohort',choices=SPECS,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();prepare(a.root,a.cohort) if a.prepare else run(a.root,a.cohort)
