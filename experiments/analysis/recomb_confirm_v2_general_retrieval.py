"""Reuse the existing natural UniRef database for qualified General100 queries."""
import argparse
from collections import defaultdict
import hashlib
import os
from pathlib import Path
import subprocess
import time
from recomb_confirm_v2_diagnostics import read,sha,write
import delepref_v8_retrieval as native
import rag_v6_retrieval as original

BASE=Path('retrieval/recomb_confirm_v2_General100')
FZ=Path('configs/recomb_confirm_v2_General100_retrieval_freeze.json')


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('General100 retrieval already frozen')
    out=root/BASE;out.mkdir(parents=True,exist_ok=True)
    panel=read(root/'reports/recomb_confirm_v2_general_native/General100_panel.json')
    if len(panel)!=100 or not all(r['WT_encoding_allowed'] for r in panel):raise ValueError('Not the qualified ordinary General100')
    queries=[dict(query_id='confirmV2q_%04d'%i,protein_id=r['protein_id'],sequence=r['sequence'],
        sequence_sha256=r['WT_sha256']) for i,r in enumerate(panel)]
    excluded=set(read(root/'retrieval/delepref_v8_main/excluded_sequence_hashes.json'))
    excluded.update(r['sequence_sha256'] for r in queries)
    write(out/'queries.json',queries);write(out/'excluded_sequence_hashes.json',sorted(excluded))
    with (out/'queries.fasta').open('w',encoding='ascii',newline='\n') as handle:
        for q in queries:handle.write('>'+q['query_id']+'\n'+q['sequence']+'\n')
    inherited=read(root/'configs/delepref_v8_main_retrieval_freeze.json')
    for path,m in inherited['database'].items():
        s=Path(path).stat()
        if (s.st_size,s.st_mtime_ns)!=(m['size'],m['mtime_ns']):raise ValueError('Natural database metadata changed')
        if 'sha256' in m and sha(path)!=m['sha256']:raise ValueError('Natural database small index changed')
    names=['analysis/recomb_confirm_v2_general_retrieval.py','analysis/recomb_confirm_v2_diagnostics.py',
        'analysis/delepref_v8_retrieval.py','analysis/rag_v6_features.py','analysis/rag_v6_retrieval.py',
        'slurm/recomb_confirm_v2_general_retrieval.sbatch',
        'reports/recomb_confirm_v2_general_native/General100_panel.json',
        str(BASE/'queries.json'),str(BASE/'queries.fasta'),str(BASE/'excluded_sequence_hashes.json')]
    write(root/FZ,dict(status='QUALIFIED_GENERAL100_NATURAL_SEARCH_FROZEN',files={p:sha(root/p) for p in names},
        database=inherited['database'],target=inherited['target'],contract=native.CONTRACT,
        mmseqs_sha256=sha(original.MMSEQS),generation_allowed=False,source_evidence_review_required=True))
    print(sha(root/FZ))


def run(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    f=read(root/FZ)
    for p,h in f['files'].items():
        if sha(root/p)!=h:raise ValueError('Frozen General retrieval input/source changed')
    for path,m in f['database'].items():
        s=Path(path).stat()
        if (s.st_size,s.st_mtime_ns)!=(m['size'],m['mtime_ns']):raise ValueError('Natural DB changed')
        if 'sha256' in m and sha(path)!=m['sha256']:raise ValueError('Natural index changed')
    if sha(original.MMSEQS)!=f['mmseqs_sha256'] or f['contract']!=native.CONTRACT:raise ValueError('Retrieval implementation differs')
    out=root/BASE
    if (out/'execution_claim.json').exists():raise FileExistsError('Search already claimed; inspect before any retry')
    write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID']));start=time.monotonic()
    qdb=str(out/'query_db');rdb=str(out/'result_db')
    commands=[
        [original.MMSEQS,'createdb',str(out/'queries.fasta'),qdb,'--threads','32'],
        [original.MMSEQS,'search',qdb,f['target'],rdb,str(out/'tmp'),'-a','1','--alignment-mode','3',
         '-e','0.001','--min-seq-id','0.20','-c','0.60','--cov-mode','2','--threads','32','-s','7.5',
         '--max-seqs','512','--split-mode','0','--split-memory-limit','160G','--remove-tmp-files','1'],
        [original.MMSEQS,'convertalis',qdb,f['target'],rdb,str(out/'hits.tsv'),'--threads','32','--format-output',native.FORMAT]]
    for i,command in enumerate(commands):
        with (out/('command_'+str(i)+'.log')).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    queries={q['query_id']:q for q in read(out/'queries.json')};groups=defaultdict(list)
    with (out/'hits.tsv').open() as handle:
        for line in handle:
            fields=line.rstrip('\n').split('\t')
            groups[fields[0]].append(native.project(fields,queries[fields[0]]['sequence']))
    profiles={};excluded=set(read(out/'excluded_sequence_hashes.json'))
    for qid,q in queries.items():
        reps,audit=native.representatives(groups[qid],excluded)
        profiles[q['protein_id']]=dict(query=q,representatives=reps,audit=audit,contract=native.CONTRACT,
            evidence_status='UNREVIEWED_NOT_FOR_GENERATION')
    write(out/'profiles_unreviewed_evidence.json',profiles)
    report=dict(status='GENERAL100_NATURAL_SEARCH_COMPLETE_EVIDENCE_SCOPE_REVIEW_PENDING',WT=100,
        raw_hits=sum(len(x) for x in groups.values()),representative_rows=sum(len(p['representatives']) for p in profiles.values()),
        profiles_sha256=sha(out/'profiles_unreviewed_evidence.json'),seconds=time.monotonic()-start,
        source_evidence_generation_allowed=False,new_model_calls=0,new_structure_calls=0,job_id=os.environ['SLURM_JOB_ID'])
    write(out/'completion.json',report);print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();(prepare if a.prepare else run)(a.root)
