"""Reuse unchanged reviewed query searches; search only corrected benign constructs."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
from recomb_scope_screen_20261017 import read,sha,write

BASE=Path('retrieval/recomb_benign_applications_20261017')
FREEZE=Path('configs/recomb_benign_application_retrieval_freeze_20261017.json')


def prepare(root):
    panel=root/'reports/recomb_benign_application_constructs_20261017/panel.json'
    review=root/'reports/recomb_benign_application_scope_20261017/audit.json'
    accepted={r['protein_id'] for r in read(review) if r['status']=='RETAIN'}
    rows=read(panel)
    if len(rows)!=23 or {r['protein_id'] for r in rows}!=accepted:raise ValueError('reviewed application scope differs')
    old=root/'retrieval/recomb_general100_applications_20261017'
    if sha(old/'profiles.json')!=read(old/'completion.json')['profiles_sha256']:raise ValueError('previous search changed')
    previous=read(old/'profiles.json');reused={};pending=[]
    for row in rows:
        p=previous.get(row['protein_id'])
        if p and p['query']['sequence_sha256']==row['sequence_sha256'] and p['query']['sequence']==row['sequence']:
            reused[row['protein_id']]=p
        else:pending.append(row)
    if len(reused)!=21 or len(pending)!=2:raise ValueError('unexpected corrected query scope')
    out=root/BASE;out.mkdir(exist_ok=False)
    write(out/'reused_profiles.json',reused);write(out/'pending_queries.json',pending)
    with (out/'queries.fasta').open('x') as f:
        for i,row in enumerate(pending):f.write('>benign_app_%02d\n%s\n'%(i,row['sequence']))
    exclusions=set(read(old/'excluded_sequence_hashes.json'));exclusions.update(r['sequence_sha256'] for r in rows)
    write(out/'excluded_sequence_hashes.json',sorted(exclusions))
    sources=[panel,review,old/'profiles.json',old/'completion.json',root/'configs/delepref_v8_main_retrieval_freeze.json',
        out/'pending_queries.json',out/'reused_profiles.json',out/'queries.fasta',out/'excluded_sequence_hashes.json',
        Path(__file__),root/'analysis/delepref_v8_retrieval.py',root/'analysis/rag_v6_features.py']
    write(root/FREEZE,dict(status='REVIEWED_QUERY_SEARCH_FROZEN_SOURCE_EVIDENCE_NOT_YET_REVIEWED',
        files={str(p.relative_to(root)):sha(p) for p in sources},WT=23,reused_queries=21,new_queries=2,
        generation_allowed=False,new_GPU_calls=0))
    print(json.dumps(dict(freeze_sha256=sha(root/FREEZE),reused_queries=21,new_queries=2)))


def run(root,checksum):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    if sha(root/FREEZE)!=checksum:raise ValueError('freeze changed')
    for p,h in read(root/FREEZE)['files'].items():
        if sha(root/p)!=h:raise ValueError('query/source identity changed')
    import delepref_v8_retrieval as native
    import rag_v6_retrieval as original
    db=read(root/'configs/delepref_v8_main_retrieval_freeze.json')
    for path,metadata in db['database'].items():
        stat=Path(path).stat()
        if (stat.st_size,stat.st_mtime_ns)!=(metadata['size'],metadata['mtime_ns']):raise ValueError('database identity changed')
        if 'sha256' in metadata and sha(path)!=metadata['sha256']:raise ValueError('database index changed')
    out=root/BASE
    if (out/'execution_claim.json').exists():raise FileExistsError('search already claimed')
    write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID']))
    query=str(out/'query_db');result=str(out/'result_db')
    commands=[
        [original.MMSEQS,'createdb',str(out/'queries.fasta'),query,'--threads','32'],
        [original.MMSEQS,'search',query,db['target'],result,str(out/'tmp'),'-a','1','--alignment-mode','3',
         '-e','0.001','--min-seq-id','0.20','-c','0.60','--cov-mode','2','--threads','32','-s','7.5',
         '--max-seqs','512','--split-mode','0','--split-memory-limit','160G','--remove-tmp-files','1'],
        [original.MMSEQS,'convertalis',query,db['target'],result,str(out/'hits.tsv'),'--threads','32','--format-output',native.FORMAT]]
    for i,cmd in enumerate(commands):
        with (out/('command_%d.log'%i)).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    pending=read(out/'pending_queries.json');queries={'benign_app_%02d'%i:r for i,r in enumerate(pending)}
    grouped={q:[] for q in queries}
    for line in (out/'hits.tsv').read_text().splitlines():
        fields=line.split('\t');grouped[fields[0]].append(native.project(fields,queries[fields[0]]['sequence']))
    profiles=read(out/'reused_profiles.json');excluded=set(read(out/'excluded_sequence_hashes.json'))
    for q,row in queries.items():
        reps,audit=native.representatives(grouped[q],excluded)
        profiles[row['protein_id']]=dict(representatives=reps,audit=audit,contract=native.CONTRACT,
            query=dict(query_id=q,protein_id=row['protein_id'],sequence=row['sequence'],sequence_sha256=row['sequence_sha256']))
    write(out/'profiles_unreviewed_evidence.json',profiles)
    write(out/'completion.json',dict(status='REVIEWED_QUERY_SEARCH_COMPLETE_EVIDENCE_SOURCE_REVIEW_REQUIRED',WT=23,
        reused_queries=21,new_queries=2,raw_new_hits=sum(map(len,grouped.values())),
        profiles_sha256=sha(out/'profiles_unreviewed_evidence.json'),freeze_sha256=checksum,
        generation_allowed=False,new_model_calls=0,new_structure_predictions=0,job_id=os.environ['SLURM_JOB_ID']))
    print(json.dumps(read(out/'completion.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
    p.add_argument('--prepare',action='store_true');p.add_argument('--freeze-sha256');a=p.parse_args()
    prepare(a.root) if a.prepare else run(a.root,a.freeze_sha256)
