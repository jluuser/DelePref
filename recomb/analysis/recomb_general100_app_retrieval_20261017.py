"""Reuse the frozen natural database and training reference for application WTs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import delepref_v8_retrieval as native
import rag_v6_retrieval as inherited
from recomb_general100_common_20261017 import ROOT,sha,write
from recomb_general100_app_prepare_20261017 import BASE as APP

BASE='retrieval/recomb_general100_applications_20261017'


def main(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('CPU Slurm required')
    out=root/BASE;out.mkdir(exist_ok=False);started=time.monotonic()
    complete=json.loads((root/APP/'preparation_completion.json').read_text())
    for name,digest in complete['files'].items():
        if sha(root/APP/name)!=digest:raise ValueError('application input changed')
    panel=json.loads((root/APP/'panel.json').read_text())
    fasta=root/APP/'queries.fasta'
    frozen=json.loads((root/'configs/delepref_v8_main_retrieval_freeze.json').read_text())
    for p,m in frozen['database'].items():
        stat=Path(p).stat()
        if (stat.st_size,stat.st_mtime_ns)!=(m['size'],m['mtime_ns']):raise ValueError('natural database metadata changed')
        if 'sha256' in m and sha(p)!=m['sha256']:raise ValueError('natural database index changed')
    excluded=set(json.loads((root/'retrieval/delepref_v8_main/excluded_sequence_hashes.json').read_text()))
    excluded.update(p['sequence_sha256'] for p in panel)
    expanded=root/'external/recomb_expanded_20261017/retrieval_exclusion_sha256.json'
    extra=json.loads(expanded.read_text())
    if not isinstance(extra,list):raise ValueError('expanded exclusion contract changed')
    excluded.update(extra)
    write(out/'excluded_sequence_hashes.json',sorted(excluded))
    training=json.loads((root/'reports/delepref_v8/train_reference_completion.json').read_text())
    write(root/'configs/recomb_general100_app_retrieval_freeze_20261017.json',dict(
        status='APPLICATION_RETRIEVAL_FROZEN_BEFORE_SEARCH',contract=native.CONTRACT,
        files={str(p):sha(p) for p in [root/APP/'panel.json',fasta,expanded,Path(__file__),
              root/'analysis/delepref_v8_retrieval.py',root/'analysis/rag_v6_features.py']},
        database=frozen['database'],target=frozen['target'],mmseqs_sha256=sha(inherited.MMSEQS),
        training_reference=training,WT=len(panel),excluded_sequences=len(excluded),
        method_outputs_used_to_set_rules=False))
    def command(args,name):
        with (out/(name+'.log')).open('w') as log:
            subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(json.dumps(dict(event='APPLICATION_RETRIEVAL_COMMAND_COMPLETE',name=name)),flush=True)
    query=str(out/'query_db')
    command([inherited.MMSEQS,'createdb',str(fasta),query,'--threads','32'],'create')
    command([inherited.MMSEQS,'search',query,frozen['target'],str(out/'result_db'),str(out/'tmp'),
         '-a','1','--alignment-mode','3','-e','0.001','--min-seq-id','0.20','-c','0.60','--cov-mode','2',
         '--threads','32','-s','7.5','--max-seqs','512','--split-mode','0','--split-memory-limit','160G','--remove-tmp-files','1'],'natural_search')
    command([inherited.MMSEQS,'convertalis',query,frozen['target'],str(out/'result_db'),str(out/'hits.tsv'),
         '--threads','32','--format-output',native.FORMAT],'natural_convert')
    queries={'appq_%03d'%i:p for i,p in enumerate(panel)};grouped={q:[] for q in queries}
    with (out/'hits.tsv').open() as f:
        for line in f:
            fields=line.rstrip('\n').split('\t')
            grouped[fields[0]].append(native.project(fields,queries[fields[0]]['sequence']))
    profiles={}
    for q,p in queries.items():
        reps,audit=native.representatives(grouped[q],excluded)
        profiles[p['protein_id']]=dict(representatives=reps,audit=audit,contract=native.CONTRACT,
            query=dict(query_id=q,protein_id=p['protein_id'],sequence=p['sequence'],sequence_sha256=p['sequence_sha256']))
    write(out/'profiles.json',profiles)
    command([inherited.MMSEQS,'easy-search',str(fasta),training['database_prefix'],str(out/'train_hits.tsv'),str(out/'tmp_train'),
        '-e','1e-3','-s','7.5','--max-seqs','100','--threads','32','--format-output',
        'query,target,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen,evalue,bits,qcov,tcov'],'training_similarity')
    command([inherited.MMSEQS,'easy-search',str(fasta),query,str(out/'self_hits.tsv'),str(out/'tmp_self'),
        '-e','1e-3','-s','7.5','--max-seqs','100','--threads','32','--format-output',
        'query,target,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen,evalue,bits,qcov,tcov'],'within_panel_similarity')
    # Raw nearest-hit tables remain authoritative; no application WT is removed
    # based on this audit, and unknown family annotations remain explicit.
    write(out/'completion.json',dict(status='APPLICATION_NATURAL_PROFILES_AND_SIMILARITY_READY',WT=len(panel),
        raw_hits=sum(map(len,grouped.values())),WT_with_representatives=sum(bool(p['representatives']) for p in profiles.values()),
        excluded_sequences=len(excluded),profiles_sha256=sha(out/'profiles.json'),
        training_hits_sha256=sha(out/'train_hits.tsv'),within_panel_hits_sha256=sha(out/'self_hits.tsv'),
        no_WT_removed_by_retrieval_or_training_similarity=True,seconds=time.monotonic()-started,
        job_id=os.environ['SLURM_JOB_ID'],new_GPU_calls=0,new_structure_predictions=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);main(p.parse_args().root)
