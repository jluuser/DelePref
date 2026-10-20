"""Launch-ready provenance wrapper around the already frozen ESMC cache runner."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import rag_v6_retrieval as r

FREEZE="configs/rag_v6_embedding_freeze.json"
BASE="reports/rag_v6_confirmation"
PREFIX="rag_v6_confirmation"


def command(root,code,shard,execute=False):
    result=[sys.executable,str(code/'cache_plm.py'),'--protocol',str(root/'configs/protocol_v2.json'),
            '--manifest',str(root/BASE/'candidates.jsonl'),'--provenance',str(root/'provenance/source_files.json'),
            '--output',str(root/'caches/embeddings'),'--shard',str(shard),'--index-prefix',PREFIX]
    return result+(['--execute'] if execute else [])


def prepare(root):
    cohort=json.loads((root/BASE/'report.json').read_text())
    pins={BASE+'/candidates.jsonl':cohort['candidates_sha256'],
          'reports/rag_v6/models.json':r.sha(root/'reports/rag_v6/models.json'),
          'reports/rag_v6/profiles.json':r.sha(root/'reports/rag_v6/profiles.json')}
    r.verify_hashes(root,pins)
    dispatch=json.loads((root/'reports/dispatch_after_pilot.json').read_text())
    code=Path(dispatch['code_snapshot'])
    r.verify_hashes(code,dispatch['code_hashes'])
    if cohort['WT']!=32 or cohort['candidates']!=384: raise ValueError('unexpected confirmation cohort')
    for shard in (0,1):
        if (root/f'caches/embeddings/{PREFIX}_shard{shard}_index.jsonl').exists():
            raise ValueError('new embedding namespace already contains an executed index')
        subprocess.run(command(root,code,shard),check=True)
    plans=[json.loads((root/f'caches/embeddings/{PREFIX}_shard{i}_plan.json').read_text()) for i in (0,1)]
    total=sum(len(p['sequences']) for p in plans)
    freeze=dict(status='RAG_V6_EMBEDDINGS_FROZEN',input_hashes=pins,code_snapshot=str(code),
                computational_code_hashes=dispatch['code_hashes'],
                protocol_sha256=r.sha(root/'configs/protocol_v2.json'),
                provenance_sha256=r.sha(root/'provenance/source_files.json'),
                source_hashes={p:r.sha(root/p) for p in ['analysis/rag_v6_embeddings.py','slurm/rag_v6_embeddings.sbatch']},
                index_prefix=PREFIX,unique_sequences=total,shards=2,max_concurrent_GPU=2,
                profile='reuse_frozen_ESMC_A40_bfloat16_float32_cache_batch4',
                structure_calls=0,paper_changes=False,training_or_parameter_changes=False)
    r.write_json(root/FREEZE,freeze,atomic=True)
    return dict(freeze_sha256=r.sha(root/FREEZE),unique_sequences=total,shard_sizes=[len(p['sequences']) for p in plans])


def execute(root,checksum,shard):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_ARRAY_TASK_ID')!=str(shard):
        raise ValueError('array allocation required')
    r.verify_hashes(root,{FREEZE:checksum}); freeze=json.loads((root/FREEZE).read_text())
    for key in ['input_hashes','source_hashes']: r.verify_hashes(root,freeze[key])
    r.verify_hashes(root,{'configs/protocol_v2.json':freeze['protocol_sha256'],'provenance/source_files.json':freeze['provenance_sha256']})
    code=Path(freeze['code_snapshot']); r.verify_hashes(code,freeze['computational_code_hashes'])
    claim=root/BASE/f'embedding_claim_{shard}.json'
    r.write_json(claim,dict(job_id=os.environ['SLURM_JOB_ID'],shard=shard,freeze_sha256=checksum),atomic=True)
    subprocess.run(command(root,code,shard,execute=True),check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    p.add_argument('--freeze-sha256');p.add_argument('--shard',type=int,choices=(0,1));a=p.parse_args()
    if a.prepare: print(json.dumps(prepare(a.root)),flush=True)
    else: execute(a.root,a.freeze_sha256,a.shard)
