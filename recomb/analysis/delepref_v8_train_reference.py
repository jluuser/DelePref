"""Build exact original training sequence reference, verified against checkpoint roster."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013')
BASE=Path('/public/home/zhangyangroup/chengshiz/keyuan.zhou/SurgEdit')
ROSTER=BASE/'data/processed/preference_training/final_readiness/corrected/splits/v3_sequence_hash_v1/train_dev_manifest.csv'
NATURAL=BASE/'data/processed/preference_training/uniref50_manifest/uniref50_formal_300k_manifest.csv'
MMSEQS=Path('/public/nfs-data/keyuan.zhou/evocompress/tools/mmseqs/bin/mmseqs')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
    raise RuntimeError('CPU Slurm allocation required')
start=time.time()
registry=json.loads((ROOT/'provenance/original_delepref_assets_20261016.json').read_text())
if sha(ROSTER)!=registry['data']['train_dev_manifest']['sha256']:raise ValueError('original roster changed')
with ROSTER.open(newline='') as f:
    train={r['protein_id']:r for r in csv.DictReader(f) if r['split']=='train'}
if len(train)!=269915:raise ValueError('original training roster count differs')
out=ROOT/'retrieval/delepref_v8_training_reference';out.mkdir(exist_ok=False)
fasta=out/'training.fasta';seen=set();rows=0
with NATURAL.open(newline='') as source, fasta.open('x') as target:
    for row in csv.DictReader(source):
        rows+=1;pid=row['protein_id']
        if pid not in train:continue
        if pid in seen:raise ValueError('duplicate training source protein')
        sequence=row['sequence']
        if hashlib.sha256(sequence.encode()).hexdigest()!=train[pid]['sequence_sha256']:
            raise ValueError('training sequence hash mismatch')
        if len(sequence)!=int(train[pid]['wildtype_length']):raise ValueError('training length mismatch')
        target.write('>'+pid+'\n'+sequence+'\n');seen.add(pid)
if seen!=set(train):raise ValueError('training source does not cover exact roster')
cmd=[str(MMSEQS),'createdb',str(fasta),str(out/'training_db'),'--dbtype','1']
result=subprocess.run(cmd,text=True,capture_output=True)
(out/'createdb.log').write_text(result.stdout+result.stderr)
if result.returncode:raise RuntimeError('MMseqs createdb failed; inspect retained log')
report=dict(status='ORIGINAL_TRAINING_REFERENCE_READY',job_id=os.environ['SLURM_JOB_ID'],
    train_WT=len(seen),source_rows=rows,roster_sha256=sha(ROSTER),natural_source_sha256=sha(NATURAL),
    fasta_path=str(fasta),fasta_sha256=sha(fasta),mmseqs_binary_sha256=sha(MMSEQS),
    database_prefix=str(out/'training_db'),seconds=time.time()-start,
    target_role='nearest_original_training_homology_audit_not_retrieval_generation_database',
    new_GPU_jobs=0,new_PLM_forwards=0,new_structure_predictions=0,paper_changes=False)
with (ROOT/'reports/delepref_v8/train_reference_completion.json').open('x') as f:
    json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(report),flush=True)
