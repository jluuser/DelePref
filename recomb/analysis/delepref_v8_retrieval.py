"""V8 natural homolog retrieval: full-target identity checks and gap-pattern preserving dereplication."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import rag_v6_features as features
import rag_v6_retrieval as original
from delepref_v8_panel import ROOT,sha,write

BASE='retrieval/delepref_v8_main'
FREEZE='configs/delepref_v8_main_retrieval_freeze.json'
FORMAT=original.FORMAT+',tseq'
CONTRACT=dict(identity_min=.2,identity_max_inclusive=1.,evalue_max=.001,known_coverage_min=.8,
    exclude_full_target_sequence_sha256=True,exclude_exact_main_WT_and_fixed_measured_sequences=True,
    do_not_exclude_newly_generated_plans=True,cluster_identity=.9,cluster_requires_same_internal_gap_pattern=True,
    representatives_max=64,proposal_minimum_homologs=1,
    quality_order='bits_desc_evalue_asc_target_ID_projection',terminal_absence='unknown',
    main_WT=100,query_coverage_search_min=.6,threads=32,split_memory_limit='160G',
    main_input_protocol_distinct_from_6WT_legacy_profile_engineering=True,
    model_outputs_used_to_set_rules=False)

def project(fields,wt):
    if len(fields)!=17:raise ValueError('expected alignment plus complete target sequence')
    result=features.project_hit(fields[:16],wt);target=fields[16].upper()
    if len(target)!=int(fields[9]):raise ValueError('complete target length mismatch')
    if fields[15].upper().replace('-','')!=target[int(fields[7])-1:int(fields[8])]:
        raise ValueError('target alignment does not reconstruct complete source sequence')
    result['target_sequence_sha256']=hashlib.sha256(target.encode()).hexdigest()
    result['complete_target_identity_checked']=True
    return result

def representatives(hits,excluded):
    selected=[];seen_targets=set();seen_projection=set();counts=Counter(raw=len(hits));processed=0
    for h in sorted(hits,key=lambda x:(-x['bits'],x['evalue'],x['target'],x['projected'])):
        processed+=1
        if not (CONTRACT['identity_min']<=h['identity']<=1 and h['evalue']<=.001 and h['known_coverage']>=.8):
            counts['quality_filtered']+=1;continue
        if h['target_sequence_sha256'] in excluded:
            counts['known_sequence_excluded']+=1;continue
        if h['target'] in seen_targets or h['projected'] in seen_projection:
            counts['exact_duplicate']+=1;continue
        seen_targets.add(h['target']);seen_projection.add(h['projected'])
        gap=tuple(i for i,c in enumerate(h['projected']) if c=='-')
        if any(tuple(i for i,c in enumerate(old['projected']) if c=='-')==gap and
               features.aligned_identity(h['projected'],old['projected'])>=.9 for old in selected):
            counts['same_gap_pattern_redundant']+=1;continue
        selected.append(h)
        if len(selected)==64:break
    counts['selected']=len(selected)
    counts['unprocessed_after_cap']=len(hits)-processed
    return selected,dict(counts)

def prepare(root):
    out=root/BASE;out.mkdir(exist_ok=False)
    main=root/'reports/delepref_v8_native_panel/main_panel.json'
    completion=json.loads((root/'reports/delepref_v8_native_panel/completion.json').read_text())
    if completion['final_WT']!=100 or sha(main)!=completion['panel_sha256']:raise ValueError('native input panel changed')
    panel=json.loads(main.read_text())
    queries=[dict(query_id='v8mainq_%04d'%i,protein_id=x['protein_id'],sequence=x['sequence'],
        sequence_sha256=x['sequence_sha256'],candidate_id=x['candidate_id']) for i,x in enumerate(panel)]
    if len({q['protein_id'] for q in queries})!=100:raise ValueError('duplicate main protein IDs')
    excluded={q['sequence_sha256'] for q in queries}
    measured=root/'external/rag_v6/preparation/primary_rows.jsonl'
    for line in measured.read_text().splitlines():
        r=json.loads(line)
        excluded.update(hashlib.sha256(r[k].encode()).hexdigest() for k in ['wt_sequence','sequence'])
    write(out/'queries.json',queries);write(out/'excluded_sequence_hashes.json',sorted(excluded))
    with (out/'queries.fasta').open('x') as f:
        for q in queries:f.write('>'+q['query_id']+'\n'+q['sequence']+'\n')
    inherited=json.loads((root/original.FREEZE).read_text())
    for p,m in inherited['database'].items():
        stat=Path(p).stat()
        if stat.st_size!=m['size'] or stat.st_mtime_ns!=m['mtime_ns']:raise ValueError('natural database metadata changed')
        if 'sha256' in m and sha(p)!=m['sha256']:raise ValueError('natural database small file changed')
    inputs={str(p):sha(p) for p in [main,root/'reports/delepref_v8_native_panel/completion.json',measured,
        root/original.FREEZE,out/'queries.json',out/'queries.fasta',out/'excluded_sequence_hashes.json']}
    sources={p:sha(root/p) for p in ['analysis/delepref_v8_retrieval.py','analysis/rag_v6_features.py','analysis/rag_v6_retrieval.py',
        'analysis/tests/test_delepref_v8_retrieval.py','slurm/delepref_v8_retrieval.sbatch']}
    write(root/FREEZE,dict(status='V8_MAIN_NATURAL_RETRIEVAL_FROZEN_BEFORE_RESULTS',contract=CONTRACT,
        inputs=inputs,sources=sources,database=inherited['database'],mmseqs_sha256=sha(original.MMSEQS),
        target=original.TARGET,WT=100,excluded_sequence_hashes=len(excluded),model_or_variant_structure_outputs_used=False))
    return dict(freeze_sha256=sha(root/FREEZE),WT=100,excluded_sequence_hashes=len(excluded))

def execute(root,checksum):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    if sha(root/FREEZE)!=checksum:raise ValueError('main retrieval freeze changed')
    frozen=json.loads((root/FREEZE).read_text())
    if frozen['contract']!=CONTRACT:raise ValueError('main retrieval contract changed')
    for p,h in {**frozen['inputs'],**frozen['sources']}.items():
        if sha(root/p)!=h:raise ValueError('retrieval input/source changed: '+p)
    for p,m in frozen['database'].items():
        stat=Path(p).stat()
        if stat.st_size!=m['size'] or stat.st_mtime_ns!=m['mtime_ns']:raise ValueError('retrieval database changed')
        if 'sha256' in m and sha(p)!=m['sha256']:raise ValueError('retrieval database small file changed')
    if sha(original.MMSEQS)!=frozen['mmseqs_sha256']:raise ValueError('MMseqs binary changed')
    out=root/BASE;write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID']))
    start=time.time()
    commands=[
        [original.MMSEQS,'createdb',str(out/'queries.fasta'),str(out/'query_db'),'--threads','32'],
        [original.MMSEQS,'search',str(out/'query_db'),frozen['target'],str(out/'result_db'),str(out/'tmp'),
         '-a','1','--alignment-mode','3','-e','0.001','--min-seq-id','0.20','-c','0.60','--cov-mode','2',
         '--threads','32','-s','7.5','--max-seqs','512','--split-mode','0','--split-memory-limit','160G','--remove-tmp-files','1'],
        [original.MMSEQS,'convertalis',str(out/'query_db'),frozen['target'],str(out/'result_db'),str(out/'hits.tsv'),
         '--threads','32','--format-output',FORMAT]]
    for i,cmd in enumerate(commands):
        r=subprocess.run(cmd,text=True,capture_output=True)
        (out/('command_%d.log'%i)).write_text(r.stdout+r.stderr)
        if r.returncode:raise RuntimeError('retrieval command failed: '+str(i))
        print(json.dumps(dict(event='MAIN_RETRIEVAL_COMMAND_COMPLETE',command_index=i)),flush=True)
    queries={q['query_id']:q for q in json.loads((out/'queries.json').read_text())}
    grouped={q:[] for q in queries}
    with (out/'hits.tsv').open() as f:
        for line in f:
            fields=line.rstrip('\n').split('\t')
            if fields[0] not in queries:raise ValueError('unknown query in retrieval results')
            grouped[fields[0]].append(project(fields,queries[fields[0]]['sequence']))
    excluded=set(json.loads((out/'excluded_sequence_hashes.json').read_text()))
    profiles={};gap_WT=0
    for qid,q in queries.items():
        reps,audit=representatives(grouped[qid],excluded)
        profiles[q['protein_id']]=dict(query=q,representatives=reps,audit=audit,
            contract=CONTRACT,weighting='quality_ordered_gap_pattern_preserving_representatives_not_independent_samples')
        gap_WT+=int(any('-' in h['projected'] for h in reps))
    write(out/'profiles.json',profiles)
    report=dict(status='V8_MAIN_NATURAL_RETRIEVAL_COMPLETE',job_id=os.environ['SLURM_JOB_ID'],seconds=time.time()-start,
        freeze_sha256=checksum,WT=100,raw_hits=sum(map(len,grouped.values())),
        WT_with_raw_hits=sum(bool(v) for v in grouped.values()),WT_with_representatives=sum(bool(v['representatives']) for v in profiles.values()),
        WT_with_observed_internal_gap=gap_WT,hits_sha256=sha(out/'hits.tsv'),profiles_sha256=sha(out/'profiles.json'),
        no_low_support_WT_dropped=True,model_parameters_changed=False,new_GPU_jobs=0,new_structure_predictions=0,paper_changes=False)
    write(out/'completion.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--prepare',action='store_true');p.add_argument('--freeze-sha256');a=p.parse_args()
    print(json.dumps(prepare(a.root)) if a.prepare else execute(a.root,a.freeze_sha256),flush=True)
