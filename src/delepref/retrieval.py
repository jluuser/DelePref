"""Sequence-only MMseqs2 projection; unaligned termini remain unknown."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from .planning import AA

FORMAT='query,target,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen,evalue,bits,qcov,tcov,qaln,taln,tseq'


def project_hit(fields,sequence):
    if len(fields)!=17:raise ValueError('Expected the 17-column MMseqs alignment including full target sequence')
    qstart,qend,qlen=int(fields[4]),int(fields[5]),int(fields[6]);tstart,tend,tlen=int(fields[7]),int(fields[8]),int(fields[9])
    qaln,taln,target=fields[14].upper(),fields[15].upper(),fields[16].upper()
    if qlen!=len(sequence) or len(qaln)!=len(taln) or not 1<=qstart<=qend<=qlen:
        raise ValueError('Query alignment coordinates mismatch')
    if qaln.replace('-','')!=sequence[qstart-1:qend]:raise ValueError('Query alignment does not reconstruct WT')
    if len(target)!=tlen or taln.replace('-','')!=target[tstart-1:tend]:raise ValueError('Full target identity mismatch')
    if not 1<=tstart<=tend<=tlen:raise ValueError('Invalid target coordinates')
    if not 0<=float(fields[2])<=1 or not np.isfinite([float(fields[k]) for k in [2,10,11]]).all():
        raise ValueError('Nonfinite alignment statistics or wrong identity units')
    projected=['.']*len(sequence);position=qstart-1
    for q,t in zip(qaln,taln):
        if q!='-':projected[position]=t if t in AA or t=='-' else '.';position+=1
    observed=[i for i,c in enumerate(projected) if c in AA]
    if observed:
        projected=[('.' if c=='-' and not observed[0]<i<observed[-1] else c) for i,c in enumerate(projected)]
    else:projected=['.']*len(sequence)
    known=sum(c in AA or c=='-' for c in projected)
    return dict(query_id=fields[0],target=fields[1],identity=float(fields[2]),evalue=float(fields[10]),bits=float(fields[11]),
        qcov=float(fields[12]),tcov=float(fields[13]),known_coverage=1-projected.count('.')/len(sequence),projected=''.join(projected),
        target_length=tlen,target_sequence_sha256=hashlib.sha256(target.encode()).hexdigest(),complete_target_identity_checked=True)


def aligned_identity(a,b):
    a,b=np.asarray(list(a)),np.asarray(list(b));known=(a!='.')&(b!='.')
    if known.mean()<.5:return -1.
    informative=known&((a!='-')|(b!='-'))
    return float(((a==b)&(a!='-')&informative).sum()/max(1,informative.sum()))


def representatives(hits,excluded=()):
    chosen=[];targets=set();projections=set();counts=Counter(raw=len(hits))
    for h in sorted(hits,key=lambda x:(-x['bits'],x['evalue'],x['target'],x['projected'])):
        if not(.2<=h['identity']<=1 and h['evalue']<=.001 and h['known_coverage']>=.8):
            counts['quality_filtered']+=1;continue
        if h['target_sequence_sha256'] in excluded:counts['known_sequence_excluded']+=1;continue
        if h['target'] in targets or h['projected'] in projections:counts['exact_duplicate']+=1;continue
        targets.add(h['target']);projections.add(h['projected']);gap=tuple(i for i,c in enumerate(h['projected']) if c=='-')
        if any(tuple(i for i,c in enumerate(old['projected']) if c=='-')==gap and aligned_identity(h['projected'],old['projected'])>=.9 for old in chosen):
            counts['same_gap_pattern_redundant']+=1;continue
        chosen.append(h)
        if len(chosen)==64:break
    counts['selected']=len(chosen)
    return chosen,dict(counts)


def search_profiles(records,database,out,*,executable='mmseqs',threads=4,reviewed_targets=None):
    exe=shutil.which(executable) or (str(Path(executable).resolve()) if Path(executable).is_file() else None)
    if not exe:raise FileNotFoundError('MMseqs2 is required for fresh search; frozen-profile inference needs no search executable')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    with (out/'queries.fasta').open('w',encoding='ascii') as f:
        for i,(pid,seq) in enumerate(records):f.write('>query_%04d\n%s\n'%(i,seq))
    command=[exe,'createdb',str(out/'queries.fasta'),str(out/'query_db'),'--threads',str(threads)]
    commands=[command,[exe,'search',str(out/'query_db'),str(database),str(out/'result_db'),str(out/'tmp'),
        '-a','1','--alignment-mode','3','-e','0.001','--min-seq-id','0.20','-c','0.60','--cov-mode','2',
        '--threads',str(threads),'-s','7.5','--max-seqs','512','--remove-tmp-files','1'],
        [exe,'convertalis',str(out/'query_db'),str(database),str(out/'result_db'),str(out/'hits.tsv'),
         '--threads',str(threads),'--format-output',FORMAT]]
    for i,cmd in enumerate(commands):
        r=subprocess.run(cmd,text=True,capture_output=True)
        (out/('command_%d.log'%i)).write_text(r.stdout+r.stderr,encoding='utf-8')
        if r.returncode:raise RuntimeError('MMseqs2 command failed; see command_%d.log'%i)
    queries={'query_%04d'%i:(pid,seq) for i,(pid,seq) in enumerate(records)};grouped={k:[] for k in queries}
    for line in (out/'hits.tsv').read_text(encoding='utf-8').splitlines():
        fields=line.split('\t');qid=fields[0]
        if qid not in queries:raise ValueError('Unknown query in MMseqs output')
        grouped[qid].append(project_hit(fields,queries[qid][1]))
    review=json.loads(Path(reviewed_targets).read_text()) if reviewed_targets else None
    excluded={hashlib.sha256(s.encode()).hexdigest() for _,s in records};profiles={}
    for qid,(pid,seq) in queries.items():
        reps,audit=representatives(grouped[qid],excluded)
        if review is not None:
            selected=[]
            for rep in reps:
                entry=review.get(rep['target'],{})
                if entry.get('status')!='RETAIN':continue
                if entry.get('representative_sequence_sha256')!=rep['target_sequence_sha256']:
                    raise ValueError('Reviewed source target hash mismatch')
                selected.append(rep)
            reps=selected
        profiles[pid]=dict(query=dict(protein_id=pid,sequence_sha256=hashlib.sha256(seq.encode()).hexdigest()),
            representatives=reps,audit=audit,evidence_status='SOURCE_REVIEWED' if review is not None else 'SEQUENCE_QUALITY_ONLY_NOT_BIOLOGICALLY_REVIEWED')
    (out/'profiles.json').write_text(json.dumps(profiles,indent=2)+'\n',encoding='utf-8')
    return profiles
