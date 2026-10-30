"""Input-only training exposure and evaluation-component audit, no model calls."""
import argparse
from collections import Counter,defaultdict
import hashlib
import os
from pathlib import Path
import subprocess
import time
from recomb_confirm_v2_diagnostics import read,sha,write
from delepref_v8_panel import MMSEQS,FORMAT,alignment,passes,Components

OUT=Path('reports/recomb_confirm_v2_candidate_homology')
FZ=Path('configs/recomb_confirm_v2_candidate_homology_freeze.json')
INPUT=Path('reports/recomb_confirm_v2_WT_candidate_audit_inputs')


def fasta_records(path):
    header=None;seq=[]
    with Path(path).open() as handle:
        for line in handle:
            if line.startswith('>'):
                if header is not None:yield header,''.join(seq)
                header=line[1:].strip().split()[0];seq=[]
            else:seq.append(line.strip())
    if header is not None:yield header,''.join(seq)


def prepare(root):
    out=root/OUT
    if (root/FZ).exists():raise FileExistsError('Homology inputs already frozen')
    dev={}
    for p in read(root/'reports/recomb_benign_scope_20261017/general_panel.json')+read(root/'reports/recomb_benign_application_constructs_20261017/panel.json'):
        seq=p['sequence'];dev[hashlib.sha256(seq.encode()).hexdigest()]=seq
        for partner in p.get('partners',[]):
            seq=partner['sequence'];dev[hashlib.sha256(seq.encode()).hexdigest()]=seq
    out.mkdir(parents=True,exist_ok=True)
    with (out/'reviewed_development.fasta').open('w') as handle:
        for i,(_,seq) in enumerate(sorted(dev.items())):handle.write('>v2dev_%04d\n%s\n'%(i,seq))
    names=['analysis/recomb_confirm_v2_candidate_homology.py','analysis/recomb_confirm_v2_diagnostics.py',
        'analysis/delepref_v8_panel.py','slurm/recomb_confirm_v2_candidate_homology.sbatch',
        str(INPUT/'WT_candidates.json'),str(INPUT/'WT_candidates.fasta'),str(INPUT/'completion.json'),
        str(OUT/'reviewed_development.fasta'),'reports/delepref_v8/train_reference_completion.json']
    write(root/FZ,dict(status='ORDINARY_WT_AUDIT_RULES_FROZEN_BEFORE_HITS',files={p:sha(root/p) for p in names},
        audit_only_not_model_release=True,training_family_exposure_not_automatically_label_leakage=True,
        search=dict(threads=16,sensitivity=7.5,evalue=.001,min_sequence_identity=.2,max_hits=512),
        definitions=dict(exact_training='complete_WT_SHA256',both80_training_flag='identity_shorter_full>=.4 and both coverage>=.8',
            query80_domain_flag='identity_shorter_full>=.4 and query coverage>=.8, including low target coverage',
            self_components='identity_shorter_full>=.3 and both coverage>=.8 plus exact SHA and known source-family',
            development_flag='identity_shorter_full>=.3 and at least one coverage>=.8')))
    print(sha(root/FZ))


def execute(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm only')
    frozen=read(root/FZ)
    for p,h in frozen['files'].items():
        if sha(root/p)!=h:raise ValueError('Input/source changed: '+p)
    out=root/OUT
    if (out/'completion.json').exists():raise FileExistsError('Homology audit already completed')
    start=time.monotonic();reference=read(root/'reports/delepref_v8/train_reference_completion.json')
    if sha(reference['fasta_path'])!=reference['fasta_sha256'] or sha(MMSEQS)!=reference['mmseqs_binary_sha256']:
        raise ValueError('Training reference/MMseqs identity changed')
    rows=read(root/INPUT/'WT_candidates.json');queries={r['audit_query_id']:r for r in rows}
    if len(queries)!=len(rows):raise ValueError('Audit query ID collision')
    options=['--threads','16','-s','7.5','-e','0.001','--min-seq-id','0.2','--max-seqs','512',
             '--split-memory-limit','24G','--format-output',FORMAT]
    for name,target in [('training',reference['database_prefix']),('self',str(root/INPUT/'WT_candidates.fasta')),
                        ('development',str(out/'reviewed_development.fasta'))]:
        run=subprocess.run([str(MMSEQS),'easy-search',str(root/INPUT/'WT_candidates.fasta'),target,
            str(out/(name+'_hits.tsv')),str(out/('tmp_'+name))]+options,capture_output=True,text=True)
        (out/(name+'_search.log')).write_text(run.stdout+run.stderr)
        if run.returncode:raise RuntimeError('MMseqs '+name+' failed; retain log')
        print('CANDIDATE_HOMOLOGY_SEARCH_COMPLETE '+name,flush=True)
    training=dict(fasta_records(reference['fasta_path']))
    if len(training)!=269915:raise ValueError('Unexpected original training roster')
    sha_to_ids=defaultdict(list)
    for key,seq in training.items():sha_to_ids[hashlib.sha256(seq.encode()).hexdigest()].append(key)
    train_hits=defaultdict(list)
    with (out/'training_hits.tsv').open() as handle:
        for line in handle:
            fields=line.rstrip('\n').split('\t');h=alignment(fields,queries)
            if h['target'] not in training or fields[15].replace('-','')!=training[h['target']][int(fields[7])-1:int(fields[8])]:
                raise ValueError('Training DB and authoritative training FASTA alignment disagree')
            train_hits[h['query']].append(h)
    dev_hits=defaultdict(list)
    for line in (out/'development_hits.tsv').read_text().splitlines():
        h=alignment(line.split('\t'),queries);dev_hits[h['query']].append(h)
    components=Components([r['candidate_id'] for r in rows]);groups=defaultdict(list)
    for r in rows:
        groups['sha',r['WT_sha256']].append(r['candidate_id'])
        if r.get('cath_family'):groups['CATH_H',r['cath_family']].append(r['candidate_id'])
        if r.get('source_family'):groups['MCSA_source_family',r['source_family']].append(r['candidate_id'])
    for ids in groups.values():
        for other in ids[1:]:components.union(ids[0],other)
    self_edges=0
    for line in (out/'self_hits.tsv').read_text().splitlines():
        h=alignment(line.split('\t'),queries)
        if h['query']!=h['target'] and passes(h,.3):
            components.union(queries[h['query']]['candidate_id'],queries[h['target']]['candidate_id']);self_edges+=1
    audit=[]
    rank=lambda h:(h['identity_shorter_full'],h['bits'],h['target'])
    for qid,q in queries.items():
        hits=train_hits[qid];query80=[h for h in hits if h['query_coverage']>=.8]
        both80=[h for h in hits if min(h['query_coverage'],h['target_coverage'])>=.8]
        audit.append(dict(candidate_id=q['candidate_id'],audit_query_id=qid,role=q['role'],WT_sha256=q['WT_sha256'],
            training_exact_sequence_ids=sha_to_ids.get(q['WT_sha256'],[]),training_hits=len(hits),hit_cap_reached=len(hits)>=512,
            nearest_returned_any=max(hits,key=rank) if hits else None,
            nearest_returned_query80=max(query80,key=rank) if query80 else None,
            nearest_returned_both80=max(both80,key=rank) if both80 else None,
            training_both80_identity40_hits=sum(passes(h,.4) for h in hits),
            training_query80_identity40_hits=sum(h['identity_shorter_full']>=.4 and h['query_coverage']>=.8 for h in hits),
            training_high_query_low_target_domain_like_hits=sum(h['identity_shorter_full']>=.4 and h['query_coverage']>=.8 and h['target_coverage']<.8 for h in hits),
            development_related_hits=sum(h['identity_shorter_full']>=.3 and max(h['query_coverage'],h['target_coverage'])>=.8 for h in dev_hits[qid]),
            evaluation_sequence_component=components.find(q['candidate_id']),model_use_allowed=False))
    write(out/'per_candidate_audit.json',audit)
    report=dict(status='INPUT_EXPOSURE_AND_COMPONENT_AUDIT_COMPLETE_NOT_MODEL_RELEASE',candidates=len(rows),
        candidate_unique_WT=len({r['WT_sha256'] for r in rows}),training_WT=len(training),
        exact_training_sequences=sum(bool(r['training_exact_sequence_ids']) for r in audit),
        candidates_with_domain_like_training_hits=sum(r['training_high_query_low_target_domain_like_hits']>0 for r in audit),
        candidates_related_to_development=sum(r['development_related_hits']>0 for r in audit),
        evaluation_sequence_components=len({r['evaluation_sequence_component'] for r in audit}),
        self_edges=self_edges,roles=dict(Counter(r['role'] for r in rows)),new_model_calls=0,
        no_variant_or_retrieval_outcome_selection=True,approximate_homology_search_not_family_disjoint_proof=True,
        seconds=time.monotonic()-start,job_id=os.environ['SLURM_JOB_ID'],
        files={p.name:sha(p) for p in out.glob('*') if p.is_file() and p.suffix in ('.json','.tsv','.fasta')})
    write(out/'completion.json',report);print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();(prepare if a.prepare else execute)(a.root)
