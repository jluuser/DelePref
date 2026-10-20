"""Input-only Binder training exposure and antigen homology audit."""
import argparse
from collections import Counter,defaultdict
import hashlib
import os
from pathlib import Path
import subprocess
import time
from recomb_confirm_v2_diagnostics import read,sha,write
from recomb_confirm_v2_candidate_homology import fasta_records
from delepref_v8_panel import MMSEQS,FORMAT,alignment,passes

BASE=Path('reports/recomb_confirm_v2_binder_exposure')
FZ=Path('configs/recomb_confirm_v2_binder_exposure_freeze.json')
INPUT=Path('reports/recomb_confirm_v2_binder_components')


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('Binder exposure already frozen')
    paths=['analysis/recomb_confirm_v2_binder_exposure.py','analysis/recomb_confirm_v2_diagnostics.py',
        'analysis/recomb_confirm_v2_candidate_homology.py','analysis/delepref_v8_panel.py',
        'slurm/recomb_confirm_v2_binder_exposure.sbatch',
        str(INPUT/'input_completion.json'),str(INPUT/'query_inputs.json'),str(INPUT/'queries.fasta'),str(INPUT/'case_inputs.json'),
        'reports/delepref_v8/train_reference_completion.json']
    write(root/FZ,dict(status='BINDER_INPUT_EXPOSURE_AND_COMPONENT_RULES_FROZEN',files={p:sha(root/p) for p in paths},
        search=dict(sensitivity=7.5,min_identity=.2,evalue=.001,max_hits=512,threads=16,memory='24G'),
        training_flags='exact complete SHA; identity_shorter_full>=.4/query80; report both80 and domain-like separately',
        antigen_edges='same reviewed UniProt accession OR min full length>=50 and identity_shorter_full>=.3/both coverage80',
        short_antigens='exact sequence/source identifier grouping only; no low-length homology edge',
        antibody_framework_identity_not_a_clonotype_or_blanket_development_exclusion=True,
        clone_proxy='ANARCI H CDR3+V/J gene-family exact and same-length CDR3>=90% identity; group proxy not proven lineage',
        hard_identity_checks='stored excluded participant hashes; complete trained design WT SHA; prior development/General100 exact design SHA',
        prospective_old_binder_antigen_definition='legacy Binder20 participants: numbered H present, non-numbered-IG participants are candidate antigen references; audit uncertainty retained',
        no_model_retrieval_or_structure_outcomes_used=True,model_release_allowed=False))
    (root/BASE).mkdir(parents=True,exist_ok=True);print(sha(root/FZ),flush=True)


def run(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm only')
    f=read(root/FZ)
    for p,h in f['files'].items():
        if sha(root/p)!=h:raise ValueError('Exposure input/source changed')
    out=root/BASE
    if (out/'execution_claim.json').exists():raise FileExistsError('Exposure audit claimed; inspect existing job')
    write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID']))
    reference=read(root/'reports/delepref_v8/train_reference_completion.json')
    if sha(reference['fasta_path'])!=reference['fasta_sha256'] or sha(MMSEQS)!=reference['mmseqs_binary_sha256']:
        raise ValueError('Original training reference or MMseqs changed')
    source=read(root/INPUT/'query_inputs.json');queries={r['query_id']:r for r in source}
    if len(queries)!=len(source):raise ValueError('Query IDs collide')
    start=time.monotonic()
    options=['--threads','16','-s','7.5','-e','0.001','--min-seq-id','0.2','--max-seqs','512',
        '--split-memory-limit','24G','--format-output',FORMAT]
    for name,target in [('training',reference['database_prefix']),('self',str(root/INPUT/'queries.fasta'))]:
        with (out/(name+'_search.log')).open('w') as log:
            subprocess.run([str(MMSEQS),'easy-search',str(root/INPUT/'queries.fasta'),target,
                str(out/(name+'_hits.tsv')),str(out/('tmp_'+name))]+options,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('BINDER_EXPOSURE_SEARCH_DONE '+name,flush=True)
    training=dict(fasta_records(reference['fasta_path']))
    if len(training)!=269915:raise ValueError('Wrong original training roster')
    exact=defaultdict(list)
    for key,seq in training.items():exact[hashlib.sha256(seq.encode()).hexdigest()].append(key)
    stats={q:dict(query_id=q,sequence_sha256=r['sequence_sha256'],uses=r['uses'],training_hits=0,
        exact_training_sequence_ids=exact.get(r['sequence_sha256'],[]),query80_identity40_hits=0,
        both80_identity40_hits=0,domain_like_training_hits=0,nearest_returned=None,nearest_query80=None,nearest_both80=None) for q,r in queries.items()}
    rank=lambda h:(h['identity_shorter_full'],h['bits'],h['target'])
    with (out/'training_hits.tsv').open() as handle:
        for line in handle:
            fields=line.rstrip('\n').split('\t');h=alignment(fields,queries);s=stats[h['query']]
            if h['target'] not in training or fields[15].replace('-','')!=training[h['target']][int(fields[7])-1:int(fields[8])]:
                raise ValueError('Training target does not reconstruct authoritative WT')
            s['training_hits']+=1
            flags={'nearest_returned':True,'nearest_query80':h['query_coverage']>=.8,
                   'nearest_both80':min(h['query_coverage'],h['target_coverage'])>=.8}
            for key,allowed in flags.items():
                if allowed and (s[key] is None or rank(h)>rank(s[key])):s[key]=h
            s['query80_identity40_hits']+=int(h['identity_shorter_full']>=.4 and h['query_coverage']>=.8)
            s['both80_identity40_hits']+=int(passes(h,.4))
            s['domain_like_training_hits']+=int(h['identity_shorter_full']>=.4 and h['query_coverage']>=.8 and h['target_coverage']<.8)
    for s in stats.values():s['returned_hit_cap_reached']=s['training_hits']>=512
    ag={q for q,r in queries.items() if any(u['kind']=='candidate' and u['role']=='fixed_antigen' for u in r['uses'])}
    dev={q for q,r in queries.items() if any(u['kind']=='development' for u in r['uses'])}
    ag_edges=[];dev_edges=[]
    with (out/'self_hits.tsv').open() as handle:
        for line in handle:
            fields=line.rstrip('\n').split('\t');h=alignment(fields,queries);q,t=h['query'],h['target']
            if t not in queries or fields[15].replace('-','')!=queries[t]['sequence'][int(fields[7])-1:int(fields[8])]:
                raise ValueError('Self-search target identity differs')
            if q==t:continue
            if min(h['query_length'],h['target_length'])>=50 and passes(h,.3):
                if q in ag and t in ag and q<t:ag_edges.append(h)
                if q in ag and t in dev:dev_edges.append(h)
    write(out/'per_query_exposure.json',stats);write(out/'antigen_sequence_edges.json',ag_edges)
    write(out/'antigen_to_development_edges.json',dev_edges)
    candidate_H={p['query_id'] for c in read(root/INPUT/'case_inputs.json') for p in c['participants'] if p['role']=='design_antibody_H'}
    summary=dict(status='BINDER_TRAINING_EXPOSURE_AND_ANTIGEN_EDGES_COMPLETE_COMPONENT_MERGE_PENDING',
        queries=len(queries),candidate_heavy_unique=len(candidate_H),antigen_unique=len(ag),
        design_exact_student_training=sum(bool(stats[q]['exact_training_sequence_ids']) for q in candidate_H),
        design_domain_like_training_exposures=sum(stats[q]['domain_like_training_hits']>0 for q in candidate_H),
        antigen_homology_edges=len(ag_edges),antigen_development_edges=len(dev_edges),
        framework_matches_not_used_as_automatic_clonotype_or_hard_development_filter=True,
        seconds=time.monotonic()-start,job_id=os.environ['SLURM_JOB_ID'],new_model_calls=0,new_structure_predictions=0,
        outputs={n:sha(out/n) for n in ['per_query_exposure.json','antigen_sequence_edges.json','antigen_to_development_edges.json']})
    write(out/'completion.json',summary);print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();prepare(a.root) if a.prepare else run(a.root)
