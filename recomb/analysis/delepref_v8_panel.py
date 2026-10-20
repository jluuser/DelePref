"""Input-only v8 cohort preparation and original-training homology audit."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013')
SOURCE=Path('/public/nfs-data/keyuan.zhou/evocompress/data/structure_240_v1/final_v1_1')
BASE='reports/delepref_v8_panel'
FREEZE='configs/delepref_v8_panel_input_protocol.json'
MMSEQS=Path('/public/nfs-data/keyuan.zhou/evocompress/tools/mmseqs/bin/mmseqs')
FORMAT='query,target,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen,evalue,bits,qcov,tcov,qaln,taln'
CONTRACT=dict(version='v8_input_panel_protocol_1',target_WT=100,length_min=150,length_max=550,
    standard_amino_acids_only=True,exclude_legacy_exact_sequences=True,exclude_legacy_PDB_entries=True,
    exclude_legacy_known_CATH_H_families=True,training_identity_shorter_full_min=.4,
    training_both_coverage_min=.8,panel_identity_shorter_full_component_min=.3,
    panel_both_coverage_component_min=.8,panel_one_per_component=True,
    order='SHA256(recomb-v8-panel-20261016:sequence_sha256:candidate_id)',
    search=dict(sensitivity=7.5,evalue=.001,min_seq_id=.2,max_seqs=512,threads=16,split_memory_limit='24G'),
    no_model_output_used=True,no_retrieval_success_selection=True,
    reference_geometry_validation='pending_before_final_confirmation_freeze',
    approximate_homology_search_not_exhaustive_family_proof=True)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def rows(p):
    with Path(p).open() as f:return [json.loads(x) for x in f if x.strip()]

def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,sort_keys=True);f.write('\n')

def families(row):
    out=set()
    for key in ['cath_h_superfamily','cath_homologous_superfamily','cath_h_superfamilies']:
        value=row.get(key)
        if isinstance(value,str):out.add(value)
        elif isinstance(value,list):out.update(str(x) for x in value)
    return {x for x in out if x and x not in ('None','NA','unmapped')}

def pdb_identity(row):
    return str(row.get('pdb_id') or row.get('domain_id','')[:4]).upper()

def pure_source(row,source_file):
    allowed=['candidate_id','protein_id','sequence','sequence_length','sequence_sha1','stratum',
        'cath_homologous_superfamily','cath_h_superfamilies','domain_id','pdb_id','chain_id',
        'coordinate_path','structure_path','archive_member','sequence_homology_component',
        'family_cluster_id','uniprot_accession','resolved_canonical_fraction','resolution_angstrom']
    out={k:row[k] for k in allowed if k in row}
    out['source_file']=str(source_file)
    sequence=out['sequence']
    if hashlib.sha1(sequence.encode()).hexdigest()!=out['sequence_sha1']:
        raise ValueError('source sequence SHA1 mismatch')
    if len(sequence)!=int(out['sequence_length']):raise ValueError('source length mismatch')
    out['sequence_sha256']=hashlib.sha256(sequence.encode()).hexdigest()
    return out

def exclusion(row,legacy):
    seq=row['sequence']
    if not CONTRACT['length_min']<=len(seq)<=CONTRACT['length_max']:return 'length'
    if not set(seq)<=set('ACDEFGHIKLMNPQRSTVWY'):return 'nonstandard_amino_acids'
    if row['sequence_sha256'] in legacy['sha']:return 'legacy_exact_sequence'
    if pdb_identity(row) in legacy['pdb']:return 'legacy_PDB_entry'
    if families(row)&set(legacy['families']):return 'legacy_CATH_H_family'
    return None

class Components:
    def __init__(self,ids):self.parent={x:x for x in ids}
    def find(self,x):
        if self.parent[x]!=x:self.parent[x]=self.find(self.parent[x])
        return self.parent[x]
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)

def alignment(fields,queries):
    if len(fields)!=16:raise ValueError('wrong MMseqs alignment schema')
    q,t,ident,alnlen,qs,qe,ql,ts,te,tl,ev,bits,qc,tc,qa,ta=fields
    qs,qe,ql,ts,te,tl=map(int,[qs,qe,ql,ts,te,tl])
    if not (0<=float(ident)<=1 and math.isfinite(float(bits)) and math.isfinite(float(ev))):raise ValueError('nonfinite/wrong-unit MMseqs fields')
    if not (1<=qs<=qe<=ql and 1<=ts<=te<=tl):raise ValueError('alignment coordinates out of range')
    if q not in queries or ql!=len(queries[q]['sequence']) or len(qa)!=len(ta):raise ValueError('alignment identity/dimension mismatch')
    if qa.replace('-','')!=queries[q]['sequence'][qs-1:qe]:raise ValueError('query alignment does not reconstruct WT')
    if sum(x!='-' for x in ta)!=te-ts+1:raise ValueError('target coordinate mismatch')
    matches=sum(a==b and a!='-' for a,b in zip(qa,ta))
    aligned=sum(a!='-' and b!='-' for a,b in zip(qa,ta))
    return dict(query=q,target=t,identity_shorter_full=matches/min(ql,tl),
        aligned_residue_identity=matches/max(1,aligned),query_coverage=(qe-qs+1)/ql,
        target_coverage=(te-ts+1)/tl,matches=matches,query_length=ql,target_length=tl,
        bits=float(bits),evalue=float(ev))

def passes(hit,identity):
    return hit['identity_shorter_full']>=identity and min(hit['query_coverage'],hit['target_coverage'])>=.8

def select(candidates,rejected,components,limit=100):
    chosen=[];used=set()
    order=sorted(candidates,key=lambda r:hashlib.sha256(
        ('recomb-v8-panel-20261016:'+r['sequence_sha256']+':'+r['candidate_id']).encode()).hexdigest())
    for row in order:
        cid=row['candidate_id'];component=components.find(cid)
        if cid in rejected or component in used:continue
        chosen.append(dict(row,panel_component=component));used.add(component)
        if len(chosen)>=limit:break
    return chosen

def prepare(root):
    out=root/BASE;out.mkdir(exist_ok=False)
    source_files=[SOURCE/x for x in ['structure_240.jsonl','pdb_reserves.jsonl','cath_reserves.jsonl']]
    old_files=[root/'manifests/development_primary.jsonl',root/'manifests/confirmation_wt_frozen.jsonl']
    legacy=dict(sha=set(),pdb=set(),families=set())
    # Only WT identity, native source and family fields are used; no variant outcome enters selection.
    for file in old_files:
        for r in rows(file):
            legacy['sha'].add(r['wt_sequence_sha256'])
            legacy['pdb'].add(str(r['pdb_id']).upper())
            legacy['families'].update(families(r))
    candidates=[];excluded=[];seen={}
    for file in source_files:
        for raw in rows(file):
            if raw.get('selection_outputs_used',False):raise ValueError('source roster used model outputs')
            row=pure_source(raw,file)
            reason=exclusion(row,legacy)
            if reason:excluded.append(dict(candidate_id=row['candidate_id'],reason=reason));continue
            path=row.get('coordinate_path') or row.get('structure_path')
            if path and not Path(path).is_file():
                excluded.append(dict(candidate_id=row['candidate_id'],reason='native_file_missing'));continue
            if not path and not row.get('archive_member'):
                excluded.append(dict(candidate_id=row['candidate_id'],reason='native_source_unresolved'));continue
            cid=row['candidate_id']
            if cid in seen:
                if seen[cid]['sequence_sha256']!=row['sequence_sha256']:raise ValueError('candidate ID collision')
                continue
            seen[cid]=row;candidates.append(row)
    external_file=root/'external/rag_v6/preparation/primary_rows.jsonl'
    external={}
    for row in rows(external_file):
        target,seq=row['target'],row['wt_sequence']
        if target in external and external[target]!=seq:raise ValueError('multiple external WT backgrounds')
        external[target]=seq
    by_sha={}
    for c in candidates:
        by_sha.setdefault(c['sequence_sha256'],dict(sequence=c['sequence'],candidate_ids=[],external_targets=[]))['candidate_ids'].append(c['candidate_id'])
    for target,seq in sorted(external.items()):
        digest=hashlib.sha256(seq.encode()).hexdigest()
        by_sha.setdefault(digest,dict(sequence=seq,candidate_ids=[],external_targets=[]))['external_targets'].append(target)
    queries=[dict(query_id='auditq_%04d'%i,sequence_sha256=digest,**r) for i,(digest,r) in enumerate(sorted(by_sha.items()))]
    write(out/'candidate_universe.json',candidates);write(out/'exclusions.json',excluded);write(out/'queries.json',queries)
    with (out/'queries.fasta').open('x') as f:
        for q in queries:f.write('>'+q['query_id']+'\n'+q['sequence']+'\n')
    reference=root/'reports/delepref_v8/train_reference_completion.json'
    sources={str(p):sha(p) for p in source_files+old_files+[external_file,reference]}
    for p in ['analysis/delepref_v8_panel.py','analysis/tests/test_delepref_v8_panel.py','slurm/delepref_v8_panel.sbatch']:
        sources[p]=sha(root/p)
    protocol=dict(status='V8_PANEL_INPUT_PROTOCOL_FROZEN_BEFORE_HOMOLOGY_RESULTS',contract=CONTRACT,
        sources=sources,prepared={str(p):sha(p) for p in [out/'candidate_universe.json',out/'queries.json',out/'queries.fasta']},
        source_rows=sum(len(rows(p)) for p in source_files),basic_candidates=len(candidates),
        queries=len(queries),external_targets=len(external),exclusion_counts=dict(Counter(x['reason'] for x in excluded)),
        native_geometry_verified=False,model_or_variant_structure_outcomes_used=False)
    write(root/FREEZE,protocol)
    return dict(freeze_sha256=sha(root/FREEZE),basic_candidates=len(candidates),queries=len(queries),exclusion_counts=protocol['exclusion_counts'])

def execute(root,checksum):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm allocation required')
    if sha(root/FREEZE)!=checksum:raise ValueError('panel protocol changed')
    protocol=json.loads((root/FREEZE).read_text())
    if protocol['contract']!=CONTRACT:raise ValueError('panel rules changed')
    for p,h in {**protocol['sources'],**protocol['prepared']}.items():
        if sha(root/p)!=h:raise ValueError('panel source/input changed: '+p)
    out=root/BASE;write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID']))
    start=time.time();reference=json.loads((root/'reports/delepref_v8/train_reference_completion.json').read_text())
    if sha(Path(reference['fasta_path']))!=reference['fasta_sha256'] or sha(MMSEQS)!=reference['mmseqs_binary_sha256']:raise ValueError('training reference/binary changed')
    fasta=out/'queries.fasta'
    options=['--threads','16','-s','7.5','-e','0.001','--min-seq-id','0.2','--max-seqs','512','--split-memory-limit','24G','--format-output',FORMAT]
    for label,target in [('train',reference['database_prefix']),('self',str(fasta))]:
        command=[str(MMSEQS),'easy-search',str(fasta),target,str(out/(label+'_hits.tsv')),str(out/('tmp_'+label))]+options
        run=subprocess.run(command,text=True,capture_output=True)
        (out/(label+'_search.log')).write_text(run.stdout+run.stderr)
        if run.returncode:raise RuntimeError(label+' homology search failed; retain logs')
        print(json.dumps(dict(event='HOMOLOGY_SEARCH_COMPLETE',target=label)),flush=True)
    queries={q['query_id']:q for q in json.loads((out/'queries.json').read_text())}
    candidates=json.loads((out/'candidate_universe.json').read_text())
    hits=defaultdict(list)
    with (out/'train_hits.tsv').open() as f:
        for line in f:
            h=alignment(line.rstrip('\n').split('\t'),queries);hits[h['query']].append(h)
    rejected=set();audits=[]
    for qid,q in queries.items():
        hs=hits[qid];barrier=[h for h in hs if passes(h,.4)]
        covered=[h for h in hs if min(h['query_coverage'],h['target_coverage'])>=.8]
        rank=lambda h:(h['identity_shorter_full'],h['bits'],h['target'])
        audits.append(dict(query_id=qid,sequence_sha256=q['sequence_sha256'],candidate_ids=q['candidate_ids'],external_targets=q['external_targets'],
            retrieved_training_hits=len(hs),at_configured_hit_cap=len(hs)>=512,threshold_hits=len(barrier),
            nearest_retrieved_any=max(hs,key=rank) if hs else None,
            nearest_retrieved_both_covered=max(covered,key=rank) if covered else None))
        if barrier:rejected.update(q['candidate_ids'])
    components=Components([x['candidate_id'] for x in candidates])
    groups=defaultdict(list)
    for row in candidates:
        cid=row['candidate_id'];groups[('sha',row['sequence_sha256'])].append(cid)
        pid=pdb_identity(row)
        if pid:groups[('pdb',pid)].append(cid)
        for family in families(row):groups[('CATH_H',family)].append(cid)
        if row.get('sequence_homology_component'):
            groups[('source_component',row['source_file'],row['sequence_homology_component'])].append(cid)
    for ids in groups.values():
        for other in ids[1:]:components.union(ids[0],other)
    self_edges=0
    with (out/'self_hits.tsv').open() as f:
        for line in f:
            h=alignment(line.rstrip('\n').split('\t'),queries)
            if h['query']==h['target'] or not passes(h,.3):continue
            qa,qb=queries[h['query']],queries[h['target']]
            for a in qa['candidate_ids']:
                for b in qb['candidate_ids']:components.union(a,b)
            self_edges+=1
    selected=select(candidates,rejected,components)
    write(out/'training_homology_audit.json',dict(training_WT=269915,queries=audits,contract=CONTRACT,
        scope='nearest returned MMseqs hits; not exhaustive pairwise or remote-homology proof'))
    write(out/'provisional_panel.json',selected)
    report=dict(status='V8_PANEL_HOMOLOGY_PREPARATION_COMPLETE_NOT_CONFIRMATION_FREEZE',
        job_id=os.environ['SLURM_JOB_ID'],seconds=time.time()-start,protocol_sha256=checksum,
        basic_candidates=len(candidates),training_threshold_rejected=len(rejected),
        eligible_components=len({components.find(x['candidate_id']) for x in candidates if x['candidate_id'] not in rejected}),
        provisional_WT=len(selected),source_types=dict(Counter(x['stratum'] for x in selected)),self_homology_edges=self_edges,
        external_targets_audited=sum(len(q['external_targets']) for q in queries.values()),
        external_targets_with_training_threshold_hits=sum(len(q['external_targets']) for q in audits if q['threshold_hits']),
        external_rows_dropped=0,native_geometry_validation_complete=False,
        next='Validate native reference mappings and final cohort/structure protocol before confirmation outputs',
        audit_sha256=sha(out/'training_homology_audit.json'),panel_sha256=sha(out/'provisional_panel.json'),
        train_hits_sha256=sha(out/'train_hits.tsv'),self_hits_sha256=sha(out/'self_hits.tsv'),
        new_GPU_jobs=0,new_structure_predictions=0,paper_changes=False)
    write(out/'completion.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--prepare',action='store_true');p.add_argument('--freeze-sha256');a=p.parse_args()
    print(json.dumps(prepare(a.root)) if a.prepare else execute(a.root,a.freeze_sha256),flush=True)
