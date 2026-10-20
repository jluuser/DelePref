"""Uniform query-conditioned alignment of unchanged reviewed natural donor fragments."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import os
from pathlib import Path
import time
import numpy as np
import Bio
from Bio.Align import PairwiseAligner,substitution_matrices
from recomb_confirm_v2_diagnostics import read,sha,write
from recomb_confirm_v2_audit_msa import records

BASE=Path('reports/recomb_confirm_v2_uniform_WT_MSA')
FZ=Path('configs/recomb_confirm_v2_uniform_WT_MSA_freeze.json')
ALPHABET=set('ACDEFGHIKLMNPQRSTVWY')


def digest(value):
    import json
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def make_aligner():
    a=PairwiseAligner();a.mode='global';a.substitution_matrix=substitution_matrices.load('BLOSUM62')
    a.open_gap_score=-10.;a.extend_gap_score=-.5
    return a


def align_donor(query,donor,aligner=None):
    if not query or not donor or set(query)-ALPHABET or set(donor)-ALPHABET:
        raise ValueError('Uniform MSA inputs must have established standard amino-acid identity')
    a=aligner or make_aligner();alignment=next(iter(a.align(query,donor)))
    coords=np.asarray(alignment.coordinates,dtype=int);result=[]
    for i in range(coords.shape[1]-1):
        q0,q1=coords[0,i:i+2];d0,d1=coords[1,i:i+2]
        if q1==q0:result.append(donor[d0:d1].lower())
        elif d1==d0:result.append('-'*(q1-q0))
        elif q1-q0==d1-d0:result.append(donor[d0:d1])
        else:raise ValueError('Unexpected pairwise alignment segment')
    aligned=''.join(result)
    if aligned.replace('-','').upper()!=donor:raise ValueError('Natural donor residues changed during alignment')
    if len(''.join(c for c in aligned if not c.islower()))!=len(query):raise ValueError('Query column count changed')
    return aligned


def aligned_rows(query,source_rows):
    if not source_rows:raise ValueError('Reviewed source MSA is empty')
    result=[('query',query)];a=make_aligner()
    for header,aligned in source_rows[1:]:
        donor=aligned.replace('-','').upper()
        result.append((header,align_donor(query,donor,a)))
    return result


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('Uniform WT MSA protocol already frozen')
    review=root/'external/recomb_confirm_v2_application_WT_MSA_reviewed'
    done=read(review/'completion.json');manifest=root/'external/recomb_confirm_v2_application_WT_MSA_staging/request_manifest.json'
    if done['unique_reviewed_queries']!=180:raise ValueError('Reviewed WT MSA roster incomplete')
    paths=['analysis/recomb_confirm_v2_uniform_msa.py','analysis/recomb_confirm_v2_audit_msa.py',
        'analysis/recomb_confirm_v2_diagnostics.py','analysis/tests/test_recomb_confirm_v2_uniform_msa.py',
        'slurm/recomb_confirm_v2_uniform_WT_MSA.sbatch',str(manifest.relative_to(root)),
        str((review/'completion.json').relative_to(root)),str((review/'review_policy.json').relative_to(root)),
        str((review/'application_review_completion.json').relative_to(root))]
    sources={str((review/'a3m'/(r['query_sha256']+'.a3m')).relative_to(root)):r['sha256'] for r in done['files']}
    for p,h in sources.items():
        if sha(root/p)!=h:raise ValueError('Reviewed WT/donor identity changed')
    matrix=substitution_matrices.load('BLOSUM62')
    write(root/FZ,dict(status='UNIFORM_REAL_DONOR_MSA_ALIGNMENT_FROZEN_BEFORE_NEW_VARIANT_STRUCTURES',
        files={p:sha(root/p) for p in paths},reviewed_sources=sources,Biopython_version=Bio.__version__,
        alignment='global affine, BLOSUM62, open -10, extend -.5 for all target/query terminal and internal gaps',
        matrix_sha256=hashlib.sha256(np.asarray(matrix,dtype=np.float64).tobytes()).hexdigest(),
        matrix_alphabet=matrix.alphabet,ties='first deterministic optimum from pinned Biopython; uniqueness not claimed',
        donor_policy='preserve every amino acid of each reviewed donor fragment; donor insertions lowercase; query-only columns are donor gaps',
        applies_equally_to_WT_and_all_six_methods=True,no_variant_specific_retrieval=True,no_native_templates=True,
        no_homolog_column_deletion_to_simulate_natural_events=True,MSA_pairing_key=-1,
        raw_standard_MSA_not_used=True,metadata_source_filter_unchanged=True,new_structure_predictions=0))
    (root/BASE).mkdir(parents=True,exist_ok=True);print(sha(root/FZ),flush=True)


def process(item):
    root_s,q,source,expected,protocol_sha=item;root=Path(root_s);path=root/source
    if sha(path)!=expected:raise ValueError('Reviewed donor MSA changed')
    rows=records(path);seq=q['sequence'];h=q['sequence_sha256']
    if rows[0][1]!=seq or hashlib.sha256(seq.encode()).hexdigest()!=h:raise ValueError('Reviewed WT query identity changed')
    output=aligned_rows(seq,rows)
    donor_hashes=[hashlib.sha256(s.replace('-','').upper().encode()).hexdigest() for _,s in output[1:]]
    identity=dict(query_sha256=h,source_MSA_sha256=expected,alignment_protocol_sha256=protocol_sha)
    key=digest(identity);dest=root/BASE/'csv'/(key+'.csv');dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['key','sequence'])
        for _,aligned in output:w.writerow([-1,aligned])
    return dict(identity,cache_key=key,CSV_relative_path=str(dest.relative_to(root)),CSV_sha256=sha(dest),
        rows=len(output),query_length=len(seq),donor_fragment_hashes=donor_hashes,
        all_donor_amino_acids_preserved=True,query_only=len(output)==1)


def run(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm only')
    f=read(root/FZ)
    for p,h in f['files'].items():
        if sha(root/p)!=h:raise ValueError('Uniform MSA source or inputs changed')
    if Bio.__version__!=f['Biopython_version']:raise ValueError('Alignment implementation version changed')
    out=root/BASE
    if (out/'execution_claim.json').exists():raise FileExistsError('Uniform WT MSA stage claimed; inspect job')
    write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID']));start=time.monotonic()
    queries=read(root/'external/recomb_confirm_v2_application_WT_MSA_staging/request_manifest.json')['queries']
    review=root/'external/recomb_confirm_v2_application_WT_MSA_reviewed'
    jobs=[]
    for q in queries:
        path=str((review/'a3m'/(q['sequence_sha256']+'.a3m')).relative_to(root))
        jobs.append((str(root),q,path,f['reviewed_sources'][path],sha(root/FZ)))
    with ProcessPoolExecutor(max_workers=8) as executor:result=list(executor.map(process,jobs))
    write(out/'csv_manifest.json',result)
    done=dict(status='UNIFORM_WT_MSA_ALIGNED_AND_DONOR_CONSERVATION_PASS_NATIVE_BOLTZ_PARSER_PENDING',
        unique_queries=len(result),rows=sum(r['rows'] for r in result),query_only=sum(r['query_only'] for r in result),
        all_reviewed_donor_fragments_preserved=True,manifest_sha256=sha(out/'csv_manifest.json'),
        freeze_sha256=sha(root/FZ),job_id=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-start,
        new_model_calls=0,new_structure_predictions=0)
    write(out/'completion.json',done);print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();prepare(a.root) if a.prepare else run(a.root)
