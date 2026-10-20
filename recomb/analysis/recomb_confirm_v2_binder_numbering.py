"""Number reviewed input antibodies with existing ANARCI, without guiding edits."""
import argparse
from collections import Counter
import hashlib
import importlib
import os
from pathlib import Path
import sys
import time
from recomb_confirm_v2_diagnostics import read,sha,write

BASE=Path('reports/recomb_confirm_v2_binder_numbering')
FZ=Path('configs/recomb_confirm_v2_binder_numbering_freeze.json')
ENV=Path('/public/home/zhangyangroup/chengshiz/anaconda3/envs/igfold310')


def domain_summary(seq,numbering,details):
    residues,start,end=numbering
    reconstructed=''.join(aa for _,aa in residues if aa!='-')
    if not 0<=start<=end<len(seq) or seq[start:end+1]!=reconstructed:raise ValueError('Numbered domain does not reconstruct exact input span')
    position=int(start);cdr=[];mapped=[]
    for (num,insertion),aa in residues:
        if aa=='-':continue
        if seq[position]!=aa:raise ValueError('Numbered residue mapping differs')
        if 105<=int(num)<=117:
            cdr.append(aa);mapped.append(dict(WT_index=position,IMGT=int(num),insertion=str(insertion).strip()))
        position+=1
    if position!=end+1 or not cdr:raise ValueError('Incomplete domain/CDR3 mapping')
    g=details.get('germlines') or {}
    def gene(which):
        v=g.get(which) or [None,None]
        return dict(model_species=v[0][0],name=v[0][1],identity=v[1]) if v[0] else None
    return dict(chain_type=details['chain_type'],start=int(start),end=int(end),domain_sequence_sha256=hashlib.sha256(reconstructed.encode()).hexdigest(),
        CDR3=''.join(cdr),CDR3_WT_positions=mapped,V_gene=gene('v_gene'),J_gene=gene('j_gene'),
        bitscore=details.get('bitscore'),HMM_species_for_numbering_only=details.get('species'),
        biological_species_not_inferred=True)


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('Numbering already frozen')
    module=importlib.import_module('anarci.anarci');package=Path(module.__file__).parent
    paths=['analysis/recomb_confirm_v2_binder_numbering.py','analysis/recomb_confirm_v2_diagnostics.py',
        'analysis/tests/test_recomb_confirm_v2_binder_numbering.py','slurm/recomb_confirm_v2_binder_numbering.sbatch',
        'reports/recomb_confirm_v2_binder_components/numbering_inputs.json',
        'reports/recomb_confirm_v2_binder_components/input_completion.json']
    assets={str(p):sha(p) for p in package.glob('*.py')}
    assets.update({str(p):sha(p) for p in (package/'dat/HMMs').glob('ALL.hmm*')})
    assets[str(ENV/'bin/hmmscan')]=sha(ENV/'bin/hmmscan')
    write(root/FZ,dict(status='EXISTING_ANARCI_IMGT_INPUT_NUMBERING_FROZEN',files={p:sha(root/p) for p in paths},
        existing_tools_and_profiles=assets,python=str(ENV/'bin/python'),hmmerpath=str(ENV/'bin'),
        scheme='imgt',bit_score_threshold=80,assign_germline=True,ncpu=4,allow=['H','K','L'],
        CDR3='rearranged IMGT105to117 including all insertion codes, original returned residue order',
        biological_species_not_inferred_from_HMM=True,no_deletion_guidance_or_selector_use=True,
        molecular_clonotype_proxy_not_proof_of_Bcell_lineage=True,model_release_allowed=False,
        references=['https://github.com/oxpig/ANARCI','https://www.imgt.org/IMGTScientificChart/Nomenclature/IMGT-FRCDRdefinition.html']))
    (root/BASE).mkdir(parents=True,exist_ok=True);print(sha(root/FZ),flush=True)


def run(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm only')
    f=read(root/FZ)
    for p,h in f['files'].items():
        if sha(root/p)!=h:raise ValueError('Numbering input/source changed')
    for p,h in f['existing_tools_and_profiles'].items():
        if sha(p)!=h:raise ValueError('Read-only ANARCI/HMMER asset changed')
    out=root/BASE
    if (out/'execution_claim.json').exists():raise FileExistsError('Numbering claimed; inspect current job')
    write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID']))
    from anarci import anarci
    queries=read(root/'reports/recomb_confirm_v2_binder_components/numbering_inputs.json')
    records=[(q['query_id'],q['sequence']) for q in queries];start=time.monotonic()
    numbered,details,hits=anarci(records,scheme='imgt',allow={'H','K','L'},hmmerpath=f['hmmerpath'],ncpu=4,
        assign_germline=True,allowed_species=None,bit_score_threshold=80,output=False)
    if len(numbered)!=len(records) or len(details)!=len(records):raise ValueError('Incomplete numbering response')
    audit=[]
    for q,domains,info in zip(queries,numbered,details):
        row=dict(query_id=q['query_id'],sequence_sha256=q['sequence_sha256'],uses=q['uses'])
        if not domains:row.update(status='NO_IG_VARIABLE_DOMAIN_NUMBERED',domains=[])
        else:
            try:
                parsed=[domain_summary(q['sequence'],n,d) for n,d in zip(domains,info)]
                if len(parsed)!=len(domains) or any(d['chain_type'] not in ('H','K','L') for d in parsed):raise ValueError('Unexpected numbered chain type')
                row.update(status='NUMBERED_IG_VARIABLE_DOMAINS_AND_EXACT_CDR3_SPANS',domains=parsed)
            except (ValueError,KeyError,TypeError,IndexError) as e:row.update(status='NUMBERING_MAPPING_HOLD',reason=str(e),domains=[])
        audit.append(row)
    write(out/'numbering_audit.json',audit)
    write(out/'raw_numbering_and_details.json',dict(query_ids=[q['query_id'] for q in queries],numbering=numbered,details=details))
    done=dict(status='EXISTING_ANARCI_NUMBERING_COMPLETE_NOT_CLONOTYPE_LINEAGE_OR_MODEL_RELEASE',
        queries=len(queries),disposition=dict(Counter(r['status'] for r in audit)),
        multi_H_domains=sum(sum(d['chain_type']=='H' for d in r['domains'])>1 for r in audit),seconds=time.monotonic()-start,
        job_id=os.environ['SLURM_JOB_ID'],new_neural_or_PLM_calls=0,new_training=False,new_structure_predictions=0,
        new_tools_installed=False,new_profiles_downloaded=False,output_sha256=sha(out/'numbering_audit.json'))
    write(out/'completion.json',done);print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();prepare(a.root) if a.prepare else run(a.root)
