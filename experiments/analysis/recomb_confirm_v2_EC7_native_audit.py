"""Native identity and input exposure checks for the reviewed EC7 supplement."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import gemmi
import numpy as np
from recomb_confirm_v2_diagnostics import read,sha,write
from recomb_edit_correspondence_20261017 import consensus
from recomb_confirm_v2_candidate_homology import fasta_records
from delepref_v8_panel import MMSEQS,FORMAT,alignment

BASE=Path('reports/recomb_confirm_v2_EC7_input_audit')
FZ=Path('configs/recomb_confirm_v2_EC7_input_audit_freeze.json')
STAGE=Path('external/recomb_confirm_v2_EC7_native_staging')


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('EC7 audit already frozen')
    row=read(root/STAGE/'candidate_WT.json')
    if not row['native_and_homology_audit_allowed'] or row['PLM_encoding_allowed']:raise ValueError('Wrong stage permission')
    out=root/BASE;out.mkdir(parents=True,exist_ok=True)
    (out/'query.fasta').write_text('>EC7_P04191\n'+row['sequence']+'\n')
    general=read(root/'reports/recomb_confirm_v2_general_native_v2/General100_panel.json')
    with (out/'General100.fasta').open('w') as f:
        for i,r in enumerate(general):f.write('>gen_%03d\n%s\n'%(i,r['sequence']))
    names=['analysis/recomb_confirm_v2_EC7_native_audit.py','analysis/recomb_confirm_v2_diagnostics.py',
        'analysis/recomb_edit_correspondence_20261017.py','analysis/recomb_confirm_v2_candidate_homology.py',
        'analysis/delepref_v8_panel.py','slurm/recomb_confirm_v2_EC7_native_audit.sbatch',
        str(STAGE/'candidate_WT.json'),str(STAGE/'annotation_protocol.json'),str(STAGE/'1SU4.cif.gz'),
        'provenance/recomb_confirm_v2_public_metadata/rcsb_entity/1SU4_1.json',
        'reports/delepref_v8/train_reference_completion.json','reports/recomb_confirm_v2_candidate_homology/reviewed_development.fasta',
        str(BASE/'query.fasta'),str(BASE/'General100.fasta')]
    write(root/FZ,dict(status='ORDINARY_EC7_NATIVE_AND_EXPOSURE_AUDIT_FROZEN_BEFORE_MODELS',files={p:sha(root/p) for p in names},
        minimum_native_CA=.85,all_exact_reviewed_active_sites_required=True,annotation_tier='UniProt reviewed Active site, separate from MCSA catalytic network',
        no_cropping_to_avoid_training_identity=True,PLM_or_variant_generation_allowed=False))
    print(sha(root/FZ),flush=True)


def run(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    f=read(root/FZ)
    for p,h in f['files'].items():
        if sha(root/p)!=h:raise ValueError('Frozen EC7 input/source changed')
    out=root/BASE
    if (out/'execution_claim.json').exists():raise FileExistsError('EC7 audit claimed; inspect current job')
    write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID']));start=time.monotonic()
    q=read(root/STAGE/'candidate_WT.json');seq=q['sequence'];path=root/q['native_path']
    if hashlib.sha256(seq.encode()).hexdigest()!=q['WT_sha256'] or sha(path)!=q['native_sha256']:raise ValueError('WT/native identity differs')
    structure=gemmi.read_structure(str(path));chain=structure[0].find_chain(q['native_chain'])
    if chain is None:raise ValueError('Native chain absent')
    entity=structure.get_entity_of(chain.get_polymer())
    if entity is None or not entity.full_sequence:raise ValueError('Native SEQRES absent')
    full=''.join(gemmi.find_tabulated_residue(s).one_letter_code.upper() for s in entity.full_sequence)
    metadata=read(root/'provenance/recomb_confirm_v2_public_metadata/rcsb_entity/1SU4_1.json')['metadata']
    if hashlib.sha256(full.encode()).hexdigest()!=metadata['polymer_sequence_sha256']:raise ValueError('Native/public entity sequence version differs')
    mp=consensus(seq,full);inverse={j:i for i,j in enumerate(mp['wt_to_edit']) if j is not None and seq[i]==full[j]}
    xyz=[None]*len(seq);residues={};mapping=[]
    for r in chain:
        if r.label_seq is None or int(r.label_seq)-1 not in inverse:continue
        i=inverse[int(r.label_seq)-1];aa=gemmi.find_tabulated_residue(r.name).one_letter_code.upper()
        ca=[a for a in r if a.name=='CA' and a.element.name=='C']
        if aa!=seq[i] or not ca:continue
        if xyz[i] is not None:raise ValueError('Ambiguous native CA correspondence')
        a=ca[0];xyz[i]=[a.pos.x,a.pos.y,a.pos.z];residues[i]=r
        mapping.append(dict(wt_index=i,auth_number=r.seqid.num,label_seq=r.label_seq,residue_name=r.name))
    sites=[]
    for feature in q['Active_site_features']:
        a=feature['location']['start'];b=feature['location']['end']
        if a.get('modifier')!='EXACT' or b.get('modifier')!='EXACT' or a['value']!=b['value']:raise ValueError('Active-site location is not a precise point')
        i=int(a['value'])-1
        if i not in residues or xyz[i] is None:raise ValueError('Declared active-site native CA/identity absent')
        sites.append(dict(wt_index=i,WT_residue=seq[i],auth_number=residues[i].seqid.num,
            description=feature.get('description'),evidences=feature.get('evidences',[]),annotation_source='UniProtKB reviewed Active site'))
    coverage=sum(x is not None for x in xyz)/len(seq)
    if coverage<.85 or not sites:raise ValueError('Native coverage or active-site criterion failed')
    native=dict(WT_sha256=q['WT_sha256'],WT_length=len(seq),native_CA=xyz,residue_mapping=mapping,catalytic=sites,
        known_CA_count=sum(x is not None for x in xyz),unknown_CA_count=sum(x is None for x in xyz),native_CA_coverage=coverage,
        native_file=str(path),native_file_sha256=sha(path),native_chain=q['native_chain'],canonical_to_entity=mp['wt_to_edit'],
        reference_assembly='1',annotation_tier=q['annotation_source'],model_use_allowed=False)
    write(out/'native_reference.json',native)
    ref=read(root/'reports/delepref_v8/train_reference_completion.json')
    if sha(ref['fasta_path'])!=ref['fasta_sha256'] or sha(MMSEQS)!=ref['mmseqs_binary_sha256']:raise ValueError('Original reference changed')
    options=['--threads','16','-s','7.5','-e','0.001','--min-seq-id','0.2','--max-seqs','512','--split-memory-limit','24G','--format-output',FORMAT]
    results={}
    for name,target in [('training',ref['database_prefix']),('development',str(root/'reports/recomb_confirm_v2_candidate_homology/reviewed_development.fasta')),('General100',str(out/'General100.fasta'))]:
        with (out/(name+'_search.log')).open('w') as log:
            subprocess.run([str(MMSEQS),'easy-search',str(out/'query.fasta'),target,str(out/(name+'_hits.tsv')),str(out/('tmp_'+name))]+options,stdout=log,stderr=subprocess.STDOUT,check=True)
        hits=[alignment(line.split('\t'),{'EC7_P04191':dict(sequence=seq)}) for line in (out/(name+'_hits.tsv')).read_text().splitlines()]
        results[name]=dict(hits=len(hits),nearest_returned=max(hits,key=lambda h:(h['identity_shorter_full'],h['bits'])) if hits else None,
            identity30_both80=sum(h['identity_shorter_full']>=.3 and min(h['query_coverage'],h['target_coverage'])>=.8 for h in hits),
            identity30_one80=sum(h['identity_shorter_full']>=.3 and max(h['query_coverage'],h['target_coverage'])>=.8 for h in hits),
            query80_identity40=sum(h['identity_shorter_full']>=.4 and h['query_coverage']>=.8 for h in hits))
    exact=[key for key,s in fasta_records(ref['fasta_path']) if hashlib.sha256(s.encode()).hexdigest()==q['WT_sha256']]
    ready=not exact and not results['development']['identity30_one80'] and not results['General100']['identity30_both80']
    done=dict(status='ORDINARY_EC7_NATIVE_AND_INPUT_EXPOSURE_PASS_CONTEXT_PROTOCOL_PENDING' if ready else 'EC7_INPUT_HOLD_NO_MODEL_RELEASE',
        accession=q['uniprot_accession'],WT_length=len(seq),native_CA_coverage=coverage,precise_active_site_labels=len(sites),
        exact_student_training_ids=exact,exposure=results,annotation_tier=q['annotation_source'],
        native_reference_sha256=sha(out/'native_reference.json'),seconds=time.monotonic()-start,
        job_id=os.environ['SLURM_JOB_ID'],new_model_calls=0,new_structure_predictions=0,model_release_allowed=False)
    write(out/'completion.json',done);print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();prepare(a.root) if a.prepare else run(a.root)
