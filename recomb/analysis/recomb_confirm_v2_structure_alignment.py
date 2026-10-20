"""Common geometry-only correspondence, distinct from original residue identity."""
import hashlib
from pathlib import Path
import re
import subprocess
import tempfile
import numpy as np
from delepref_v8_structure_metrics import write_ca_pdb,relationship_metrics


def parse_alignment(output, n_pred, n_ref):
    lines=output.splitlines()
    marker=next((i for i,line in enumerate(lines) if 'denotes residue pairs' in line),None)
    if marker is None or len(lines)<marker+4:
        raise ValueError("USalign alignment block missing")
    first,last=lines[marker+1].strip(),lines[marker+3].strip()
    if not re.fullmatch('[A-Z-]+',first) or not re.fullmatch('[A-Z-]+',last) or len(first)!=len(last):
        raise ValueError("Invalid USalign alignment strings")
    pi=ri=0;pairs=[]
    for a,b in zip(first,last):
        if a!='-' and b!='-':pairs.append((ri,pi))
        pi+=a!='-';ri+=b!='-'
    if (pi,ri)!=(n_pred,n_ref):
        raise ValueError("Alignment does not cover the supplied finite coordinate sequences")
    return pairs


def aligned_geometry(reference,prediction,binary):
    ref=np.asarray(reference,dtype=float);pred=np.asarray(prediction,dtype=float)
    ri=np.flatnonzero(np.isfinite(ref).all(1));pi=np.flatnonzero(np.isfinite(pred).all(1))
    if min(len(ri),len(pi))<3:
        return dict(status='INSUFFICIENT_FINITE_COORDINATES',aligned_CA=0,
                    full_WT_length=len(ref),not_edit_identity_mapping=True)
    with tempfile.TemporaryDirectory(prefix='recomb_v2_structure_') as temp:
        t=Path(temp);write_ca_pdb(t/'pred.pdb',pred[pi]);write_ca_pdb(t/'ref.pdb',ref[ri])
        result=subprocess.run([str(binary),str(t/'pred.pdb'),str(t/'ref.pdb'),'-u',str(len(ref)),
            '-mol','prot','-m',str(t/'matrix.txt')],capture_output=True,text=True,check=True,timeout=120)
        output=result.stdout;pairs=parse_alignment(output,len(pi),len(ri))
        native=np.array([ri[a] for a,b in pairs],dtype=int)
        edited=np.array([pi[b] for a,b in pairs],dtype=int)
        match=re.search(r'TM-score=\s*([\d.]+) \(normalized by user-specified L=([\d.]+) and d0=([\d.]+)\)',output)
        if not match or float(match[2])!=len(ref):raise ValueError('Full-WT normalizer not preserved')
        matrix=[]
        for line in (t/'matrix.txt').read_text().splitlines():
            x=line.split()
            if len(x)==5 and x[0] in ('0','1','2'):matrix.append([float(v) for v in x[1:]])
        if len(matrix)!=3:raise ValueError('USalign transform missing')
        matrix=np.asarray(matrix)
        transformed=np.round(pred[edited],3)@matrix[:,1:].T+matrix[:,0]
        d0=max(.5,1.24*np.cbrt(len(ref)-15.)-1.8) if len(ref)>21 else .5
        computed=float(np.sum(1/(1+np.sum((transformed-np.round(ref[native],3))**2,axis=1)/d0**2))/len(ref))
        if abs(computed-float(match[1]))>3e-5:
            raise ValueError('USalign score and returned alignment/transform disagree')
        metrics=relationship_metrics(ref,pred[edited],native)
        return dict(status='COMMON_STRUCTURE_ALIGNMENT_COMPLETE',full_WT_TM=computed,
            aligned_CA=len(pairs),native_CA_available=len(ri),predicted_CA_available=len(pi),
            full_WT_length=len(ref),aligned_full_WT_coverage=len(pairs)/len(ref),
            aligned_observed_native_coverage=len(pairs)/len(ri),
            structure_mapped_CA_lddt=metrics['retained_ca_lddt_lower'],
            structure_mapped_native_contact_recovery=metrics['contact_recovery_lower'],
            native_contacts=metrics['native_contacts'],
            native_to_prediction_pairs=[[int(a),int(b)] for a,b in zip(native,edited)],
            not_edit_identity_mapping=True,unaligned_relations='not_recovered_in_structural_comparison',
            unknown_native_regions='excluded_from_observed_native_contact_denominator_reported_by_coverage',
            USalign_output_sha256=hashlib.sha256(output.encode()).hexdigest())
