"""Check only reviewed application constructs against native polymer sequences."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from recomb_application_scope_20261017 import values
from recomb_scope_screen_20261017 import read,sha,write


def audit(root):
    rows=read(root/'reports/recomb_general100_applications_20261017/panel.json')
    reviewed={r['protein_id'] for r in read(root/'reports/recomb_benign_application_scope_20261017/audit.json') if r['status']=='RETAIN'}
    out=root/'reports/recomb_benign_application_identity_20261017';out.mkdir(exist_ok=False)
    reports=[]
    for row in rows:
        if row['protein_id'] not in reviewed:continue
        path=Path(row['native_reference_path'])
        if sha(path)!=row['native_reference_sha256']:raise ValueError('native input changed')
        with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:data=MMCIF2Dict(f)
        sequences=dict(zip(values(data,'_entity_poly.entity_id'),values(data,'_entity_poly.pdbx_seq_one_letter_code_can')))
        chain_entities={}
        for chain,entity in zip(values(data,'_atom_site.auth_asym_id'),values(data,'_atom_site.label_entity_id')):
            if entity in sequences:chain_entities.setdefault(chain,set()).add(entity)
        parts=[dict(chain_id=row['design_chain'],sequence=row['sequence'],role='design')]
        parts.extend(dict(p,role='partner') for p in row.get('partners',[]))
        details=[]
        for part in parts:
            ids=chain_entities.get(part['chain_id'],set())
            native=''.join(sequences[next(iter(ids))].split()) if len(ids)==1 else ''
            query=part['sequence']
            occurrences=[i for i in range(max(0,len(native)-len(query)+1)) if native.startswith(query,i)] if query else []
            exact=len(ids)==1 and len(occurrences)==1 and not(set(native)-set('ACDEFGHIKLMNPQRSTVWYX'))
            details.append(dict(chain_id=part['chain_id'],role=part['role'],WT_length=len(query),native_canonical_length=len(native),
                WT_sha256=hashlib.sha256(query.encode()).hexdigest(),native_canonical_sha256=hashlib.sha256(native.encode()).hexdigest(),
                status='EXACT_UNIQUE_CONSTRUCT' if exact else 'NEEDS_EXISTING_AUDITED_ALIGNMENT',
                native_start_0based=occurrences[0] if exact else None))
        reports.append(dict(protein_id=row['protein_id'],dataset=row['dataset'],chains=details,
            status='EXACT_CHAIN_IDENTITIES' if all(x['status']=='EXACT_UNIQUE_CONSTRUCT' for x in details) else 'MAPPING_REVIEW_REQUIRED',
            source_sha256=row['native_reference_sha256'],generation_released=False))
    write(out/'audit.json',reports)
    summary=dict(status='REVIEWED_APPLICATION_EXACT_IDENTITY_FIRST_PASS',reviewed_inputs=len(reports),
        disposition=dict(Counter(r['status'] for r in reports)),
        by_dataset={d:dict(Counter(r['status'] for r in reports if r['dataset']==d)) for d in ['Binder20','Enzyme20']},
        nonexact_not_excluded_for_performance=True,next='Reuse existing audited canonical-to-native alignments for nonexact constructs; retrieval source review still required',
        new_model_calls=0,new_structure_predictions=0)
    write(out/'completion.json',summary);print(json.dumps(summary))


if __name__=='__main__':audit(Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
