"""Bundle already-cached context evidence for native-qualified ordinary inputs."""
from collections import Counter
import json
from pathlib import Path
from recomb_confirm_v2_diagnostics import read,sha,write
from recomb_confirm_v2_expansion_review import entity_scope

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/recomb_confirm_v2_context_evidence'


def main():
    if (OUT/'completion.json').exists():raise FileExistsError('Context evidence already bundled')
    audit=read(ROOT/'reports/recomb_confirm_v2_enzyme_native/native_audit.json')
    qualified={r['candidate_id'] for r in audit if r['status']=='NATIVE_AND_ALL_CATALYTIC_IDENTITIES_PASS_CONTEXT_PENDING'}
    enzymes=[r for r in read(ROOT/'reports/recomb_confirm_v2_expansion_review/MCSA_metadata_review.json') if 'MCSA|'+r['candidate_id'] in qualified]
    binders=read(ROOT/'external/recomb_confirm_v2_SAbDab_native_staging/cases.json')
    pdbs={r['reference_pdb_id'].upper() for r in enzymes}|{r['pdb'].upper() for r in binders}
    prior=set(read(ROOT/'reports/recomb_benign_retrieval_scope_20261017/inventory.json')['positively_reviewed_source_taxa'])
    public=ROOT/'provenance/recomb_confirm_v2_public_metadata'
    entries={};files={};statuses=Counter()
    for path in (public/'rcsb_entity').glob('*.json'):
        pdb=path.stem.split('_',1)[0]
        if pdb not in pdbs:continue
        d=read(path)
        if d.get('status')!='SUCCEEDED':continue
        eid=path.stem.split('_',1)[1];m=d['metadata'];status=entity_scope(m,prior)
        entries.setdefault(pdb,{})[eid]=dict(metadata=m,scope_status=status,source_url=d['source_url'],
            evidence_path=path.relative_to(ROOT).as_posix(),evidence_sha256=sha(path))
        files[path.relative_to(ROOT).as_posix()]=sha(path);statuses[status]+=1
    notes={}
    for r in enzymes:
        p=public/'uniprot'/(r['uniprot_accession']+'.json');d=read(p)
        if d.get('status')!='SUCCEEDED':raise ValueError('Approved canonical source metadata absent')
        m=d['metadata']
        notes['MCSA|'+r['candidate_id']]=dict(accession=r['uniprot_accession'],organism=m.get('organism'),
            canonical_sequence_sha256=m['canonical_sequence_sha256'],proteinDescription=m.get('proteinDescription'),
            functional_comments=m.get('functional_comments'),source_url=d['source_url'],metadata_sha256=sha(p))
        files[p.relative_to(ROOT).as_posix()]=sha(p)
    write(OUT/'entity_context_metadata.json',entries);write(OUT/'canonical_enzyme_context_notes.json',notes)
    done=dict(status='CACHED_NATIVE_CONTEXT_EVIDENCE_BUNDLED_NOT_MODEL_RELEASE',enzyme_inputs=len(enzymes),
        binder_cases=len(binders),PDB=len(pdbs),PDB_with_entity_metadata=len(entries),entity_scope=dict(statuses),
        raw_sequences_saved=False,new_requests=0,new_model_calls=0,evidence_files=files,
        outputs={n:sha(OUT/n) for n in ['entity_context_metadata.json','canonical_enzyme_context_notes.json']})
    write(OUT/'completion.json',done)
    print({k:v for k,v in done.items() if k not in ('evidence_files','outputs')},flush=True)


if __name__=='__main__':main()
