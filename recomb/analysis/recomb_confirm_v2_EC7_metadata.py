"""Check ordinary public EC7 candidates to document the cached M-CSA coverage gap."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request
from recomb_confirm_v2_diagnostics import read,write,sha
from recomb_confirm_v2_public_metadata import uniprot_projection
from recomb_confirm_v2_audit_msa import source_class
from recomb_scope_screen_20261017 import HAZARD,VIRAL,PATHOGEN
from recomb_confirm_v2_design_function_review import PRODUCT
from recomb_confirm_v2_function_annotation_supplement import ALIASES

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'provenance/recomb_confirm_v2_EC7_metadata'
ACCESSIONS=['P06576','P04191']


def get(accession):
    path=OUT/(accession+'.json')
    if path.exists():return read(path)
    url='https://rest.uniprot.org/uniprotkb/'+accession+'.json'
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'RECOMB-catalytic-input-metadata/2.0'}),timeout=45) as h:raw=h.read()
    d=json.loads(raw)
    if d.get('primaryAccession')!=accession:raise ValueError('Canonical accession differs')
    m=uniprot_projection(d)
    text=json.dumps({k:m.get(k) for k in ['proteinDescription','keywords','functional_comments']},ensure_ascii=False)
    flags=[p.pattern for p in [HAZARD,PRODUCT,ALIASES] if p.search(text)]
    org=json.dumps(m.get('organism'),ensure_ascii=False)
    if VIRAL.search(org) or PATHOGEN.search(org):flags.append('source_outside_scope')
    ordinary=source_class(d)=='SOURCE_GROUP_ACCEPTED_FOR_REVIEWED_WT_REFERENCE' and not flags
    features=[r for r in d.get('features',[]) if r.get('type') in ('Active site','Modified residue','Binding site','Chain','Transit peptide')]
    result=dict(status='ORDINARY_EC7_METADATA_CANDIDATE_NATIVE_AND_ANNOTATION_PROTOCOL_PENDING' if ordinary else 'HOLD_METADATA_ONLY',
        accession=accession,metadata=m,features=features,
        PDB_cross_references=[r for r in d.get('uniProtKBCrossReferences',[]) if r.get('database')=='PDB'],
        source_url=url,source_response_sha256=hashlib.sha256(raw).hexdigest(),
        scope_flags=flags,raw_sequence_saved=False,model_use_allowed=False,
        not_automatically_part_of_Enzyme80=True)
    write(path,result);return result


def main():
    if (OUT/'completion.json').exists():raise FileExistsError('Reuse existing EC7 metadata')
    write(OUT/'request_manifest.json',dict(accessions=ACCESSIONS,purpose='Cached124 M-CSA inputs cover EC1to6; check gap before final enzyme roster',
        metadata_only=True,no_sequence_or_variant_generation=True))
    with ThreadPoolExecutor(max_workers=2) as ex:results=list(ex.map(get,ACCESSIONS))
    summary=dict(status='EC7_PUBLIC_METADATA_CHECK_COMPLETE_NOT_MODEL_RELEASE',candidates=len(results),
        rows=[dict(accession=r['accession'],status=r['status'],length=r['metadata']['sequence_length'],
            active_sites=sum(x['type']=='Active site' for x in r['features']),PDB_references=len(r['PDB_cross_references'])) for r in results],
        new_model_calls=0,new_structure_predictions=0,raw_sequences_saved=False)
    write(OUT/'completion.json',summary);print(summary,flush=True)


if __name__=='__main__':main()
