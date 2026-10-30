"""Stage one ordinary reviewed EC7 WT/reference for native audits, without models."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import urllib.request
from recomb_confirm_v2_diagnostics import read,sha,write
from recomb_confirm_v2_expansion_review import entity_scope

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'external/recomb_confirm_v2_EC7_native_staging'


def main():
    if (OUT/'completion.json').exists():raise FileExistsError('EC7 native input staging already completed')
    meta=read(ROOT/'provenance/recomb_confirm_v2_EC7_metadata/P04191.json')
    if meta['status']!='ORDINARY_EC7_METADATA_CANDIDATE_NATIVE_AND_ANNOTATION_PROTOCOL_PENDING':raise ValueError('Ordinary source/function not reviewed')
    sites=[f for f in meta['features'] if f['type']=='Active site']
    if len(sites)!=1:raise ValueError('Do not reinterpret binding sites as active-site labels')
    if not any(r['id']=='1SU4' for r in meta['PDB_cross_references']):raise ValueError('Chosen published native reference absent from curated cross-references')
    entity=read(ROOT/'provenance/recomb_confirm_v2_public_metadata/rcsb_entity/1SU4_1.json')
    if entity['status']!='SUCCEEDED' or entity_scope(entity['metadata'],set())!='PASS_SOURCE_AND_FUNCTION_METADATA':
        raise ValueError('Native protein source/function not positively reviewed')
    ids=entity['metadata'].get('polymer_identifiers',{}).get('uniprot_ids',[])
    if 'P04191' not in ids:raise ValueError('Native chain does not map to canonical candidate')
    url='https://rest.uniprot.org/uniprotkb/P04191.json'
    with urllib.request.urlopen(url,timeout=45) as h:raw=h.read()
    data=json.loads(raw);seq=data['sequence']['value']
    if data['primaryAccession']!='P04191' or hashlib.sha256(seq.encode()).hexdigest()!=meta['metadata']['canonical_sequence_sha256']:
        raise ValueError('Canonical public WT version changed')
    if not 150<=len(seq)<=1022 or set(seq)-set('ACDEFGHIKLMNPQRSTVWY'):raise ValueError('WT outside original model input limits')
    coordinates=OUT/'1SU4.cif.gz';nativeurl='https://files.rcsb.org/download/1SU4.cif.gz'
    if not coordinates.exists():
        with urllib.request.urlopen(nativeurl,timeout=60) as h:compressed=h.read()
        if not re.search(rb'_entry\.id\s+1SU4\s',gzip.decompress(compressed),re.I):raise ValueError('Native entry identity differs')
        coordinates.parent.mkdir(parents=True,exist_ok=True);coordinates.write_bytes(compressed)
    write(OUT/'candidate_WT.json',dict(candidate_id='UniProtEC7|P04191',sequence=seq,WT_sha256=hashlib.sha256(seq.encode()).hexdigest(),
        WT_length=len(seq),uniprot_accession='P04191',native_PDB='1SU4',native_chain='A',
        native_path='external/recomb_confirm_v2_EC7_native_staging/1SU4.cif.gz',native_sha256=sha(coordinates),
        Active_site_features=sites,annotation_source='UniProtKB reviewed active site with recorded evidence, separate from MCSA complete catalytic network',
        native_and_homology_audit_allowed=True,PLM_encoding_allowed=False,deletion_generation_allowed=False,structure_prediction_allowed=False))
    write(OUT/'annotation_protocol.json',dict(status='EC7_ANNOTATION_TIER_FROZEN_BEFORE_NATIVE_AND_METHOD_OUTPUTS',
        candidate='P04191',native_reference='1SU4, curated UniProt cross-reference; fixed prior to any generated variants',
        labels='all exact Active site positions from reviewed entry; never substitute Binding site or synthesize missing catalytic labels',
        retain_all_declared_positions_and_missing_correspondence=True,
        inclusion_requires='ordinary source, all active-site identities and native CA, native coverage>=.85, no exact student-training WT, development/General exposure audit',
        source_tier_reported_separately_from_MCSA=True,no_functional_activity_claim_from_native_or_prediction=True,
        model_release_allowed=False))
    done=dict(status='ORDINARY_EC7_WT_AND_NATIVE_STAGED_AUDITS_PENDING',accession='P04191',WT_length=len(seq),
        Active_site_annotations=1,source_URL=url,native_URL=nativeurl,native_bytes=coordinates.stat().st_size,
        candidate_sha256=sha(OUT/'candidate_WT.json'),native_sha256=sha(coordinates),
        canonical_response_sha256=hashlib.sha256(raw).hexdigest(),new_model_calls=0,new_structure_predictions=0)
    write(OUT/'completion.json',done);print(done,flush=True)


if __name__=='__main__':main()
