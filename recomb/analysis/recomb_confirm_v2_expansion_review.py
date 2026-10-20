"""Per-candidate metadata/identity review; native and family release is separate."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import re
from recomb_confirm_v2_diagnostics import read,write,sha
from recomb_confirm_v2_audit_msa import source_class,CONCERN

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'reports/recomb_confirm_v2_expansion_buffers'
PUBLIC=ROOT/'provenance/recomb_confirm_v2_public_metadata'
OUT=ROOT/'reports/recomb_confirm_v2_expansion_review'


def name_of(description):
    if not isinstance(description,dict):return str(description or '')
    n=description.get('recommendedName',{}).get('fullName',{}).get('value')
    if n:return n
    for group in ('submissionNames','alternativeNames'):
        for row in description.get(group,[]):
            if row.get('fullName',{}).get('value'):return row['fullName']['value']
    return ''


def entity_scope(e,prior_taxa):
    if not e:return 'HOLD_METADATA_MISSING'
    description=e.get('entity',{}).get('pdbx_description','')
    if not description:return 'HOLD_FUNCTION_NAME_MISSING'
    if CONCERN.search(description):return 'HOLD_FUNCTION_REVIEW'
    sources=e.get('sources') or []
    if not sources:return 'HOLD_SOURCE_MISSING'
    status=[]
    for s in sources:
        organism=dict(scientificName=s.get('ncbi_scientific_name') or s.get('scientific_name',''),
            lineage=[p['name'] for p in s.get('taxonomy_lineage',[])])
        result=source_class(dict(organism=organism,proteinDescription={'name':description}))
        if s.get('ncbi_taxonomy_id') in prior_taxa and result=='HOLD_SOURCE_NOT_POSITIVELY_REVIEWED':
            result='SOURCE_GROUP_ACCEPTED_FOR_REVIEWED_WT_REFERENCE'
        status.append(result)
    return 'PASS_SOURCE_AND_FUNCTION_METADATA' if all(s=='SOURCE_GROUP_ACCEPTED_FOR_REVIEWED_WT_REFERENCE' for s in status) else 'HOLD_SOURCE_REVIEW'


def main():
    if not (BASE/'metadata_completion.json').exists():raise RuntimeError('Wait for the existing metadata fetch; do not submit it again')
    if (OUT/'completion.json').exists():raise FileExistsError('Expansion review already saved')
    old_general=read(ROOT/'reports/recomb_benign_scope_20261017/general_audit.json')
    old_app=read(ROOT/'reports/recomb_benign_application_scope_20261017/audit.json')
    historical_WT={r['WT_sha256'] for r in old_general+old_app}
    blocked_WT={r['WT_sha256'] for r in old_general+old_app if r['status']=='EXCLUDE'}
    prior_taxa=set(read(ROOT/'reports/recomb_benign_retrieval_scope_20261017/inventory.json')['positively_reviewed_source_taxa'])
    entities=defaultdict(dict)
    for path in (PUBLIC/'rcsb_entity').glob('*.json'):
        r=read(path)
        if r.get('status')=='SUCCEEDED':
            pdb,eid=r['key'].split('_',1);entities[pdb][eid]=r['metadata']
    def bychain(pdb,chain):
        matches=[e for e in entities[pdb].values() if chain in e.get('polymer_identifiers',{}).get('auth_asym_ids',[])]
        return matches[0] if len(matches)==1 else None
    cath=[]
    for r in read(BASE/'CATH600_metadata_buffer.json'):
        e=bychain(r['PDB'],r['chain']);status=entity_scope(e,prior_taxa)
        if r['WT_sha256'] in blocked_WT:status='EXCLUDED_PREVIOUS_SCOPE'
        elif r['WT_sha256'] in historical_WT:status='HISTORICAL_NOT_NEW_CONFIRMATION'
        cath.append(dict(r,metadata_review_status=status,
            description=e.get('entity',{}).get('pdbx_description') if e else None,
            source_organisms=[s.get('ncbi_scientific_name') for s in e.get('sources',[])] if e else [],
            native_coordinate_and_homology_review_pending=True,model_use_allowed=False))
    enzymes=[]
    for r in read(BASE/'MCSA_coordinate_pass_metadata_buffer.json'):
        accession=r.get('uniprot_accession');path=PUBLIC/'uniprot'/(accession+'.json')
        metadata=read(path)['metadata'] if path.exists() and read(path).get('status')=='SUCCEEDED' else {}
        entry=dict(organism=metadata.get('organism',{}),proteinDescription=metadata.get('proteinDescription'),
            keywords=metadata.get('keywords'),comments=metadata.get('functional_comments'))
        status=source_class(entry);exact=metadata.get('canonical_sequence_sha256')==r['WT_sha256']
        if r['WT_sha256'] in blocked_WT:result='EXCLUDED_PREVIOUS_SCOPE'
        elif r['WT_sha256'] in historical_WT:result='HISTORICAL_NOT_NEW_CONFIRMATION'
        elif status!='SOURCE_GROUP_ACCEPTED_FOR_REVIEWED_WT_REFERENCE':result=status
        elif not exact:result='HOLD_CANONICAL_IDENTITY_VERSION'
        else:result='PASS_SOURCE_FUNCTION_AND_CANONICAL_IDENTITY'
        enzymes.append(dict(r,metadata_review_status=result,
            source_organism=entry['organism'].get('scientificName'),canonical_identity_exact=exact,
            canonical_description=name_of(entry['proteinDescription']),
            native_annotation_and_family_review_pending=True,model_use_allowed=False))
    binders=[]
    for r in read(BASE/'SAbDab_metadata_buffer.json'):
        pdb=r['pdb'].upper();heavy=r['prospective_design_chain'];light=r.get('fixed_light_chain')
        partners=[c.strip() for c in re.split(r'\s*\|\s*',r['antigen_chain']) if c.strip()]
        if light not in ('','NA','None',None):partners.append(light)
        chains=[heavy]+list(dict.fromkeys(partners))
        evidence=[];antigen_accessions=set();status='PASS_SOURCE_AND_CHAIN_ROLE_METADATA'
        for chain in chains:
            e=bychain(pdb,chain);scope=entity_scope(e,prior_taxa)
            if e and chain==heavy and scope=='HOLD_SOURCE_REVIEW':
                sources=e.get('sources') or []
                # Curated SAbDab H-chain numbering is evidence of antibody role;
                # synthetic construct is not silently treated as an organism.
                if sources and all(s.get('ncbi_scientific_name')=='synthetic construct' for s in sources) and not CONCERN.search(e.get('entity',{}).get('pdbx_description','')):
                    scope='PASS_CURATED_SABDAB_SYNTHETIC_ANTIBODY_ROLE'
            if scope not in ('PASS_SOURCE_AND_FUNCTION_METADATA','PASS_CURATED_SABDAB_SYNTHETIC_ANTIBODY_ROLE'):
                status='HOLD_MODELLED_CHAIN_SOURCE_OR_FUNCTION'
            if e and chain in partners and chain!=light:
                antigen_accessions.update(e.get('polymer_identifiers',{}).get('uniprot_ids',[]))
            evidence.append(dict(auth_chain=chain,role='design_antibody' if chain==heavy else 'fixed_partner',
                description=e.get('entity',{}).get('pdbx_description') if e else None,scope=scope))
        if len(chains)!=len(set(chains)):status='HOLD_CHAIN_ROLE_COLLISION'
        # Other biological context is reviewed during assembly construction;
        # no sequence or model is released by this metadata stage.
        binders.append(dict(r,metadata_review_status=status,chain_evidence=evidence,
            antigen_reference_accessions=sorted(antigen_accessions),model_use_allowed=False,
            native_construct_full_assembly_clonotype_review_pending=True))
    write(OUT/'CATH_metadata_review.json',cath);write(OUT/'MCSA_metadata_review.json',enzymes)
    write(OUT/'SAbDab_metadata_review.json',binders)
    report=dict(status='PER_CANDIDATE_METADATA_REVIEW_COMPLETE_NATIVE_FAMILY_RELEASE_PENDING',
        CATH=dict(Counter(r['metadata_review_status'] for r in cath)),
        MCSA=dict(Counter(r['metadata_review_status'] for r in enzymes)),
        SAbDab=dict(Counter(r['metadata_review_status'] for r in binders)),
        raw_sequences_exported=False,new_model_inputs_released=0,new_model_calls=0,
        source_policy_sha256=sha(ROOT/'external/recomb_confirm_v2_WT_MSA_reviewed/review_policy.json'),
        prior_reviewed_taxa_also_respected=True)
    write(OUT/'completion.json',report);print(json.dumps(report))


if __name__=='__main__':main()
