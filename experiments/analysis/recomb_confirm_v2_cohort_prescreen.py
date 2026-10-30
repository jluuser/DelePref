"""Cached metadata prescreen only; construct/family review still required."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import re
from recomb_confirm_v2_diagnostics import read,sha,write
from recomb_confirm_v2_audit_msa import source_class

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'provenance/recomb_confirm_v2_public_metadata'
ANTIBODY=re.compile(r'antibody|nanobody|\bVHH\b|\bFab\b|heavy chain|light chain|immunoglobulin.*chain',re.I)


def entity(key):
    path=PUBLIC/'rcsb_entity'/(key+'.json')
    return read(path)['metadata'] if path.exists() and read(path).get('status')=='SUCCEEDED' else None


def entity_scope(e):
    if e is None:return 'HOLD_ENTITY_METADATA_MISSING'
    name=e.get('entity',{}).get('pdbx_description')
    if not name:return 'HOLD_FUNCTION_NAME_MISSING'
    sources=e.get('sources') or []
    statuses=[]
    for source in sources:
        organism=dict(scientificName=source.get('ncbi_scientific_name') or source.get('scientific_name',''),
                      lineage=[v['name'] for v in source.get('taxonomy_lineage',[])])
        status=source_class(dict(organism=organism,proteinDescription=dict(name=name)))
        if status=='HOLD_SOURCE_NOT_POSITIVELY_REVIEWED' and organism['scientificName']=='synthetic construct' and ANTIBODY.search(name):
            status='SOURCE_GROUP_ACCEPTED_FOR_REVIEWED_WT_REFERENCE'
        statuses.append(status)
    return 'METADATA_SCOPE_PASS' if statuses and all(s=='SOURCE_GROUP_ACCEPTED_FOR_REVIEWED_WT_REFERENCE' for s in statuses) else (
        'HOLD_FUNCTION_REVIEW' if 'HOLD_FUNCTION_REVIEW' in statuses else 'HOLD_SOURCE_REVIEW')


def description(e):
    return e.get('entity',{}).get('pdbx_description') if e else None


def main():
    out=ROOT/'reports/recomb_confirm_v2_cohort_prescreen'
    if (out/'completion.json').exists():raise FileExistsError('Prescreen already completed')
    inventory=read(ROOT/'reports/recomb_confirm_v2_inventory/application_candidate_metadata.json')
    supplemental={r['candidate_id']:r['metadata'] for r in read(ROOT/'reports/recomb_confirm_v2_inventory/application_context_supplement.json')}
    dev=read(ROOT/'reports/recomb_benign_application_scope_20261017/audit.json')
    excluded_WT={r['WT_sha256'] for r in dev if r['status']=='EXCLUDE'}
    previous_WT={r['WT_sha256'] for r in dev}
    rows=[]
    for r in inventory:
        extra=supplemental[r['candidate_id']];m={**r['metadata'],**extra};pdb=str(m.get('pdb_id','')).upper()
        result=dict(candidate_id=r['candidate_id'],cohort=r['cohort'],protein_id=r['protein_id'],
            WT_sha256=r['WT_sha256'],PDB=pdb,prior_development_or_application20_WT=r['WT_sha256'] in previous_WT,
            model_use_allowed=False)
        if r['WT_sha256'] in excluded_WT:
            result['status']='EXCLUDED_BY_EXISTING_SCOPE';rows.append(result);continue
        if r['cohort']=='enzyme':
            accession=m['uniprot_accession'];path=PUBLIC/'uniprot'/(accession+'.json')
            u=read(path)['metadata'] if path.exists() and read(path).get('status')=='SUCCEEDED' else {}
            status=source_class(dict(organism=u.get('organism',{}),proteinDescription=u.get('proteinDescription'),
                                    keywords=u.get('keywords'),comments=u.get('functional_comments')))
            exact=u.get('canonical_sequence_sha256')==r['WT_sha256']
            result.update(accession=accession,organism=u.get('organism',{}).get('scientificName'),
                protein_name=m.get('enzyme_name'),EC=m.get('ec_numbers'),canonical_identity_exact=exact,
                native_entity_scope=entity_scope(entity(pdb+'_'+str(m.get('pdb_entity_id','1')))))
            result['status']=('SOURCE_AND_CANONICAL_PASS_FAMILY_REVIEW_PENDING' if exact and
                status=='SOURCE_GROUP_ACCEPTED_FOR_REVIEWED_WT_REFERENCE' else status if
                status!='SOURCE_GROUP_ACCEPTED_FOR_REVIEWED_WT_REFERENCE' else 'HOLD_CANONICAL_IDENTITY')
        else:
            design=entity(pdb+'_'+str(m.get('compressed_entity_id','')))
            partner=entity(pdb+'_'+str(m.get('partner_entity_id','')))
            real_binder=bool(ANTIBODY.search(description(design) or ''))
            result.update(design_description=description(design),partner_description=description(partner),
                design_source_status=entity_scope(design),partner_source_status=entity_scope(partner),
                design_is_antibody_candidate=real_binder,existing_interaction_category=m.get('audited_interaction_category'),
                design_auth_chain=m.get('compressed_auth_chain_id'),partner_auth_chain=m.get('partner_auth_chain_id'),
                full_binder_assembly_review_required=True)
            result['status']=('SOURCE_PASS_TRUE_BINDER_CONSTRUCT_REVIEW_PENDING' if real_binder and
                entity_scope(design)=='METADATA_SCOPE_PASS' and entity_scope(partner)=='METADATA_SCOPE_PASS' else
                'GENERAL_INTERFACE_NOT_TRUE_BINDER' if not real_binder else 'HOLD_SOURCE_OR_FUNCTION_REVIEW')
        rows.append(result)
    general=[];entity_by_pdb=defaultdict(list)
    for path in (PUBLIC/'rcsb_entity').glob('*.json'):
        e=read(path)
        if e.get('status')=='SUCCEEDED':entity_by_pdb[e['key'].split('_')[0]].append(e['metadata'])
    old=read(ROOT/'reports/recomb_benign_scope_20261017/general_audit.json')
    excluded={r['WT_sha256'] for r in old if r['status']=='EXCLUDE'}
    for r in read(ROOT/'reports/recomb_confirm_v2_inventory/general_candidate_metadata.json'):
        pid=r['protein_id']
        if r.get('domain_id'):pdb=r['domain_id'][:4].upper();chain=r['domain_id'][4]
        elif pid.startswith('curated_pdb|'):p=pid.split('|');pdb=p[2].upper();chain=p[3]
        else:continue
        candidates=[e for e in entity_by_pdb[pdb] if chain in e.get('polymer_identifiers',{}).get('auth_asym_ids',[])]
        e=candidates[0] if len(candidates)==1 else None
        status=entity_scope(e)
        if r['sequence_sha256'] in excluded:status='EXCLUDED_BY_EXISTING_SCOPE'
        general.append(dict(candidate_id=r['candidate_id'],protein_id=pid,WT_sha256=r['sequence_sha256'],
            PDB=pdb,chain=chain,description=description(e),source_scope_status=status,
            source_organisms=[s.get('ncbi_scientific_name') for s in e.get('sources',[])] if e else [],
            original100=r['previously_in_original100'],native_PASS=r['native_audit'].get('status')=='PASS',
            family_component=r.get('sequence_homology_component'),model_use_allowed=False))
    write(out/'application_prescreen.json',rows);write(out/'general_prescreen.json',general)
    summary=dict(status='PRELIMINARY_METADATA_PRESCREEN_NOT_MODEL_RELEASE',
        enzyme=dict(Counter(r['status'] for r in rows if r['cohort']=='enzyme')),
        binding=dict(Counter(r['status'] for r in rows if r['cohort']=='binding_interface')),
        general=dict(Counter(r['source_scope_status'] for r in general)),
        general_new_native_PASS_metadata_pass=sum(not r['original100'] and r['native_PASS'] and r['source_scope_status']=='METADATA_SCOPE_PASS' for r in general),
        new_model_scope_approvals=0,raw_sequences_saved=False)
    write(out/'completion.json',summary);print(json.dumps(summary))


if __name__=='__main__':main()
