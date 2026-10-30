"""Design-target functional scope check after source review, before any variants."""
from collections import Counter,defaultdict
from pathlib import Path
import re
from recomb_confirm_v2_diagnostics import read,write,sha
from recomb_scope_screen_20261017 import HAZARD,VIRAL

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'provenance/recomb_confirm_v2_public_metadata'
REVIEW=ROOT/'reports/recomb_confirm_v2_expansion_review'
OUT=ROOT/'reports/recomb_confirm_v2_design_function_review'
PRODUCT=re.compile(r'cytolysin|hemolysin|haemolysin|enterotoxin|exotoxin|endotoxin|'
    r'lipopolysaccharide|\blipid[ -]?a\b|ribosome.inactivating|rRNA.{0,20}N.glycosyl|'
    r'\babrin\b|\bsaporin\b|\bgelonin\b|\bmodeccin\b|\bvolkensin\b|'
    r'botulin|tetanus|diphtheria|anthrax|saxitoxin|\bprion\b',re.I)


def main():
    if (OUT/'completion.json').exists():raise FileExistsError('Functional scope review already recorded')
    entities=defaultdict(list)
    for path in (PUBLIC/'rcsb_entity').glob('*.json'):
        d=read(path)
        if d.get('status')=='SUCCEEDED':entities[d['key'].split('_')[0]].append(d['metadata'])
    def evidence(pdb,accession=None):
        texts=[];sources=[]
        if accession:
            p=PUBLIC/'uniprot'/(accession+'.json')
            if p.exists():texts.append(read(p).get('metadata',{}));sources.append(str(p.relative_to(ROOT)))
        p=PUBLIC/'rcsb_entry'/(pdb+'.json')
        if p.exists():texts.append(read(p).get('metadata',{}));sources.append(str(p.relative_to(ROOT)))
        for e in entities[pdb]:
            texts.append(dict(entity=e.get('entity'),sources=e.get('sources')))
        return __import__('json').dumps(texts,ensure_ascii=False),sources
    result=[]
    for filename,kind,passing in [('CATH_metadata_review.json','general_domain','PASS_SOURCE_AND_FUNCTION_METADATA'),
            ('MCSA_metadata_review.json','enzyme','PASS_SOURCE_FUNCTION_AND_CANONICAL_IDENTITY'),
            ('SAbDab_metadata_review.json','binder','PASS_SOURCE_AND_CHAIN_ROLE_METADATA')]:
        for r in read(REVIEW/filename):
            if r['metadata_review_status']!=passing:continue
            pdb=str(r.get('PDB') or r.get('reference_pdb_id') or r.get('pdb')).upper()
            text,files=evidence(pdb,r.get('uniprot_accession'))
            matches=[]
            for name,pattern in [('declared_toxin_virulence_resistance',HAZARD),('viral_or_phage_context',VIRAL),
                                 ('specific_toxic_product_or_high_risk_context',PRODUCT)]:
                found=pattern.search(text)
                if found:matches.append(dict(category=name,matched_term=found.group(0)))
            cid=('CATH|'+r['domain_id'] if kind=='general_domain' else
                 'MCSA|'+str(r['candidate_id']) if kind=='enzyme' else
                 'SAbDab|'+pdb+'|'+r['prospective_design_chain'])
            result.append(dict(candidate_id=cid,
                kind=kind,PDB=pdb,WT_sha256=r.get('WT_sha256'),
                status='HOLD_FUNCTIONAL_SCOPE_NO_VARIANT_USE' if matches else 'FUNCTIONAL_SCOPE_PASS_NATIVE_AND_FAMILY_PENDING',
                flags=matches,evidence_files=files,model_use_allowed=False,
                note='Ordinary source does not override concerning function; no harmful candidate was optimized'))
    for r in read(ROOT/'reports/recomb_confirm_v2_cohort_prescreen/application_prescreen.json'):
        if r['cohort']!='binding_interface' or r.get('existing_interaction_category')=='antibody_antigen':continue
        if r.get('design_source_status')!='METADATA_SCOPE_PASS' or r.get('partner_source_status')!='METADATA_SCOPE_PASS':continue
        text,files=evidence(r['PDB']);matches=[]
        for name,pattern in [('declared_toxin_virulence_resistance',HAZARD),('viral_or_phage_context',VIRAL),
                             ('specific_toxic_product_or_high_risk_context',PRODUCT)]:
            found=pattern.search(text)
            if found:matches.append(dict(category=name,matched_term=found.group(0)))
        result.append(dict(candidate_id='PPIchain|'+r['candidate_id'],kind='general_chain',PDB=r['PDB'],WT_sha256=r['WT_sha256'],
            status='HOLD_FUNCTIONAL_SCOPE_NO_VARIANT_USE' if matches else 'FUNCTIONAL_SCOPE_PASS_NATIVE_AND_FAMILY_PENDING',
            flags=matches,evidence_files=files,model_use_allowed=False))
    write(OUT/'per_candidate_function_review.json',result)
    report=dict(status='DESIGN_FUNCTION_SCOPE_REVIEW_COMPLETE_BEFORE_VARIANTS',
        counts={k:dict(Counter(r['status'] for r in result if r['kind']==k)) for k in ['general_domain','enzyme','binder','general_chain']},
        new_model_calls=0,new_model_release=False,raw_sequences_saved=False,
        review_source_sha256=sha(Path(__file__)),old_scope_policy_respected=True)
    write(OUT/'completion.json',report);print(report)


if __name__=='__main__':main()
