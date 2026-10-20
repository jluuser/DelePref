"""Complete cached-domain function annotations before releasing any design targets."""
import concurrent.futures
from collections import Counter,defaultdict
from pathlib import Path
import json
import re
from recomb_confirm_v2_diagnostics import read,write,sha
from recomb_confirm_v2_public_metadata import get
from recomb_scope_screen_20261017 import HAZARD
from recomb_confirm_v2_design_function_review import PRODUCT

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'provenance/recomb_confirm_v2_public_metadata'
BASE=ROOT/'reports/recomb_confirm_v2_design_function_review'
ALIASES=re.compile(r'ribosome.inactivating|trichosanthin|\babrin\b|\bsaporin\b|\bgelonin\b|'
                   r'\bmodeccin\b|\bvolkensin\b|\bmomordin\b|\bcucurmosin\b|'
                   r'\bcurcin\b|\bluffin\b|\bbouganin\b|\bdianthin\b|cytolyt',re.I)


def main():
    if (BASE/'function_annotation_supplement_completion.json').exists():raise FileExistsError('Function supplement already complete')
    initial=read(BASE/'per_candidate_function_review.json')
    entities=defaultdict(list)
    for p in (PUBLIC/'rcsb_entity').glob('*.json'):
        e=read(p)
        if e.get('status')=='SUCCEEDED':entities[e['key'].split('_')[0]].append(e['metadata'])
    candidate_accessions={}
    for row in initial:
        accessions=set()
        for e in entities[row['PDB']]:accessions.update(e.get('polymer_identifiers',{}).get('uniprot_ids',[]))
        candidate_accessions[row['candidate_id']]=sorted(a for a in accessions if re.fullmatch('[A-Z0-9]+',a))
    ids=sorted({a for values in candidate_accessions.values() for a in values})
    fetched=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        for record in ex.map(lambda a:get('uniprot',a),ids):
            fetched.append(record)
            if len(fetched)%25==0:print('FUNCTION_REFERENCE_METADATA '+str(len(fetched))+'/'+str(len(ids)),flush=True)
    revised=[]
    for row in initial:
        result=dict(row);records=[];new_flags=[]
        for accession in candidate_accessions[row['candidate_id']]:
            path=PUBLIC/'uniprot'/(accession+'.json')
            if not path.exists():continue
            d=read(path)
            if d.get('status')!='SUCCEEDED':continue
            m=d['metadata'];text=json.dumps({k:m.get(k) for k in ['proteinDescription','keywords','functional_comments']},ensure_ascii=False)
            for name,pattern in [('curated_toxin_virulence_or_resistance_annotation',HAZARD),
                                 ('toxic_product_or_high_risk_context',PRODUCT),('toxic_function_alias',ALIASES)]:
                found=pattern.search(text)
                if found:new_flags.append(dict(category=name,matched_term=found.group(0),accession=accession))
            records.append(dict(accession=accession,metadata_sha256=sha(path),source_url=d['source_url']))
        result['curated_reference_annotations']=records
        result['flags']=row['flags']+new_flags
        if result['flags']:result['status']='HOLD_FUNCTIONAL_SCOPE_NO_VARIANT_USE'
        elif not records and row['kind']=='general_domain':result['status']='HOLD_CURATED_DOMAIN_FUNCTION_METADATA_UNAVAILABLE'
        result['model_use_allowed']=False;revised.append(result)
    write(BASE/'per_candidate_function_review_v2.json',revised)
    report=dict(status='CURATED_FUNCTION_ANNOTATION_SUPPLEMENT_COMPLETE_NO_MODEL_RELEASE',
        reference_accessions=len(ids),metadata_status=dict(Counter(r['status'] for r in fetched)),
        reused=sum(r['reused'] for r in fetched),counts={k:dict(Counter(r['status'] for r in revised if r['kind']==k))
            for k in ['general_domain','general_chain','enzyme','binder']},
        new_model_calls=0,raw_sequences_saved=False,source_sha256=sha(Path(__file__)))
    write(BASE/'function_annotation_supplement_completion.json',report);print(report)


if __name__=='__main__':main()
