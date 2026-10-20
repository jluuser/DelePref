"""Reuse homolog metadata, audit current source/identity, and retain no raw sequences."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request
from recomb_confirm_v2_diagnostics import read,write,sha
from recomb_confirm_v2_audit_msa import SOURCE_GROUPS,MODEL_SOURCES,CONCERN
from recomb_scope_screen_20261017 import HAZARD,VIRAL,PATHOGEN
from recomb_confirm_v2_design_function_review import PRODUCT
from recomb_confirm_v2_function_annotation_supplement import ALIASES

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/recomb_confirm_v2_General100_evidence_review'


def request_json(url):
    last=None
    for attempt in range(3):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'RECOMB-reference-source-identity-audit/2.0'})
            with urllib.request.urlopen(req,timeout=60) as r:raw=r.read()
            return json.loads(raw),hashlib.sha256(raw).hexdigest()
        except Exception as e:
            last=repr(e)
            if attempt<2:time.sleep(2)
    raise RuntimeError(last)


def batch(ids):
    name=hashlib.sha256('|'.join(ids).encode()).hexdigest()+'.json'
    path=OUT/'uniref_batches'/name
    if path.exists():
        d=read(path)
        if d.get('status')=='SUCCEEDED':return d
    url='https://rest.uniprot.org/uniref/search?'+urllib.parse.urlencode(dict(
        query=' OR '.join('id:'+i for i in ids),format='json',size=100))
    try:
        payload,h=request_json(url);rows=[]
        for entry in payload.get('results',[]):
            if entry['id'] not in ids:raise ValueError('Unexpected metadata target')
            rep=entry.get('representativeMember',{});seq=rep.get('sequence',{}).get('value')
            rows.append(dict(target=entry['id'],updated=entry.get('updated'),
                organism=rep.get('organismName'),taxon_id=rep.get('organismTaxId'),
                protein_name=rep.get('proteinName'),member_id=rep.get('memberId'),
                representative_sequence_sha256=hashlib.sha256(seq.encode()).hexdigest() if seq else None))
        result=dict(status='SUCCEEDED',requested_ids=ids,rows=rows,source_url=url,
            source_response_sha256=h,raw_sequences_saved=False)
    except Exception as e:result=dict(status='FAILED',requested_ids=ids,rows=[],source_url=url,error=repr(e))
    write(path,result);return result


def taxonomy(taxon):
    path=OUT/'taxonomy'/(str(taxon)+'.json')
    if path.exists():
        r=read(path)
        if r.get('status')=='SUCCEEDED':return r
    url='https://rest.uniprot.org/taxonomy/'+str(taxon)
    try:
        d,h=request_json(url)
        if int(d['taxonId'])!=int(taxon):raise ValueError('Taxon identity differs')
        result=dict(status='SUCCEEDED',taxon_id=int(taxon),scientificName=d.get('scientificName'),
            lineage=[p['scientificName'] for p in d.get('lineage',[])],source_url=url,source_response_sha256=h)
    except Exception as e:result=dict(status='FAILED',taxon_id=int(taxon),error=repr(e))
    write(path,result);return result


def main():
    if (OUT/'completion.json').exists():raise FileExistsError('Evidence metadata review already completed')
    inventory=read(OUT/'inventory.json');targets=inventory['targets'];records={};reused=0
    for folder in [ROOT/'reports/recomb_benign_retrieval_scope_20261017/batches',
                   ROOT/'reports/recomb_benign_app_retrieval_scope_20261017/batches',
                   ROOT/'reports/recomb_benign_app_retrieval_scope_20261017',
                   OUT/'uniref_batches']:
        for path in sorted(folder.glob('*.json')):
            value=read(path)
            if not isinstance(value,dict):continue
            if value.get('errors') or value.get('status')=='FAILED':continue
            for r in value.get('rows',[]):
                if r.get('target') in targets and r.get('representative_sequence_sha256'):
                    records[r['target']]=dict(r,metadata_cache_path=str(path.relative_to(ROOT)).replace('\\','/'))
    reused=len(records);missing=sorted(set(targets)-set(records))
    for folder in ['uniref_batches','taxonomy']:(OUT/folder).mkdir(parents=True,exist_ok=True)
    write(OUT/'policy.json',dict(status='REFERENCE_SOURCE_POLICY_FROZEN_BEFORE_RAG_OUTPUTS',
        source_groups=SOURCE_GROUPS,explicit_model_sources=MODEL_SOURCES,
        function_patterns=[p.pattern for p in [CONCERN,HAZARD,VIRAL,PATHOGEN,PRODUCT,ALIASES]],
        full_representative_hash_must_match=True,unknown_function_or_source='omit evidence and disclose; not classify as dangerous',
        applied_to_already_approved_WT_only=True,design_target_scope_approval=False,
        max_representatives_unchanged=64,inventory_sha256=sha(OUT/'inventory.json'),source_sha256=sha(Path(__file__))))
    batches=[missing[i:i+32] for i in range(0,len(missing),32)]
    errors=[]
    with ThreadPoolExecutor(max_workers=6) as ex:
        for i,d in enumerate(ex.map(batch,batches),1):
            if d['status']=='FAILED':errors.append(dict(ids=d['requested_ids'],error=d['error']))
            for r in d['rows']:records[r['target']]=r
            if i%10==0:
                write(OUT/'progress.json',dict(stage='UniRef_metadata',done=i,total=len(batches),cached_targets=reused))
                print('REFERENCE_METADATA '+str(i)+'/'+str(len(batches)),flush=True)
    taxids=sorted({int(r['taxon_id']) for r in records.values() if r.get('taxon_id')})
    taxa={}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for i,d in enumerate(ex.map(taxonomy,taxids),1):
            taxa[d['taxon_id']]=d
            if i%50==0:
                write(OUT/'progress.json',dict(stage='taxonomy',done=i,total=len(taxids),cached_targets=reused))
                print('REFERENCE_TAXONOMY '+str(i)+'/'+str(len(taxids)),flush=True)
    audits={}
    for target,expected in targets.items():
        r=records.get(target,dict(target=target));t=taxa.get(r.get('taxon_id'),{})
        name=r.get('protein_name') or '';organism=r.get('organism') or ''
        lineage=t.get('lineage',[]);rich=[]
        # Reuse curated accession metadata only when its full sequence identity agrees.
        accession=target.removeprefix('UniRef50_')
        path=ROOT/'provenance/recomb_confirm_v2_public_metadata/uniprot'/(accession+'.json')
        if path.exists():
            k=read(path);m=k.get('metadata',{})
            if k.get('status')=='SUCCEEDED' and m.get('canonical_sequence_sha256')==r.get('representative_sequence_sha256'):
                rich=[m.get(x) for x in ['proteinDescription','keywords','functional_comments']]
        text=name+' '+json.dumps(rich,ensure_ascii=False)
        flags=[p.pattern for p in [CONCERN,HAZARD,PRODUCT,ALIASES] if p.search(text)]
        status,reason='HOLD','source_or_metadata_not_positive'
        if flags or VIRAL.search(organism+' '+text) or PATHOGEN.search(organism):
            status,reason='EXCLUDE','out_of_requested_reference_scope'
        elif len(expected)!=1 or r.get('representative_sequence_sha256')!=expected[0]:
            reason='representative_full_sequence_identity_unconfirmed'
        elif not name or re.search(r'uncharacterized|hypothetical|unknown protein',name,re.I):
            reason='function_metadata_insufficient'
        elif t.get('status')!='SUCCEEDED':reason='taxonomy_metadata_unavailable'
        elif any(g in lineage for g in SOURCE_GROUPS) or any(organism.startswith(s) for s in MODEL_SOURCES):
            status,reason='RETAIN','positive_reference_source_function_and_full_sequence_identity'
        else:reason='source_not_in_reviewed_reference_groups'
        audits[target]=dict(r,status=status,reason=reason,reviewed_lineage=lineage,
            taxonomy_file_sha256=sha(OUT/'taxonomy'/(str(r['taxon_id'])+'.json')) if r.get('taxon_id') in taxa else None,
            curated_function_cache_reused=bool(rich))
    write(OUT/'target_metadata_audit.json',audits)
    queries={pid:dict(representatives=len(ids),retained=sum(audits[t]['status']=='RETAIN' for t in ids))
        for pid,ids in inventory['query_targets'].items()}
    write(OUT/'per_query_scope_counts.json',queries)
    result=dict(status='GENERAL100_REFERENCE_METADATA_REVIEW_COMPLETE_PROFILE_BUILD_PENDING',
        unique_targets=len(targets),reused_target_metadata=reused,unique_taxa=len(taxids),
        disposition=dict(Counter(r['status'] for r in audits.values())),
        hold_reasons=dict(Counter(r['reason'] for r in audits.values() if r['status']=='HOLD')),
        queries=len(queries),queries_with_retained_evidence=sum(r['retained']>0 for r in queries.values()),
        errors=errors,raw_sequences_saved=False,new_model_calls=0,profile_generation_allowed=False,
        inventory_sha256=sha(OUT/'inventory.json'),audit_sha256=sha(OUT/'target_metadata_audit.json'))
    write(OUT/'completion.json',result);print(result,flush=True)


if __name__=='__main__':main()
