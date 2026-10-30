"""Fetch public source/function metadata only; never releases model inputs."""
import concurrent.futures
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request
from recomb_confirm_v2_diagnostics import read,write

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'provenance/recomb_confirm_v2_public_metadata'


def uniprot_projection(d):
    seq=d.get('sequence',{}).get('value','')
    return dict(accession=d.get('primaryAccession'),organism=d.get('organism'),
        proteinDescription=d.get('proteinDescription'),keywords=d.get('keywords',[]),
        functional_comments=[c for c in d.get('comments',[]) if c.get('commentType') in
            ('FUNCTION','CATALYTIC ACTIVITY','COFACTOR','SUBUNIT','SUBCELLULAR LOCATION','DISEASE','TOXIC DOSE')],
        entryType=d.get('entryType'),sequence_length=len(seq),
        canonical_sequence_sha256=hashlib.sha256(seq.encode()).hexdigest() if seq else None)


def rcsb_entity_projection(d):
    seq=d.get('entity_poly',{}).get('pdbx_seq_one_letter_code_can','')
    return dict(polymer_identifiers=d.get('rcsb_polymer_entity_container_identifiers'),
        sources=d.get('rcsb_entity_source_organism'),entity=d.get('rcsb_polymer_entity'),
        sequence_length=len(seq),polymer_sequence_sha256=hashlib.sha256(seq.encode()).hexdigest() if seq else None)


def get(kind,key):
    if not re.fullmatch('[A-Za-z0-9_-]+',key):raise ValueError('Invalid metadata identifier')
    path=OUT/kind/(key+'.json')
    if path.exists() and read(path).get('status')=='SUCCEEDED':return dict(kind=kind,key=key,reused=True,status='SUCCEEDED')
    cached=ROOT/'provenance/recomb_functional_annotations_20261017/uniprot'/(key+'.json')
    url=('https://rest.uniprot.org/uniprotkb/'+key+'.json') if kind=='uniprot' else (
        'https://data.rcsb.org/rest/v1/core/entry/'+key if kind=='rcsb_entry' else
        'https://data.rcsb.org/rest/v1/core/polymer_entity/'+key.replace('_','/'))
    errors=[]
    for attempt in range(3):
        try:
            if kind=='uniprot' and cached.exists():
                raw=cached.read_bytes();reused_source=True
            else:
                request=urllib.request.Request(url,headers={'User-Agent':'RECOMB-metadata-audit/2.0'})
                with urllib.request.urlopen(request,timeout=25) as response:raw=response.read()
                reused_source=False
            data=json.loads(raw)
            if kind=='uniprot':projected=uniprot_projection(data)
            elif kind=='rcsb_entity':projected=rcsb_entity_projection(data)
            else:projected=dict(entry_id=data.get('rcsb_id'),title=data.get('struct',{}).get('title'),
                polymer_entity_ids=data.get('rcsb_entry_container_identifiers',{}).get('polymer_entity_ids',[]))
            value=dict(status='SUCCEEDED',kind=kind,key=key,source_url=url,
                source_response_sha256=hashlib.sha256(raw).hexdigest(),metadata=projected,
                raw_sequence_saved=False,review_status='UNREVIEWED_NOT_MODEL_INPUT',
                existing_source_cache_reused=reused_source,
                fetched_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write(path,value);return dict(kind=kind,key=key,reused=False,status='SUCCEEDED')
        except Exception as error:
            errors.append(repr(error))
            if attempt<2:time.sleep(1+attempt)
    write(path,dict(status='FAILED',kind=kind,key=key,source_url=url,errors=errors))
    return dict(kind=kind,key=key,reused=False,status='FAILED')


def main():
    rows=read(ROOT/'reports/recomb_confirm_v2_inventory/application_candidate_metadata.json')
    general=read(ROOT/'reports/recomb_confirm_v2_inventory/general_candidate_metadata.json')
    audit=read(ROOT/'reports/recomb_benign_scope_20261017/general_audit.json')
    app_audit=read(ROOT/'reports/recomb_benign_application_scope_20261017/audit.json')
    blocked_accessions=set();blocked_pdb=set()
    for r in audit+app_audit:
        if r['status']!='EXCLUDE':continue
        pid=r['protein_id']
        if pid.startswith('Enzyme20|'):blocked_accessions.add(pid.split('|')[-1])
        match=re.search(r'(?:AB-Bind:|SAbDab:)([A-Za-z0-9]{4})',pid)
        if match:blocked_pdb.add(match[1].upper())
        for annotation in r.get('annotations',[]):
            accession=annotation.get('accession')
            if accession:blocked_accessions.add(accession)
    todo=set()
    for r in rows:
        m=r['metadata'];pdb=str(m.get('pdb_id','')).upper();accession=m.get('uniprot_accession')
        if accession and accession not in blocked_accessions:todo.add(('uniprot',accession))
        if pdb and pdb not in blocked_pdb:todo.add(('rcsb_entry',pdb))
    for r in general:
        if r['previously_in_original100']:continue
        pid=r['protein_id'];pdb=r.get('pdb_id')
        if pid.startswith('curated_pdb|'):
            parts=pid.split('|');accession=parts[1];pdb=parts[2]
            if accession not in blocked_accessions:todo.add(('uniprot',accession))
        if r.get('domain_id'):pdb=r['domain_id'][:4]
        if pdb and str(pdb).upper() not in blocked_pdb:todo.add(('rcsb_entry',str(pdb).upper()))
    for r in audit:
        if r['status']!='HOLD':continue
        for a in r.get('annotations',[]):
            if a.get('accession') and a['accession'] not in blocked_accessions:todo.add(('uniprot',a['accession']))
    OUT.mkdir(parents=True,exist_ok=True)
    write(OUT/'request_manifest.json',dict(requests=[dict(kind=k,key=v) for k,v in sorted(todo)],
        metadata_only=True,no_model_input_release=True,blocked_accessions=sorted(blocked_accessions),
        blocked_pdb=sorted(blocked_pdb)))
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        for result in executor.map(lambda task:get(*task),sorted(todo)):
            results.append(result)
            if len(results)%20==0:
                write(OUT/'progress.json',dict(completed=len(results),initial_total=len(todo),
                      dispositions=dict(Counter(r['status'] for r in results))))
                print('PUBLIC_METADATA_COMPLETE '+str(len(results))+'/'+str(len(todo)),flush=True)
    entities=set()
    for r in results:
        if r['kind']=='rcsb_entry' and r['status']=='SUCCEEDED':
            for entity in read(OUT/'rcsb_entry'/(r['key']+'.json'))['metadata']['polymer_entity_ids']:
                entities.add(('rcsb_entity',r['key']+'_'+entity))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        for result in executor.map(lambda task:get(*task),sorted(entities)):
            results.append(result)
            if len(results)%20==0:
                write(OUT/'progress.json',dict(completed=len(results),total=len(todo)+len(entities),
                      dispositions=dict(Counter(r['status'] for r in results))))
                print('PUBLIC_ENTITY_METADATA_COMPLETE '+str(len(results)),flush=True)
    write(OUT/'completion.json',dict(status='METADATA_FETCH_FINISHED_PENDING_INDIVIDUAL_REVIEW',
        requests=len(results),by_status=dict(Counter(r['status'] for r in results)),
        reused=sum(r['reused'] for r in results),raw_sequences_saved=False,
        new_scope_approvals=0,new_model_calls=0,results=results))
    print('PUBLIC_METADATA_FETCH_FINISHED',flush=True)


if __name__=='__main__':main()
