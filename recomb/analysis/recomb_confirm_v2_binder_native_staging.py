"""Stage native coordinates only for explicitly source/function-reviewed binders."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request
from recomb_confirm_v2_diagnostics import read,sha,write

BASE=Path('external/recomb_confirm_v2_SAbDab_native_staging')


def prepare(root):
    out=root/BASE
    if (out/'request_manifest.json').exists():raise FileExistsError('Reuse the existing native manifest')
    source=root/'reports/recomb_confirm_v2_expansion_review/SAbDab_metadata_review.json'
    scopepath=root/'reports/recomb_confirm_v2_design_function_review/per_candidate_function_review_v2.json'
    scope={r['candidate_id']:r for r in read(scopepath)}
    cases=[]
    for r in read(source):
        pdb=r['pdb'].upper();cid='SAbDab|'+pdb+'|'+r['prospective_design_chain']
        if r['metadata_review_status']!='PASS_SOURCE_AND_CHAIN_ROLE_METADATA':continue
        if scope.get(cid,{}).get('status')!='FUNCTIONAL_SCOPE_PASS_NATIVE_AND_FAMILY_PENDING':continue
        identity='|'.join([pdb,r['prospective_design_chain'],str(r.get('fixed_light_chain')),r['antigen_chain']])
        key=hashlib.sha256(identity.encode()).hexdigest()
        cases.append(dict(r,candidate_id=cid,native_case_key=key,native_input_audit_allowed=True,
            model_use_allowed=False,full_construct_and_family_audit_pending=True))
    if len(cases)!=351:raise ValueError('Latest approved Binder scope differs')
    keys=[r['native_case_key'] for r in cases]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate SAbDab native chain-role tuple')
    write(out/'cases.json',cases)
    write(out/'request_manifest.json',dict(status='APPROVED_BINDER_NATIVE_INPUT_AUDIT_ONLY',
        PDB=sorted({r['pdb'].upper() for r in cases}),cases=len(cases),
        scope_sha256=sha(scopepath),metadata_sha256=sha(source),cases_sha256=sha(out/'cases.json'),
        new_model_calls=0,model_release_allowed=False,source_sha256=sha(Path(__file__))))
    print(dict(cases=len(cases),PDB=len({r['pdb'].upper() for r in cases})),flush=True)


def cache_inventory(root):
    out=root/BASE;manifest=read(out/'request_manifest.json');allowed=set(manifest['PDB']);cached={}
    inventory=read(root/'reports/recomb_confirm_v2_inventory/application_candidate_metadata.json')
    for r in inventory:
        m=r.get('metadata',{});pdb=str(m.get('pdb_id','')).upper()
        if pdb not in allowed:continue
        source=Path(r['source_file'])
        if sha(source)!=r['source_file_sha256']:raise ValueError('Existing native source metadata changed')
        ref=read(source).get('reference',{})
        path=ref.get('reference_coordinate_path') or ref.get('reference_structure_path')
        if path and Path(path).is_file():cached.setdefault(pdb,dict(path=path,sha256=sha(path),source_metadata=str(source)))
    # Native enzyme/whole-chain inventories also contain immutable complete PDB files.
    for r in read(root/'reports/recomb_confirm_v2_native_audit_release/eligible_WT_native_audit_inputs.json'):
        pdb=str(r.get('native_reference_pdb','')).upper();path=r.get('native_reference_path')
        if pdb in allowed and path and Path(path).is_file():cached.setdefault(pdb,dict(path=path,sha256=sha(path)))
    write(out/'cached_native_files.json',dict(status='CACHED_NATIVE_CANDIDATES_NOT_YET_CONSTRUCT_VERIFIED',
        files=cached,PDB_requested=len(allowed),cached_PDB=len(cached),manifest_sha256=sha(out/'request_manifest.json')))
    print(dict(cached_PDB=len(cached),missing_PDB=len(allowed)-len(cached)),flush=True)


def fetch(root):
    out=root/BASE
    if (out/'completion.json').exists():raise FileExistsError('Native staging already complete')
    manifest=read(out/'request_manifest.json');cached=read(out/'cached_native_files.json')
    if sha(out/'request_manifest.json')!=cached['manifest_sha256']:raise ValueError('Native request identity differs')
    (out/'coordinates').mkdir(parents=True,exist_ok=True);(out/'records').mkdir(parents=True,exist_ok=True)
    def one(pdb):
        if not re.fullmatch('[A-Z0-9]{4}',pdb):raise ValueError('Invalid approved PDB ID')
        record=out/'records'/(pdb+'.json')
        if record.exists():
            d=read(record)
            if d['status']=='SUCCEEDED' and (d.get('server_cached') or sha(root/d['path'])==d['sha256']):return d
        if pdb in cached['files']:
            d=dict(cached['files'][pdb],PDB=pdb,status='SUCCEEDED',server_cached=True)
            write(record,d);return d
        url='https://files.rcsb.org/download/'+pdb+'.cif.gz';errors=[]
        for attempt in range(3):
            try:
                req=urllib.request.Request(url,headers={'User-Agent':'RECOMB-native-input-audit/2.0'})
                with urllib.request.urlopen(req,timeout=80) as r:raw=r.read();etag=r.headers.get('ETag')
                data=gzip.decompress(raw)
                if not re.search(rb'_entry\.id\s+[\x22\x27]?'+pdb.encode()+rb'[\x22\x27]?\s',data,re.I):
                    raise ValueError('Downloaded native entry identity differs')
                path=out/'coordinates'/(pdb+'.cif.gz');path.write_bytes(raw)
                d=dict(status='SUCCEEDED',PDB=pdb,path=path.relative_to(root).as_posix(),sha256=sha(path),bytes=len(raw),
                    source_url=url,ETag=etag,server_cached=False,fetched_at=datetime.now(timezone.utc).isoformat())
                write(record,d);return d
            except Exception as e:
                errors.append(repr(e))
                if attempt<2:time.sleep(2)
        d=dict(status='FAILED',PDB=pdb,source_url=url,errors=errors);write(record,d);return d
    results=[]
    with ThreadPoolExecutor(max_workers=4) as ex:
        for r in ex.map(one,manifest['PDB']):
            results.append(r)
            if len(results)%20==0:
                write(out/'progress.json',dict(completed=len(results),PDB=len(manifest['PDB']),disposition=dict(Counter(x['status'] for x in results))))
                print('BINDER_NATIVE_STAGED '+str(len(results))+'/'+str(len(manifest['PDB'])),flush=True)
    report=dict(status='BINDER_NATIVE_COORDINATES_STAGED_CONSTRUCT_AND_FAMILY_AUDIT_PENDING',
        PDB=len(results),cases=manifest['cases'],disposition=dict(Counter(r['status'] for r in results)),
        cached_PDB=sum(r.get('server_cached',False) for r in results),
        downloaded_bytes=sum(r.get('bytes',0) for r in results),new_model_calls=0,new_structure_predictions=0,
        request_manifest_sha256=sha(out/'request_manifest.json'),model_release_allowed=False,records=results)
    write(out/'completion.json',report);print({k:v for k,v in report.items() if k!='records'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','inventory','fetch']);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    a=p.parse_args();{'prepare':prepare,'inventory':cache_inventory,'fetch':fetch}[a.mode](a.root)
