"""Download the five predeclared public MaveDB assays for retrospective audit."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request

URNS=[f'urn:mavedb:00001219-a-{i}' for i in range(1,5)]+['urn:mavedb:00000100-b-1']
API='https://api.mavedb.org/api/v1/score-sets/'


def fetch(urn,out):
    record=dict(urn=urn,files={},failure=None)
    try:
        for suffix,extension in [('', 'json'),('/scores','csv')]:
            url=API+urn+suffix
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'RECOMB-retrospective-input-audit'}),timeout=30) as response:
                payload=response.read(5*1024*1024+1)
            if len(payload)>5*1024*1024:raise ValueError('fixed per-assay size cap exceeded')
            filename=urn.replace(':','_')+'.'+extension
            (out/filename).write_bytes(payload)
            record['files'][filename]=dict(url=url,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
            if extension=='json':
                metadata=json.loads(payload);record['metadata_keys']=list(metadata)
            else:
                text='\n'.join(line for line in payload.decode('utf-8-sig').splitlines() if not line.startswith('#'))
                rows=list(csv.DictReader(io.StringIO(text)))
                record['rows']=len(rows);record['columns']=list(rows[0]) if rows else []
                record['simple_protein_deletion_HGVS_rows']=sum(bool(re.fullmatch(r'p\.[A-Za-z*]{1,3}[0-9]+del',row.get('hgvs_pro',''))) for row in rows)
    except Exception as error:record['failure']=repr(error)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    with ThreadPoolExecutor(max_workers=5) as executor:records=list(executor.map(lambda urn:fetch(urn,a.output),URNS))
    report=dict(status='MAVEDB_INPUTS_STAGED' if all(x['failure'] is None for x in records) else 'MAVEDB_STAGING_PARTIAL',
                assays=records,training_or_generation=False,exact_legacy_1645_reconstruction_audit_pending=True,
                official_API_documentation='https://www.mavedb.org/docs/mavedb/programmatic-access/api-quickstart.html')
    (a.output/'staging_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)
