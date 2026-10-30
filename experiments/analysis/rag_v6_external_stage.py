"""Stage official external evaluation inputs without running or selecting variants."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013')
REPO='https://api.github.com/repos/OATML-Markslab/ProteinGym'
DATA='https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_indels.zip'


def request(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'RECOMB-reproducible-data-staging'}),timeout=30)


def stage(output=None):
    out=Path(output) if output is not None else ROOT/'external/rag_v6'
    out.mkdir(parents=True,exist_ok=False)
    report=dict(status='STAGING',official_repository='https://github.com/OATML-Markslab/ProteinGym',
                official_data_version='v1.3',new_model_fits=0,variant_generation=0,
                legacy_1645_rows_restored=False,files={},failures=[])
    try:
        with request(REPO+'/commits/main') as response: commit=json.load(response)['sha']
        report['repository_commit']=commit
        url=f'https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/{commit}/reference_files/DMS_indels.csv'
        with request(url) as response: payload=response.read(5*1024*1024)
        records=list(csv.DictReader(io.StringIO(payload.decode('utf-8-sig'))))
        if not records or 'DMS_id' not in records[0]:raise ValueError('unexpected official reference schema')
        (out/'DMS_indels.csv').write_bytes(payload)
        report['files']['DMS_indels.csv']=dict(url=url,sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload),rows=len(records),fields=list(records[0]))
    except Exception as error: report['failures'].append(dict(stage='reference_metadata',error=repr(error)))
    try:
        partial=out/'DMS_ProteinGym_indels.zip.partial';digest=hashlib.sha256();total=0
        with request(DATA) as response,partial.open('xb') as handle:
            length=response.headers.get('Content-Length')
            if length and int(length)>128*1024*1024:raise ValueError('compressed download exceeds fixed 128MiB limit')
            while True:
                block=response.read(1024*1024)
                if not block:break
                total+=len(block)
                if total>128*1024*1024:raise ValueError('download exceeds fixed compressed limit')
                digest.update(block);handle.write(block)
        with zipfile.ZipFile(partial) as archive:
            members=archive.infolist()
            if any(Path(x.filename).is_absolute() or '..' in Path(x.filename).parts for x in members):
                raise ValueError('unsafe archive path')
            member_names=[x.filename for x in members]
        final=out/'DMS_ProteinGym_indels.zip';partial.rename(final)
        report['files'][final.name]=dict(url=DATA,sha256=digest.hexdigest(),bytes=total,members=member_names,
                                       uncompressed_bytes=sum(x.file_size for x in members),extracted=False)
    except Exception as error:report['failures'].append(dict(stage='raw_indel_archive',error=repr(error)))
    report['status']='EXTERNAL_INPUTS_STAGED' if not report['failures'] else 'EXTERNAL_STAGING_PARTIAL'
    report['interpretation']='Official versioned inputs; original AAAI ProteinGym/MaveDB row cohort requires explicit reconstruction audit before calling it replicated.'
    (out/'staging_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='files'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    stage(args.output)
