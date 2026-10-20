"""Fetch only eligible assay CSV ranges from official PG_v1.3 score archive."""
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import urllib.request
import zipfile
import zlib

URL='https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/zero_shot_indels_scores.zip'
LOCAL=Path(__file__).resolve().parents[1]
OUT=LOCAL/'caches/recomb_public_baselines_20261017'


def byte_range(start,end):
    if end-start+1>65536:
        chunks=[];total=None
        for pos in range(start,end+1,65536):
            b,total=byte_range(pos,min(pos+65535,end));chunks.append(b)
        return b''.join(chunks),total
    request=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request,timeout=15) as r:
                if r.status!=206 or not r.headers['Content-Range'].startswith(f'bytes {start}-{end}/'):
                    raise ValueError('HTTP range identity/size mismatch')
                body=r.read(end-start+1)
                if len(body)!=end-start+1:raise ValueError('truncated remote ZIP member')
                return body,int(r.headers['Content-Range'].split('/')[-1])
        except (TimeoutError,OSError):
            if attempt==2:raise


class RemoteZip(io.RawIOBase):
    def __init__(self):self.pos=0;_,self.size=byte_range(0,3)
    def seekable(self):return True
    def seek(self,p,whence=0):self.pos=p if whence==0 else self.pos+p if whence==1 else self.size+p;return self.pos
    def tell(self):return self.pos
    def read(self,n=-1):
        n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
        if n<=0:return b''
        b,_=byte_range(self.pos,self.pos+n-1);self.pos+=len(b);return b


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=LOCAL/'caches/recomb_expanded_20261017'
    rows=[json.loads(l) for l in (data/'rows.jsonl').read_text().splitlines()]
    metadata=json.loads((data/'assay_metadata.json').read_text());wanted={}
    for r in rows:wanted.setdefault(r['assay_id'],set()).add(r['mutant_sequence'])
    with zipfile.ZipFile(RemoteZip()) as z:members={Path(i.filename).stem:i for i in z.infolist() if i.filename.endswith('.csv')}
    def fetch(assay):
        info=members[assay];path=OUT/(assay+'.csv')
        if path.exists():raw=path.read_bytes()
        else:
            header,_=byte_range(info.header_offset,info.header_offset+29)
            fields=struct.unpack('<4s5H3L2H',header);assert fields[0]==b'PK\x03\x04'
            name_len,extra_len=fields[-2:];start=info.header_offset+30+name_len+extra_len
            compressed,_=byte_range(start,start+info.compress_size-1)
            raw=zlib.decompress(compressed,-15) if info.compress_type==8 else compressed
        assert len(raw)==info.file_size and zlib.crc32(raw)&0xffffffff==info.CRC
        if not path.exists():path.write_bytes(raw)
        reader=csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
        if 'mutated_sequence' not in reader.fieldnames:raise ValueError('no sequence join key')
        matched={};conflicts=0
        for r in reader:
            seq=r['mutated_sequence']
            if seq in wanted[assay]:
                if seq in matched and r!=matched[seq]:conflicts+=1
                matched[seq]=r
        if conflicts:raise ValueError('conflicting score rows: '+assay)
        return dict(assay_id=assay,source_member=info.filename,compressed_bytes=info.compress_size,
                    raw_sha256=hashlib.sha256(raw).hexdigest(),columns=reader.fieldnames,
                    eligible_sequences=len(wanted[assay]),matched_sequences=len(matched)),matched
    reports=[];scores={}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for report,matched in pool.map(fetch,sorted(wanted)):
            reports.append(report);scores[report['assay_id']]=matched
            print('PUBLIC_ASSAY_SCORES '+report['assay_id']+' '+str(report['matched_sequences']),flush=True)
    joined=[]
    for row in rows:
        source=scores[row['assay_id']].get(row['mutant_sequence'])
        numeric={}
        if source:
            for k,v in source.items():
                if k in ('mutated_sequence','mutant','DMS_score','DMS_score_bin'):continue
                try:
                    value=float(v)
                    if math.isfinite(value):numeric[k]=value
                except (ValueError,TypeError):continue
        joined.append(dict(row_id=row['row_id'],assay_id=row['assay_id'],scores=numeric))
    with (OUT/'public_score_matrix.jsonl').open('w') as f:
        for r in joined:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
    report=dict(status='OFFICIAL_PG_V1_3_SEQUENCE_ALIGNED_PUBLIC_SCORES_FETCHED',url=URL,
        exact_assay_and_full_mutant_sequence_join=True,WT_reference_sha256='09534e16eeeb80ce22243f6cada151db654c7330117f2a97980f6707da87c7cd',
        assay_files=reports,assays=len(reports),observations=len(joined),
        downloaded_compressed_member_bytes=sum(x['compressed_bytes'] for x in reports),
        matrix_sha256=hashlib.sha256((OUT/'public_score_matrix.jsonl').read_bytes()).hexdigest(),
        columns=sorted(set(k for r in joined for k in r['scores'])),
        missing_sequences=sum(not r['scores'] for r in joined),new_model_downloads=0)
    (OUT/'coverage_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',newline='\n')
    print(json.dumps({k:v for k,v in report.items() if k!='assay_files'}),flush=True)


if __name__=='__main__':main()
