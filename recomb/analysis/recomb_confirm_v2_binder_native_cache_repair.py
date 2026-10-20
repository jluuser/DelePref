"""Replace derived-assembly cache candidates by their original public mmCIFs."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import gzip
import hashlib
from pathlib import Path
import re
import time
import urllib.request
from recomb_confirm_v2_diagnostics import read,sha,write

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'external/recomb_confirm_v2_SAbDab_native_staging'


def main():
    if (BASE/'coordinate_manifest_v2.json').exists():raise FileExistsError('Coordinate repair already completed')
    old=read(BASE/'completion.json');bad=[r for r in old['records'] if r.get('server_cached')]
    if len(bad)!=9:raise ValueError('Cached-source repair inventory differs')
    def one(r):
        pdb=r['PDB'];url='https://files.rcsb.org/download/'+pdb+'.cif.gz'
        path=BASE/'coordinates'/(pdb+'.cif.gz');provenance=BASE/'original_source_repairs'/(pdb+'.json')
        if provenance.exists():
            d=read(provenance)
            if sha(ROOT/d['path'])==d['sha256']:return d
        errors=[]
        for i in range(3):
            try:
                with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'RECOMB-native-source-identity-repair/2.0'}),timeout=60) as h:data=h.read();etag=h.headers.get('ETag')
                full=gzip.decompress(data)
                if not re.search(rb'_entry\.id\s+[\x22\x27]?'+pdb.encode()+rb'[\x22\x27]?\s',full,re.I):raise ValueError('Original native entry identity differs')
                if path.exists() and path.read_bytes()!=data:raise ValueError('Do not overwrite a different staged original')
                path.write_bytes(data)
                d=dict(status='SUCCEEDED',PDB=pdb,path=path.relative_to(ROOT).as_posix(),sha256=sha(path),bytes=len(data),
                    server_cached=False,source_url=url,ETag=etag,fetched_at=datetime.now(timezone.utc).isoformat(),
                    previous_derived_cache=dict(path=r['path'],sha256=r['sha256'],reason='generated assembly entry identifier XXXX; old cache preserved'))
                write(provenance,d);return d
            except Exception as e:
                errors.append(repr(e))
                if i<2:time.sleep(2)
        raise RuntimeError(str(dict(PDB=pdb,errors=errors)))
    with ThreadPoolExecutor(max_workers=4) as ex:replacements=list(ex.map(one,bad))
    mapping={r['PDB']:r for r in replacements}
    records=[mapping.get(r['PDB'],r) for r in old['records']]
    write(BASE/'coordinate_manifest_v2.json',dict(status='ORIGINAL_NATIVE_COORDINATES_WITH_SAME_APPROVED_CASES',
        cases=351,PDB=265,records=records,old_completion_sha256=sha(BASE/'completion.json'),
        original_case_manifest_sha256=sha(BASE/'cases.json'),corrected_native_sources=9,
        old_freeze_and_derived_caches_preserved=True,scope_or_selection_rule_changed=False,new_model_calls=0))
    print(dict(originals_repaired=9,downloaded_bytes=sum(r['bytes'] for r in replacements),all_other_native_files_reused=256),flush=True)


if __name__=='__main__':main()
