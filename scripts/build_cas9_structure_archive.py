"""Export frozen Cas9 complex structures, records and reference inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from build_structure_archive import add_bytes, add_file, canonical_json, normalize, sha256_file


CASES = ('SpCas9', 'SaCas9', 'Nme1Cas9', 'GeoCas9', 'FnCas9')


def build(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError('Output must be outside research source')
    if output.exists():
        raise FileExistsError('Use a new output path: ' + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    records, references, analyses = [], [], []
    with tarfile.open(output, 'w:gz', compresslevel=1) as tar:
        for case in CASES:
            folder = root / ('reports/recomb_cas9_20261020' if case == 'SpCas9'
                             else 'reports/recomb_cas9_panel_20261020/' + case)
            for record_path in sorted((folder/'records').glob('*.json')):
                raw = record_path.read_bytes(); record = json.loads(raw.decode('utf8'))
                task = record['task_id']
                payload = canonical_json(normalize(record, root))
                name = 'records/{}/{}.json'.format(case, task)
                add_bytes(tar, name, payload)
                item = dict(case=case, physical_id=task, status=record['status'], record=name,
                    original_record_sha256=hashlib.sha256(raw).hexdigest(),
                    normalized_record_sha256=hashlib.sha256(payload).hexdigest())
                if record['status'] == 'SUCCEEDED':
                    source = folder/'structures'/(task+'.cif')
                    artifact = add_file(tar, root, source, 'predictions/{}/{}.cif'.format(case,task))
                    if artifact['sha256'] != record['sha256']:
                        raise ValueError('Cas9 prediction checksum mismatch: ' + task)
                    item['structure'] = artifact
                records.append(item)
            for source in sorted((folder/'reference').glob('*')):
                if not source.is_file() or not source.name.endswith(('.json','.cif.gz','.pdb.gz')):
                    continue
                name = 'references/{}/{}'.format(case,source.name)
                raw = source.read_bytes()
                if source.suffix == '.json':
                    payload = canonical_json(normalize(json.loads(raw.decode('utf8')),root))
                    add_bytes(tar,name,payload)
                    references.append(dict(case=case,path=name,original_sha256=hashlib.sha256(raw).hexdigest(),
                                           sha256=hashlib.sha256(payload).hexdigest()))
                else:
                    item=add_file(tar,root,source,name); item['case']=case; references.append(item)
            native=folder/'native_reference.json'
            if native.is_file():
                raw=native.read_bytes(); payload=canonical_json(normalize(json.loads(raw.decode('utf8')),root))
                name='references/{}/native_reference.json'.format(case)
                add_bytes(tar,name,payload)
                references.append(dict(case=case,path=name,original_sha256=hashlib.sha256(raw).hexdigest(),
                                       sha256=hashlib.sha256(payload).hexdigest()))
        panel = root/'reports/recomb_cas9_panel_20261020'
        for filename in ('all90_main_per_seed.csv','panel_summary.json','logical_requests.json','physical_requests.json'):
            source=panel/filename; raw=source.read_bytes()
            payload=canonical_json(normalize(json.loads(raw.decode('utf-8-sig')),root)) if filename.endswith('.json') else raw
            name='analysis/'+filename
            add_bytes(tar,name,payload)
            analyses.append(dict(path=name,original_sha256=hashlib.sha256(raw).hexdigest(),
                                 sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload)))
        if len(records)!=88:
            raise ValueError('Expected 88 Cas9 prediction records, found {}'.format(len(records)))
        manifest=dict(schema='delepref-cas9-structures-v1',cases=list(CASES),
                      prediction_records=records,native_references=references,analysis_files=analyses,
                      source_paths='Server-local paths normalized; original checksums retained.')
        add_bytes(tar,'manifest.json',canonical_json(manifest))
    print(json.dumps(dict(status='EXPORTED',records=len(records),references=len(references),
                          bytes=output.stat().st_size,sha256=sha256_file(output))),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();build(a.root,a.out)
