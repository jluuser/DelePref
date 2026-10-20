"""Check query/donor identities and the actual Boltz CSV parser, CPU only."""
import csv
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
from boltz.data.parse.csv import parse_csv
from recomb_confirm_v2_diagnostics import read,sha,write

root=Path(__file__).resolve().parents[1]
base=root/'reports/recomb_confirm_v2_uniform_WT_MSA'
if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
if (base/'parser_preflight.json').exists():raise FileExistsError('Uniform native parser preflight already complete')
done=read(base/'completion.json');manifest=base/'csv_manifest.json'
if sha(manifest)!=done['manifest_sha256']:raise ValueError('Uniform WT MSA manifest changed')
if sha(root/'configs/recomb_confirm_v2_uniform_WT_MSA_freeze.json')!=done['freeze_sha256']:raise ValueError('Uniform MSA protocol changed')
frozen=read(root/'configs/recomb_confirm_v2_uniform_WT_MSA_freeze.json')
for p,h in frozen['files'].items():
    if sha(root/p)!=h:raise ValueError('Uniform alignment frozen source changed')
query_bysha={q['sequence_sha256']:q['sequence'] for q in read(root/'external/recomb_confirm_v2_application_WT_MSA_staging/request_manifest.json')['queries']}
result=[]
for record in read(manifest):
    path=root/record['CSV_relative_path']
    if sha(path)!=record['CSV_sha256']:raise ValueError('Aligned CSV identity changed')
    with path.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    q=query_bysha[record['query_sha256']]
    if rows[0]['sequence']!=q:raise ValueError('CSV query identity differs')
    donor_hashes=[hashlib.sha256(r['sequence'].replace('-','').upper().encode()).hexdigest() for r in rows[1:]]
    if donor_hashes!=record['donor_fragment_hashes']:raise ValueError('CSV donor amino-acid content changed')
    msa=parse_csv(path)
    if len(msa.sequences)<1 or any(int(s['res_end'])-int(s['res_start'])!=len(q) for s in msa.sequences):
        raise ValueError('Native Boltz parser query width differs')
    if any(int(s['taxonomy'])!=-1 for s in msa.sequences):raise ValueError('Artificial MSA pairing introduced')
    result.append(dict(query_sha256=record['query_sha256'],input_rows=len(rows),native_parsed_unique_rows=len(msa.sequences),
        width=len(q),lowercase_insertion_AA=sum(c.islower() for r in rows[1:] for c in r['sequence'])))
source=Path(inspect.getfile(parse_csv))
freeze=dict(status='UNIFORM_WT_MSA_NATIVE_PARSER_AND_DONOR_IDENTITY_PASS',
    alignment_freeze_sha256=done['freeze_sha256'],CSV_manifest_sha256=sha(manifest),
    preflight_source_sha256=sha(Path(__file__)),Boltz_version=importlib.metadata.version('boltz'),
    native_parser_path=str(source),native_parser_sha256=sha(source),
    native_MSA_representation='unchanged native Boltz parsing; lowercase donor insertions follow its insertion-count representation',
    new_model_calls=0,new_structure_predictions=0)
write(root/'configs/recomb_confirm_v2_uniform_WT_MSA_parser_freeze.json',freeze)
write(base/'parser_preflight.json',dict(status=freeze['status'],queries=len(result),results=result,
    native_parser_freeze_sha256=sha(root/'configs/recomb_confirm_v2_uniform_WT_MSA_parser_freeze.json'),
    job_id=os.environ['SLURM_JOB_ID'],new_model_calls=0,new_structure_predictions=0))
print(freeze['status'],len(result),flush=True)
