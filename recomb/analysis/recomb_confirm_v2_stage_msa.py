"""Stage standard WT MSAs for reviewed development constructs; no prediction release."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'caches/msa_runtime'))
from recomb_confirm_v2_diagnostics import read,sha,write


def main():
    out=ROOT/'external/recomb_confirm_v2_WT_MSA_staging'
    if (out/'completion.json').exists():raise FileExistsError('MSA staging already completed')
    panel=read(ROOT/'reports/recomb_benign_application_constructs_20261017/panel.json')
    allowed={r['protein_id'] for r in read(ROOT/'reports/recomb_benign_application_scope_20261017/audit.json') if r['status']=='RETAIN'}
    if len(panel)!=23 or {r['protein_id'] for r in panel}!=allowed:
        raise ValueError('Not the positively reviewed 23 development constructs')
    queries={}
    for p in panel:
        chains=[dict(sequence=p['sequence'],chain_id=p.get('design_chain','A'),role='design')]
        chains += [dict(c,role='partner') for c in p.get('partners',[])]
        for c in chains:
            s=c['sequence'];h=hashlib.sha256(s.encode()).hexdigest()
            queries.setdefault(h,dict(sequence=s,sequence_sha256=h,uses=[]))['uses'].append(
                dict(protein_id=p['protein_id'],chain_id=c['chain_id'],role=c['role']))
    ordered=sorted(queries)
    vendor=ROOT/'caches/boltz_mmseqs2_official.py'
    write(out/'request_manifest.json',dict(status='REVIEWED_WT_QUERIES_ONLY_MSA_NOT_YET_REVIEWED',
        unique_sequences=len(queries),queries=[queries[h] for h in ordered],
        official_helper_sha256=sha(vendor),provider='https://api.colabfold.com',
        use_env=True,use_filter=True,use_pairing=False,prediction_release_allowed=False))
    spec=importlib.util.spec_from_file_location('recomb_official_mmseqs',vendor)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    start=time.monotonic()
    print('REVIEWED_WT_STANDARD_MSA_REQUEST_START '+str(len(queries)),flush=True)
    lines=module.run_mmseqs2([queries[h]['sequence'] for h in ordered],prefix=str(out/'official_unpaired'),
        use_env=True,use_filter=True,use_pairing=False,host_url='https://api.colabfold.com')
    if not isinstance(lines,list) or len(lines)!=len(queries):raise ValueError('Unexpected official MSA return')
    files=[]
    for h,msa in zip(ordered,lines):
        path=out/'a3m'/ (h+'.a3m');path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(msa,encoding='utf-8')
        files.append(dict(sequence_sha256=h,path=str(path),sha256=sha(path),
            records=sum(line.startswith('>') for line in msa.splitlines()),
            evidence_review='STAGED_UNREVIEWED_NO_PREDICTION_RELEASE'))
    write(out/'completion.json',dict(status='WT_MSA_STAGED_PENDING_SOURCE_REVIEW_AND_MULTICHAIN_PROFILE',
        development_constructs=23,unique_WT_and_partner_sequences=len(queries),
        seconds=time.monotonic()-start,files=files,new_structure_predictions=0,
        prediction_release_allowed=False))
    print('WT_MSA_STAGING_COMPLETE',flush=True)


if __name__=='__main__':main()
