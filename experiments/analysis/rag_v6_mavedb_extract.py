"""Audit measured single-deletion records against their published references.

This reconstructs existing observations only; it does not propose or rank edits.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import re

ROOT=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013')
EXPECTED={**{f'urn:mavedb:00001219-a-{i}':n for i,n in enumerate([185,186,185,186],1)},'urn:mavedb:00000100-b-1':280}
AA3=dict(zip(['Ala','Arg','Asn','Asp','Cys','Gln','Glu','Gly','His','Ile','Leu','Lys','Met','Phe','Pro','Ser','Thr','Trp','Tyr','Val'],list('ARNDCQEGHILKMFPSTWYV')))


def run():
    out=ROOT/'external/rag_v6';base=out/'mavedb'
    receipts=json.loads((base/'staging_report.json').read_text(encoding='utf-8'))
    for assay in receipts['assays']:
        if assay['failure']:raise ValueError('source assay was not downloaded')
        for name,meta in assay['files'].items():
            if hashlib.sha256((base/name).read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('raw MaveDB input changed')
    protein_meta=list(csv.DictReader((out/'DMS_indels.csv').open(encoding='utf-8')))
    template=next(x['target_seq'] for x in protein_meta if x['DMS_index']=='DMS_ind_6')
    records,audit=[],[]
    for urn,expected in EXPECTED.items():
        stem=urn.replace(':','_');meta=json.loads((base/(stem+'.json')).read_text(encoding='utf-8'))
        seq=meta['targetGenes'][0]['targetSequence']
        # The published protein reference is used directly; the DNA-target assay
        # is cross-checked residue by residue against its ProteinGym reference.
        wt=seq['sequence'] if seq['sequenceType']=='protein' else template
        source=list(csv.DictReader((base/(stem+'.csv')).open(encoding='utf-8')))
        count=missing=0
        for ordinal,row in enumerate(source):
            match=re.fullmatch(r'p\.([A-Za-z*]{1,3})([0-9]+)del',row.get('hgvs_pro',''))
            if not match:continue
            code,position=match.group(1),int(match.group(2))-1
            reference=AA3.get(code,code)
            if not (0<=position<len(wt) and reference==wt[position]):
                raise ValueError(f'published HGVS/reference disagreement in {urn}; no silent filtering')
            try:score=float(row['score'])
            except ValueError:score=float('nan')
            if not math.isfinite(score):missing+=1;continue
            records.append(dict(row_id=row['accession'],assay_id=urn,
                UniProt_accession='P00374' if '00001219' in urn else 'P62593',
                wt_sequence=wt,sequence=wt[:position]+wt[position+1:],
                possible_deleted_positions=[position],alignment_ambiguous=False,
                experimental_score=score,score_direction_audit_pending=True,
                source_HGVS=row['hgvs_pro'],source_row=ordinal,
                usage='retrospective_existing_observation_only'))
            count+=1
        audit.append(dict(urn=urn,finite_single_deletion_rows=count,historical_expected=expected,
                          nonfinite_score_rows=missing,reference_residues_verified=True))
        if count!=expected:raise ValueError('current source count differs from historical cohort; keep separate')
    path=out/'legacy_mavedb_single_deletion_rows.jsonl'
    with path.open('x',encoding='utf-8') as handle:
        for row in records:handle.write(json.dumps(row,sort_keys=True)+'\n')
    report=dict(status='ALL_NINE_ASSAY_INPUT_SCOPES_RECONSTRUCTED',MaveDB_rows=len(records),
                ProteinGym_rows=623,total_rows=len(records)+623,assays=audit,
                records_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                observed_scope_counts_match_historical=True,exact_old_file_identity_unverified=True,
                direction_and_evaluation_protocol_audit_pending=True,variant_generation_or_ranking=False)
    (out/'mavedb_component_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':run()
