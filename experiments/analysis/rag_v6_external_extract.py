"""Reconstruct the four ProteinGym assay inputs named in the historical table.

Only retrospective measured rows are staged; no variant is generated or ranked.
Ambiguous deletion maps are retained explicitly rather than filtered by outcomes.
"""
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013')
EXPECTED={'DMS_ind_3':1,'DMS_ind_6':258,'DMS_ind_40':314,'DMS_ind_61':50}


def run():
    base=ROOT/'external/rag_v6'
    receipt=json.loads((base/'staging_report.local_recovery.json').read_text())
    for name,meta in receipt['files'].items():
        if hashlib.sha256((base/name).read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('official input changed')
    metadata=list(csv.DictReader((base/'DMS_indels.csv').open()))
    records,summary=[],[]
    with zipfile.ZipFile(base/'DMS_ProteinGym_indels.zip') as archive:
        for meta in metadata:
            index=meta['DMS_index']
            if index not in EXPECTED:continue
            names=[p for p in archive.namelist() if Path(p).name==meta['DMS_filename']]
            if len(names)!=1:raise ValueError('ambiguous official assay member')
            wt=meta['target_seq']; count=ambiguous=0
            for ordinal,row in enumerate(csv.DictReader(io.StringIO(archive.read(names[0]).decode('utf-8-sig')))):
                edited=row['mutated_sequence']
                if len(edited)!=len(wt)-1:continue
                maps=[i for i in range(len(wt)) if wt[:i]+wt[i+1:]==edited]
                if not maps:continue
                value=float(row['DMS_score'])
                identity=f'{index}:{ordinal}'
                records.append(dict(row_id=identity,DMS_index=index,DMS_id=meta['DMS_id'],
                    UniProt_ID=meta['UniProt_ID'],wt_sequence=wt,sequence=edited,
                    possible_deleted_positions=maps,alignment_ambiguous=len(maps)>1,
                    DMS_score=value,score_direction='ProteinGym_DMS_score_already_higher_is_better',
                    source_member=names[0],source_row=ordinal,usage='retrospective_evaluation_only'))
                count+=1;ambiguous+=len(maps)>1
            summary.append(dict(DMS_index=index,observed_single_deletion_rows=count,
                                historical_expected=EXPECTED[index],ambiguous_maps=ambiguous))
    output=base/'legacy_proteingym_single_deletion_rows.jsonl'
    with output.open('x',encoding='utf-8') as handle:
        for row in records:handle.write(json.dumps(row,sort_keys=True)+'\n')
    report=dict(status='PROTEINGYM_COMPONENT_RECONSTRUCTED',rows=len(records),assays=summary,
                records_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                historical_counts_match=all(x['observed_single_deletion_rows']==x['historical_expected'] for x in summary),
                MaveDB_assays_still_needed=5,legacy_1645_rows_restored=False,
                variants_generated=0,models_fit=0,variants_ranked=0,
                ambiguity_policy='retain_all_possible_maps; scoring policy must be frozen before external evaluation')
    (base/'legacy_component_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':run()
