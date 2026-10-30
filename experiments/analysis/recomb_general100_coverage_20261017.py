"""Describe fixed-panel input coverage using already downloaded annotations."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/recomb_general100_input_coverage_20261017'


def main():
    panel=json.loads((ROOT/'reports/delepref_v8_native_panel/main_panel.json').read_text(encoding='utf-8'))
    native={r['protein_id']:r for r in json.loads((ROOT/'reports/delepref_v8_native_annotations/per_WT_native_features.json').read_text(encoding='utf-8'))}
    mapped={r['protein_id']:r for r in json.loads((ROOT/'reports/recomb_functional_annotations_20261017/mapped_features.json').read_text(encoding='utf-8'))}
    entries={}
    for p in (ROOT/'provenance/recomb_functional_annotations_20261017/uniprot').glob('*.json'):
        entries[p.stem]=json.loads(p.read_text(encoding='utf-8'))
    rows=[]
    for p in panel:
        pid=p['protein_id'];annotation=mapped[pid]
        possible=[entries[a] for a in annotation.get('candidate_accessions',[]) if a in entries]
        taxa={str(e.get('organism',{}).get('taxonId')) for e in possible}
        kingdoms={e.get('organism',{}).get('lineage',['UNKNOWN'])[0] for e in possible if e.get('organism',{}).get('lineage')}
        taxon=next(iter(taxa)) if len(taxa)==1 else 'UNKNOWN_OR_AMBIGUOUS'
        kingdom=next(iter(kingdoms)) if len(kingdoms)==1 else 'UNKNOWN_OR_AMBIGUOUS'
        n=native[pid];helix=len(n['deposited_helix_positions']);sheet=len(n['deposited_sheet_positions'])
        secondary=('helix_and_sheet_records' if helix and sheet else 'helix_records_only' if helix else
                   'sheet_records_only' if sheet else 'no_deposited_secondary_records')
        unique=entries.get(annotation.get('accession')) if annotation.get('mapping_status')=='UNIQUE_EXACT_SUBSEQUENCE_MAP' else None
        enzyme=any(c.get('commentType')=='CATALYTIC ACTIVITY' for c in (unique or {}).get('comments',[]))
        domain_features=[f for f in annotation.get('features',[]) if f['type']=='Domain']
        rows.append(dict(protein_id=pid,WT_sha256=p['sequence_sha256'],source=p['stratum'],length=p['sequence_length'],
            native_CA_coverage=p['wt_CA_coverage'],cath_classes=p.get('cath_h_superfamilies',[]),
            deposited_secondary_record_type=secondary,helix_positions=helix,sheet_positions=sheet,
            taxon_id=taxon,superkingdom=kingdom,taxonomy_source='PDBe/UniProt chain candidates, consensus if all candidates agree',
            functional_mapping_status=annotation['mapping_status'],has_unambiguous_catalytic_activity_annotation=enzyme,
            mapped_domain_feature_count=len(domain_features),
            domain_architecture_scope='isolated_CATH_domain_construct' if p['stratum']=='cath_domain' else
                 ('multiple_mapped_domain_annotations' if len(domain_features)>1 else 'chain_construct_domain_count_not_established')))
    report=dict(status='FIXED_GENERAL100_INPUT_ONLY_COVERAGE_DESCRIBED',WT=100,
        source=dict(Counter(r['source'] for r in rows)),
        length_bins={'150-199':sum(150<=r['length']<=199 for r in rows),'200-349':sum(200<=r['length']<=349 for r in rows),
                     '350-529':sum(350<=r['length']<=529 for r in rows)},
        superkingdom=dict(Counter(r['superkingdom'] for r in rows)),
        deposited_secondary_record_type=dict(Counter(r['deposited_secondary_record_type'] for r in rows)),
        functional_mapping_status=dict(Counter(r['functional_mapping_status'] for r in rows)),
        unambiguously_annotated_catalytic_activity_WT=sum(r['has_unambiguous_catalytic_activity_annotation'] for r in rows),
        multiple_mapped_domains_chain_WT=sum(r['domain_architecture_scope']=='multiple_mapped_domain_annotations' for r in rows),
        no_roster_members_changed=True,no_retrieval_or_model_outcomes_read=True,new_GPU_calls=0,
        caveats=['Deposited HELIX/SHEET record types are not CATH class or a new DSSP assignment',
                 'Missing functional/domain annotation is unknown, not absence',
                 'General100 contains53 chain and47 domain constructs, length150-529; not all protein classes'])
    OUT.mkdir(exist_ok=False)
    (OUT/'per_WT.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    report['per_WT_sha256']=hashlib.sha256((OUT/'per_WT.json').read_bytes()).hexdigest()
    (OUT/'completion.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
