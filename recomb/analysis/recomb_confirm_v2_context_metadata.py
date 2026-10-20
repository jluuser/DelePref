"""Supplement cached candidate metadata with chain roles, never variant records."""
import argparse
from pathlib import Path
from recomb_confirm_v2_diagnostics import read,sha,write


def main(root):
    base=root/'reports/recomb_confirm_v2_inventory';out=base/'application_context_supplement.json'
    if out.exists():raise FileExistsError('Context metadata already exported')
    rows=[]
    for r in read(base/'application_candidate_metadata.json'):
        source=Path(r['source_file'])
        if sha(source)!=r['source_file_sha256']:raise ValueError('Candidate source changed')
        ref=read(source)['reference']
        keys=['audited_interaction_category','complex_category_evidence','interaction_category_rule_v1',
            'compressed_auth_chain_id','compressed_entity_id','partner_auth_chain_id','partner_entity_id',
            'biological_assembly_id','biological_relevance_approved','assembly_oligomeric_count',
            'coordinate_compressed_chain_ca_coverage','coordinate_partner_chain_ca_coverage',
            'coordinate_chain_ca_coverage','coordinate_coordinate_valid','reference_coordinate_path',
            'reference_coordinate_sha256','ec_class','uniprot_accession','pdb_id','auth_chain_id','pdb_entity_id',
            'coordinate_selected_label_asym_id','catalytic_residue_count']
        rows.append(dict(candidate_id=r['candidate_id'],cohort=r['cohort'],source_file_sha256=r['source_file_sha256'],
            metadata={k:ref[k] for k in keys if k in ref}))
    write(out,rows);print(dict(metadata_rows=len(rows),raw_sequences_exported=False,model_inputs_released=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
