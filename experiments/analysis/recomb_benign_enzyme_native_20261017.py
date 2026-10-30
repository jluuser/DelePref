"""Reuse existing consensus native mappings only for positively reviewed enzymes."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.SeqUtils import seq1
from recomb_scope_screen_20261017 import read,sha,write


def norm(value):return '' if value in (None,'.','?') else str(value)


def build(root):
    base=Path('/public/nfs-data/keyuan.zhou/evocompress/results/formal_v9/revision_20260906_abc')
    mapping=base/'B/e2_predictive_v2/reference_coordinate_coverage_consensus_v3.json'
    cohort=base/'A/E2_canonical/task_cohorts_canonical_v2/task_cohort.jsonl'
    refs={r['protein_id']:r for r in read(mapping)['rows']}
    old={r['protein_id']:r for r in (json.loads(line) for line in cohort.read_text().splitlines())}
    panel=root/'reports/recomb_benign_application_constructs_20261017/panel.json'
    allowed={r['protein_id']:r for r in read(root/'reports/recomb_benign_application_scope_20261017/audit.json') if r['status']=='RETAIN'}
    out=root/'reports/recomb_benign_enzyme_native_20261017';out.mkdir(exist_ok=False)
    output=[]
    for row in read(panel):
        if row['dataset']!='Enzyme20':continue
        if row['protein_id'] not in allowed or row['sequence_sha256']!=allowed[row['protein_id']]['WT_sha256']:
            raise ValueError('unreviewed or changed enzyme input')
        pid=row['protein_id'].split('|')[-1];wt=row['sequence'];ref=refs[pid]
        if old[pid]['wt_sequence']!=wt or old[pid]['wt_sequence_sha256']!=row['sequence_sha256'] or ref['wt_length']!=len(wt):
            raise ValueError('canonical WT identity changed')
        if ref['alignment_backend']!='all_optimal_alignment_consensus':raise ValueError('wrong legacy mapping contract')
        path=Path(row['native_reference_path'])
        if sha(path)!=row['native_reference_sha256']:raise ValueError('native file changed')
        with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:d=MMCIF2Dict(f)
        atom={k.split('.')[-1]:v for k,v in d.items() if k.startswith('_atom_site.')}
        coordinates={}
        for i in range(len(atom['id'])):
            if atom['label_atom_id'][i]!='CA' or atom.get('pdbx_PDB_model_num',['1']*len(atom['id']))[i]!='1' or atom.get('label_alt_id',['.']*len(atom['id']))[i] not in ('.','?','A'):continue
            key=tuple(norm(atom[t][i]) for t in ['label_asym_id','label_seq_id','auth_asym_id','auth_seq_id','pdbx_PDB_ins_code'])
            coordinates[key]=[float(atom[t][i]) for t in ['Cartn_x','Cartn_y','Cartn_z']]
        xyz=[None]*len(wt);mapped=[]
        for residue in ref['residues']:
            if residue.get('wt_index') is None:continue
            i=int(residue['wt_index'])
            code=seq1(residue['monomer'],custom_map={'MSE':'M'},undef_code='X')
            if not 0<=i<len(wt) or code!=wt[i]:raise ValueError('legacy consensus residue identity mismatch')
            key=tuple(norm(residue.get(t)) for t in ['label_chain','label_seq_id','auth_chain','auth_seq_id','insertion_code'])
            if key in coordinates:xyz[i]=coordinates[key]
            mapped.append(residue)
        for site in ref['catalytic']:
            if site['status']=='PASS' and wt[int(site['wt_index'])]!=site['wt_residue']:
                raise ValueError('catalytic correspondence mismatch')
        output.append(dict(protein_id=row['protein_id'],WT_sha256=row['sequence_sha256'],WT_length=len(wt),
            native_CA=xyz,consensus_residue_mapping=mapped,catalytic=ref['catalytic'],
            known_CA_count=sum(v is not None for v in xyz),unknown_CA_count=sum(v is None for v in xyz),
            alignment_backend=ref['alignment_backend'],optimal_alignment_count=ref['optimal_alignment_count'],
            excluded_nonconsensus_entity_positions=ref['excluded_nonconsensus_entity_positions'],
            native_file_sha256=row['native_reference_sha256']))
    if len(output)!=11:raise ValueError('screened enzyme count changed')
    write(out/'native_references.json',output)
    summary=dict(status='SCREENED_ENZYME_LEGACY_CONSENSUS_MAPPING_VERIFIED',WT=len(output),
        all_canonical_WT_identities_match=True,all_mapped_residue_identities_match=True,
        no_new_alignment_or_variant_prediction=True,unknown_coordinates_preserved=True,
        native_CA_coverage=[round(r['known_CA_count']/r['WT_length'],4) for r in output],
        source_mapping_sha256=sha(mapping),source_cohort_sha256=sha(cohort),panel_sha256=sha(panel),
        output_sha256=sha(out/'native_references.json'),new_model_calls=0,new_structure_predictions=0)
    write(out/'completion.json',summary);print(json.dumps(summary))


if __name__=='__main__':build(Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
