"""Audit native polymer chain identities and annotations without generating edits."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from recomb_scope_screen_20261017 import classify, describe, read, sha, write


def values(data, key):
    value = data.get(key, [])
    return value if isinstance(value, list) else [value]


def audit(root):
    out = root / 'reports/recomb_benign_application_scope_20261017'
    out.mkdir(exist_ok=False)
    policy = read(root / 'configs/recomb_benign_scope_20261017.json')
    reviewed = set(policy['application_reviewed_ids'])
    metadata_path = root / 'provenance/recomb_benign_scope_20261017/enzyme_function_metadata.json'
    entries = {x['primaryAccession']: x for x in read(metadata_path)['data']['results']}
    rows = read(root / 'reports/recomb_general100_applications_20261017/panel.json')
    result = []
    for row in rows:
        path = Path(row['native_reference_path'])
        if sha(path) != row['native_reference_sha256']:
            raise ValueError('native source changed')
        with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as f:
            data = MMCIF2Dict(f)
        entity_types = dict(zip(values(data, '_entity.id'), values(data, '_entity.type')))
        names = dict(zip(values(data, '_entity.id'), values(data, '_entity.pdbx_description')))
        organisms = {}
        for category, name_key in [('gen', 'pdbx_gene_src_scientific_name'), ('nat', 'pdbx_organism_scientific')]:
            for entity, organism in zip(values(data, '_entity_src_' + category + '.entity_id'),
                                        values(data, '_entity_src_' + category + '.' + name_key)):
                organisms.setdefault(entity, set()).add(organism)
        chain_entities = {}
        for chain, entity in zip(values(data, '_atom_site.auth_asym_id'), values(data, '_atom_site.label_entity_id')):
            if entity_types.get(entity) == 'polymer':
                chain_entities.setdefault(chain, set()).add(entity)
        chains = [row['design_chain']] + [p['chain_id'] for p in row.get('partners', [])]
        chain_audit = {chain: sorted(chain_entities.get(chain, [])) for chain in chains}
        complete = all(len(v) == 1 for v in chain_audit.values())
        entities = sorted({entity for ids in chain_audit.values() for entity in ids})
        metadata = [dict(entity_id=e, organism='; '.join(sorted(organisms.get(e, []))),
                         description=names.get(e), keywords=[]) for e in entities]
        if row['dataset'] == 'Enzyme20':
            accession = row['protein_id'].split('|')[-1]
            if accession in entries:
                metadata.append(describe(entries[accession]))
            else:
                complete = False
        status, reason = classify(metadata, row['protein_id'] in reviewed and complete)
        if not complete and status != 'EXCLUDE':
            status, reason = 'HOLD', 'incomplete_exact_polymer_chain_or_function_metadata'
        result.append(dict(protein_id=row['protein_id'], dataset=row['dataset'], WT_sha256=row['sequence_sha256'],
            status=status, reason=reason, chain_to_polymer=chain_audit, annotations=metadata,
            native_source_sha256=row['native_reference_sha256'], executable=False,
            next_gate='retrieval evidence-source audit and frozen screened inputs'))
    write(out / 'audit.json', result)
    summary = dict(status='APPLICATION_CHAIN_AND_FUNCTION_SCOPE_AUDITED_NOT_RELEASED',
        dataset_dispositions={name: dict(Counter(x['status'] for x in result if x['dataset'] == name)) for name in ['Binder20', 'Enzyme20']},
        policy_sha256=sha(root / 'configs/recomb_benign_scope_20261017.json'),
        enzyme_metadata_sha256=sha(metadata_path), new_GPU_calls=0, new_structure_calls=0,
        application_generation_released=False)
    write(out / 'completion.json', summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
    audit(p.parse_args().root)
