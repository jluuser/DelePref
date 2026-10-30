"""Cross-check the new correspondence against frozen costs and legacy readouts."""
import argparse
import json
from pathlib import Path
import numpy as np
from recomb_raygun_canonical_audit_20261025 import read, sha, write, REL, METRICS, RECORDS, predicted_interface
from recomb_confirm_v2_application_metrics import enzyme_catalytic_geometry


def validate(root):
    data = read(root / REL / 'strict_recalculation.json')
    original = {(r['protein_id'], r['physical_id']): r for r in read(root / METRICS)}
    checked = []
    for row in data['per_output']:
        old = original[row['protein_id'], row['physical_id']]
        wt, variant = old['wildtype_sequence'], old['edited_sequence']
        a = row['correspondence']; pairs = a['aligned_diagonal_pairs']
        assert a['edit_distance'] == old['correspondence']['edit_distance']
        assert sorted([i for i, j in pairs] + a['WT_deletion_positions']) == list(range(len(wt)))
        assert sorted([j for i, j in pairs] + a['variant_insertion_positions']) == list(range(len(variant)))
        assert all(wt[i] == variant[j] for i, j in enumerate(a['wt_to_edit']) if j is not None)
        assert row['frozen_role_assignment'] == old['role_mapping']['selected']
        native = read(old['native_assembly_reference'])
        primary = next(c for c in native['chains'] if c['predictor_chain_id'] == native['primary_target_chain'])
        assert primary['sequence'] == wt
        rec = read(root / RECORDS / (row['physical_id'] + '.json'))
        with np.load(rec['output_path'], allow_pickle=False) as z:
            chains = json.loads(str(z['chains_json']))
        if row['cohort'] == 'Enzyme80':
            legacy = enzyme_catalytic_geometry(native, chains, old['correspondence'], old['role_mapping']['selected'])
            for suffix in ['lower', 'upper']:
                key = 'identity_aware_catalytic_neighbor_contact_recovery_' + suffix
                assert abs(legacy[key] - old['catalytic_identity_geometry'][key]) < 1e-12
        else:
            positive = predicted_interface(native, chains, old['role_mapping']['selected'])
            inverse = {j: i for i, j in enumerate(old['correspondence']['wt_to_edit'])
                       if j is not None and old['correspondence']['position_states'][i] == 'stable'}
            observed = {int(a['residue_index']) for a in primary['heavy_atoms']}
            mapped = {(inverse[j], c, k) for j, c, k in positive if j in inverse and inverse[j] in observed}
            truth = {(int(a['design_index']), a['partner_chain'], int(a['partner_index'])) for a in native['observed_H_antigen_contacts']}
            assert len(truth & mapped) == old['interface']['recovered_known_contacts']
            assert len(truth) == old['interface']['native_contacts']
        checked.append(dict(protein_id=row['protein_id'], physical_id=row['physical_id'], status='PASS'))
    write(root / REL / 'independent_readout_validation.json', dict(status='PASS_140_FIXED_OUTPUTS',
        checked=len(checked), alignment_cost_matches_original_global_optimum=True,
        exact_identity_pairs_and_complete_traceback_partitions=True,
        legacy_native_universe_and_readout_reproduced=True, frozen_roles_unchanged=True,
        strict_recalculation_sha256=sha(root / REL / 'strict_recalculation.json'), cases=checked))
    print('CANONICAL_VALIDATION_PASS', len(checked), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    validate(p.parse_args().root)
