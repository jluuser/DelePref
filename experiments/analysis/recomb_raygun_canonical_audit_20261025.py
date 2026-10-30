"""CPU-only re-evaluation of fixed Raygun outputs under exact-identity NW.

The existing ambiguity envelopes are preserved; they are NOT structural-alignment
bounds. This script writes a separate mapping audit, never overwrites old metrics,
and never calls a predictor or chooses a new generated sequence.
"""
from array import array
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

AA = set('ACDEFGHIKLMNPQRSTVWY')
REL = Path('reports/recomb_raygun_canonical_20261025')
METRICS = Path('reports/recomb_confirm_v2_application_evaluation_serialization_v5/main/per_output_metrics.json')
RECORDS = Path('reports/recomb_confirm_v2_application_structures_hardware_v3/records')
PROTOCOL = dict(algorithm='Needleman-Wunsch global minimum unit-edit alignment',
                match_cost=0, substitution_cost=1, gap_cost=1, terminal_gaps='penalized',
                traceback_from='C terminus', tie_priority=['diagonal', 'WT deletion', 'variant insertion'],
                correspondence='only identical standard amino-acid diagonal pairs are credited',
                ambiguous_X_B_Z_not_credited=True, mapping_chosen_without_structure_or_metric_scores=True,
                structure_and_generator_outputs='unchanged; main seed20261017, fixed20pct, all140 targets',
                molecular_chain_roles='original frozen role assignment, not refitted for the new correspondence',
                contact_recovery='fixed observed-native denominator; unmatched identities get zero credit',
                interface_F1='2 exact-identity native-contact hits / (native contacts + all observed generated contacts)',
                common_structure_columns='original archived structural correspondence; unchanged')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, d):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def canonical_alignment(wt, variant):
    """NW with fixed costs and explicit, deterministic traceback tie resolution."""
    n, m = len(wt), len(variant)
    matrix = [array('I', range(m + 1))]
    for i, a in enumerate(wt, 1):
        previous = matrix[-1]; row = array('I', [i] + [0] * m)
        for j, b in enumerate(variant, 1):
            row[j] = min(previous[j - 1] + (a != b), previous[j] + 1, row[j - 1] + 1)
        matrix.append(row)
    i, j = n, m
    pairs = []; deleted = []; inserted = []
    while i or j:
        if i and j and matrix[i][j] == matrix[i - 1][j - 1] + (wt[i - 1] != variant[j - 1]):
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i and matrix[i][j] == matrix[i - 1][j] + 1:
            deleted.append(i - 1); i -= 1
        else:
            assert j and matrix[i][j] == matrix[i][j - 1] + 1
            inserted.append(j - 1); j -= 1
    pairs.reverse(); mapping = [None] * n
    substitutions = []
    for i, j in pairs:
        if wt[i] == variant[j] and wt[i] in AA:
            mapping[i] = j
        else:
            substitutions.append(i)
    assert len(set(j for j in mapping if j is not None)) == sum(j is not None for j in mapping)
    return dict(method='canonical_global_NW_exact_identity', wt_to_edit=mapping,
                aligned_diagonal_pairs=pairs, WT_deletion_positions=sorted(deleted),
                variant_insertion_positions=sorted(inserted), unmatched_diagonal_positions=substitutions,
                edit_distance=int(matrix[n][m]), exact_identity_pairs=sum(j is not None for j in mapping))


def consensus_exact(wt, variant, correspondence):
    return [j if state == 'stable' and j is not None and wt[i] == variant[j] and wt[i] in AA else None
            for i, (state, j) in enumerate(zip(correspondence['position_states'], correspondence['wt_to_edit']))]


def ca_array(points):
    return np.asarray([p if p is not None else [np.nan] * 3 for p in points], dtype=float)


def enzyme_readout(native, output, mapping):
    primary = next(c for c in native['chains'] if c['predictor_chain_id'] == native['primary_target_chain'])
    reference = ca_array(primary['native_CA']); prediction = ca_array(output['coordinates'])
    sites = sorted({int(s['wt_index']) for s in native['catalytic']})
    for s in native['catalytic']:
        assert primary['sequence'][int(s['wt_index'])] == s['WT_residue']
    contacts = set()
    for i in sites:
        if not np.isfinite(reference[i]).all():
            continue
        distances = np.linalg.norm(reference - reference[i], axis=1)
        for j in np.flatnonzero(np.isfinite(distances) & (distances < 8.)):
            if abs(i - int(j)) >= 3:
                contacts.add(tuple(sorted((i, int(j)))))
    hit = missing = 0
    for i, j in contacts:
        a, b = mapping[i], mapping[j]
        if a is None or b is None:
            continue
        if not np.isfinite(prediction[[a, b]]).all():
            missing += 1; continue
        hit += float(np.linalg.norm(prediction[a] - prediction[b])) < 8.
    assert contacts
    return dict(complete_catalytic_AA_identity=int(all(mapping[i] is not None for i in sites)),
                identity_aware_catalytic_neighbor_contact_recovery=hit / len(contacts),
                catalytic_sites=sites, identical_catalytic_sites=[i for i in sites if mapping[i] is not None],
                native_neighbor_contacts=len(contacts), recovered_neighbor_contacts=int(hit),
                missing_mapped_predicted_neighbor_contacts=missing)


def predicted_interface(native, chains, role_map):
    from scipy.spatial import cKDTree
    primary = native['primary_target_chain']; design = chains[role_map[primary]]
    atoms = [a for a in design['heavy_atoms'] if a.get('element', '').upper() not in ('H', 'D') and np.isfinite(a['xyz']).all()]
    points = np.asarray([a['xyz'] for a in atoms]); positive = set()
    for partner in [c for c in native['chains'] if c['role'] == 'fixed_antigen']:
        cid = partner['predictor_chain_id']; known = {int(a['residue_index']) for a in partner['heavy_atoms']}
        pa = [a for a in chains[role_map[cid]]['heavy_atoms'] if int(a['residue_index']) in known
              and a.get('element', '').upper() not in ('H', 'D') and np.isfinite(a['xyz']).all()]
        tree = cKDTree(np.asarray([a['xyz'] for a in pa]))
        for a, nearby in zip(atoms, tree.query_ball_point(points, 5.)):
            for k in nearby:
                # Keep the original strict (<5 A) contact convention.
                if np.linalg.norm(np.asarray(a['xyz']) - np.asarray(pa[k]['xyz'])) < 5.:
                    positive.add((int(a['residue_index']), cid, int(pa[k]['residue_index'])))
    return positive


def binder_readout(native, positives, mapping):
    truth = {(int(a['design_index']), a['partner_chain'], int(a['partner_index'])) for a in native['observed_H_antigen_contacts']}
    primary = next(c for c in native['chains'] if c['predictor_chain_id'] == native['primary_target_chain'])
    observed = {int(a['residue_index']) for a in primary['heavy_atoms']}
    inverse = {j: i for i, j in enumerate(mapping) if j is not None}
    mapped = {(inverse[j], c, k) for j, c, k in positives if j in inverse and inverse[j] in observed}
    hit = len(truth & mapped)
    # Unmatched generated residues cannot obtain native identity credit, but their
    # observed partner contacts still enter the predicted-positive F1 denominator.
    predicted = sum(j not in inverse or inverse[j] in observed for j, _, _ in positives)
    assert truth
    return dict(interface_contact_recovery=hit / len(truth), interface_F1=2 * hit / (len(truth) + predicted),
                native_contacts=len(truth), recovered_contacts=hit, observed_predicted_contacts=predicted)


def audit(root):
    start = time.monotonic(); base = root / REL; base.mkdir(parents=True, exist_ok=True)
    if (base / 'strict_recalculation.json').exists():
        raise FileExistsError('Audit already exists; inspect it, do not blindly rerun')
    write(base / 'strict_protocol_before_readout.json', PROTOCOL)
    original = read(root / METRICS)
    rows = [r for r in original if r['method'] == 'raygun_native' and r['budget'] == .2
            and r['evaluation_role'] == 'primary20pct' and r['seed'] == 20261017]
    assert len(rows) == 140
    artifact_audit = {r['physical_id']: r for r in read(root / METRICS.parent / 'prediction_identity_audit.json')}
    results = []; assets = {}
    for count, row in enumerate(rows, 1):
        native_path = Path(row['native_assembly_reference']); native = read(native_path)
        assert sha(native_path) == row['native_assembly_reference_sha256']
        record = read(root / RECORDS / (row['physical_id'] + '.json'))
        assert record['status'] == row['prediction_status'] == 'SUCCEEDED'
        artifact = Path(record['output_path']); digest = sha(artifact)
        assert digest == record['output_sha256'] == artifact_audit[row['physical_id']]['artifact_sha256']
        with np.load(artifact, allow_pickle=False) as z:
            chains = json.loads(str(z['chains_json']))
        roles = row['role_mapping']['selected']; primary = native['primary_target_chain']
        output = chains[roles[primary]]; wt = row['wildtype_sequence']; variant = row['edited_sequence']
        assert output['sequence'] == variant
        a = canonical_alignment(wt, variant); strict = a['wt_to_edit']
        consensus = consensus_exact(wt, variant, row['correspondence'])
        if row['cohort'] == 'Enzyme80':
            score = enzyme_readout(native, output, strict)
            check = enzyme_readout(native, output, consensus)
            legacy = {k: row['catalytic_identity_geometry'][k] for k in ['identity_aware_catalytic_neighbor_contact_recovery_lower', 'identity_aware_catalytic_neighbor_contact_recovery_upper']}
        else:
            contacts = predicted_interface(native, chains, roles)
            score = binder_readout(native, contacts, strict); check = binder_readout(native, contacts, consensus)
            legacy = row['interface']
        results.append(dict(protein_id=row['protein_id'], cohort=row['cohort'], physical_id=row['physical_id'],
                            WT_sha256=row['WT_sha256'], edited_sequence_sha256=row['edited_sequence_sha256'],
                            correspondence=a, strict_canonical=score, strict_consensus_diagnostic=check,
                            legacy_identification_bounds=legacy, frozen_role_assignment=roles))
        assets[str(native_path)] = row['native_assembly_reference_sha256']; assets[str(artifact)] = digest
        if count % 20 == 0:
            print('CANONICAL_READOUT', count, '/140', round(time.monotonic() - start, 2), flush=True)
    summaries = []
    for cohort in ['Enzyme80', 'Binder60']:
        group = [r for r in results if r['cohort'] == cohort]
        keys = ['complete_catalytic_AA_identity', 'identity_aware_catalytic_neighbor_contact_recovery'] if cohort == 'Enzyme80' else ['interface_contact_recovery', 'interface_F1']
        summaries.append(dict(cohort=cohort, N=len(group),
            strict_canonical={k: float(np.mean([r['strict_canonical'][k] for r in group])) for k in keys},
            strict_consensus_diagnostic={k: float(np.mean([r['strict_consensus_diagnostic'][k] for r in group])) for k in keys}))
    write(base / 'strict_recalculation.json', dict(protocol=PROTOCOL, summaries=summaries, per_output=results,
          source_metrics_sha256=sha(root / METRICS), source_script_sha256=sha(__file__),
          input_assets_sha256=assets, seconds=time.monotonic() - start, new_predictions=0, new_model_calls=0,
          primary_policy_selection='pending user protocol clarification; this is a separate readout audit'))
    print('STRICT_AUDIT_COMPLETE', json.dumps(summaries), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    audit(p.parse_args().root)
