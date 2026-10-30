"""Versioned readout-only correction: missing structure is not lost AA identity."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path

PARENT = Path('reports/recomb_confirm_v2_application_evaluation_hardware_v3')
OUT = Path('reports/recomb_confirm_v2_application_evaluation_missingness_v4')
FZ = Path('configs/recomb_confirm_v2_application_missingness_freeze.json')
CONFIRM = Path('configs/recomb_confirm_v2_application_hardware_v3_confirmation_freeze.json')
ENDPOINT = 'identity_aware_catalytic_neighbor_contact_recovery'


def correct_failed_enzyme_row(row, native):
    if row['cohort'] != 'Enzyme80' or row['prediction_status'] == 'SUCCEEDED':
        return row
    primary = next(c for c in native['chains'] if c['predictor_chain_id'] == native['primary_target_chain'])
    states = row['correspondence']['position_states']
    mapping = row['correspondence']['wt_to_edit']
    sequence = row['edited_sequence']
    sites = {int(s['wt_index']) for s in native['catalytic']}

    def index(i):
        return mapping[i] if states[i] == 'stable' else None

    def identity(i):
        j = index(i)
        return None if j is None else j < len(sequence) and sequence[j] == primary['sequence'][i]

    def finite(point):
        return point is not None and all(math.isfinite(v) for v in point)

    contacts = set()
    reference = primary['native_CA']
    for i in sites:
        if not finite(reference[i]):
            continue
        for j in range(len(reference)):
            if abs(i-j) >= 3 and finite(reference[j]) and math.dist(reference[i], reference[j]) < 8.:
                contacts.add(tuple(sorted((i, j))))
    disrupted = unknown = 0
    for i, j in contacts:
        if states[i] == 'deleted' or states[j] == 'deleted':
            disrupted += 1
        elif any(p in sites and identity(p) is False for p in (i, j)):
            disrupted += 1
        else:
            unknown += 1
    result = copy.deepcopy(row)
    g = result['catalytic_identity_geometry']
    g.update(identity_retained_catalytic_sites=sum(identity(i) is True for i in sites),
             identity_aware_catalytic_neighbor_contacts=len(contacts),
             identity_aware_catalytic_neighbor_contact_recovery_lower=0. if contacts else None,
             identity_aware_catalytic_neighbor_contact_recovery_upper=unknown/len(contacts) if contacts else None,
             deleted_or_substituted_native_neighbor_contacts=disrupted,
             unknown_native_neighbor_contacts=unknown,
             catalytic_pair_CA_distance_MAE_A=None,
             known_identity_preserved_site_pairs=0,
             catalytic_pair_coverage=0. if len(sites) > 1 else None)
    result['missing_geometry_correction'] = dict(
        version='readout_missingness_v4', no_structure_repredicted=True,
        input_AA_identity_is_separate_from_unavailable_coordinates=True,
        raw_failed_prediction_status_preserved=True)
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def run(root, phase, report=False):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Readout correction requires CPU Slurm')
    frozen = read(root/FZ)
    if sha(root/CONFIRM) != frozen['parent_confirmation_sha256']:
        raise ValueError('Parent scientific confirmation changed')
    for name, expected in frozen['files'].items():
        if sha(root/name) != expected:
            raise ValueError('Missingness adapter source changed: '+name)
    from recomb_confirm_v2_application_evaluate_hardware_v3 import (
        summary_entry, paired_bounds, endpoint_bounds, seed_mean)
    import numpy as np
    source = root/PARENT/phase
    done = read(source/'completion.json')
    for name, expected in done['outputs'].items():
        if sha(source/name) != expected:
            raise ValueError('Raw phase artifact changed: '+name)
    out = root/OUT/phase
    if (out/'completion.json').exists():
        saved = read(out/'completion.json')
        if saved['raw_parent_metrics_completion_sha256'] != sha(source/'completion.json'):
            raise ValueError('Corrected phase has different raw parent')
        for name, expected in saved['outputs'].items():
            if sha(out/name) != expected:
                raise ValueError('Corrected phase artifact changed')
    else:
        raw = read(source/'per_output_metrics.json')
        rows = []
        corrected = 0
        for row in raw:
            if row['cohort'] == 'Enzyme80' and row['prediction_status'] != 'SUCCEEDED':
                path = Path(row['native_assembly_reference'])
                if sha(path) != row['native_assembly_reference_sha256']:
                    raise ValueError('Native reference changed')
                fixed = correct_failed_enzyme_row(row, read(path))
                assert fixed['catalytic_identity'] == row['catalytic_identity']
                corrected += 1
                rows.append(fixed)
            else:
                rows.append(row)
        protocol = read(root/'reports/recomb_confirm_v2_application_catalog_hardware_v3/statistical_protocol.json')
        tables = read(source/'mean_tables.json')
        for table in tables:
            if table['cohort'] == 'Enzyme80':
                selected = [r for r in rows if r['cohort'] == 'Enzyme80' and r['method'] == table['method']
                            and r['budget'] == table['budget'] and r['seed'] == 20261017]
                table['endpoints'][ENDPOINT] = summary_entry(selected, ENDPOINT)
        comparisons = read(source/'paired_comparisons.json')
        for item in comparisons:
            if item['cohort'] == 'Enzyme80' and item['endpoint'] == ENDPOINT:
                selected = [r for r in rows if r['cohort'] == 'Enzyme80' and r['budget'] == item['budget']
                            and r['method'] != 'WT' and r['seed'] == 20261017]
                by = {(r['protein_id'], r['method']): r for r in selected}
                ids = sorted({r['protein_id'] for r in selected})
                item.update(paired_bounds(ids, item['method'], item['versus'], by, ENDPOINT,
                                          protocol['component_by_protein_id']))
        repeats = read(source/'within_WT_three_seed_summaries.json')
        for item in repeats:
            if item['cohort'] != 'Enzyme80':
                continue
            selected = [r for r in rows if r['protein_id'] == item['protein_id'] and r['method'] == item['method']
                        and (r['budget'] == .2 or r['method'] == 'WT')]
            bounds = [endpoint_bounds(r, ENDPOINT) for r in selected]
            lower, upper = seed_mean(bounds)
            exact = len(bounds) == 3 and all(a is not None and a == b for a, b in bounds)
            item['endpoints'][ENDPOINT] = dict(mean_lower=lower, mean_upper=upper,
                point_seed_SD=float(np.std([a for a, b in bounds], ddof=1)) if exact else None,
                point_worst_seed=min(a for a, b in bounds) if exact else None,
                uncertain_seeds_not_replaced_by_point_estimates=not exact)
        outputs = dict(per_output_metrics=rows, mean_tables=tables, paired_comparisons=comparisons,
                       within_WT_three_seed_summaries=repeats,
                       prediction_identity_audit=read(source/'prediction_identity_audit.json'))
        out.mkdir(parents=True, exist_ok=True)
        for name, value in outputs.items():
            write(out/(name+'.json'), value)
        summary = dict(done)
        summary.update(raw_parent_metrics_completion_sha256=sha(source/'completion.json'),
            missingness_source_freeze_sha256=sha(root/FZ),
            failed_enzyme_logical_rows_corrected=corrected,
            primary_AA_identity_and_Binder_endpoints_unchanged=True,
            no_new_model_calls=True, no_new_structure_predictions=True,
            outputs={name+'.json': sha(out/(name+'.json')) for name in outputs})
        write(out/'completion.json', summary)
    if report:
        if phase != 'all':
            raise ValueError('Final report requires all2160 corrected metrics')
        import recomb_confirm_v2_application_repeat_report_hardware_v3 as parent
        parent.EVALUATION = OUT/'all'
        parent.OUT = Path('reports/recomb_confirm_v2_application_final_report_missingness_v4')
        parent.run(root)
        p = root/parent.OUT/'completion.json'
        value = read(p)
        value['missingness_source_freeze_sha256'] = sha(root/FZ)
        value['raw_parent_all_metrics_completion_sha256'] = sha(source/'completion.json')
        write(p, value)
    print(read(out/'completion.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=['WT', 'main', 'all'], required=True)
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    run(args.root, args.phase, args.report)
