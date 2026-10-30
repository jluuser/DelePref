"""Input-only application selection and cache inventory; never reads quality scores."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import yaml
from recomb_general100_common_20261017 import ROOT, sha, sequence_sha, write
import delepref_unlimited_planning as planning

BASE = 'reports/recomb_general100_applications_20261017'
SURG = Path('/public/home/zhangyangroup/chengshiz/keyuan.zhou/SurgEdit')
BINDER = SURG / 'results/experiments/student_v3_step004000_v3_25_binder50_fair_contact_comparison'
LEGACY = Path('/public/nfs-data/keyuan.zhou/evocompress/results/iclr_completion_20260910/E1_E4')
EC_QUOTA = {'1': 4, '2': 4, '3': 4, '4': 3, '5': 2, '6': 2, '7': 1}


def csv_rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def choose_binder(rows):
    selected = []
    for source, count in [('AB-Bind', 8), ('SAbDab', 12)]:
        eligible = [x for x in rows if x['source'] == source and len(json.loads(x['binder_chain_ids_json'])) == 1]
        key = lambda x: hashlib.sha256(('recomb-binder20-20261017|' + x['case_id'] + '|' + x['wildtype_sequence_sha256']).encode()).hexdigest()
        selected.extend(sorted(eligible, key=key)[:count])
    if len(selected) != 20:
        raise ValueError('Binder20 input quota unavailable')
    return selected


def choose_enzyme(groups):
    # Published input family annotations take precedence. Unknown annotations
    # stay unknown; accession uniqueness is not described as family disjointness.
    selected, seen_families = [], set()
    for ec, quota in EC_QUOTA.items():
        eligible = [g for g in groups if str(g['reference']['ec_class']) == ec
                    and str(g['reference'].get('coordinate_coordinate_valid')).lower() == 'true']
        eligible.sort(key=lambda g: hashlib.sha256(('recomb-enzyme20-20261017|' + g['wt_sequence_sha256']).encode()).hexdigest())
        picked = []
        for g in eligible:
            family = g['reference'].get('homology_component') or g['reference'].get('cath_h_superfamily')
            if family and family in seen_families:
                continue
            picked.append(g)
            if family:
                seen_families.add(family)
            if len(picked) == quota:
                break
        if len(picked) != quota:
            raise ValueError('EC input quota unavailable: ' + ec)
        selected.extend(picked)
    if len(selected) != 20 or len({g['wt_sequence_sha256'] for g in selected}) != 20:
        raise ValueError('Enzyme20 input identity mismatch')
    return selected


def main(root):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('CPU Slurm allocation required')
    out = root / BASE
    out.mkdir(exist_ok=False)
    cohort = csv_rows(BINDER / 'cohort.csv')
    selected_binder = choose_binder(cohort)
    manifest = {int(x['case_id']): x for x in csv_rows(SURG / 'results/experiments/student_v3_step004000_binder/binder_manifest.csv')}
    plans = {(int(x['case_id']), float(x['budget'])): x for x in csv_rows(BINDER / 'scisor_plans.csv')}
    predictions = csv_rows(BINDER / 'prediction_plan_manifest.csv')
    inputs, panels, baseline_outputs, cache_inventory = {}, [], [], []
    for x in selected_binder:
        case = int(x['case_id']); original = manifest[case]
        wt = x['wildtype_sequence']; pid = 'Binder20|' + x['protein_id']
        if sha(x['wt_source_yaml']) != x['wt_source_yaml_sha256'] or sequence_sha(wt) != x['wildtype_sequence_sha256']:
            raise ValueError('Binder source changed')
        document = yaml.safe_load(Path(x['wt_source_yaml']).read_text())
        proteins = [r['protein'] for r in document['sequences']]
        design_chain = json.loads(x['binder_chain_ids_json'])[0]
        by_id = {p['id']: p for p in proteins}
        if by_id[design_chain]['sequence'] != wt:
            raise ValueError('single design chain is not exact WT')
        partners = [dict(chain_id=p, sequence=by_id[p]['sequence'], sequence_sha256=sequence_sha(by_id[p]['sequence']))
                    for p in json.loads(x['target_chain_ids_json'])]
        ref = Path(original['structure_path'])
        if sha(ref) != original['structure_sha256']:
            raise ValueError('native Binder reference changed')
        row = dict(protein_id=pid, original_protein_id=x['protein_id'], case_id=case, dataset='Binder20', source=x['source'],
                   sequence=wt, sequence_sha256=sequence_sha(wt), sequence_length=len(wt), design_chain=design_chain,
                   partners=partners, native_reference_path=str(ref), native_reference_sha256=sha(ref),
                   original_WT_yaml=x['wt_source_yaml'], budget=.2, target_length=len(wt)-planning.delete_count(len(wt), .2))
        panels.append(row)
        inputs[str(ref)] = sha(ref); inputs[x['wt_source_yaml']] = sha(x['wt_source_yaml'])
        old = plans[(case, .2)]
        intervals = tuple(map(tuple, json.loads(old['deleted_intervals'])))
        mask = planning.validate_plan(len(wt), .2, intervals)
        if ''.join(c for c, deleted in zip(wt, mask) if not deleted) != old['edited_sequence']:
            raise ValueError('Binder SCISOR mapping does not reconstruct')
        baseline_outputs.append(dict(row, method='SCISOR_U90_S_native', edited_sequence=old['edited_sequence'],
             edited_sequence_sha256=old['edited_sequence_sha256'], deleted_intervals=list(map(list, intervals)),
             generation_seed=int(old['condition_seed']), reused_generation=True, source_path=str(BINDER/'scisor_plans.csv')))
        for method in ('WT', 'SCISOR_U90_S_native'):
            lookup = [p for p in predictions if int(p['case_id']) == case and
                      ((method == 'WT' and p['method'].lower() == 'wt') or
                       (method != 'WT' and 'scisor' in p['method'].lower() and float(p['budget']) == .2))]
            for p in lookup:
                rid = p['record_id']; directory = BINDER/'boltz_outputs/boltz_results_boltz_inputs/predictions'/rid
                structure = directory/(rid+'_model_0.cif')
                if structure.is_file():
                    cache_inventory.append(dict(protein_id=pid, method=method, sequence_sha256=p['edited_sequence_sha256'],
                        structure_path=str(structure), structure_sha256=sha(structure), yaml_path=p['yaml_path'],
                        yaml_sha256=sha(p['yaml_path']), profile='Binder_Boltz2_r1_s50_seed20260726',
                        cache_status='SOURCE_FOUND_PROFILE_AUDIT_REQUIRED'))
    groups = []
    for path in sorted((LEGACY/'Raygun_native/groups').glob('E2_*.json')):
        g = json.loads(path.read_text()); g['_source_path'] = str(path); groups.append(g)
    selected_enzyme = choose_enzyme(groups)
    old_final = []
    with (LEGACY/'FINAL/per_candidate.jsonl').open() as f:
        for line in f:
            z = json.loads(line)
            if z.get('experiment') == 'E2' and float(z.get('budget', -1)) == .2 and int(z.get('candidate_index', -1)) == 0:
                old_final.append(z)
    for g in selected_enzyme:
        ref = g['reference']; wt = g['wt_sequence']; pid = 'Enzyme20|' + g['protein_id']
        path = Path(ref['reference_coordinate_path'])
        if sha(path) != ref['reference_coordinate_sha256'] or sequence_sha(wt) != g['wt_sequence_sha256']:
            raise ValueError('Enzyme source identity changed')
        row = dict(protein_id=pid, original_protein_id=g['protein_id'], dataset='Enzyme20', ec_class=str(ref['ec_class']),
            source='M-CSA actual frozen Enzyme120', sequence=wt, sequence_length=len(wt), sequence_sha256=sequence_sha(wt),
            design_chain=ref['auth_chain_id'], partners=[], reference=ref, native_reference_path=str(path),
            native_reference_sha256=sha(path), budget=.2, target_length=len(wt)-planning.delete_count(len(wt), .2),
            family_annotation_status='KNOWN' if ref.get('homology_component') or ref.get('cath_h_superfamily') else 'UNKNOWN_REQUIRES_INPUT_SIMILARITY_AUDIT')
        panels.append(row); inputs[str(path)] = sha(path); inputs[g['_source_path']] = sha(g['_source_path'])
        for rec in g['existing_records']:
            if float(rec['budget_requested']) == .2 and int(rec.get('candidate_index', -1)) == 0:
                if len(rec['sequence']) != row['target_length']:
                    raise ValueError('cached Raygun length differs from common budget')
                baseline_outputs.append(dict(row, method='Raygun_500_native', edited_sequence=rec['sequence'],
                    edited_sequence_sha256=sequence_sha(rec['sequence']), reused_generation=True,
                    source_path=g['_source_path'], native_candidate_record=rec))
        for rec in old_final:
            if rec['protein_id'] != g['protein_id']:
                continue
            structure = Path(rec['prediction_path'])
            if structure.is_file():
                cache_inventory.append(dict(protein_id=pid, source_method=rec['method'], sequence_sha256=rec['sequence_sha256'],
                    structure_path=str(structure), structure_sha256=sha(structure), profile='Enzyme_Boltz2_r3_s200_seed20260908',
                    cache_status='SOURCE_FOUND_PROFILE_AND_SEQUENCE_AUDIT_REQUIRED'))
    if len(panels) != 40:
        raise ValueError('application roster incomplete')
    write(out/'panel.json', panels); write(out/'reused_baseline_outputs.json', baseline_outputs)
    write(out/'cache_inventory.json', cache_inventory)
    (out/'queries.fasta').write_text(''.join('>appq_%03d\n%s\n' % (i, p['sequence']) for i,p in enumerate(panels)))
    sources = [BINDER/'cohort.csv', BINDER/'scisor_plans.csv', BINDER/'prediction_plan_manifest.csv',
               SURG/'results/experiments/student_v3_step004000_binder/binder_manifest.csv']
    inputs.update({str(p): sha(p) for p in sources})
    write(root/'configs/recomb_general100_app_inputs_freeze_20261017.json', dict(
         status='APPLICATION_INPUTS_FROZEN_BEFORE_NEW_GENERATION', inputs=inputs,
         panel_sha256=sha(out/'panel.json'), input_script_sha256=sha(Path(__file__)),
         Binder20=20, Enzyme20=20, no_quality_scores_used_for_selection=True,
         no_functional_site_generation_protection=True, qmax=None, complete_structure_prediction_limit=None,
         allowed_GPU_types=['A40','A100'], GPU_concurrency_limit=2, paper_paused=True))
    report=dict(status='APPLICATION_ROSTER_AND_CACHE_SOURCES_PREPARED', Binder20=20, Enzyme20=20,
          selected_partner_count_histogram={str(k):sum(len(p['partners'])==k for p in panels if p['dataset']=='Binder20') for k in (1,2)},
          excluded_multiple_binder_chains=26, native_partner_chains_already_separate_in_old_YAML=True,
          provisional_cache_records=len(cache_inventory), generation_records_reused=len(baseline_outputs),
          no_cache_declared_compatible_without_identity_audit=True,
          files={p.name:sha(p) for p in out.iterdir() if p.is_file()}, job_id=os.environ['SLURM_JOB_ID'],
          new_structure_predictions=0, new_GPU_calls=0)
    write(out/'preparation_completion.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    main(p.parse_args().root)
