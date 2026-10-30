"""Frozen E9 winner input identities, exact compatible caches, and ESMFold release."""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path

from recomb_confirm_v2_E9_completion import read, sha, write, digest, spans

BASE = Path('reports/recomb_confirm_v2_E9_catalog')
INPUT_FREEZE = Path('configs/recomb_confirm_v2_E9_catalog_inputs_freeze.json')
CONFIRM = Path('configs/recomb_confirm_v2_E9_confirmation_freeze.json')
GENERAL = Path('reports/recomb_confirm_v2_General100_catalog')
SCIENCE_SHA = '7c42d417fea99f5aee5b90b2094c5a15b43532e78de2ce743555e72f55f6a04d'
WINNERS = Path('configs/recomb_confirm_v2_E9_selected_winners_freeze.json')


def explicit_correspondence(sequence, intervals):
    # Same explicit-edit construction as the frozen General output assembler.
    deleted = set()
    for a, b in intervals:
        if not 0 <= a <= b < len(sequence) or deleted.intersection(range(a, b+1)):
            raise ValueError('Invalid explicit deletion intervals')
        deleted.update(range(a, b+1))
    if len(deleted) != max(1, min(len(sequence)-1, round(len(sequence)*.2))):
        raise ValueError('E9 deletion budget differs')
    edited = ''.join(c for i, c in enumerate(sequence) if i not in deleted)
    mapping, j = [], 0
    for i in range(len(sequence)):
        mapping.append(None if i in deleted else j)
        if i not in deleted:
            j += 1
    return edited, dict(method='explicit_native_deletion_plan', wt_to_edit=mapping,
        position_states=['deleted' if i in deleted else 'stable' for i in range(len(sequence))],
        stable_positions=len(edited), certain_deleted_positions=len(deleted), ambiguous_positions=0)


def physical_request(sequence, profile, weights_sha256):
    sequence_sha = hashlib.sha256(sequence.encode()).hexdigest()
    signature = dict(sequence_sha256=sequence_sha, weights_sha256=weights_sha256,
                     profile_sha256=digest(profile), seed=int(profile['seed']), sample=0)
    chains = [dict(id='A', sequence=sequence)]
    return dict(signature, task_id='esmfold_'+digest(signature), predictor='esmfold',
                profile_name='esmfold', sequence=sequence, chains=chains,
                canonical_chains_sha256=digest(chains), category='general_esmfold',
                wt_variant='variant', requires_all_atom_output=False)


def compatible_identity(saved, request):
    keys = ['sequence_sha256', 'weights_sha256', 'profile_sha256', 'seed', 'sample']
    if any(saved.get(k) != request[k] for k in keys) or saved.get('sequence') != request['sequence']:
        return False
    if saved.get('predictor', 'esmfold') != 'esmfold':
        return False
    if 'chains' in saved and saved['chains'] != request['chains']:
        return False
    if 'canonical_chains_sha256' in saved and saved['canonical_chains_sha256'] != request['canonical_chains_sha256']:
        return False
    return True


def derive_predictor(source):
    substitutions = {
        "BASE=Path('reports/recomb_confirm_v2_General100_structures')": "BASE=Path('reports/recomb_confirm_v2_E9_structures')",
        "CAT=Path('reports/recomb_confirm_v2_General100_catalog')": "CAT=Path('reports/recomb_confirm_v2_E9_catalog')",
        "FREEZE=Path('configs/recomb_confirm_v2_General100_confirmation_freeze.json')": "FREEZE=Path('configs/recomb_confirm_v2_E9_confirmation_freeze.json')"}
    derived = source
    for old, new in substitutions.items():
        if derived.count(old) != 1:
            raise ValueError('Original predictor namespace discovery differs')
        derived = derived.replace(old, new)
    functions = lambda s: {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(s).body if isinstance(n, ast.FunctionDef)}
    if functions(derived) != functions(source):
        raise ValueError('Derived predictor changed a scientific function')
    return derived


def prepare(root):
    if (root/INPUT_FREEZE).exists() or (root/CONFIRM).exists():
        raise FileExistsError('E9 catalog already prepared/released')
    if sha(root/'configs/recomb_confirm_v2_General100_confirmation_freeze.json') != SCIENCE_SHA:
        raise ValueError('General scientific parent changed')
    frozen = read(root/WINNERS)
    for name, expected in frozen['files'].items():
        if sha(root/name) != expected:
            raise ValueError('Frozen E9 candidate/winner source changed: '+name)
    names = ['analysis/recomb_confirm_v2_E9_catalog.py', 'analysis/recomb_confirm_v2_E9_evaluate.py',
             'analysis/recomb_confirm_v2_E9_completion.py', 'analysis/recomb_confirm_v2_diagnostics.py',
             'analysis/tests/test_recomb_confirm_v2_E9_catalog.py',
             'slurm/recomb_confirm_v2_E9_catalog.sbatch', 'slurm/recomb_confirm_v2_E9_ESMfold.sbatch',
             'slurm/recomb_confirm_v2_E9_evaluate.sbatch',
             'analysis/recomb_confirm_v2_general_predict.py', 'analysis/recomb_confirm_v2_general_evaluate.py',
             'analysis/recomb_confirm_v2_general_outputs.py', 'analysis/recomb_confirm_v2_structure_alignment.py',
             'analysis/recomb_benign_metrics_20261017.py', 'analysis/delepref_v8_structure_metrics.py',
             'analysis/recomb_confirm_v2_application_serialization.py', 'code/prediction_journal.py',
             str(WINNERS), 'configs/recomb_confirm_v2_General100_confirmation_freeze.json',
             'reports/recomb_confirm_v2_general_native_v2/General100_panel.json',
             'reports/recomb_confirm_v2_E9_completion/selected_structure_outputs.json',
             'reports/delepref_v8_predictors/esmfold_preflight.json',
             'reports/recomb_confirm_v2_General100_evaluation/completion.json',
             'reports/recomb_confirm_v2_General100_evaluation/per_output_metrics.json']
    names += [str(GENERAL/n) for n in ['logical_requests.json', 'physical_requests.json',
                                     'predictor_profiles.json', 'statistical_protocol.json']]
    write(root/INPUT_FREEZE, dict(status='E9_CACHE_CATALOG_AND_UNCHANGED_PREDICTOR_SOURCE_INPUTS_FROZEN',
        files={n: sha(root/n) for n in names}, source_winner_freeze_sha256=sha(root/WINNERS),
        logical_winners=40, fixed_WT=20, folding_allowed=False,
        post_outcome_exploratory=True, no_quality_based_cache_or_case_filter=True))
    print(sha(root/INPUT_FREEZE), flush=True)


def build(root, expected_freeze):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('E9 catalog requires CPU Slurm')
    if sha(root/INPUT_FREEZE) != expected_freeze:
        raise ValueError('E9 catalog source/input freeze changed')
    frozen = read(root/INPUT_FREEZE)
    for name, expected in frozen['files'].items():
        if sha(root/name) != expected:
            raise ValueError('Frozen E9 catalog source/input changed: '+name)
    for name, expected in read(root/WINNERS)['files'].items():
        if sha(root/name) != expected:
            raise ValueError('Frozen E9 candidate/winner artifact changed: '+name)
    out = root/BASE;out.mkdir(parents=True, exist_ok=True)
    fd = os.open(out/'execution_claim.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, 'w') as f:
        json.dump(dict(job_id=os.environ['SLURM_JOB_ID'], input_freeze_sha256=expected_freeze), f)
    winners = read(root/'reports/recomb_confirm_v2_E9_completion/selected_structure_outputs.json')
    panel = {r['protein_id']: r for r in read(root/'reports/recomb_confirm_v2_general_native_v2/General100_panel.json')}
    profile = read(root/GENERAL/'predictor_profiles.json')['esmfold']
    preflight = read(root/'reports/delepref_v8_predictors/esmfold_preflight.json')
    weight = next(r['sha256'] for r in preflight['files'] if r['path'].endswith('pytorch_model.bin'))
    physical, logical = {}, []
    for row in winners:
        pid = row['protein_id'];native = panel[pid]
        if row['wildtype_sequence'] != native['sequence']:
            raise ValueError('E9 WT differs from the positively reviewed General panel')
        edited, mapping = explicit_correspondence(native['sequence'], spans(row))
        if edited != row['edited_sequence'] or hashlib.sha256(edited.encode()).hexdigest() != row['edited_sequence_sha256']:
            raise ValueError('Frozen E9 winner reconstruction differs')
        p = physical_request(edited, profile, weight);physical.setdefault(p['task_id'], p)
        item = dict(protein_id=pid, cohort='general', WT_sha256=native['WT_sha256'],
            wildtype_sequence=native['sequence'], edited_sequence=edited, edited_sequence_sha256=row['edited_sequence_sha256'],
            method=row['method'], budget=.2, evaluation_role='post_outcome_E9_same_seed_completion',
            predictor='esmfold', profile_name='esmfold', profile_family='esmfold', seed=p['seed'], sample=0,
            physical_id=p['task_id'], correspondence=mapping,
            source_path='reports/recomb_confirm_v2_E9_completion/selected_structure_outputs.json',
            selection_metadata={k: row.get(k) for k in ['generation_seed', 'raw_candidate_rank', 'scorer_score',
                'scorer_rank', 'generation_kind', 'retrieval_source', 'retrieval_seed_spans', 'paired_retrieval_slots',
                'original_policy_slots', 'common_seed_sha256', 'pool_sha256']},
            native_to_prediction_chains=[dict(native_chain_id=native['reference_chain'], prediction_chain_id='A', role='design')])
        item['request_id'] = digest({k: item[k] for k in ['protein_id', 'method', 'budget', 'evaluation_role', 'predictor', 'seed', 'physical_id']})
        logical.append(item)
    if len(logical) != 40 or len({r['request_id'] for r in logical}) != 40 or len({r['protein_id'] for r in logical}) != 20:
        raise ValueError('Fixed20 two-winner logical roster differs')
    import numpy as np
    reuse, indices = {}, {}
    folders = ['reports/recomb_confirm_v2_General100_structures', 'reports/recomb_benign_structures_20261017',
               'reports/delepref_v8_structures', 'reports/recomb_confirm_v2_span_structures']
    for p in physical.values():
        for folder in folders:
            directory = root/folder
            journal = next((directory/n for n in ['prediction_journal.jsonl', 'budget_ledger.jsonl'] if (directory/n).exists()), None)
            if journal is None:
                continue
            if str(journal) not in indices:
                raw = journal.read_bytes()
                if raw and not raw.endswith(b'\n'):
                    raise ValueError('Historical cache journal is incomplete')
                indices[str(journal)] = {r['task_id']: r for line in raw.splitlines() for r in [json.loads(line)]}
            for name in [p['task_id'], 'esmfold_'+p['sequence_sha256']]:
                record = directory/'records'/(name+'.json')
                if not record.exists():
                    continue
                saved = read(record)
                if not compatible_identity(saved, p):
                    continue
                if saved.get('status') != 'SUCCEEDED':
                    raise ValueError('Matching historical failed identity requires technical audit: '+str(record))
                event = indices[str(journal)].get(saved.get('task_id', name), {})
                if event.get('state') not in ('succeeded', 'cache_hit') or event.get('output_sha') != saved['output_sha256']:
                    raise ValueError('Matching cache success/journal differs')
                for a, b in [('sequence_sha', 'sequence_sha256'), ('predictor_weights_sha', 'weights_sha256'),
                             ('profile_sha', 'profile_sha256'), ('seed', 'seed'), ('sample', 'sample')]:
                    if event.get(a) != p[b]:
                        raise ValueError('Matching cache journal input identity differs')
                artifact = Path(saved['output_path'])
                if sha(artifact) != saved['output_sha256']:
                    raise ValueError('Matching cache artifact SHA changed')
                with np.load(artifact, allow_pickle=False) as z:
                    if str(z['sequence']) != p['sequence'] or z['coordinates'].shape != (len(p['sequence']), 3):
                        raise ValueError('Matching cache sequence/coordinates differ')
                reuse[p['task_id']] = dict(record_path=str(record), record_sha256=sha(record),
                    artifact_path=str(artifact), artifact_sha256=saved['output_sha256'],
                    journal_path=str(journal), journal_sha256=sha(journal))
                break
            if p['task_id'] in reuse:
                break
    protocol = copy.deepcopy(read(root/'reports/recomb_confirm_v2_E9_completion/statistical_protocol.json'))
    protocol.update(status='POST_OUTCOME_E9_MATCHED_SEED_FIXED20', WT=20,
        methods=['delepref_unlimited', 'rag_unlimited', 'policy_completion', 'simple_legal_completion'],
        requested_comparisons=[['policy_completion', 'simple_legal_completion']],
        source_selected_winners_freeze_sha256=sha(root/WINNERS),
        no_new_success_threshold=True, fixed_edit_layer_preserved=True, no_full_candidate_pool_predictions=True)
    for name, value in [('logical_requests.json', logical), ('physical_requests.json', sorted(physical.values(), key=lambda p: p['task_id'])),
                        ('compatible_cache.json', reuse), ('predictor_profiles.json', {'esmfold': profile}),
                        ('statistical_protocol.json', protocol)]:
        write(out/name, value)
    original = (root/'analysis/recomb_confirm_v2_general_predict.py').read_text()
    derived = root/'analysis/recomb_confirm_v2_E9_predict.py'
    if derived.exists():
        raise FileExistsError('Derived E9 worker already exists')
    derived.write_text(derive_predictor(original))
    proof = dict(status='E9_FIXED40_INPUTS_EXACT_PROFILE_CACHE_AND_SCIENTIFIC_FUNCTION_AST_PASS',
        logical=40, WT=20, physical=len(physical), compatible_cache=len(reuse), missing_physical=len(physical)-len(reuse),
        parent_predictor_sha256=sha(root/'analysis/recomb_confirm_v2_general_predict.py'),
        derived_predictor_sha256=sha(derived), all_predictor_function_ASTs_identical=True,
        profile_sha256=digest(profile), source_winners_freeze_sha256=sha(root/WINNERS),
        cache_identity_includes_sequence_weights_profile_seed_sample_single_chain=True,
        no_quality_filter_or_winner_reselection=True, new_structure_predictions=0)
    write(out/'completion.json', proof)
    names = list(frozen['files'])+[str(INPUT_FREEZE), 'analysis/recomb_confirm_v2_E9_predict.py']
    names += [str(BASE/n) for n in ['logical_requests.json', 'physical_requests.json', 'compatible_cache.json',
                                  'predictor_profiles.json', 'statistical_protocol.json', 'completion.json']]
    names += [panel[pid]['wt_native_json'] for pid in {r['protein_id'] for r in logical}]
    names += list(read(root/WINNERS)['files'])
    released_files = {str(root/n): sha(root/n) for n in sorted(set(names))}
    for item in read(root/'configs/recomb_confirm_v2_General100_confirmation_freeze.json')['files']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Original General scientific dependency changed: '+item['path'])
        released_files[item['path']] = item['sha256']
    write(root/CONFIRM, dict(status='E9_SELECTED_WINNER_STRUCTURAL_SUPPLEMENT_RELEASED',
        files=[dict(path=n, sha256=h) for n, h in sorted(released_files.items())],
        fold_allowed=True, all_selected_outputs_frozen=True, post_outcome_exploratory=True,
        logical=40, fixed_WT=20, physical=len(physical), cache=len(reuse), missing=len(physical)-len(reuse),
        only_original_ESMFold_A100_profile=True, no_full_candidate_pool_folding=True,
        no_new_training_or_WT_ESMC_encoding=True, new_structure_predictions=0))
    print(json.dumps(dict(**proof, final_confirmation_sha256=sha(root/CONFIRM))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--freeze-sha256')
    args = parser.parse_args()
    prepare(args.root) if args.prepare else build(args.root, args.freeze_sha256)
