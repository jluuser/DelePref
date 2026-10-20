"""Independent completed-scope checks for E9/E10, without model execution."""
import argparse
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from transformers.models.esm.configuration_esm import get_default_vocab_list


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def run(root):
    out = root/'reports/recomb_targeted_supplements_20261019'
    base = root/'reports/recomb_confirm_v2_E10_timing_padding_v2'
    frozen = read(root/'configs/recomb_confirm_v2_E10_timing_padding_v2_freeze.json')
    pins = {**frozen['files'], **frozen['original_dependencies'], **frozen['native_external_code'], **frozen['assets_sha256']}
    for name, expected in pins.items():
        if sha(root/name) != expected:
            raise ValueError('Frozen input/model/source changed: '+name)
    for name, item in frozen['target_database'].items():
        s = Path(name).stat()
        if (s.st_size, s.st_mtime_ns) != (item['size'], item['mtime_ns']):
            raise ValueError('Frozen target DB changed')
        if 'sha256' in item and sha(name) != item['sha256']:
            raise ValueError('Frozen target index changed')
    rows = read(base/'all_request_measurements.json')
    by = {(r['protein_id'], r['method'], r['scenario']): r for r in rows}
    if len(rows) != 240 or len(by) != 240:
        raise ValueError('The complete requested matrix is not present')
    ray = {(r['protein_id'], r['budget']): r for r in read(root/'reports/recomb_confirm_v2_E10_native_fidelity/all40_historical_Raygun_generation_hardware.json')}
    l_path = root/'reports/recomb_confirm_v2_General100_scisor_L/all_conditions.json'
    original_l = {(r['protein_id'], r['budget']): r for r in read(l_path) if r['mode'] == 'native'}
    vocab = {s: i for i, s in enumerate(get_default_vocab_list())}
    reviewed = read(root/'reports/recomb_confirm_v2_General100_evidence_review/target_metadata_audit.json')
    totals = collections.Counter();matches = collections.Counter();joins = []
    for i, case in enumerate(frozen['cases']):
        pid = case['protein_id'];sequence = case['wildtype_sequence']
        for method in frozen['methods']:
            d = base/'requests'/('%02d_%s' % (i, case['WT_sha256'][:12]))/method
            if method == 'scisor_L_native':
                tokens = torch.load(d/'actual_cold_native_WT_cache.pt', map_location='cpu', weights_only=True)
                if tokens.tolist() != [[vocab['<cls>']]+[vocab[c] for c in sequence]+[vocab['<eos>']]]:
                    raise ValueError('Native L reused WT tokens are not the exact WT input')
                totals['native_L_WT_token_caches_verified'] += 1
            for scenario, beta in [('cold', .2), ('cached', .3)]:
                value = by[pid, method, scenario]
                for name, expected in value['output_files_sha256'].items():
                    if sha(d/scenario/name) != expected:
                        raise ValueError('Actual request output changed')
                selected = read(d/scenario/'selected_output.json')
                ident = read(d/scenario/'request_identity.json')
                pool = read(d/scenario/'candidate_pool.json')
                if ident['WT_sha256'] != case['WT_sha256'] or ident['method'] != method or ident['beta'] != beta:
                    raise ValueError('Actual request input differs')
                if hashlib.sha256(selected['edited_sequence'].encode()).hexdigest() != value['selected_sequence_sha256']:
                    raise ValueError('Actual final sequence differs')
                matched = value['selected_sequence_matches_scientific_output']
                if matched:
                    matches[method] += 1
                comp, stages = value['components'], value['stages']
                if method in ['delepref_unlimited', 'rag_unlimited', 'gap_frequency', 'terminal']:
                    if len(pool) != 8 or not matched:
                        raise ValueError('Head-family scientific K8 output not reproduced')
                    totals['head_Policy_calls'] += comp['actual_Policy_WT_calls']
                    totals['head_WT_encoder_batches'] += stages['actual_WT_encoder_batches']
                else:
                    if method == 'scisor_L_native':
                        old = original_l[pid, beta]
                        if len(pool) != 1 or comp['native_K'] != 1 or old['wildtype_sequence'] != sequence or old['condition_seed'] != selected['seed']:
                            raise ValueError('Native L input/seed/K1 protocol differs')
                        totals['L_native_calls'] += 1
                        hardware = old['actual_GPU'];old_sha = old['edited_sequence_sha256']
                    else:
                        old = ray[pid, beta]
                        if len(pool) != 500 or any(len(x['edited_sequence']) != len(selected['edited_sequence']) for x in pool):
                            raise ValueError('Ray all500 output lengths differ')
                        totals['Ray_requests'] += 1
                        totals['Ray_generated'] += comp['candidates_generated']
                        totals['Ray_scored'] += comp['candidates_scored']
                        totals['Ray_retained_top50_memberships'] += len(comp['retained_top50_indices'])
                        if old['original_generation_seed'] != selected['seed']:
                            raise ValueError('Ray original seed differs')
                        hardware = old['original_actual_GPU'];old_sha = old['original_selected_sequence_sha256']
                    if old_sha != case['scientific_final_sequences'][method][str(beta)]['edited_sequence_sha256']:
                        raise ValueError('Historical native record is not the frozen reference')
                    if not matched and hardware != 'NVIDIA A40':
                        raise ValueError('Unresolved same-hardware native difference')
                    joins.append(dict(protein_id=pid, method=method, budget=beta, scenario=scenario,
                        historical_generation_GPU=hardware, timing_GPU=ident['actual_GPU'],
                        same_scientific_sequence=matched, same_input_and_seed=True,
                        historical_sequence_sha256=old_sha, actual_sequence_sha256=value['selected_sequence_sha256']))
                if scenario == 'cold' and method in ['rag_unlimited', 'gap_frequency']:
                    totals['uncached_queries'] += 1
                    totals['original_evidence_semantic_matches'] += int(stages['uncached_retrieval']['generation_evidence_semantics_match_original'])
                    profile = read(d/'cold_uncached_search/reviewed_profile.json')
                    if profile['query']['sequence_sha256'] != case['WT_sha256']:
                        raise ValueError('Query identity differs')
                    for rep in profile['representatives']:
                        q = reviewed.get(rep['target'], {})
                        if q.get('status') != 'RETAIN' or q.get('representative_sequence_sha256') != rep['target_sequence_sha256']:
                            raise ValueError('An unreviewed or mismatched representative was modeled')
    required = dict(head_Policy_calls=160, head_WT_encoder_batches=80, L_native_calls=40,
        Ray_requests=40, Ray_generated=20000, Ray_scored=20000, Ray_retained_top50_memberships=2000,
        uncached_queries=40, native_L_WT_token_caches_verified=20)
    if any(totals[k] != v for k, v in required.items()) or len(joins) != 80:
        raise ValueError('Requested workload/provenance scope is incomplete')
    for file_name, content in [('E10_native_gpu_sequence_identity_join.json', joins)]:
        with (out/file_name).open('x') as stream:json.dump(content, stream, indent=2)
    with (out/'E10_native_gpu_sequence_identity_join.csv').open('x', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(joins[0]));w.writeheader();w.writerows(joins)
    p = subprocess.run(['/opt/gridview/slurm/bin/sacct', '-j', '1121929,1122028',
        '--format=JobID,State,ExitCode,Submit,Eligible,Start,End,ElapsedRaw,NodeList,AllocTRES', '-P'],
        capture_output=True, text=True, check=True)
    for tag in ['1121929|COMPLETED|0:0|', '1122028|COMPLETED|0:0|']:
        if tag not in p.stdout:raise ValueError('Final Slurm success not proven')
    (out/'E10_final_sacct.txt').write_text(p.stdout)
    row = next(x for x in csv.DictReader(p.stdout.splitlines(), delimiter='|') if x['JobID'] == '1121929')
    queue = (datetime.datetime.fromisoformat(row['Start'])-datetime.datetime.fromisoformat(row['Submit'])).total_seconds()
    clock = dict(source='same Slurm controller raw timestamps; no cross-host subtraction',
        submit_raw=row['Submit'], start_raw=row['Start'], end_raw=row['End'],
        queue_seconds_including_brief_resource_hold=queue, Slurm_elapsed_seconds=int(row['ElapsedRaw']),
        deployment_preflight=read(base/'deployment_preflight.json'))
    (out/'E10_queue_and_deployment_context.json').write_text(json.dumps(clock, indent=2)+'\n')
    result = read(root/'reports/recomb_confirm_v2_E10_result/completion.json')
    for name, expected in result['outputs'].items():
        if sha(root/'reports/recomb_confirm_v2_E10_result'/name) != expected:
            raise ValueError('Final readout output changed')
    value = dict(status='E10_ALL240_AND_FROZEN_METHOD_ASSETS_INDEPENDENT_INTEGRITY_PASS',
        WT=20, methods=6, scenarios=2, actual_requests=240, actual_method_processes=120,
        actual_workload_counts=dict(totals), scientific_sequence_matches_by_method=dict(matches),
        historical_native_conditions_joined=80, native_mismatches_by_method=dict(collections.Counter(x['method'] for x in joins if not x['same_scientific_sequence'])),
        all_native_mismatches_historical_A40=True, unexplained_same_hardware_native_mismatches=0,
        frozen_source_model_sha256=pins, native_L_original_conditions_sha256=sha(l_path),
        main_timing_freeze_sha256=sha(root/'configs/recomb_confirm_v2_E10_timing_padding_v2_freeze.json'),
        result_completion_sha256=sha(root/'reports/recomb_confirm_v2_E10_result/completion.json'),
        job_queue_and_deployment_context=clock, new_model_calls=0, new_structure_predictions=0,
        verifier_source_sha256=sha(Path(__file__)))
    with (out/'E10_final_integrity_verification.json').open('x') as stream:json.dump(value, stream, indent=2)
    print(json.dumps({k: v for k, v in value.items() if k != 'frozen_source_model_sha256'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--root', type=Path, required=True)
    run(p.parse_args().root)
