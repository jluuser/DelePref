"""Measured process-cold and same-WT cached requests on one allocated A100.

The parent measures actual subprocess/event elapsed time, including imports,
model load, uncached query search and serialized final outputs. E8 is untouched.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013')
BASE = Path('reports/recomb_confirm_v2_E10_timing')
FREEZE = Path('configs/recomb_confirm_v2_E10_timing_freeze.json')
EVENT = 'RECOMB_E10_EVENT '
V7PY = '/public/home/zhangyangroup/chengshiz/keyuan.zhou/evocompress/.venv-v7/bin/python'
RAYPY = '/public/home/zhangyangroup/chengshiz/anaconda3/envs/delepath/bin/python'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def exclusive(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)


def prepare(root):
    from recomb_confirm_v2_E10_requests import METHODS
    if (root/FREEZE).exists():
        raise FileExistsError('E10 timing already frozen; inspect exact process/job claims')
    fidelity_dir = root/'reports/recomb_confirm_v2_E10_fidelity'
    fidelity = read(fidelity_dir/'completion.json')
    for name, h in fidelity['outputs'].items():
        if sha(fidelity_dir/name) != h:
            raise ValueError('Completed fidelity evidence differs')
    if not (fidelity_dir/'diagnosis_zh.md').exists():
        raise ValueError('Interpret the actual fidelity controls before full timing')
    clock_qa = read(root/BASE/'protocol_QA_completion.json')
    if clock_qa['status'] != 'E10_REAL_CHILD_PROCESS_CLOCK_BOUNDARIES_PASS' or clock_qa['returncode'] != 0:
        raise ValueError('cu9 process clock protocol QA has not passed')
    for name, h in clock_qa['source_sha256'].items():
        if sha(root/name) != h:
            raise ValueError('Clock protocol changed after its cu9 tests: '+name)
    adapter = read(root/'reports/recomb_confirm_v2_E10_adapter_check/completion.json')
    if adapter['main_method_conditions'] != 80 or not adapter['status'].endswith('EQUIVALENCE_PASS'):
        raise ValueError('Fixed20 scientific adapter equivalence is incomplete')
    adapter_source = read(root/'configs/recomb_confirm_v2_E10_adapter_check_freeze.json')
    if sha(root/'configs/recomb_confirm_v2_E10_adapter_check_freeze.json') != adapter['freeze_sha256']:
        raise ValueError('Completed adapter source freeze differs')
    for name, h in adapter['outputs'].items():
        if sha(root/'reports/recomb_confirm_v2_E10_adapter_check'/name) != h:
            raise ValueError('Completed adapter proof differs: '+name)
    for name, h in adapter_source['files'].items():
        if sha(root/name) != h:
            raise ValueError('Verified standalone adapter dependency changed: '+name)
    old = read(root/'configs/recomb_confirm_v2_General100_GPU_costs_freeze.json')
    parent = read(root/'configs/delepref_v8_baseline_freeze.json')
    retrieval = read(root/'configs/recomb_confirm_v2_General100_retrieval_freeze.json')
    records = {r['protein_id']: r for r in read(root/'reports/recomb_confirm_v2_general_native/encoder_inputs.json')}
    cases = [dict(r, wildtype_sequence=records[r['protein_id']]['wildtype_sequence'])
             for r in read(root/'reports/recomb_confirm_v2_General100_costs/cases.json')]
    if len(cases) != 20 or any(hashlib.sha256(r['wildtype_sequence'].encode()).hexdigest() != r['WT_sha256'] for r in cases):
        raise ValueError('Original fixed20 ordinary request identities differ')
    files = ['analysis/recomb_confirm_v2_E10_timing.py', 'analysis/tests/test_recomb_confirm_v2_E10_timing.py', 'analysis/recomb_confirm_v2_E10_requests.py',
        'analysis/recomb_confirm_v2_E10_fidelity.py', 'slurm/recomb_confirm_v2_E10_timing.sbatch',
        'configs/recomb_confirm_v2_E10_adapter_check_freeze.json',
        'reports/recomb_confirm_v2_E10_adapter_check/completion.json',
        'reports/recomb_confirm_v2_E10_adapter_check/head_adapter_equivalence.json',
        'reports/recomb_confirm_v2_E10_fidelity/completion.json',
        'reports/recomb_confirm_v2_E10_fidelity/diagnosis_zh.md',
        'reports/recomb_confirm_v2_E10_fidelity/fidelity_summary.json',
        'configs/recomb_confirm_v2_General100_retrieval_freeze.json',
        'analysis/recomb_benign_profiles_20261017.py', 'analysis/delepref_v8_retrieval.py',
        'analysis/rag_v6_retrieval.py', 'analysis/rag_v6_features.py',
        'reports/recomb_confirm_v2_General100_evidence_review/target_metadata_audit.json',
        'retrieval/recomb_confirm_v2_General100/excluded_sequence_hashes.json']
    files += [str(BASE/'protocol_QA_completion.json'), 'configs/recomb_confirm_v2_E10_timing_protocol_QA_freeze.json']
    files += list(old['files'])
    files += list(adapter_source['files'])
    sources = {str(p): sha(root/p) for p in sorted(set(files))}
    for p, h in {**old['files'], **parent['inputs'], **parent['code'], **old['Raygun_external_code'],
                 **old['SCISOR_external_code'], **old['assets_sha256']}.items():
        if sha(p if Path(p).is_absolute() else root/p) != h:
            raise ValueError('Original scientific code/asset differs: '+p)
    write(root/FREEZE, dict(status='E10_FIXED20_SIX_METHOD_TRUE_REQUEST_TIMING_FROZEN',
        files=sources, original_dependencies={**parent['inputs'], **parent['code']},
        native_external_code={**old['Raygun_external_code'], **old['SCISOR_external_code']},
        assets_sha256=old['assets_sha256'], target_database=retrieval['database'],
        cases=cases, methods=list(METHODS), cold_budget=.2, cached_budget=.3,
        method_processes=120, request_outputs=240, same_physical_allocated_A100=True,
        method_order='cyclic rotation by frozen case index',
        authoritative_latency='parent monotonic from subprocess launch/request signal through final-output completion event',
        first_request='new process and WT cache miss; existing model/target-database assets ready',
        cached_request='same loaded process and WT, beta .3; reuse actual cold-request evidence/representation where native API supports',
        target_database_offline_build_not_repeated=True, uncached_query_createdb_search_convertalis_included=True,
        deployment_bundle_preflight_and_Slurm_queue_time_separate=True,
        OS_filesystem_cache_may_already_be_warm=True,
        native_L_K=1, native_Raygun_generated_and_scored=500, native_Raygun_retained=50,
        weights_environment_or_scientific_output_changes=False, new_structure_predictions=0, new_training=False,
        V7_python=V7PY, Raygun_python=RAYPY))
    (root/BASE).mkdir(parents=True, exist_ok=True)
    print(sha(root/FREEZE), flush=True)


def worker(root, checksum, index, method):
    # Parent launch timestamp predates this file's imports and argument parsing.
    cold_internal_start = time.perf_counter()
    if sha(root/FREEZE) != checksum:
        raise ValueError('Timed worker freeze differs')
    frozen = read(root/FREEZE)
    case = frozen['cases'][index]
    from recomb_confirm_v2_E10_requests import HEAD_METHODS, reviewed_uncached_search, head_request, load_native, native_request
    from recomb_confirm_v2_E9_completion import digest
    if method not in frozen['methods']:
        raise ValueError('Method not frozen')
    directory = root/BASE/'requests'/('%02d_%s' % (index, case['WT_sha256'][:12]))/method
    directory.mkdir(parents=True, exist_ok=False)
    exclusive(directory/'execution_claim.json', dict(job_id=os.environ['SLURM_JOB_ID'], method=method,
        protein_id=case['protein_id'], freeze_sha256=checksum, process_id=os.getpid()))
    import torch
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name():
        raise RuntimeError('All E10 request measurements require the same allocated A100')
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    record = dict(protein_id=case['protein_id'], wildtype_sequence=case['wildtype_sequence'], budget=.2)
    if hashlib.sha256(record['wildtype_sequence'].encode()).hexdigest() != case['WT_sha256']:
        raise ValueError('Actual request sequence differs')
    before = time.perf_counter()
    if method in HEAD_METHODS:
        import delepref_v8_runtime as rt
        import delepref_v8_main as parent
        gpu, api = rt.load(root)
        cpu, cpuapi = rt.load(root, device='cpu', cached_heads_only=True)
        state = dict(gpu=gpu, api=api, cpu=cpu, cpuapi=cpuapi, chunk=None, profile=None)
    else:
        state = load_native(root, method)
    torch.cuda.synchronize()
    setup = time.perf_counter()-before
    measurements = []
    with torch.inference_mode():
        for scenario, beta in [('cold', frozen['cold_budget']), ('cached', frozen['cached_budget'])]:
            if scenario == 'cached':
                if sys.stdin.readline().strip() != 'RUN_CACHED_REQUEST':
                    raise RuntimeError('Parent did not issue the further request')
                request_started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
            else:
                request_started = cold_internal_start
            stages = {}
            if scenario == 'cold':
                stages['model_setup_seconds'] = setup
            if method in HEAD_METHODS:
                if state['profile'] is None:
                    if method in ('rag_unlimited', 'gap_frequency'):
                        profile, search = reviewed_uncached_search(root, case, directory/'cold_uncached_search')
                        state['profile'] = profile
                        stages['uncached_retrieval'] = search
                    else:
                        state['profile'] = {'representatives': []}
                if state['chunk'] is None:
                    before = time.perf_counter()
                    before_batches = state['gpu'].model.encoder.forward_batches
                    encoded = state['api'].encode_chunk(state['gpu'], [record])
                    torch.cuda.synchronize()
                    cache = dict(embeddings={k.protein_id: v.detach().cpu() for k, v in encoded.encoded.embeddings.items()},
                        sequences={record['protein_id']: record['wildtype_sequence']})
                    state['chunk'] = parent.cached_chunk(state['cpuapi'], cache, [record])
                    torch.save(cache, directory/'actual_cold_WT_cache.pt')
                    stages['WT_encoding_transfer_and_cache_seconds'] = time.perf_counter()-before
                    stages['actual_WT_encoder_batches'] = state['gpu'].model.encoder.forward_batches-before_batches
                    stages['actual_WT_cache_sha256'] = sha(directory/'actual_cold_WT_cache.pt')
                    del encoded
                else:
                    stages['actual_WT_encoder_batches'] = 0
                    stages['actual_WT_cache_sha256'] = sha(directory/'actual_cold_WT_cache.pt')
                result = head_request(state['cpu'], state['cpuapi'], state['chunk'], record, state['profile'], method, beta)
            else:
                native_key = 'reusable_embedding' if method == 'raygun_native' else 'reusable_tokens'
                native_cache_hit = state[native_key] is not None
                result = native_request(state, record, beta)
                if scenario == 'cold':
                    torch.save(state[native_key].detach().cpu(), directory/'actual_cold_native_WT_cache.pt')
                stages['native_input_cache_hit'] = native_cache_hit
                stages['native_input_cache_sha256'] = sha(directory/'actual_cold_native_WT_cache.pt')
                stages['actual_native_WT_embedding_calls'] = int(method == 'raygun_native' and not native_cache_hit)
            torch.cuda.synchronize()
            before = time.perf_counter()
            output_dir = directory/scenario
            write(output_dir/'candidate_pool.json', result['candidate_pool'])
            write(output_dir/'selected_output.json', result['selected'])
            write(output_dir/'request_identity.json', dict(protein_id=case['protein_id'], WT_sha256=case['WT_sha256'],
                beta=beta, method=method, freeze_sha256=checksum, actual_GPU=torch.cuda.get_device_name(),
                CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'), SLURMD_NODENAME=os.environ.get('SLURMD_NODENAME'),
                actual_reviewed_profile_sha256=digest(state['profile']) if method in HEAD_METHODS else None,
                actual_WT_cache_sha256=stages.get('actual_WT_cache_sha256', stages.get('native_input_cache_sha256')),
                selected_sequence_sha256=result['selected']['edited_sequence_sha256']))
            stages['final_output_serialization_seconds'] = time.perf_counter()-before
            internal_elapsed = time.perf_counter()-request_started
            measurements.append(dict(scenario=scenario, method=method, beta=beta, protein_id=case['protein_id'],
                internal_request_seconds=internal_elapsed,
                peak_GPU_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                peak_GPU_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                stages=stages, components={k: v for k, v in result.items() if k not in ['candidate_pool', 'selected']},
                selected_sequence_sha256=result['selected']['edited_sequence_sha256'],
                output_files_sha256={n: sha(output_dir/n) for n in ['candidate_pool.json', 'selected_output.json', 'request_identity.json']}))
            write(output_dir/'measurement.json', measurements[-1])
            print(EVENT+json.dumps(dict(scenario=scenario, measurement=measurements[-1])), flush=True)
    write(directory/'worker_completion.json', dict(status='TWO_ACTUAL_REQUESTS_PRODUCED', freeze_sha256=checksum,
        method=method, protein_id=case['protein_id'], measurements=measurements, no_structures_predicted=True))


def observe_process(command, cwd, env, log_path, before_cached_request=None):
    """Time completed output events; exclude idle time between the two requests."""
    parent_started = time.perf_counter()
    process = subprocess.Popen(command, cwd=str(cwd), env=env, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    observed, warm_started = [], None
    with Path(log_path).open('w') as log:
        for line in process.stdout:
            log.write(line);log.flush()
            if not line.startswith(EVENT):
                continue
            event = json.loads(line[len(EVENT):])
            now = time.perf_counter()
            value = event['measurement']
            if event['scenario'] == 'cold' and not observed:
                value['parent_observed_request_seconds'] = now-parent_started
                if before_cached_request is not None:
                    before_cached_request()
                warm_started = time.perf_counter()
                process.stdin.write('RUN_CACHED_REQUEST\n');process.stdin.flush()
            elif event['scenario'] == 'cached' and warm_started is not None and len(observed) == 1:
                value['parent_observed_request_seconds'] = now-warm_started
            else:
                process.kill()
                process.wait()
                process.stdin.close();process.stdout.close()
                raise ValueError('Unexpected request completion event order')
            observed.append(value)
    returncode = process.wait()
    process.stdin.close();process.stdout.close()
    return returncode, observed


def execute(root, checksum):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('E10 complete request timing requires Slurm')
    if sha(root/FREEZE) != checksum:
        raise ValueError('E10 timing freeze differs')
    frozen = read(root/FREEZE)
    out = root/BASE
    exclusive(out/'execution_claim.json', dict(job_id=os.environ['SLURM_JOB_ID'], freeze_sha256=checksum))
    preflight_started = time.perf_counter()
    for p, h in {**frozen['files'], **frozen['original_dependencies'], **frozen['native_external_code'], **frozen['assets_sha256']}.items():
        if sha(p if Path(p).is_absolute() else root/p) != h:
            raise ValueError('E10 deployment bundle changed: '+p)
    for p, m in frozen['target_database'].items():
        stat = Path(p).stat()
        if (stat.st_size, stat.st_mtime_ns) != (m['size'], m['mtime_ns']):
            raise ValueError('Existing database changed')
    write(out/'deployment_preflight.json', dict(seconds=time.perf_counter()-preflight_started,
        excluded_from_per_request_latency=True, target_database_rebuilt=False))
    results = []
    for index, case in enumerate(frozen['cases']):
        methods = frozen['methods']
        offset = index % len(methods)
        for method in methods[offset:]+methods[:offset]:
            directory = out/'requests'/('%02d_%s' % (index, case['WT_sha256'][:12]))/method
            directory.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled')
            interpreter = frozen['V7_python']
            if method == 'raygun_native':
                interpreter = frozen['Raygun_python']
                env['PYTHONPATH'] = '/public/nfs-data/keyuan.zhou/evocompress/results/formal_v9/external_raygun/tools/fair_esm_site:/public/nfs-data/keyuan.zhou/evocompress/external/raygun/repo:'+env.get('PYTHONPATH', '')
            command = [interpreter, '-u', str(root/'analysis/recomb_confirm_v2_E10_timing.py'), '--root', str(root),
                '--freeze-sha256', checksum, '--case-index', str(index), '--worker-method', method]
            log_path = directory.parent/(method+'_process.log')
            returncode, observed = observe_process(command, root, env, log_path)
            if returncode or [r['scenario'] for r in observed] != ['cold', 'cached']:
                write(out/'process_failure.json', dict(method=method, case_index=index, returncode=returncode,
                    log_path=str(log_path), no_blind_restart=True))
                raise RuntimeError('Actual E10 request process failed; inspect source/log before any retry')
            write(directory/'parent_observed_requests.json', observed)
            results.extend(observed)
            write(out/'progress.json', dict(finished_method_processes=len(results)//2, total_method_processes=120,
                measured_requests=len(results), total_requests=240))
            print('E10_TRUE_TWO_REQUESTS_COMPLETE '+case['protein_id']+' '+method, flush=True)
    if len(results) != 240 or len({(r['protein_id'], r['method'], r['scenario']) for r in results}) != 240:
        raise ValueError('E10 fixed20 six-method two-scenario coverage differs')
    write(out/'all_request_measurements.json', results)
    columns = ['protein_id', 'method', 'scenario', 'beta', 'parent_observed_request_seconds',
               'internal_request_seconds', 'peak_GPU_allocated_bytes', 'peak_GPU_reserved_bytes', 'selected_sequence_sha256']
    with (out/'complete_request_table.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader();writer.writerows({k: r[k] for k in columns} for r in results)
    # Analysis and human-readable delivery are a separate verified stage.
    names = ['all_request_measurements.json', 'complete_request_table.csv', 'deployment_preflight.json']
    write(out/'measurement_completion.json', dict(status='E10_ALL240_ACTUAL_REQUEST_MEASUREMENTS_COMPLETE',
        fixed_WT=20, methods=6, scenarios=2, method_processes=120, freeze_sha256=checksum,
        no_new_structure_predictions=True, no_scientific_winner_replacement=True,
        outputs={n: sha(out/n) for n in names}, job_id=os.environ['SLURM_JOB_ID']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--freeze-sha256')
    parser.add_argument('--worker-method')
    parser.add_argument('--case-index', type=int)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.root)
    elif args.worker_method:
        worker(args.root, args.freeze_sha256, args.case_index, args.worker_method)
    else:
        execute(args.root, args.freeze_sha256)
