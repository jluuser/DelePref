"""Actual standalone shrinking requests using the original scientific components.

No structures are predicted here. Timed outputs are versioned independently of
the completed quality experiment. Offline assets and reviewed evidence are reused.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import time

from recomb_confirm_v2_diagnostics import read, sha, write
from recomb_confirm_v2_E9_completion import digest

HEAD_METHODS = ('delepref_unlimited', 'rag_unlimited', 'gap_frequency', 'terminal')
METHODS = HEAD_METHODS+('scisor_L_native', 'raygun_native')


def reviewed_uncached_search(root, case, directory):
    """Execute real query search; only existing positively reviewed hits can pass."""
    import delepref_v8_retrieval as native
    import rag_v6_retrieval as original
    from recomb_benign_profiles_20261017 import filtered_profile
    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=False)
    frozen = read(root/'configs/recomb_confirm_v2_General100_retrieval_freeze.json')
    for path, metadata in frozen['database'].items():
        stat = Path(path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (metadata['size'], metadata['mtime_ns']):
            raise ValueError('Existing read-only target database changed')
        if 'sha256' in metadata and sha(path) != metadata['sha256']:
            raise ValueError('Existing target index changed')
    if sha(original.MMSEQS) != frozen['mmseqs_sha256'] or frozen['contract'] != native.CONTRACT:
        raise ValueError('Original MMseqs or projection contract differs')
    query_id = 'E10q_'+case['WT_sha256'][:20]
    sequence = case['wildtype_sequence']
    if hashlib.sha256(sequence.encode()).hexdigest() != case['WT_sha256']:
        raise ValueError('First-query sequence identity differs')
    query = dict(query_id=query_id, protein_id=case['protein_id'], sequence=sequence,
                 sequence_sha256=case['WT_sha256'])
    fasta = directory/'query.fasta'
    fasta.write_text('>'+query_id+'\n'+sequence+'\n', encoding='ascii')
    qdb, rdb = str(directory/'query_db'), str(directory/'result_db')
    commands = [
        [original.MMSEQS, 'createdb', str(fasta), qdb, '--threads', '32'],
        [original.MMSEQS, 'search', qdb, frozen['target'], rdb, str(directory/'tmp'),
         '-a', '1', '--alignment-mode', '3', '-e', '0.001', '--min-seq-id', '0.20',
         '-c', '0.60', '--cov-mode', '2', '--threads', '32', '-s', '7.5', '--max-seqs', '512',
         '--split-mode', '0', '--split-memory-limit', '160G', '--remove-tmp-files', '1'],
        [original.MMSEQS, 'convertalis', qdb, frozen['target'], rdb, str(directory/'hits.tsv'),
         '--threads', '32', '--format-output', native.FORMAT]]
    stages = []
    for index, command in enumerate(commands):
        before = time.perf_counter()
        with (directory/('command_'+str(index)+'.log')).open('w') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        stages.append(dict(command=command, seconds=time.perf_counter()-before))
    before = time.perf_counter()
    hits = []
    for line in (directory/'hits.tsv').read_text().splitlines():
        fields = line.split('\t')
        if fields[0] != query_id:
            raise ValueError('Unexpected query identity in uncached search')
        hits.append(native.project(fields, sequence))
    excluded = set(read(root/'retrieval/recomb_confirm_v2_General100/excluded_sequence_hashes.json'))
    representatives, audit = native.representatives(hits, excluded)
    unreviewed = dict(query=query, representatives=representatives, audit=audit, contract=native.CONTRACT)
    reviewed = read(root/'reports/recomb_confirm_v2_General100_evidence_review/target_metadata_audit.json')
    profile = filtered_profile(unreviewed, dict(sequence_sha256=case['WT_sha256']), reviewed)
    profile['evidence_status'] = 'SOURCE_FUNCTION_AND_FULL_SEQUENCE_IDENTITY_REVIEWED'
    profile['scope_filter'] = 'confirm_v2_reference_source_groups_and_function_review'
    old = read(root/'retrieval/recomb_confirm_v2_General100/profiles.json')[case['protein_id']]
    def semantic_rows(p):
        return [(r['target'], r['target_sequence_sha256'], r['projected']) for r in p['representatives']]
    metadata = dict(query_database_creation_seconds=stages[0]['seconds'],
        natural_database_search_seconds=stages[1]['seconds'], alignment_export_seconds=stages[2]['seconds'],
        projection_representative_and_review_filter_seconds=time.perf_counter()-before,
        measured_uncached_query_total_seconds=time.perf_counter()-started, commands=stages,
        raw_hits=len(hits), representatives_before_review=len(representatives), reviewed_representatives=len(profile['representatives']),
        generation_evidence_semantics_match_original=semantic_rows(profile) == semantic_rows(old),
        new_profile_semantic_sha256=digest(semantic_rows(profile)), original_profile_semantic_sha256=digest(semantic_rows(old)),
        hits_sha256=sha(directory/'hits.tsv'), reviewed_profile_sha256=digest(profile),
        existing_target_database_reused=True, target_database_construction_per_request=False,
        unknown_source_hits_not_modeled=True)
    write(directory/'reviewed_profile.json', profile)
    write(directory/'search_measurement.json', metadata)
    return profile, metadata


def head_request(runtime, api, chunk, record, profile, method, beta):
    """Standalone adapter to the exact original planning/Scorer functions.

    The CPU preflight compares all fixed20 pools/winners with scientific output.
    Only the requested branch runs; unrelated baseline branches are not charged.
    """
    import numpy as np
    import delepref_v8_main as parent
    import delepref_unlimited_planning as planning
    import delepref_unlimited_scorer as scoring
    from delepref_v8_controls import condition_seed, terminal_proposals
    from recomb_gap_frequency_control_20261017 import proposals as gap_proposals
    from recomb_general100_mechanisms_20261017 import select_rows
    if method not in HEAD_METHODS:
        raise ValueError('Not a frozen DelePref-family method')
    pid, sequence = record['protein_id'], record['wildtype_sequence']
    template = dict(record, budget=beta, sequence_length=len(sequence), sequence_sha256=api.sequence_sha256(sequence),
        generation_seed=13, condition_seed=condition_seed(13, pid, sequence, beta),
        student_version=runtime.checkpoint['student_version'],
        source_checkpoint_sha256=runtime.source_checkpoint_sha256, selected_checkpoint_step=4000)
    before = time.perf_counter()
    embedding = chunk.encoded.embeddings[chunk.encoded.key_for(pid, sequence)]
    logits = runtime.model.propose_from_embeddings([embedding], [beta])[0].residue_logits.detach().cpu().double().numpy()
    logit_seconds = time.perf_counter()-before
    before = time.perf_counter()
    scorer_calls = 0
    diagnostic = {}
    if method == 'terminal':
        items = terminal_proposals(len(sequence), beta)
        generation_seconds = time.perf_counter()-before
        before = time.perf_counter()
        pool = scoring.score_candidates(runtime, api, chunk, parent.make_rows(api, template, items, len(sequence), logits))
        score_seconds = time.perf_counter()-before
        scorer_calls += 1
    else:
        decoder = runtime.config['decoder']
        items, cost = planning.policy_proposals(logits, beta, template['condition_seed'],
            perturbation_scale=float(decoder.get('perturbation_scale', 1.)),
            max_attempts=8*int(decoder.get('max_attempts_multiplier', 16)))
        if len(items) != 8:
            raise ValueError('Original Policy did not produce complete unique K8')
        generation_seconds = time.perf_counter()-before
        before = time.perf_counter()
        original = scoring.score_candidates(runtime, api, chunk,
            parent.make_rows(api, template, items, len(sequence), logits, cost))
        score_seconds = time.perf_counter()-before
        scorer_calls += 1
        pool = original
        diagnostic['original_generation'] = cost
        if method in ('rag_unlimited', 'gap_frequency'):
            native = [dict(item, row=row) for item, row in zip(items, original)]
            before = time.perf_counter()
            if method == 'rag_unlimited':
                retrieved, attempts = planning.retrieved_proposals(logits, beta, profile, template['condition_seed'])
                diagnostic['retrieval_attempts'] = attempts
            else:
                retrieved, gap = gap_proposals(logits, beta, profile)
                diagnostic['gap_guidance'] = gap
            mixed, fallback = planning.mix_candidates(native, retrieved)
            generation_seconds += time.perf_counter()-before
            diagnostic['fallback'] = fallback
            if not fallback:
                before = time.perf_counter()
                pool = scoring.score_candidates(runtime, api, chunk,
                    parent.make_rows(api, template, mixed, len(sequence), logits,
                        cost if method == 'rag_unlimited' else None))
                score_seconds += time.perf_counter()-before
                scorer_calls += 1
    if len(pool) != 8 or len({r['deletion_mask_sha256'] for r in pool}) != 8:
        raise ValueError('Standalone scientific request must return complete unique K8')
    winner = select_rows(api, sequence, pool, method)
    return dict(method=method, beta=beta, candidate_pool=pool, selected=winner,
        policy_logits_seconds=logit_seconds, generation_seconds=generation_seconds,
        scoring_seconds=score_seconds, actual_Policy_WT_calls=1, actual_Scorer_batches=scorer_calls,
        candidates_returned=8, diagnostic=diagnostic)


def load_native(root, method):
    """Original official L/Raygun loaders, in their existing environments."""
    import importlib.util
    import sys
    import torch
    if method == 'scisor_L_native':
        sys.path.insert(0, '/public/home/zhangyangroup/chengshiz/keyuan.zhou/SurgEdit')
        from recomb_confirm_v2_scisor_L_loader_v2 import load
        source = read(root/'configs/recomb_confirm_v2_General100_scisor_L_freeze.json')
        model, payload, _ = load(root)
        del payload
        model = model.to('cuda').eval()
        model.p0 = torch.load(source['p0']['path'], map_location='cuda', weights_only=True)
        rate = source['alpha_rate']
        model.alpha = lambda t: (1-t)**rate
        model.beta = lambda t: rate/(1-t)
        model.log_alpha = lambda t: -torch.log(model.alpha(t))
        return dict(method=method, model=model, source=source, reusable_tokens=None)
    if method != 'raygun_native':
        raise ValueError('Unsupported native method')
    filename = '/public/nfs-data/keyuan.zhou/evocompress/results/formal_v9/external_raygun/tools/raygun_worker.py'
    spec = importlib.util.spec_from_file_location('E10_official_native_raygun', filename)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    torch, _, model, encoder, alphabet, incompatible = native.load_raygun_models('cuda')
    contract = read(root/'reports/recomb_confirm_v2_General100_raygun_v2/raygun_checkpoint_load_contract.json')
    if sorted(incompatible.missing_keys) != sorted(contract['missing_keys']) or list(incompatible.unexpected_keys) != contract['unexpected_keys']:
        raise ValueError('Original Raygun checkpoint-load contract differs')
    tasks = {r['protein_id']: r for r in read(root/'reports/recomb_confirm_v2_General100_raygun_v2/raygun_tasks.json') if r['budget'] == .2}
    return dict(method=method, native=native, model=model, encoder=encoder, alphabet=alphabet,
                reusable_embedding=None, tasks=tasks)


def native_request(state, record, beta):
    import random
    import numpy as np
    import torch
    from delepref_v8_controls import condition_seed
    from delepref_unlimited_planning import delete_count, validate_plan
    pid, sequence = record['protein_id'], record['wildtype_sequence']
    method = state['method']
    encoding_before = time.perf_counter()
    if method == 'scisor_L_native':
        from delepref_v8_scisor_mapping import mapped_plan
        model = state['model']
        if state['reusable_tokens'] is None:
            state['reusable_tokens'] = torch.tensor(model.tokenizer(sequence).input_ids, device='cuda').unsqueeze(0)
        tokens = state['reusable_tokens']
        torch.cuda.synchronize()
        encoding_seconds = time.perf_counter()-encoding_before
        seed = condition_seed(13, pid, sequence, beta)
        random.seed(seed);np.random.seed(seed % (2**32-1));torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        before = time.perf_counter()
        result, preserved = model.shrink_sequence(tokens,
            torch.tensor([delete_count(len(sequence), beta)], device='cuda'), temperature=1.)
        item = mapped_plan(sequence, beta, preserved[0], result[0].detach().cpu().tolist(), model.tokenizer)
        mask = validate_plan(len(sequence), beta, item['deleted_intervals'])
        torch.cuda.synchronize()
        generation_seconds = time.perf_counter()-before
        winner = dict(item, protein_id=pid, method=method, budget=beta, seed=seed,
            deletion_mask_sha256=hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest(),
            edited_sequence_sha256=hashlib.sha256(item['edited_sequence'].encode()).hexdigest())
        return dict(method=method, beta=beta, candidate_pool=[winner], selected=winner,
            input_tokenization_seconds=encoding_seconds, generation_seconds=generation_seconds,
            scoring_seconds=0., candidates_generated=1, candidates_scored_by_external_Scorer=0,
            native_K=1, reuse_scope='loaded_model_and_WT_tokens; no unsupported latent-representation cache')
    native, model, encoder, alphabet = (state[k] for k in ['native', 'model', 'encoder', 'alphabet'])
    seed = native.stable_seed(20260830, 'official-native', state['tasks'][pid]['dataset'], pid, beta)
    random.seed(seed);np.random.seed(seed & 0xffffffff);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    if state['reusable_embedding'] is None:
        state['reusable_embedding'], _ = native._esm_embeddings(torch, encoder, alphabet, [sequence], 'cuda')
    embedding = state['reusable_embedding']
    torch.cuda.synchronize()
    encoding_seconds = time.perf_counter()-encoding_before
    target = len(sequence)-delete_count(len(sequence), beta)
    before = time.perf_counter()
    generated, _ = native._generate_sequences(torch, encoder, alphabet, model, embedding[:1],
        [target]*500, [1.]*500, 'N/A', 'cuda', batch_size=8)
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter()-before
    if len(generated) != 500 or any(len(s) != target for s in generated):
        raise ValueError('Native Raygun count/length differs')
    before = time.perf_counter()
    scores, native_scoring_seconds = native._pll_scores(torch, encoder, alphabet, generated, 'cuda', batch_size=8)
    torch.cuda.synchronize()
    scoring_seconds = time.perf_counter()-before
    if len(scores) != 500 or not np.isfinite(scores).all():
        raise ValueError('Native Raygun must score all500 sequences')
    order = sorted(range(500), key=lambda i: (-scores[i], i))
    pool = [dict(candidate_index=i, edited_sequence=s, edited_sequence_sha256=hashlib.sha256(s.encode()).hexdigest(),
                 native_score=float(scores[i])) for i, s in enumerate(generated)]
    winner = dict(pool[order[0]], protein_id=pid, method=method, budget=beta, seed=seed)
    return dict(method=method, beta=beta, candidate_pool=pool, selected=winner,
        WT_encoding_seconds=encoding_seconds, generation_seconds=generation_seconds,
        scoring_seconds=scoring_seconds, native_reported_scoring_seconds=native_scoring_seconds,
        candidates_generated=500, candidates_scored=500, retained_top50_indices=order[:50],
        native_score_definition='upstream unmasked adjusted PLL log-score sum; unchanged',
        reuse_scope='loaded_models_and_actual_WT_ESM2_embedding')
