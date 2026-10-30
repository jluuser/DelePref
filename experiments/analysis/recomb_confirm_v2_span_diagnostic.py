"""Decoder-cap-only 2x2 development comparison; never changes the main task."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np

from recomb_confirm_v2_diagnostics import read, sha, write


def q3_complete(values, beta, forced=None, kept=None):
    from delepref_rag_planning import constrained_dp
    n = len(values)
    z = np.zeros(n, dtype=bool)
    return constrained_dp(values, beta, z if forced is None else forced,
                          z if kept is None else kept, qmax=3, max_span=max(1, n//2))


def policy(values, beta, condition_seed, scale, max_attempts):
    import torch
    from delepref_unlimited_planning import mask_for
    base = torch.as_tensor(np.asarray(values, dtype=np.float64).copy())
    result, seen, duplicates = [], set(), 0
    for attempt in range(max_attempts):
        if attempt == 0 or scale == 0:
            noisy = base
        else:
            g = torch.Generator(device="cpu").manual_seed(int(condition_seed)+attempt*1_000_003)
            noisy = base + scale*torch.randn(base.shape, generator=g, dtype=base.dtype)
        spans, _ = q3_complete(noisy.numpy(), beta)
        if spans in seen:
            duplicates += 1
            continue
        seen.add(spans)
        result.append(dict(spans=spans, kind="original_policy_q3_diagnostic",
                           policy_energy=float(values[mask_for(len(values), spans)].mean())))
        if len(result) == 8:
            break
    if len(result) != 8:
        raise RuntimeError("Incomplete unique diagnostic K8")
    return result, dict(attempts=attempt+1, duplicates=duplicates, requested=8, returned=8)


def retrieval(values, beta, profile, condition_seed):
    from delepref_unlimited_planning import seed_sets, mask_for
    from delepref_rag_planning import InfeasibleCompletion
    n = len(values)
    result, attempts, seen = [], [], set()
    # Use the current unlimited seed sets unchanged: only the final decoder cap changes.
    for item in seed_sets(profile, n, beta):
        forced = mask_for(n, item["spans"])
        kept = np.zeros(n, dtype=bool)
        for a,b in item["spans"]:
            kept[a-1] = True; kept[b+1] = True
        for refinement in range(2):
            if item["complete_natural_plan"] and refinement:
                continue
            noisy = values
            if refinement:
                salt = int(hashlib.sha256((item["target"]+str(item["spans"])).encode()).hexdigest()[:8],16)
                noisy = values + np.random.default_rng((condition_seed+salt)%(2**32)).standard_normal(n)
            try:
                spans, _ = q3_complete(noisy, beta, forced, kept)
            except InfeasibleCompletion as error:
                attempts.append(dict(target=item["target"], seed_spans=item["spans"],
                                     status="infeasible", reason=str(error), refinement=refinement))
                continue
            duplicate = spans in seen
            attempts.append(dict(target=item["target"], seed_spans=item["spans"],
                                 status="duplicate" if duplicate else "accepted", refinement=refinement))
            if duplicate:
                continue
            seen.add(spans)
            result.append(dict(spans=spans, source_target=item["target"], seed_spans=item["spans"],
                all_reliable_events_used=item["all_reliable_events_used"],
                kind="retrieval_direct" if item["complete_natural_plan"] else (
                    "retrieval_refined" if refinement else "retrieval_completed")))
            if len(result) == 4:
                return result, attempts
    return result, attempts


def main(root):
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU Slurm allocation required")
    frozen = read(root / "configs/recomb_confirm_v2_span_freeze.json")
    for name,digest in frozen["files"].items():
        if sha(root/name) != digest:
            raise ValueError("Diagnostic input/source changed: "+name)
    out = root/"reports/recomb_confirm_v2_span_diagnostic"
    if (out/"planning_completion.json").exists():
        raise FileExistsError("Diagnostic already complete")
    import torch
    import delepref_v8_runtime as rt
    import delepref_unlimited_planning as unlimited
    from delepref_v8_rag_unlimited import cached_chunk, make_rows
    from delepref_v8_main import controls
    torch.set_num_threads(4)
    start = time.monotonic()
    panel = read(root/"reports/recomb_benign_scope_20261017/general_panel.json")
    approved = {p["protein_id"]:p for p in panel}
    ids = read(root/"reports/recomb_benign_final_subsets_20261017/subsets.json")["completion_ablation"]
    if len(ids) != 20 or not set(ids) <= set(approved):
        raise ValueError("Development subset is not the reviewed frozen20")
    provenance = read(root/"reports/delepref_v8_main/encoding_completion.json")
    cache_path = root/"reports/delepref_v8_main/wt_cache.pt"
    if sha(cache_path) != provenance["cache_sha256"]:
        raise ValueError("WT encoder cache differs")
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    if cache["checkpoint_sha256"] != rt.CHECKPOINT_SHA or not cache["real_encoder"]:
        raise ValueError("Not the formal model real WT cache")
    for key in ("sequences", "embeddings"):
        cache[key] = {p:v for p,v in cache[key].items() if p in ids}
    records = [dict(protein_id=p, wildtype_sequence=approved[p]["sequence"], budget=.2) for p in ids]
    runtime,api = rt.load(root, device="cpu", cached_heads_only=True)
    chunk = cached_chunk(api, cache, records)
    profiles = read(root/"retrieval/recomb_benign_general_20261017/profiles.json")
    old = read(root/"reports/recomb_benign_plans_20261017/all_selections.json")
    inherited = [r for r in old if r["protein_id"] in ids and r["budget"] == .2
                 and r["generation_seed"] == 13 and r["method"] in ("delepref_unlimited", "rag_unlimited")]
    if len(inherited) != 40:
        raise ValueError("Missing existing unlimited arm")
    candidates, selections, diagnostics = [], [], []
    with torch.inference_mode():
        for record in records:
            pid,seq = record["protein_id"],record["wildtype_sequence"]
            template = dict(record, sequence_length=len(seq), sequence_sha256=api.sequence_sha256(seq),
                generation_seed=13, condition_seed=controls.condition_seed(13,pid,seq,.2),
                student_version=runtime.checkpoint["student_version"],
                source_checkpoint_sha256=runtime.source_checkpoint_sha256, selected_checkpoint_step=4000)
            embedding = chunk.encoded.embeddings[chunk.encoded.key_for(pid,seq)]
            logits = runtime.model.propose_from_embeddings([embedding],[.2])[0].residue_logits.detach().cpu().double().numpy()
            cfg = runtime.config["decoder"]
            items,cost = policy(logits,.2,template["condition_seed"],float(cfg.get("perturbation_scale",1.)),
                                8*int(cfg.get("max_attempts_multiplier",16)))
            def score(items):
                rows = make_rows(api,template,items,len(seq),logits,cost)
                for row in rows:
                    if row["span_count"] > 3:
                        raise ValueError("Diagnostic cap violated")
                    row.update(qmax=3, decoder="exact_additive_q3_development_diagnostic",
                               generation_reason="only_decoder_cap_changed_not_native_model_reproduction")
                return api.score_candidates(runtime,chunk,rows)
            original = score(items)
            native = [dict(item,row=row) for item,row in zip(items,original)]
            proposed,attempts = retrieval(logits,.2,profiles[pid],template["condition_seed"])
            mixed,fallback = unlimited.mix_candidates(native,proposed)
            rag = [dict(r) for r in original] if fallback else score(mixed)
            for method,pool in [("delepref_q3_diagnostic",original),("rag_q3_diagnostic",rag)]:
                if len(pool) != 8 or len({r["deletion_mask_sha256"] for r in pool}) != 8:
                    raise ValueError("Invalid unique K8")
                for row in pool:
                    row["method"] = method
                best = dict(min(pool,key=lambda r:r["scorer_rank"]))
                mask = unlimited.validate_plan(len(seq),.2,json.loads(best["deleted_intervals"]))
                edited = "".join(a for a,d in zip(seq,mask) if not d)
                best.update(edited_sequence=edited, edited_sequence_sha256=controls.sequence_sha(edited))
                selections.append(best);candidates.extend(pool)
            diagnostics.append(dict(protein_id=pid, fallback=fallback, original_generation=cost,
                                    retrieval_attempts=attempts, retrieved_unique=len(proposed)))
            write(out/"partial_progress.json",dict(completed_WT=len(diagnostics),expected_WT=20,
                                                   last_protein_id=pid,job_id=os.environ["SLURM_JOB_ID"]))
            print("SPAN_DIAGNOSTIC_WT_COMPLETE "+pid,flush=True)
    write(out/"q3_candidate_scores.json",candidates)
    write(out/"q3_selected_outputs.json",selections)
    write(out/"inherited_unlimited_outputs.json",inherited)
    write(out/"generation_diagnostics.json",diagnostics)
    write(out/"planning_completion.json",dict(status="DEVELOPMENT_2X2_OUTPUTS_FROZEN",WT=20,
        inherited_unlimited_outputs=40,new_q3_outputs=len(selections),new_candidates=len(candidates),
        extra_PLM_forwards=0,new_training=False,main_task_qmax=None,diagnostic_qmax=3,
        not_original_native_decoder_reproduction=True,seconds=time.monotonic()-start,
        job_id=os.environ["SLURM_JOB_ID"],files={p.name:sha(p) for p in out.glob("*.json")}))
    print("SPAN_DIAGNOSTIC_PLANNING_COMPLETE",flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True)
    main(parser.parse_args().root)
