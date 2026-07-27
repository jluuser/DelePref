#!/usr/bin/env python3
"""Generate strict-budget delepref plans with seeded perturbation plus DP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import replace
from typing import Dict, List, Mapping, Optional, Sequence


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if os.path.join(REPO_ROOT, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from train_student_pilot import as_dict, build_delepref, load_config


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate exact-budget delepref deletion plans")
    parser.add_argument("--config", default="configs/deletion_preference_student.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_csv", required=True, help="CSV with protein_id, wildtype_sequence, and beta/target_delete_fraction")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--diagnostics_json", default=None)
    parser.add_argument("--num_candidates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    return parser.parse_args(argv)


def _load_inputs(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    seen = set()
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            protein_id = str(row.get("protein_id") or "").strip()
            sequence = str(row.get("wildtype_sequence") or "").strip()
            beta_raw = row.get("beta") or row.get("target_delete_fraction")
            if not protein_id or not sequence or beta_raw in (None, ""):
                raise ValueError("input row {} is missing protein_id, wildtype_sequence, or beta".format(row_number))
            beta = float(beta_raw)
            if not math.isfinite(beta) or beta < 0.0 or beta > 1.0:
                raise ValueError("input row {} beta must be finite and in [0,1]".format(row_number))
            key = (protein_id, hashlib.sha256(sequence.encode("utf-8")).hexdigest(), beta)
            if key in seen:
                raise ValueError(
                    "input row {} duplicates protein_id + sequence hash + beta and would collide proposal IDs".format(
                        row_number
                    )
                )
            seen.add(key)
            rows.append({"protein_id": protein_id, "wildtype_sequence": sequence, "beta": beta})
    if not rows:
        raise ValueError("input CSV contains no proteins")
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    import torch

    from delepref.preference_core import (
        candidate_from_proposal,
        make_student_proposal_mask_row,
        write_student_proposal_masks,
    )
    from delepref.student_model import (
        generate_strict_budget_plans,
        load_student_checkpoint,
        proposal_from_decoded_plan,
        seed_everything,
        validate_strict_plan,
    )

    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("delepref ESMC generation requires CUDA visible in the current runtime")
    config = load_config(args.config)
    policy_config = as_dict(config.get("policy"))
    decoder_config = as_dict(config.get("decoder"))
    seed = int(args.seed if args.seed is not None else decoder_config.get("seed", 13))
    requested_k = int(
        args.num_candidates if args.num_candidates is not None else decoder_config.get("num_candidates", 8)
    )
    if requested_k < 1:
        raise ValueError("num_candidates must be positive")
    seed_everything(seed)
    device = torch.device("cuda")
    model = build_delepref(config, device)
    payload = load_student_checkpoint(args.checkpoint, model=model, load_optimizer=False)
    model.eval()
    inputs = _load_inputs(args.input_csv)
    encoded = model.encode_unique_wildtypes(
        [(row["protein_id"], row["wildtype_sequence"]) for row in inputs]
    )
    output_rows: List[Mapping[str, object]] = []
    diagnostics: List[Mapping[str, object]] = []
    qmax = int(policy_config.get("qmax", 3))
    if qmax != 3:
        raise ValueError("delepref final generation requires Qmax=3")
    min_span = int(policy_config.get("min_span_len", 1))
    max_fraction = float(policy_config.get("max_span_len_fraction", 0.5))
    with torch.no_grad():
        for input_index, row in enumerate(inputs):
            protein_id = str(row["protein_id"])
            sequence = str(row["wildtype_sequence"])
            beta = float(row["beta"])
            embedding = encoded.embeddings[encoded.key_for(protein_id, sequence)]
            policy_output = model.propose_from_embeddings([embedding], [beta])[0]
            max_span = max(min_span, min(len(sequence) - 1, int(math.floor(max_fraction * len(sequence)))))
            result = generate_strict_budget_plans(
                policy_output.residue_logits,
                beta,
                num_candidates=requested_k,
                qmax=qmax,
                min_span_len=min_span,
                max_span_len=max_span,
                seed=seed + input_index,
                perturbation_scale=float(decoder_config.get("perturbation_scale", 1.0)),
                max_attempts=requested_k * int(decoder_config.get("max_attempts_multiplier", 16)),
            )
            proposals = []
            for rank, decoded in enumerate(result.plans, start=1):
                validate_strict_plan(
                    len(sequence),
                    beta,
                    decoded.deleted_intervals,
                    qmax=qmax,
                    min_span_len=min_span,
                    max_span_len=max_span,
                )
                proposals.append(
                    proposal_from_decoded_plan(
                        protein_id=protein_id,
                        wildtype_sequence=sequence,
                        beta=beta,
                        decoded=decoded,
                        rank=rank,
                        generation_reason=result.reason,
                    )
                )
            if proposals:
                candidates = [candidate_from_proposal(proposal) for proposal in proposals]
                scorer_output = model.score_plans_from_embeddings(candidates, encoded)
                scorer_scores = scorer_output["preference_score"].detach().float().cpu().tolist()
                reranked = sorted(
                    zip(proposals, scorer_scores),
                    key=lambda item: (-float(item[1]), item[0].deleted_intervals),
                )
                for rerank, (proposal, scorer_score) in enumerate(reranked, start=1):
                    metadata = dict(proposal.metadata or {})
                    metadata.update(
                        {
                            "raw_policy_score": proposal.proposal_score,
                            "preference_scorer_score": float(scorer_score),
                            "requested_k": requested_k,
                            "returned_k": len(proposals),
                            "generation_attempts": result.attempts,
                            "duplicate_decodes": result.duplicate_count,
                            "exhaustion_reason": result.reason,
                            "online_mutant_evaluation": False,
                            "student_confidence_available": False,
                            "student_confidence_used_for_ranking": False,
                        }
                    )
                    reranked_proposal = replace(
                        proposal,
                        proposal_score=float(scorer_score),
                        proposal_rank=rerank,
                        metadata=metadata,
                    )
                    output_rows.append(
                        make_student_proposal_mask_row(
                            reranked_proposal,
                            proposal_model=model.policy.policy_model_name,
                            proposal_checkpoint=os.path.abspath(args.checkpoint),
                            proposal_mode="delepref_strict_dp_reranked",
                            metadata={"checkpoint_step": int(payload.get("step", 0))},
                        )
                    )
            diagnostics.append(
                {
                    "protein_id": protein_id,
                    "sequence_length": len(sequence),
                    "beta": beta,
                    "requested_k": requested_k,
                    "returned_k": len(result.plans),
                    "attempts": result.attempts,
                    "duplicate_decodes": result.duplicate_count,
                    "reason": result.reason,
                    "all_returned_plans_strict": True,
                }
            )
    write_student_proposal_masks(args.output_csv, output_rows)
    diagnostics_path = args.diagnostics_json or os.path.splitext(args.output_csv)[0] + "_diagnostics.json"
    with open(diagnostics_path, "w") as handle:
        json.dump(
            {
                "status": "ok",
                "requested_k": requested_k,
                "input_count": len(inputs),
                "unique_wildtypes_encoded": encoded.unique_count,
                "proposal_rows": len(output_rows),
                "diagnostics": diagnostics,
                "claim_boundary": "The method ranks candidate deletion plans by teacher-derived deletion perturbation preference and optional auxiliary risk evidence.",
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print(json.dumps({"status": "ok", "output_csv": args.output_csv, "diagnostics_json": diagnostics_path}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
