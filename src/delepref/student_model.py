"""Strict-budget deletion-preference student delepref.

The delepref student encodes each unique wild-type once, then reuses the frozen
residue embeddings in both the boundary-aware preference scorer and the
budget-conditioned proposal policy.  Deletion mutants are never encoded.

The model ranks deletion plans by teacher-derived deletion perturbation
preference.  It does not establish that a deleted region has no function.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import os
import random
import weakref
from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .preference_core import (
    COORDINATE_SYSTEM,
    DeletionCandidate,
    DeletionProposal,
    PreferencePair,
    TinyProteinEncoder,
    TinyProteinTokenizer,
    canonical_deleted_intervals,
    deleted_length,
    deletion_mask_from_intervals,
    parse_deleted_intervals,
    residue_embeddings_from_backend,
)


STUDENT_VERSION = "strict_budget_delepref_v1"
FINAL_QMAX = 3


@dataclass(frozen=True, order=True)
class WildTypeKey:
    """Batch deduplication key required by the delepref interface."""

    protein_id: str
    sequence_sha256: str


@dataclass
class EncodedWildTypes:
    """Frozen residue embeddings indexed by protein id plus sequence hash."""

    embeddings: Dict[WildTypeKey, torch.Tensor]
    sequences: Dict[WildTypeKey, str]
    input_count: int

    @property
    def unique_count(self) -> int:
        return len(self.embeddings)

    def key_for(self, protein_id: str, sequence: str) -> WildTypeKey:
        key = wildtype_key(protein_id, sequence)
        stored = self.sequences.get(key)
        if stored is None:
            raise KeyError("wild type was not encoded: {}".format(key))
        if stored != sequence:
            raise RuntimeError("sequence hash collision for protein {}".format(protein_id))
        return key

    def for_candidate(self, candidate: DeletionCandidate) -> torch.Tensor:
        return self.embeddings[self.key_for(candidate.protein_id, candidate.wildtype_sequence)]


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(str(sequence).encode("utf-8")).hexdigest()


def wildtype_key(protein_id: str, sequence: str) -> WildTypeKey:
    protein = str(protein_id).strip()
    sequence_text = str(sequence).strip()
    if not protein:
        raise ValueError("protein_id must be non-empty")
    if not sequence_text:
        raise ValueError("wildtype_sequence must be non-empty")
    return WildTypeKey(protein_id=protein, sequence_sha256=sequence_sha256(sequence_text))


def strict_delete_count(sequence_length: int, beta: float) -> int:
    """Return D=max(1,min(L-1,round(beta*L))) for a deletable sequence."""

    length = int(sequence_length)
    if length < 2:
        raise ValueError("strict deletion requires L >= 2 so at least one residue is retained")
    beta_value = float(beta)
    if not math.isfinite(beta_value):
        raise ValueError("beta must be finite")
    if beta_value < 0.0 or beta_value > 1.0:
        raise ValueError("beta must be in [0, 1]")
    return max(1, min(length - 1, int(round(beta_value * float(length)))))


def _normalize_wildtype_inputs(
    protein_ids_or_records: Sequence[object],
    wildtype_sequences: Optional[Sequence[str]],
) -> List[Tuple[str, str]]:
    if wildtype_sequences is not None:
        if len(protein_ids_or_records) != len(wildtype_sequences):
            raise ValueError("protein_ids and wildtype_sequences must have equal length")
        return [(str(protein_id), str(sequence)) for protein_id, sequence in zip(protein_ids_or_records, wildtype_sequences)]

    records: List[Tuple[str, str]] = []
    for item in protein_ids_or_records:
        if isinstance(item, DeletionCandidate):
            records.append((item.protein_id, item.wildtype_sequence))
        elif isinstance(item, Mapping):
            records.append((str(item.get("protein_id") or ""), str(item.get("wildtype_sequence") or "")))
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            records.append((str(item[0]), str(item[1])))
        else:
            raise TypeError("wild-type records must be candidates, mappings, or (protein_id, sequence) pairs")
    return records


class FrozenWTEncoder(nn.Module):
    """One frozen protein encoder shared by all delepref trainable heads."""

    def __init__(
        self,
        encoder: nn.Module,
        tokenizer: object,
        *,
        hidden_size: int,
        max_length: int = 1022,
    ):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.hidden_size = int(hidden_size)
        self.max_length = int(max_length)
        self.forward_batches = 0
        self.unique_wildtypes_encoded = 0
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    @property
    def backend(self) -> nn.Module:
        return self.encoder

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self.encoder.eval()
        return self

    def _device(self) -> torch.device:
        try:
            return next(self.encoder.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def encode_unique_wildtypes(
        self,
        protein_ids_or_records: Sequence[object],
        wildtype_sequences: Optional[Sequence[str]] = None,
        *,
        microbatch_size: Optional[int] = None,
    ) -> EncodedWildTypes:
        records = _normalize_wildtype_inputs(protein_ids_or_records, wildtype_sequences)
        unique: MutableMapping[WildTypeKey, str] = {}
        for protein_id, sequence in records:
            key = wildtype_key(protein_id, sequence)
            previous = unique.get(key)
            if previous is not None and previous != sequence:
                raise RuntimeError("sequence hash collision for protein {}".format(protein_id))
            if len(sequence) > self.max_length:
                raise ValueError(
                    "wild type {} length {} exceeds max_length {}; delepref never silently truncates coordinates".format(
                        protein_id, len(sequence), self.max_length
                    )
                )
            unique[key] = sequence
        if not unique:
            return EncodedWildTypes(embeddings={}, sequences={}, input_count=0)
        ordered = sorted(unique.items(), key=lambda item: item[0])
        batch_size = len(ordered) if microbatch_size is None else int(microbatch_size)
        if batch_size < 1:
            raise ValueError("encoder microbatch_size must be >= 1")
        residue_batches: List[torch.Tensor] = []
        for start in range(0, len(ordered), batch_size):
            chunk = ordered[start : start + batch_size]
            residue_batches.extend(
                residue_embeddings_from_backend(
                    self.encoder,
                    [sequence for _key, sequence in chunk],
                    self._device(),
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    freeze=True,
                )
            )
            self.forward_batches += 1
        self.unique_wildtypes_encoded += len(ordered)
        embeddings = {key: embedding.detach() for (key, _sequence), embedding in zip(ordered, residue_batches)}
        return EncodedWildTypes(
            embeddings=embeddings,
            sequences=dict(ordered),
            input_count=len(records),
        )

    @classmethod
    def tiny_for_tests(cls, *, hidden_size: int = 32, max_length: int = 1022) -> "FrozenWTEncoder":
        return cls(
            TinyProteinEncoder(hidden_size=hidden_size),
            TinyProteinTokenizer(),
            hidden_size=hidden_size,
            max_length=max_length,
        )


class _SharedEncoderHead(nn.Module):
    """Head base class that exposes, but does not re-register, the encoder."""

    def __init__(self, encoder: FrozenWTEncoder):
        super().__init__()
        object.__setattr__(self, "_encoder_ref", weakref.ref(encoder))

    @property
    def encoder(self) -> FrozenWTEncoder:
        encoder = self._encoder_ref()
        if encoder is None:
            raise RuntimeError("shared encoder has been released")
        return encoder


def _candidate_intervals(candidate: DeletionCandidate, length: int, *, qmax: Optional[int] = None) -> Tuple[Tuple[int, int], ...]:
    intervals = parse_deleted_intervals(candidate.deleted_intervals, length)
    if qmax is not None and len(intervals) > int(qmax):
        raise ValueError("candidate {} has {} spans > Qmax {}".format(candidate.plan_id, len(intervals), qmax))
    actual_deleted = deleted_length(intervals)
    if actual_deleted != int(candidate.deleted_length):
        raise ValueError(
            "candidate {} deleted_length {} != interval total {}".format(
                candidate.plan_id, candidate.deleted_length, actual_deleted
            )
        )
    if actual_deleted >= int(length):
        raise ValueError("candidate {} deletes the entire wild type".format(candidate.plan_id))
    return intervals


class BoundaryAwarePreferenceScorer(_SharedEncoderHead):
    """Deletion-plan scorer using WT pools, span boundaries, and anchors."""

    def __init__(
        self,
        encoder: FrozenWTEncoder,
        *,
        hidden_size: int,
        head_hidden_dim: int = 256,
        span_hidden_dim: int = 128,
        qmax: int = FINAL_QMAX,
        dropout: float = 0.1,
        scorer_model_name: str = "esmc600m_boundary_aware_preference_scorer_delepref",
    ):
        super().__init__(encoder)
        self.hidden_size = int(hidden_size)
        self.qmax = max(1, int(qmax))
        self.scorer_model_name = str(scorer_model_name)
        span_input_dim = self.hidden_size * 5 + 8
        self.span_phi = nn.Sequential(
            nn.LayerNorm(span_input_dim),
            nn.Linear(span_input_dim, int(span_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(span_hidden_dim), int(span_hidden_dim)),
            nn.GELU(),
        )
        plan_input_dim = self.hidden_size * 3 + int(span_hidden_dim) + 4
        hidden = max(16, int(head_hidden_dim))
        self.plan_head = nn.Sequential(
            nn.LayerNorm(plan_input_dim),
            nn.Linear(plan_input_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, max(8, hidden // 2)),
            nn.GELU(),
            nn.Linear(max(8, hidden // 2), 1),
        )

    def _feature_dtype(self) -> torch.dtype:
        return next(self.span_phi.parameters()).dtype

    def _span_feature(self, residue_embeddings: torch.Tensor, start: int, end: int) -> torch.Tensor:
        length = int(residue_embeddings.shape[0])
        hidden = int(residue_embeddings.shape[1])
        zero = residue_embeddings.new_zeros(hidden)
        left = residue_embeddings[start - 1] if start > 0 else zero
        right = residue_embeddings[end + 1] if end + 1 < length else zero
        span_pool = residue_embeddings[start : end + 1].mean(dim=0)
        denom_position = float(max(1, length - 1))
        geometry = residue_embeddings.new_tensor(
            [
                float(start) / denom_position,
                float(end) / denom_position,
                float(end - start + 1) / float(length),
                (0.5 * float(start + end)) / denom_position,
            ]
        )
        left_terminal = start == 0
        right_terminal = end == length - 1
        boundary_type = residue_embeddings.new_tensor(
            [
                float(left_terminal and not right_terminal),
                float(right_terminal and not left_terminal),
                float(left_terminal and right_terminal),
                float(not left_terminal and not right_terminal),
            ]
        )
        raw = torch.cat([left, right, left - right, left * right, span_pool, geometry, boundary_type], dim=-1)
        return raw.to(dtype=self._feature_dtype())

    def score_plans_from_embeddings(
        self,
        candidates: Sequence[DeletionCandidate],
        encoded_wildtypes: EncodedWildTypes,
    ) -> Dict[str, torch.Tensor]:
        if not candidates:
            device = next(self.parameters()).device
            empty = torch.empty(0, device=device)
            return {"preference_score": empty}
        features: List[torch.Tensor] = []
        feature_dtype = self._feature_dtype()
        for candidate in candidates:
            residue_embeddings = encoded_wildtypes.for_candidate(candidate)
            length = int(residue_embeddings.shape[0])
            intervals = _candidate_intervals(candidate, length, qmax=self.qmax)
            mask = deletion_mask_from_intervals(
                length,
                intervals,
                device=residue_embeddings.device,
                dtype=residue_embeddings.dtype,
            )
            deleted_count = mask.sum()
            retained_count = (1.0 - mask).sum()
            if float(retained_count.detach().cpu()) < 1.0:
                raise ValueError("scorer requires at least one retained residue")
            deleted_pool = (residue_embeddings * mask.unsqueeze(-1)).sum(dim=0) / deleted_count
            retained_pool = (residue_embeddings * (1.0 - mask).unsqueeze(-1)).sum(dim=0) / retained_count
            global_pool = residue_embeddings.mean(dim=0)
            span_raw = torch.stack(
                [self._span_feature(residue_embeddings, start, end) for start, end in intervals], dim=0
            )
            span_encoded = self.span_phi(span_raw)
            span_aggregate = span_encoded.mean(dim=0)
            terminal_count = sum(int(start == 0) + int(end == length - 1) for start, end in intervals)
            scalars = residue_embeddings.new_tensor(
                [
                    float(len(intervals)) / float(self.qmax),
                    float(candidate.deleted_length) / float(length),
                    float(candidate.target_delete_fraction),
                    float(terminal_count) / 2.0,
                ]
            )
            features.append(
                torch.cat(
                    [
                        global_pool.to(feature_dtype),
                        deleted_pool.to(feature_dtype),
                        retained_pool.to(feature_dtype),
                        span_aggregate,
                        scalars.to(feature_dtype),
                    ],
                    dim=-1,
                )
            )
        raw = self.plan_head(torch.stack(features, dim=0)).squeeze(-1)
        return {"preference_score": raw}


@dataclass
class ProposalOutput:
    residue_logits: torch.Tensor
    raw_residue_logits: torch.Tensor
    activity_logits: torch.Tensor
    centers: torch.Tensor
    lengths_fraction: torch.Tensor
    span_prior: torch.Tensor
    beta: float


def _soft_span_prior(
    length: int,
    centers: torch.Tensor,
    lengths_fraction: torch.Tensor,
    activity_logits: torch.Tensor,
) -> torch.Tensor:
    positions = torch.linspace(0.0, 1.0, steps=int(length), device=centers.device, dtype=centers.dtype)
    if int(length) == 1:
        positions.zero_()
    prior = torch.zeros_like(positions)
    for slot in range(int(centers.numel())):
        sigma = (lengths_fraction[slot].clamp_min(1.0 / float(max(1, length))) / 2.0).clamp_min(
            1.0 / float(max(1, length))
        )
        value = activity_logits[slot].sigmoid() * torch.exp(
            -0.5 * ((positions - centers[slot]) / sigma).pow(2)
        )
        prior = torch.maximum(prior, value)
    return prior.clamp(0.0, 1.0)


class StrictBudgetProposalPolicy(_SharedEncoderHead):
    """Budget-conditioned span generator and residue refiner over shared WT embeddings."""

    policy_arch = "strict_span_refiner_delepref"

    def __init__(
        self,
        encoder: FrozenWTEncoder,
        *,
        hidden_size: int,
        budget_hidden_dim: int = 64,
        policy_hidden_dim: int = 256,
        span_hidden_dim: int = 256,
        qmax: int = FINAL_QMAX,
        min_span_len: int = 1,
        max_span_len_fraction: float = 0.5,
        dropout: float = 0.1,
        policy_model_name: str = "esmc600m_strict_span_refiner_policy_delepref",
    ):
        super().__init__(encoder)
        self.hidden_size = int(hidden_size)
        self.qmax = max(1, int(qmax))
        self.min_span_len = max(1, int(min_span_len))
        self.max_span_len_fraction = max(1.0e-6, min(1.0, float(max_span_len_fraction)))
        self.policy_model_name = str(policy_model_name)
        budget_hidden = int(budget_hidden_dim)
        self.budget_encoder = nn.Sequential(
            nn.Linear(1, budget_hidden),
            nn.GELU(),
            nn.Linear(budget_hidden, budget_hidden),
            nn.GELU(),
        )
        self.span_generator = nn.Sequential(
            nn.LayerNorm(self.hidden_size + budget_hidden + 4),
            nn.Linear(self.hidden_size + budget_hidden + 4, int(span_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(span_hidden_dim), self.qmax * 3),
        )
        self.mask_refiner = nn.Sequential(
            nn.LayerNorm(self.hidden_size + budget_hidden + 5),
            nn.Linear(self.hidden_size + budget_hidden + 5, int(policy_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(policy_hidden_dim), max(8, int(policy_hidden_dim) // 2)),
            nn.GELU(),
            nn.Linear(max(8, int(policy_hidden_dim) // 2), 1),
        )

    def _feature_dtype(self) -> torch.dtype:
        return next(self.budget_encoder.parameters()).dtype

    def _one_from_embedding(self, residue_embeddings: torch.Tensor, beta: float) -> ProposalOutput:
        length = int(residue_embeddings.shape[0])
        strict_delete_count(length, beta)
        dtype = self._feature_dtype()
        embeddings = residue_embeddings.to(dtype=dtype)
        device = embeddings.device
        beta_tensor = torch.tensor([[float(beta)]], device=device, dtype=dtype)
        budget_features = self.budget_encoder(beta_tensor).view(-1)
        positions = torch.linspace(0.0, 1.0, steps=length, device=device, dtype=dtype)
        terminal_distance = torch.minimum(positions, 1.0 - positions)
        global_pool = embeddings.mean(dim=0)
        summary = torch.tensor(
            [
                float(length) / float(max(1, self.encoder.max_length)),
                float(beta),
                float(self.qmax) / 8.0,
                float(self.min_span_len) / float(length),
            ],
            device=device,
            dtype=dtype,
        )
        span_raw = self.span_generator(torch.cat([global_pool, budget_features, summary], dim=-1)).view(
            self.qmax, 3
        )
        activity_logits = span_raw[:, 0]
        centers = span_raw[:, 1].sigmoid()
        min_fraction = min(1.0, float(self.min_span_len) / float(length))
        max_fraction = max(min_fraction, min(self.max_span_len_fraction, float(length - 1) / float(length)))
        lengths_fraction = min_fraction + span_raw[:, 2].sigmoid() * (max_fraction - min_fraction)
        prior = _soft_span_prior(length, centers, lengths_fraction, activity_logits)
        scalar_features = torch.stack(
            [
                positions,
                terminal_distance,
                prior,
                torch.full_like(positions, float(beta)),
                torch.full_like(positions, float(length) / float(max(1, self.encoder.max_length))),
            ],
            dim=-1,
        )
        refiner_input = torch.cat(
            [embeddings, budget_features.unsqueeze(0).expand(length, -1), scalar_features], dim=-1
        )
        raw_logits = self.mask_refiner(refiner_input).squeeze(-1)
        residue_logits = raw_logits + 0.35 * torch.logit(prior.clamp(1.0e-4, 1.0 - 1.0e-4))
        return ProposalOutput(
            residue_logits=residue_logits,
            raw_residue_logits=raw_logits,
            activity_logits=activity_logits,
            centers=centers,
            lengths_fraction=lengths_fraction,
            span_prior=prior,
            beta=float(beta),
        )

    def propose_from_embeddings(
        self,
        residue_embeddings: Sequence[torch.Tensor],
        betas: Sequence[float],
    ) -> List[ProposalOutput]:
        if len(residue_embeddings) != len(betas):
            raise ValueError("residue_embeddings and betas must have equal length")
        return [self._one_from_embedding(embedding, beta) for embedding, beta in zip(residue_embeddings, betas)]


def _normalized_preference(value: object) -> str:
    text = str(value or "").strip().lower()
    return {
        "a": "a_better",
        "prefer_a": "a_better",
        "b": "b_better",
        "prefer_b": "b_better",
        "equal": "tie",
    }.get(text, text)


def validate_strict_preference_pair(pair: PreferencePair, *, beta_tolerance: float = 1.0e-8) -> None:
    """Fail fast unless ranking compares the same protein and exact budget."""

    if pair.protein_id != pair.a.protein_id or pair.protein_id != pair.b.protein_id:
        raise ValueError("rank pair {} crosses protein_id".format(pair.pair_id))
    if pair.a.protein_id != pair.b.protein_id:
        raise ValueError("rank pair {} crosses protein_id".format(pair.pair_id))
    if pair.a.wildtype_sequence != pair.b.wildtype_sequence:
        raise ValueError("rank pair {} uses different wild-type sequences".format(pair.pair_id))
    beta_a = float(pair.a.target_delete_fraction)
    beta_b = float(pair.b.target_delete_fraction)
    if not math.isfinite(beta_a) or not math.isfinite(beta_b) or abs(beta_a - beta_b) > float(beta_tolerance):
        raise ValueError("rank pair {} crosses beta: {} vs {}".format(pair.pair_id, beta_a, beta_b))
    if int(pair.a.deleted_length) != int(pair.b.deleted_length):
        raise ValueError("rank pair {} crosses deleted_length".format(pair.pair_id))
    length = len(pair.a.wildtype_sequence)
    expected = strict_delete_count(length, beta_a)
    for side, candidate in (("A", pair.a), ("B", pair.b)):
        intervals = _candidate_intervals(candidate, length)
        actual = deleted_length(intervals)
        if actual != expected:
            raise ValueError(
                "rank pair {} side {} deleted_length {} != strict D {}".format(pair.pair_id, side, actual, expected)
            )


def _weighted_mean(
    values: Sequence[torch.Tensor],
    weights: Sequence[torch.Tensor],
    reference: torch.Tensor,
) -> torch.Tensor:
    if not values:
        return reference * 0.0
    stacked = torch.stack(list(values))
    weight_tensor = torch.stack(list(weights)).to(device=stacked.device, dtype=stacked.dtype)
    # Normalize by the valid item count, not by sum(weights).  Pair-quality
    # weights must change the absolute loss/gradient even for batch_size=1.
    return (stacked * weight_tensor).sum() / float(len(values))


def scorer_pairwise_objective_from_embeddings(
    scorer: BoundaryAwarePreferenceScorer,
    pairs: Sequence[PreferencePair],
    encoded_wildtypes: EncodedWildTypes,
    *,
    tie_weight_scale: float = 0.1,
) -> Dict[str, object]:
    for pair in pairs:
        validate_strict_preference_pair(pair)
    candidates: List[DeletionCandidate] = []
    for pair in pairs:
        candidates.extend([pair.a, pair.b])
    output = scorer.score_plans_from_embeddings(candidates, encoded_wildtypes)
    score_a = output["preference_score"][0::2]
    score_b = output["preference_score"][1::2]
    terms: List[torch.Tensor] = []
    weights: List[torch.Tensor] = []
    num_rank = 0
    num_tie = 0
    for index, pair in enumerate(pairs):
        preference = _normalized_preference(pair.preference)
        if preference == "invalid" or not pair.is_valid_pair or float(pair.loss_weight) <= 0.0:
            continue
        delta = score_a[index] - score_b[index]
        quality_weight = delta.new_tensor(float(pair.loss_weight))
        if preference == "a_better":
            terms.append(F.softplus(-delta))
            weights.append(quality_weight)
            num_rank += 1
        elif preference == "b_better":
            terms.append(F.softplus(delta))
            weights.append(quality_weight)
            num_rank += 1
        elif preference == "tie":
            terms.append(delta.pow(2) * float(tie_weight_scale))
            weights.append(quality_weight)
            num_tie += 1
        else:
            raise ValueError("unknown preference {!r}".format(pair.preference))
    reference = score_a.sum() + score_b.sum()
    return {
        "loss": _weighted_mean(terms, weights, reference),
        "score_a": score_a,
        "score_b": score_b,
        "num_rank_used": num_rank,
        "num_tie_used": num_tie,
    }


def _matched_slots(output: ProposalOutput, intervals: Sequence[Tuple[int, int]], length: int) -> List[int]:
    count = len(intervals)
    if count > int(output.centers.numel()):
        raise ValueError("candidate span count exceeds proposal Qmax")
    target_centers = [0.5 * float(start + end) / float(max(1, length - 1)) for start, end in intervals]
    target_lengths = [float(end - start + 1) / float(length) for start, end in intervals]
    predicted_centers = output.centers.detach().float().cpu().tolist()
    predicted_lengths = output.lengths_fraction.detach().float().cpu().tolist()
    best_cost: Optional[float] = None
    best_slots: Optional[Tuple[int, ...]] = None
    for slots in itertools.permutations(range(len(predicted_centers)), count):
        cost = sum(
            (predicted_centers[slot] - target_centers[index]) ** 2
            + (predicted_lengths[slot] - target_lengths[index]) ** 2
            for index, slot in enumerate(slots)
        )
        if best_cost is None or cost < best_cost - 1.0e-15 or (
            abs(cost - best_cost) <= 1.0e-15 and slots < (best_slots or slots)
        ):
            best_cost = cost
            best_slots = slots
    return list(best_slots or ())


def policy_condition_identity(candidate: DeletionCandidate) -> Tuple[object, ...]:
    """Identity of one budget-conditioned WT policy forward."""

    return (
        candidate.protein_id,
        sequence_sha256(candidate.wildtype_sequence),
        float(candidate.target_delete_fraction),
    )


def policy_candidate_identity(candidate: DeletionCandidate) -> Tuple[object, ...]:
    """Plan identity used to deduplicate policy supervision.

    Perturbation/plan IDs are intentionally excluded: aliases of the same
    exact deletion mask must not create repeated imitation targets.
    """

    intervals = _candidate_intervals(candidate, len(candidate.wildtype_sequence))
    return policy_condition_identity(candidate) + (intervals,)


def joint_objective_normalization_counts(pairs: Sequence[PreferencePair]) -> Dict[str, int]:
    """Return additive denominators for batch/accumulation-invariant training.

    ``_weighted_mean`` deliberately divides by item count rather than the sum
    of quality weights.  These counts let the DDP trainer reconstruct exactly
    the same component means across rank and gradient-accumulation boundaries.
    Condition-grouped batching guarantees that imitation/budget identities do
    not straddle microbatches; the trainer checks that invariant explicitly.
    """

    scorer_items = 0
    rank_items = 0
    tie_items = 0
    imitation = {}
    conditions = set()
    for pair in pairs:
        validate_strict_preference_pair(pair)
        condition_a = policy_condition_identity(pair.a)
        if condition_a != policy_condition_identity(pair.b):
            raise ValueError("strict pair does not map to one policy condition")
        conditions.add(condition_a)
        preference = _normalized_preference(pair.preference)
        if preference == "invalid" or not pair.is_valid_pair or float(pair.loss_weight) <= 0.0:
            continue
        scorer_items += 1
        preferred = None
        if preference == "a_better":
            rank_items += 1
            preferred = pair.a
        elif preference == "b_better":
            rank_items += 1
            preferred = pair.b
        elif preference == "tie":
            tie_items += 1
        else:
            raise ValueError("unknown preference {!r}".format(pair.preference))
        if preferred is not None and bool(pair.policy_positive_eligible):
            identity = policy_candidate_identity(preferred)
            imitation[identity] = max(float(pair.loss_weight), imitation.get(identity, -math.inf))
    return {
        "scorer": scorer_items,
        "rank": rank_items,
        "tie": tie_items,
        "imitation": len(imitation),
        "budget": len(conditions),
        "structure": len(imitation),
    }


def normalized_selected_residue_energy(
    policy: StrictBudgetProposalPolicy,
    candidate: DeletionCandidate,
    output: ProposalOutput,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(sum selected z_i / D, sum selected z_i)`` for one plan."""

    logits = output.residue_logits
    length = int(logits.numel())
    intervals = _candidate_intervals(candidate, length, qmax=policy.qmax)
    expected = strict_delete_count(length, output.beta)
    if int(candidate.deleted_length) != expected:
        raise ValueError("candidate deleted_length does not equal strict budget D")
    if not math.isclose(
        float(candidate.target_delete_fraction), float(output.beta), rel_tol=0.0, abs_tol=1.0e-8
    ):
        raise ValueError("candidate beta does not match policy output beta")
    mask = deletion_mask_from_intervals(length, intervals, device=logits.device, dtype=logits.dtype)
    selected_sum = (logits * mask).sum()
    return selected_sum / float(expected), selected_sum


def proposal_condition_budget_loss(output: ProposalOutput) -> torch.Tensor:
    """Budget loss for one unique ``(protein, sequence_sha256, beta)`` condition."""

    logits = output.residue_logits
    length = int(logits.numel())
    expected = strict_delete_count(length, output.beta)
    expected_fraction = logits.sigmoid().sum() / float(length)
    target_fraction = logits.new_tensor(float(expected) / float(length))
    return (expected_fraction - target_fraction).pow(2)


def proposal_candidate_losses(
    policy: StrictBudgetProposalPolicy,
    candidate: DeletionCandidate,
    output: ProposalOutput,
) -> Dict[str, torch.Tensor]:
    """One candidate's deduplicated positive-supervision components."""

    logits = output.residue_logits
    length = int(logits.numel())
    intervals = _candidate_intervals(candidate, length, qmax=policy.qmax)
    expected = strict_delete_count(length, output.beta)
    if int(candidate.deleted_length) != expected:
        raise ValueError("candidate deleted_length does not equal strict budget D")
    mask = deletion_mask_from_intervals(length, intervals, device=logits.device, dtype=logits.dtype)
    mask_bce = F.binary_cross_entropy_with_logits(logits, mask, reduction="sum") / float(length)
    slots = _matched_slots(output, intervals, length)
    activity_targets = torch.zeros_like(output.activity_logits)
    if slots:
        activity_targets[torch.tensor(slots, device=activity_targets.device, dtype=torch.long)] = 1.0
    activity_bce = F.binary_cross_entropy_with_logits(
        output.activity_logits, activity_targets, reduction="sum"
    ) / float(policy.qmax)
    if slots:
        center_targets = logits.new_tensor(
            [0.5 * float(start + end) / float(max(1, length - 1)) for start, end in intervals]
        )
        length_targets = logits.new_tensor([float(end - start + 1) / float(length) for start, end in intervals])
        slot_index = torch.tensor(slots, device=logits.device, dtype=torch.long)
        center_loss = F.smooth_l1_loss(output.centers[slot_index], center_targets, reduction="sum") / float(len(slots))
        length_loss = F.smooth_l1_loss(
            output.lengths_fraction[slot_index], length_targets, reduction="sum"
        ) / float(len(slots))
    else:
        center_loss = logits.sum() * 0.0
        length_loss = logits.sum() * 0.0
    predicted_activity_fraction = output.activity_logits.sigmoid().sum() / float(policy.qmax)
    target_activity_fraction = logits.new_tensor(float(len(intervals)) / float(policy.qmax))
    structure_aux_loss = F.smooth_l1_loss(
        predicted_activity_fraction, target_activity_fraction, reduction="mean"
    )
    imitation_loss = mask_bce + activity_bce + center_loss + length_loss
    normalized_energy, selected_energy_sum = normalized_selected_residue_energy(policy, candidate, output)
    return {
        "normalized_energy": normalized_energy,
        "selected_energy_sum": selected_energy_sum,
        "imitation_loss": imitation_loss,
        "mask_bce": mask_bce,
        "activity_bce": activity_bce,
        "center_loss": center_loss,
        "length_loss": length_loss,
        "budget_aux_loss": proposal_condition_budget_loss(output),
        "structure_aux_loss": structure_aux_loss,
    }


def proposal_pairwise_objective_from_embeddings(
    policy: StrictBudgetProposalPolicy,
    pairs: Sequence[PreferencePair],
    encoded_wildtypes: EncodedWildTypes,
    *,
    rank_weight: float = 1.0,
    imitation_weight: float = 0.2,
    mask_weight: float = 1.0,
    activity_weight: float = 0.25,
    center_weight: float = 0.25,
    length_weight: float = 0.25,
    budget_aux_weight: float = 0.1,
    structure_aux_weight: float = 0.05,
    tie_weight_scale: float = 0.1,
    margin: float = 0.0,
) -> Dict[str, object]:
    """Strict same-protein/same-count proposal ranking and imitation objective."""

    for pair in pairs:
        validate_strict_preference_pair(pair)
    reference = next(policy.parameters()).sum()
    if not pairs:
        zero = reference * 0.0
        return {
            "loss": zero,
            "rank_loss": zero,
            "tie_loss": zero,
            "imitation_loss": zero,
            "mask_bce": zero,
            "activity_bce": zero,
            "center_loss": zero,
            "length_loss": zero,
            "budget_aux_loss": zero,
            "structure_aux_loss": zero,
            "num_rank_used": 0,
            "num_tie_used": 0,
            "num_imitation_used": 0,
            "num_imitation_references": 0,
            "num_imitation_duplicates_removed": 0,
            "num_span_count_supervision_used": 0,
            "num_budget_conditions": 0,
            "num_budget_condition_references": 0,
            "num_budget_duplicates_removed": 0,
            "num_candidate_energies": 0,
            "num_negative_candidate_references": 0,
            "num_negative_supervision_used": 0,
            "num_aux_candidates": 0,
            "num_matched_spans": 0,
            "num_mask_residues": 0,
        }

    output_cache: Dict[Tuple[object, ...], ProposalOutput] = {}
    energy_cache: Dict[Tuple[object, ...], Tuple[torch.Tensor, torch.Tensor]] = {}

    def output_for(candidate: DeletionCandidate) -> ProposalOutput:
        key = policy_condition_identity(candidate)
        cached = output_cache.get(key)
        if cached is not None:
            return cached
        wt_key = encoded_wildtypes.key_for(candidate.protein_id, candidate.wildtype_sequence)
        beta = float(candidate.target_delete_fraction)
        output = policy.propose_from_embeddings([encoded_wildtypes.embeddings[wt_key]], [beta])[0]
        output_cache[key] = output
        return output

    def energy_for(candidate: DeletionCandidate) -> Tuple[torch.Tensor, torch.Tensor]:
        key = policy_candidate_identity(candidate)
        cached = energy_cache.get(key)
        if cached is not None:
            return cached
        energy = normalized_selected_residue_energy(policy, candidate, output_for(candidate))
        energy_cache[key] = energy
        return energy

    rank_terms: List[torch.Tensor] = []
    rank_weights: List[torch.Tensor] = []
    tie_terms: List[torch.Tensor] = []
    tie_weights: List[torch.Tensor] = []
    imitation_targets: Dict[Tuple[object, ...], Tuple[DeletionCandidate, torch.Tensor]] = {}
    condition_references = Counter()
    negative_candidate_references = 0
    imitation_references = 0
    num_rank = 0
    num_tie = 0
    num_imitation = 0
    num_matched_spans = 0
    num_mask_residues = 0

    for pair in pairs:
        condition_a = policy_condition_identity(pair.a)
        condition_b = policy_condition_identity(pair.b)
        if condition_a != condition_b:
            raise ValueError("strict pair does not map to one policy condition")
        output_for(pair.a)
        condition_references[condition_a] += 1
        preference = _normalized_preference(pair.preference)
        if preference == "invalid" or not pair.is_valid_pair or float(pair.loss_weight) <= 0.0:
            continue
        energy_a, _sum_a = energy_for(pair.a)
        energy_b, _sum_b = energy_for(pair.b)
        quality_weight = energy_a.new_tensor(float(pair.loss_weight))
        if preference == "a_better":
            rank_terms.append(F.softplus(-(energy_a - energy_b - float(margin))))
            rank_weights.append(quality_weight)
            preferred_candidate = pair.a
            negative_candidate_references += 1
            num_rank += 1
        elif preference == "b_better":
            rank_terms.append(F.softplus(-(energy_b - energy_a - float(margin))))
            rank_weights.append(quality_weight)
            preferred_candidate = pair.b
            negative_candidate_references += 1
            num_rank += 1
        elif preference == "tie":
            tie_terms.append((energy_a - energy_b).pow(2) * float(tie_weight_scale))
            tie_weights.append(quality_weight)
            num_tie += 1
            preferred_candidate = None
        else:
            raise ValueError("unknown preference {!r}".format(pair.preference))
        if preferred_candidate is not None and bool(pair.policy_positive_eligible):
            imitation_references += 1
            key = policy_candidate_identity(preferred_candidate)
            previous = imitation_targets.get(key)
            if previous is None or float(quality_weight.detach().cpu()) > float(previous[1].detach().cpu()):
                imitation_targets[key] = (preferred_candidate, quality_weight)

    imitation_components: Dict[str, List[torch.Tensor]] = {
        "imitation_loss": [],
        "mask_bce": [],
        "activity_bce": [],
        "center_loss": [],
        "length_loss": [],
    }
    imitation_weights: List[torch.Tensor] = []
    structure_terms: List[torch.Tensor] = []
    structure_weights: List[torch.Tensor] = []
    for key in sorted(imitation_targets):
        preferred_candidate, quality_weight = imitation_targets[key]
        preferred = proposal_candidate_losses(policy, preferred_candidate, output_for(preferred_candidate))
        weighted_imitation = (
            float(mask_weight) * preferred["mask_bce"]
            + float(activity_weight) * preferred["activity_bce"]
            + float(center_weight) * preferred["center_loss"]
            + float(length_weight) * preferred["length_loss"]
        )
        imitation_components["imitation_loss"].append(weighted_imitation)
        for name in ("mask_bce", "activity_bce", "center_loss", "length_loss"):
            imitation_components[name].append(preferred[name])
        imitation_weights.append(quality_weight)
        structure_terms.append(preferred["structure_aux_loss"])
        structure_weights.append(quality_weight)
        num_imitation += 1
        num_matched_spans += len(preferred_candidate.deleted_intervals)
        num_mask_residues += len(preferred_candidate.wildtype_sequence)

    rank_loss = _weighted_mean(rank_terms, rank_weights, reference)
    tie_loss = _weighted_mean(tie_terms, tie_weights, reference)
    imitation_loss = _weighted_mean(imitation_components["imitation_loss"], imitation_weights, reference)
    component_means = {
        name: _weighted_mean(values, imitation_weights, reference)
        for name, values in imitation_components.items()
        if name != "imitation_loss"
    }
    condition_outputs = [output_cache[key] for key in sorted(output_cache)]
    budget_aux_loss = _weighted_mean(
        [proposal_condition_budget_loss(output) for output in condition_outputs],
        [reference.new_tensor(1.0) for _output in condition_outputs],
        reference,
    )
    structure_aux_loss = _weighted_mean(structure_terms, structure_weights, reference)
    total = (
        float(rank_weight) * (rank_loss + tie_loss)
        + float(imitation_weight) * imitation_loss
        + float(budget_aux_weight) * budget_aux_loss
        + float(structure_aux_weight) * structure_aux_loss
    )
    return {
        "loss": total,
        "rank_loss": rank_loss,
        "tie_loss": tie_loss,
        "imitation_loss": imitation_loss,
        "mask_bce": component_means["mask_bce"],
        "activity_bce": component_means["activity_bce"],
        "center_loss": component_means["center_loss"],
        "length_loss": component_means["length_loss"],
        "budget_aux_loss": budget_aux_loss,
        "structure_aux_loss": structure_aux_loss,
        "num_rank_used": num_rank,
        "num_tie_used": num_tie,
        "num_imitation_used": num_imitation,
        "num_imitation_references": imitation_references,
        "num_imitation_duplicates_removed": imitation_references - num_imitation,
        "num_span_count_supervision_used": num_imitation,
        "num_budget_conditions": len(output_cache),
        "num_budget_condition_references": sum(condition_references.values()),
        "num_budget_duplicates_removed": sum(condition_references.values()) - len(output_cache),
        "num_candidate_energies": len(energy_cache),
        "num_negative_candidate_references": negative_candidate_references,
        "num_negative_supervision_used": 0,
        "num_aux_candidates": num_imitation,
        "num_matched_spans": num_matched_spans,
        "num_mask_residues": num_mask_residues,
        "num_unique_policy_forwards": len(output_cache),
    }


class DeletionPreferenceStudent(nn.Module):
    """Shared frozen WT encoder with scorer and proposal heads."""

    def __init__(
        self,
        encoder: FrozenWTEncoder,
        *,
        hidden_size: int,
        scorer_hidden_dim: int = 256,
        span_feature_dim: int = 128,
        budget_hidden_dim: int = 64,
        policy_hidden_dim: int = 256,
        policy_span_hidden_dim: int = 256,
        qmax: int = FINAL_QMAX,
        min_span_len: int = 1,
        max_span_len_fraction: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.scorer = BoundaryAwarePreferenceScorer(
            encoder,
            hidden_size=hidden_size,
            head_hidden_dim=scorer_hidden_dim,
            span_hidden_dim=span_feature_dim,
            qmax=qmax,
            dropout=dropout,
        )
        self.policy = StrictBudgetProposalPolicy(
            encoder,
            hidden_size=hidden_size,
            budget_hidden_dim=budget_hidden_dim,
            policy_hidden_dim=policy_hidden_dim,
            span_hidden_dim=policy_span_hidden_dim,
            qmax=qmax,
            min_span_len=min_span_len,
            max_span_len_fraction=max_span_len_fraction,
            dropout=dropout,
        )
        if self.scorer.encoder is not self.encoder or self.policy.encoder is not self.encoder:
            raise RuntimeError("delepref scorer and policy must share exactly one FrozenWTEncoder")

    @classmethod
    def tiny_for_tests(
        cls,
        *,
        hidden_size: int = 32,
        qmax: int = FINAL_QMAX,
        min_span_len: int = 1,
        max_span_len_fraction: float = 1.0,
        dropout: float = 0.0,
    ) -> "DeletionPreferenceStudent":
        encoder = FrozenWTEncoder.tiny_for_tests(hidden_size=hidden_size)
        return cls(
            encoder,
            hidden_size=hidden_size,
            scorer_hidden_dim=64,
            span_feature_dim=32,
            budget_hidden_dim=16,
            policy_hidden_dim=64,
            policy_span_hidden_dim=64,
            qmax=qmax,
            min_span_len=min_span_len,
            max_span_len_fraction=max_span_len_fraction,
            dropout=dropout,
        )

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self.encoder.encoder.eval()
        return self

    def encode_unique_wildtypes(
        self,
        protein_ids_or_records: Sequence[object],
        wildtype_sequences: Optional[Sequence[str]] = None,
        *,
        microbatch_size: Optional[int] = None,
    ) -> EncodedWildTypes:
        return self.encoder.encode_unique_wildtypes(
            protein_ids_or_records,
            wildtype_sequences,
            microbatch_size=microbatch_size,
        )

    def score_plans_from_embeddings(
        self,
        candidates: Sequence[DeletionCandidate],
        encoded_wildtypes: EncodedWildTypes,
    ) -> Dict[str, torch.Tensor]:
        return self.scorer.score_plans_from_embeddings(candidates, encoded_wildtypes)

    def propose_from_embeddings(
        self,
        residue_embeddings: Sequence[torch.Tensor],
        betas: Sequence[float],
    ) -> List[ProposalOutput]:
        return self.policy.propose_from_embeddings(residue_embeddings, betas)

    def joint_objective(
        self,
        pairs: Sequence[PreferencePair],
        *,
        encoder_microbatch_size: Optional[int] = None,
        scorer_tie_weight_scale: float = 0.1,
        proposal_kwargs: Optional[Mapping[str, float]] = None,
        scorer_loss_weight: float = 1.0,
        proposal_loss_weight: float = 1.0,
    ) -> Dict[str, object]:
        records: List[DeletionCandidate] = []
        for pair in pairs:
            records.extend([pair.a, pair.b])
        encoded = self.encode_unique_wildtypes(records, microbatch_size=encoder_microbatch_size)
        scorer_info = scorer_pairwise_objective_from_embeddings(
            self.scorer, pairs, encoded, tie_weight_scale=scorer_tie_weight_scale
        )
        policy_info = proposal_pairwise_objective_from_embeddings(
            self.policy, pairs, encoded, **dict(proposal_kwargs or {})
        )
        total = float(scorer_loss_weight) * scorer_info["loss"] + float(proposal_loss_weight) * policy_info["loss"]
        return {
            "loss": total,
            "scorer": scorer_info,
            "proposal": policy_info,
            "encoded_wildtypes": encoded,
        }

    def forward(
        self,
        pairs: Sequence[PreferencePair],
        *,
        encoder_microbatch_size: Optional[int] = None,
        scorer_tie_weight_scale: float = 0.1,
        proposal_kwargs: Optional[Mapping[str, float]] = None,
        scorer_loss_weight: float = 1.0,
        proposal_loss_weight: float = 1.0,
    ) -> Dict[str, object]:
        """DDP-safe entry point for the joint formal-training objective."""

        return self.joint_objective(
            pairs,
            encoder_microbatch_size=encoder_microbatch_size,
            scorer_tie_weight_scale=scorer_tie_weight_scale,
            proposal_kwargs=proposal_kwargs,
            scorer_loss_weight=scorer_loss_weight,
            proposal_loss_weight=proposal_loss_weight,
        )


class InfeasibleStrictBudgetError(ValueError):
    pass


@dataclass(frozen=True)
class StrictDecodedPlan:
    deleted_intervals: Tuple[Tuple[int, int], ...]
    deleted_length: int
    score: float
    perturbed_score: float
    seed: int


@dataclass(frozen=True)
class StrictGenerationResult:
    plans: Tuple[StrictDecodedPlan, ...]
    requested_count: int
    attempts: int
    duplicate_count: int
    reason: str


def _feasible_remaining(
    available_length: int,
    deleted_count: int,
    span_cap: int,
    min_span_len: int,
    max_span_len: int,
) -> bool:
    if deleted_count == 0:
        return True
    if available_length <= 0 or span_cap <= 0:
        return False
    for spans in range(1, int(span_cap) + 1):
        if spans * int(min_span_len) <= deleted_count <= spans * int(max_span_len):
            if deleted_count + spans - 1 <= available_length:
                return True
    return False


def validate_strict_plan(
    sequence_length: int,
    beta: float,
    intervals: Sequence[Tuple[int, int]],
    *,
    qmax: int,
    min_span_len: int,
    max_span_len: int,
) -> None:
    length = int(sequence_length)
    expected = strict_delete_count(length, beta)
    parsed = parse_deleted_intervals(intervals, length)
    if len(parsed) > int(qmax):
        raise ValueError("plan exceeds Qmax")
    previous_end = -2
    for start, end in parsed:
        span_length = int(end) - int(start) + 1
        if span_length < int(min_span_len) or span_length > int(max_span_len):
            raise ValueError("plan violates span length bounds")
        if int(start) <= previous_end + 1:
            raise ValueError("spans must be maximal runs separated by a retained residue")
        previous_end = int(end)
    actual = deleted_length(parsed)
    if actual != expected:
        raise ValueError("plan deletes {} residues, expected {}".format(actual, expected))
    if actual >= length:
        raise ValueError("plan deletes the entire sequence")


def strict_budget_dp_decode(
    residue_logits: torch.Tensor,
    beta: float,
    *,
    qmax: int,
    min_span_len: int = 1,
    max_span_len: Optional[int] = None,
) -> Tuple[Tuple[Tuple[int, int], ...], float]:
    """Exact maximum-score deletion plan under the strict delepref invariants.

    ``B[q,i,d]`` is the best score in the first ``i`` residues with exactly
    ``q`` completed spans and ``d`` deletions. ``R`` is the same state with the
    last residue retained; ``E`` ends a span at ``i-1``.  The span transition
    is a max over a bounded length window.  Along diagonals ``c=i-d`` that max
    is maintained by a monotone deque, giving O(L*D*Qmax) time rather than a
    grow/prune heuristic or an O(max_span_len) inner loop.

    Ties are deterministic without an epsilon: an E-window tie keeps the
    earliest start, B ties prefer the retained transition, and final ties use
    fewer spans.  No strictly lower-scoring plan is ever accepted as a tie.
    """

    values = residue_logits.detach().to(device="cpu", dtype=torch.float64).view(-1)
    length = int(values.numel())
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("strict DP requires finite residue logits")
    deleted_count_value = strict_delete_count(length, beta)
    span_cap = int(qmax)
    min_len = int(min_span_len)
    if span_cap < 1:
        raise ValueError("qmax must be a positive integer")
    if min_len < 1:
        raise ValueError("min_span_len must be a positive integer")
    max_len = int(max_span_len) if max_span_len is not None else deleted_count_value
    if max_len < 1:
        raise ValueError("max_span_len must be a positive integer")
    max_len = min(max_len, length - 1)
    if max_len < min_len:
        raise InfeasibleStrictBudgetError("max_span_len is smaller than min_span_len")
    if not _feasible_remaining(length, deleted_count_value, span_cap, min_len, max_len):
        raise InfeasibleStrictBudgetError(
            "no strict plan for L={}, D={}, Qmax={}, span=[{},{}]".format(
                length, deleted_count_value, span_cap, min_len, max_len
            )
        )
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("strict DP requires numpy for the exact O(L*D*Q) implementation") from exc
    logits = values.numpy()
    prefix = np.zeros(length + 1, dtype=np.float64)
    prefix[1:] = np.cumsum(logits, dtype=np.float64)
    negative_infinity = float("-inf")
    best = np.full(
        (span_cap + 1, length + 1, deleted_count_value + 1),
        negative_infinity,
        dtype=np.float64,
    )
    choice = np.full(
        (span_cap + 1, length + 1, deleted_count_value + 1),
        -1,
        dtype=np.int32,
    )
    best[0, :, 0] = 0.0
    choice[0, :, 0] = 0

    for span_count in range(1, span_cap + 1):
        end_score = np.full((length + 1, deleted_count_value + 1), negative_infinity, dtype=np.float64)
        end_length = np.zeros((length + 1, deleted_count_value + 1), dtype=np.int32)
        # c is retained residue count in the current prefix: c = i - d.
        for retained_count in range(0, length + 1):
            window = deque()
            first_i = max(1, retained_count)
            last_i = min(length, retained_count + deleted_count_value)
            for prefix_length in range(first_i, last_i + 1):
                deleted_in_prefix = prefix_length - retained_count
                start_to_add = prefix_length - min_len
                if start_to_add >= 0:
                    previous_deleted = start_to_add - retained_count
                    if 0 <= previous_deleted <= deleted_count_value:
                        previous_prefix = 0 if start_to_add == 0 else start_to_add - 1
                        previous_score = best[span_count - 1, previous_prefix, previous_deleted]
                        if math.isfinite(float(previous_score)):
                            candidate_value = float(previous_score) - float(prefix[start_to_add])
                            # Strict comparison preserves the earlier start on an exact tie.
                            while window and candidate_value > window[-1][1]:
                                window.pop()
                            window.append((start_to_add, candidate_value))
                minimum_start = prefix_length - max_len
                while window and window[0][0] < minimum_start:
                    window.popleft()
                if window and 0 <= deleted_in_prefix <= deleted_count_value:
                    start, window_value = window[0]
                    end_score[prefix_length, deleted_in_prefix] = float(prefix[prefix_length]) + window_value
                    end_length[prefix_length, deleted_in_prefix] = prefix_length - start

        for deleted_in_prefix in range(1, deleted_count_value + 1):
            candidates = end_score[:, deleted_in_prefix]
            cumulative = np.maximum.accumulate(candidates)
            previous = np.empty_like(cumulative)
            previous[0] = negative_infinity
            previous[1:] = cumulative[:-1]
            strict_improvement = candidates > previous
            reachable = np.isfinite(cumulative)
            best[span_count, :, deleted_in_prefix] = cumulative
            choice[span_count, reachable, deleted_in_prefix] = 0
            improvement_positions = np.nonzero(strict_improvement & reachable)[0]
            choice[span_count, improvement_positions, deleted_in_prefix] = end_length[
                improvement_positions, deleted_in_prefix
            ]

    selected_span_count = 0
    selected_score = negative_infinity
    for span_count in range(1, span_cap + 1):
        score = float(best[span_count, length, deleted_count_value])
        if score > selected_score:
            selected_score = score
            selected_span_count = span_count
    if selected_span_count == 0 or not math.isfinite(selected_score):
        raise InfeasibleStrictBudgetError("strict DP found no feasible plan")
    reconstructed: List[Tuple[int, int]] = []
    span_count = selected_span_count
    prefix_length = length
    remaining_deleted = deleted_count_value
    while span_count > 0 or remaining_deleted > 0:
        if prefix_length <= 0:
            raise RuntimeError("strict DP backtracking reached an invalid state")
        transition = int(choice[span_count, prefix_length, remaining_deleted])
        if transition == 0:
            prefix_length -= 1
            continue
        if transition < 0:
            raise RuntimeError("strict DP backtracking encountered an unreachable state")
        start = prefix_length - transition
        reconstructed.append((start, prefix_length - 1))
        remaining_deleted -= transition
        span_count -= 1
        prefix_length = 0 if start == 0 else start - 1
    intervals = tuple(reversed(reconstructed))
    validate_strict_plan(
        length,
        beta,
        intervals,
        qmax=span_cap,
        min_span_len=min_len,
        max_span_len=max_len,
    )
    reconstructed_score = sum(float(prefix[end + 1] - prefix[start]) for start, end in intervals)
    if not math.isclose(reconstructed_score, selected_score, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise RuntimeError("strict DP backtracking score mismatch")
    return intervals, selected_score


def generate_strict_budget_plans(
    residue_logits: torch.Tensor,
    beta: float,
    *,
    num_candidates: int,
    qmax: int,
    min_span_len: int = 1,
    max_span_len: Optional[int] = None,
    seed: int = 13,
    perturbation_scale: float = 1.0,
    max_attempts: Optional[int] = None,
) -> StrictGenerationResult:
    """Generate up to K unique legal plans using seeded perturbation plus DP."""

    requested = max(0, int(num_candidates))
    if requested == 0:
        return StrictGenerationResult(plans=(), requested_count=0, attempts=0, duplicate_count=0, reason="requested_zero")
    base = residue_logits.detach().to(device="cpu", dtype=torch.float64).view(-1)
    length = int(base.numel())
    if length < 2:
        return StrictGenerationResult(
            plans=(),
            requested_count=requested,
            attempts=0,
            duplicate_count=0,
            reason="sequence_too_short",
        )
    max_len = min(
        int(max_span_len) if max_span_len is not None else strict_delete_count(length, beta),
        length - 1,
    )
    attempts_limit = max(requested, int(max_attempts) if max_attempts is not None else requested * 16)
    plans: List[StrictDecodedPlan] = []
    seen = set()
    duplicates = 0
    reason = "complete"
    for attempt in range(attempts_limit):
        attempt_seed = int(seed) + attempt * 1_000_003
        if attempt == 0 or float(perturbation_scale) == 0.0:
            perturbed = base.clone()
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(attempt_seed)
            noise = torch.randn(base.shape, generator=generator, dtype=base.dtype)
            perturbed = base + float(perturbation_scale) * noise
        try:
            intervals, perturbed_score = strict_budget_dp_decode(
                perturbed,
                beta,
                qmax=qmax,
                min_span_len=min_span_len,
                max_span_len=max_len,
            )
        except InfeasibleStrictBudgetError:
            if not plans:
                return StrictGenerationResult(
                    plans=(),
                    requested_count=requested,
                    attempts=attempt + 1,
                    duplicate_count=duplicates,
                    reason="infeasible_constraints",
                )
            reason = "infeasible_constraints_after_partial_generation"
            break
        if intervals in seen:
            duplicates += 1
            continue
        seen.add(intervals)
        validate_strict_plan(
            length,
            beta,
            intervals,
            qmax=qmax,
            min_span_len=min_span_len,
            max_span_len=max_len,
        )
        original_score = sum(float(base[start : end + 1].sum().item()) for start, end in intervals)
        plans.append(
            StrictDecodedPlan(
                deleted_intervals=intervals,
                deleted_length=deleted_length(intervals),
                score=original_score,
                perturbed_score=float(perturbed_score),
                seed=attempt_seed,
            )
        )
        if len(plans) >= requested:
            break
    if len(plans) < requested and reason == "complete":
        reason = "unique_plan_space_exhausted_or_attempt_limit"
    return StrictGenerationResult(
        plans=tuple(plans),
        requested_count=requested,
        attempts=min(attempts_limit, 1 if requested == 0 else (attempt + 1)),
        duplicate_count=duplicates,
        reason=reason,
    )


def proposal_from_decoded_plan(
    *,
    protein_id: str,
    wildtype_sequence: str,
    beta: float,
    decoded: StrictDecodedPlan,
    rank: int,
    generation_reason: str,
) -> DeletionProposal:
    length = len(wildtype_sequence)
    mask = deletion_mask_from_intervals(length, decoded.deleted_intervals, dtype=torch.float32)
    fraction = float(decoded.deleted_length) / float(length)
    span_lengths = [end - start + 1 for start, end in decoded.deleted_intervals]
    stable_suffix = hashlib.sha256(
        "{}\0{}\0{:.17g}".format(protein_id, sequence_sha256(wildtype_sequence), float(beta)).encode("utf-8")
    ).hexdigest()[:12]
    return DeletionProposal(
        protein_id=protein_id,
        proposal_id="{}__delepref_strict__{}__{:04d}".format(protein_id, stable_suffix, int(rank)),
        wildtype_sequence=wildtype_sequence,
        target_delete_fraction=float(beta),
        deleted_intervals=decoded.deleted_intervals,
        deleted_length=decoded.deleted_length,
        deletion_fraction=fraction,
        proposal_score=float(decoded.score),
        proposal_logprob=float(decoded.score),
        # The delepref student has no supervised confidence target.  The legacy
        # output contract still requires this numeric field, so it is a
        # neutral unavailable placeholder and is never used for ranking.
        proposal_confidence=0.0,
        proposal_rank=int(rank),
        deletion_mask=tuple(int(value) for value in mask.tolist()),
        budget_error=0.0,
        span_count=len(decoded.deleted_intervals),
        fragmentation=float(max(0, len(decoded.deleted_intervals) - 1)) / float(decoded.deleted_length),
        mean_span_length=float(sum(span_lengths)) / float(len(span_lengths)),
        max_span_length=max(span_lengths),
        generation_policy="delepref_seeded_perturbation_strict_dp",
        policy_arch="strict_span_refiner_delepref",
        metadata={
            "student_version": STUDENT_VERSION,
            "coordinate_system": COORDINATE_SYSTEM,
            "decoder": "strict_dynamic_programming",
            "decoder_seed": decoded.seed,
            "generation_reason": generation_reason,
            "student_confidence_available": False,
            "proposal_confidence_used_for_ranking": False,
            "deleted_intervals": canonical_deleted_intervals(decoded.deleted_intervals),
        },
    )


def trainable_gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    squared = 0.0
    for parameter in parameters:
        if not parameter.requires_grad or parameter.grad is None:
            continue
        value = parameter.grad.detach().float().norm(2).item()
        squared += float(value) ** 2
    return math.sqrt(squared)


def head_only_state_dict(model: DeletionPreferenceStudent) -> Dict[str, torch.Tensor]:
    """Checkpoint schema that intentionally excludes all frozen encoder weights."""

    return {key: value for key, value in model.state_dict().items() if not key.startswith("encoder.")}


def save_student_checkpoint(
    path: str,
    *,
    model: DeletionPreferenceStudent,
    optimizer: Optional[object],
    step: int,
    config: Mapping[str, object],
    metadata: Optional[Mapping[str, object]] = None,
) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {
        "student_version": STUDENT_VERSION,
        "step": int(step),
        "student_head_state_dict": head_only_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "config": dict(config),
        "metadata": {**dict(metadata or {}), "save_encoder_weights": False, "online_mutant_evaluation": False},
        "encoder": {
            "frozen": True,
            "weights_saved": False,
            "backend_type": getattr(model.encoder.encoder, "backend_type", type(model.encoder.encoder).__name__),
            "hidden_size": int(model.encoder.hidden_size),
            "max_length": int(model.encoder.max_length),
        },
        "scorer_model_name": model.scorer.scorer_model_name,
        "policy_model_name": model.policy.policy_model_name,
        "coordinate_system": COORDINATE_SYSTEM,
    }
    temporary_path = path + ".tmp"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def load_student_checkpoint(
    path: str,
    *,
    model: DeletionPreferenceStudent,
    optimizer: Optional[object] = None,
    load_optimizer: bool = True,
) -> Mapping[str, object]:
    payload = torch.load(path, map_location="cpu")
    if payload.get("student_version") != STUDENT_VERSION:
        raise ValueError("not a delepref checkpoint")
    state = payload.get("student_head_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("delepref checkpoint is missing student_head_state_dict")
    if any(str(key).startswith("encoder.") for key in state):
        raise ValueError("delepref checkpoint must not contain encoder weights")
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected_list = list(unexpected)
    non_encoder_missing = [key for key in missing if not key.startswith("encoder.")]
    if unexpected_list or non_encoder_missing:
        raise RuntimeError(
            "delepref checkpoint mismatch: missing={} unexpected={}".format(
                non_encoder_missing[:20], unexpected_list[:20]
            )
        )
    if optimizer is not None and load_optimizer and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
