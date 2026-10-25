"""Minimal student deletion-preference scorer for V1 scorer protocol.

This module implements the current DelePref student scorer interface for
counterfactual deletion preference distillation. It ranks candidate deletion
plans by teacher-derived perturbation preference and optional auxiliary
metadata; it does not treat teacher preference as a ground-truth functional
label.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


COORDINATE_SYSTEM = "0_based_closed_inclusive"

STUDENT_PREFERENCE_SCORE_COLUMNS = [
    "protein_id",
    "perturbation_id",
    "plan_id",
    "target_delete_fraction",
    "deleted_intervals",
    "num_intervals",
    "deleted_length",
    "deletion_fraction",
    "coordinate_system",
    "preference_score",
    "preference_confidence",
    "scorer_model",
    "scorer_checkpoint",
    "scorer_mode",
    "scoring_status",
    "metadata_json",
]

PREFERENCE_VALUES = {"a_better", "b_better", "tie", "invalid"}


@dataclass(frozen=True)
class DeletionCandidate:
    """One deletion plan for one wild-type protein."""

    protein_id: str
    perturbation_id: str
    plan_id: str
    wildtype_sequence: str
    deleted_intervals: Tuple[Tuple[int, int], ...]
    deleted_length: int
    deletion_fraction: float
    target_delete_fraction: float
    budget_bin: str = ""
    generation_policy: str = ""


@dataclass(frozen=True)
class PreferencePair:
    """One quality-aware pairwise teacher preference."""

    pair_id: str
    protein_id: str
    budget_bin: str
    preference: str
    preference_confidence: float
    loss_weight: float
    preference_margin: float
    is_valid_pair: bool
    construction_policy: str
    quality_status: str
    quality_reason: str
    a: DeletionCandidate
    b: DeletionCandidate
    pair_quality_bucket: str = ""
    policy_positive_eligible: bool = True
    scorer_training_weight: float = 0.0
    metadata: Optional[Mapping[str, object]] = None


@dataclass(frozen=True)
class DeletionProposal:
    """One student-generated deletion proposal."""

    protein_id: str
    proposal_id: str
    wildtype_sequence: str
    target_delete_fraction: float
    deleted_intervals: Tuple[Tuple[int, int], ...]
    deleted_length: int
    deletion_fraction: float
    proposal_score: float
    proposal_logprob: float
    proposal_confidence: float
    proposal_rank: int = 0
    deletion_mask: Tuple[int, ...] = ()
    budget_error: float = 0.0
    span_count: int = 0
    fragmentation: float = 0.0
    mean_span_length: float = 0.0
    max_span_length: int = 0
    generation_policy: str = ""
    policy_arch: str = ""
    metadata: Optional[Mapping[str, object]] = None


def _as_bool(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None or str(value).strip() == "":
        return float(default)
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _json_loads(value: object, default: object) -> object:
    if value is None or str(value).strip() == "":
        return default
    try:
        return json.loads(str(value))
    except (TypeError, ValueError):
        return default


def _normalize_preference(value: object) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "a": "a_better",
        "prefer_a": "a_better",
        "preferred_a": "a_better",
        "b": "b_better",
        "prefer_b": "b_better",
        "preferred_b": "b_better",
        "equal": "tie",
        "none": "tie",
        "ambiguous": "tie",
    }
    return aliases.get(text, text)


def parse_deleted_intervals(value: object, sequence_length: Optional[int] = None) -> Tuple[Tuple[int, int], ...]:
    """Parse 0-based closed-inclusive deletion intervals.

    Accepted forms are a JSON string such as ``"[[3,5],[10,12]]"`` or a Python
    sequence of two-item integer lists/tuples. Intervals must be sorted and
    non-overlapping under closed-inclusive coordinates.
    """

    data = _json_loads(value, None) if isinstance(value, str) else value
    if not isinstance(data, (list, tuple)):
        raise ValueError("deleted_intervals must be a list")
    intervals: List[Tuple[int, int]] = []
    previous_end: Optional[int] = None
    previous_start: Optional[int] = None
    for index, item in enumerate(data):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("interval {} must be [start, end]".format(index))
        start, end = item
        if isinstance(start, bool) or isinstance(end, bool):
            raise ValueError("interval {} start/end must be integers".format(index))
        start = int(start)
        end = int(end)
        if start < 0:
            raise ValueError("interval {} start must be nonnegative".format(index))
        if end < start:
            raise ValueError("interval {} end must be >= start".format(index))
        if sequence_length is not None and end >= int(sequence_length):
            raise ValueError("interval {} end must be < sequence_length".format(index))
        if previous_start is not None and start < previous_start:
            raise ValueError("deleted_intervals must be sorted")
        if previous_end is not None and start <= previous_end:
            raise ValueError("deleted_intervals must not overlap")
        intervals.append((start, end))
        previous_start = start
        previous_end = end
    if not intervals:
        raise ValueError("deleted_intervals must contain at least one interval")
    return tuple(intervals)


def canonical_deleted_intervals(intervals: Sequence[Tuple[int, int]]) -> str:
    return json.dumps([[int(start), int(end)] for start, end in intervals], separators=(",", ":"))


def deleted_length(intervals: Sequence[Tuple[int, int]]) -> int:
    return int(sum(int(end) - int(start) + 1 for start, end in intervals))


def deletion_mask_from_intervals(
    sequence_length: int,
    intervals: Sequence[Tuple[int, int]],
    *,
    device: object = None,
    dtype: object = None,
) -> torch.Tensor:
    """Return a length-L deletion mask with 1.0 on deleted residues."""

    mask = torch.zeros(int(sequence_length), device=device, dtype=dtype or torch.float32)
    for start, end in intervals:
        start = int(start)
        end = int(end)
        if start < 0 or end < start or end >= int(sequence_length):
            raise ValueError("deleted interval [{}, {}] out of bounds for length {}".format(start, end, sequence_length))
        mask[start : end + 1] = 1.0
    return mask


def intervals_from_deletion_mask(mask: object, sequence_length: Optional[int] = None) -> Tuple[Tuple[int, int], ...]:
    """Convert a binary deletion mask into closed-inclusive intervals."""

    values = mask.detach().cpu().view(-1).tolist() if torch.is_tensor(mask) else list(mask)  # type: ignore[arg-type]
    if sequence_length is not None and len(values) != int(sequence_length):
        raise ValueError("mask length {} != sequence_length {}".format(len(values), sequence_length))
    intervals: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index, value in enumerate(values):
        is_deleted = float(value) >= 0.5
        if is_deleted and start is None:
            start = index
        elif not is_deleted and start is not None:
            intervals.append((start, index - 1))
            start = None
    if start is not None:
        intervals.append((start, len(values) - 1))
    if not intervals:
        raise ValueError("mask must delete at least one residue")
    return tuple(intervals)


class _EncoderConfig:
    def __init__(self, hidden_size: int):
        self.hidden_size = int(hidden_size)


class _ESMCOutput:
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class ESMCOfficialModelWrapper(nn.Module):
    """Adapter exposing official ESMC as the minimal HF-like encoder contract."""

    backend_type = "esmc_official_safetensors"

    def __init__(self, esmc_model: nn.Module, *, hidden_size: int = 1152, max_length: int = 1022):
        super().__init__()
        self.esmc_model = esmc_model
        self.tokenizer = getattr(esmc_model, "tokenizer", None)
        if self.tokenizer is None:
            raise ValueError("official ESMC model wrapper requires model.tokenizer")
        self.config = _EncoderConfig(hidden_size=int(hidden_size))
        self.max_length = int(max_length)

    def forward(self, **encoded) -> _ESMCOutput:
        sequence_tokens = encoded.get("input_ids")
        if sequence_tokens is None:
            sequence_tokens = encoded.get("sequence_tokens")
        if sequence_tokens is None:
            raise ValueError("ESMCOfficialModelWrapper requires input_ids or sequence_tokens")
        output = self.esmc_model(sequence_tokens=sequence_tokens)
        return _ESMCOutput(output.embeddings)


def normalize_esmc_state_key(key: str) -> Optional[str]:
    if key.endswith("._extra_state") or key.endswith("_extra_state"):
        return None
    if key.startswith("esmc."):
        key = key[len("esmc.") :]
    if ".attn.layernorm_qkv.layer_norm_weight" in key:
        key = key.replace(".attn.layernorm_qkv.layer_norm_weight", ".attn.layernorm_qkv.0.weight")
    elif ".attn.layernorm_qkv.layer_norm_bias" in key:
        key = key.replace(".attn.layernorm_qkv.layer_norm_bias", ".attn.layernorm_qkv.0.bias")
    elif ".attn.layernorm_qkv.weight" in key:
        key = key.replace(".attn.layernorm_qkv.weight", ".attn.layernorm_qkv.1.weight")
    elif ".ffn.layer_norm_weight" in key:
        key = key.replace(".ffn.layer_norm_weight", ".ffn.0.weight")
    elif ".ffn.layer_norm_bias" in key:
        key = key.replace(".ffn.layer_norm_bias", ".ffn.0.bias")
    elif ".ffn.fc1_weight" in key:
        key = key.replace(".ffn.fc1_weight", ".ffn.1.weight")
    elif ".ffn.fc2_weight" in key:
        key = key.replace(".ffn.fc2_weight", ".ffn.3.weight")
    elif key.startswith("lm_head."):
        key = "sequence_head." + key[len("lm_head.") :]
    return key


def load_esmc_official_safetensors_model(
    snapshot_path: str,
    *,
    dtype: str = "bfloat16",
    device: object = "cpu",
    max_length: int = 1022,
    require_cuda: bool = True,
) -> ESMCOfficialModelWrapper:
    """Load ESMC-600M from official ESM package plus local safetensors."""

    import inspect

    from esm.models.esmc import ESMC
    try:
        from esm.tokenization import get_esmc_model_tokenizers
    except ImportError:
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

        def get_esmc_model_tokenizers():
            return EsmSequenceTokenizer()
    from safetensors.torch import load_file

    safetensors_path = os.path.join(snapshot_path, "model.safetensors")
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError("ESMC safetensors not found: {}".format(safetensors_path))
    torch_device = torch.device(device)
    if bool(require_cuda) and torch_device.type != "cuda":
        raise RuntimeError("ESMC-600M student requires CUDA for this pipeline; got device {}".format(torch_device))
    try:
        sequence_tokenizer = get_esmc_model_tokenizers()
    except AttributeError as error:
        if "can't set attribute" not in str(error):
            raise
        from .encoder_compat import make_sequence_tokenizer

        sequence_tokenizer = make_sequence_tokenizer()
    esmc_kwargs = {
        "d_model": 1152,
        "n_heads": 18,
        "n_layers": 36,
        "tokenizer": sequence_tokenizer,
    }
    if "use_flash_attn" in inspect.signature(ESMC).parameters:
        esmc_kwargs["use_flash_attn"] = False
    model = ESMC(**esmc_kwargs).eval()
    state = load_file(safetensors_path, device="cpu")
    mapped: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for key, tensor in state.items():
        target_key = normalize_esmc_state_key(key)
        if target_key is None:
            skipped.append(key)
            continue
        mapped[target_key] = tensor
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "ESMC state_dict mapping mismatch: missing={} unexpected={} skipped={}".format(
                list(missing)[:20],
                list(unexpected)[:20],
                len(skipped),
            )
        )
    model.to(torch_device)
    if str(dtype).lower() in {"bf16", "bfloat16"}:
        model.to(torch.bfloat16)
    elif str(dtype).lower() in {"fp16", "float16"}:
        model.to(torch.float16)
    model.eval()
    return ESMCOfficialModelWrapper(model, hidden_size=1152, max_length=max_length)


def residue_embeddings_from_backend(
    model: object,
    sequences: Sequence[str],
    device: torch.device,
    *,
    tokenizer: Optional[object] = None,
    max_length: int = 1022,
    freeze: bool = True,
) -> List[torch.Tensor]:
    """Return one length-aligned residue embedding tensor per sequence."""

    if not sequences:
        return []
    backend_type = getattr(model, "backend_type", "")
    token_source = tokenizer or getattr(model, "tokenizer", None)
    if token_source is None:
        raise ValueError("protein encoder backend requires a tokenizer")
    if backend_type == "esmc_official_safetensors":
        encoded = token_source(list(sequences), return_tensors="pt", padding=True)
    else:
        try:
            encoded = token_source(
                list(sequences),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(max_length) + 2,
                add_special_tokens=True,
                return_special_tokens_mask=True,
            )
        except TypeError:
            encoded = token_source(
                list(sequences),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(max_length) + 2,
                add_special_tokens=True,
            )
    special_tokens_mask = encoded.pop("special_tokens_mask", None)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    if special_tokens_mask is not None:
        special_tokens_mask = special_tokens_mask.to(device)
    context = torch.no_grad() if freeze else _NullContext()
    was_training = bool(getattr(model, "training", False))
    if freeze and hasattr(model, "eval"):
        model.eval()
    with context:
        if backend_type == "esmc_official_safetensors" and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**encoded)
        else:
            outputs = model(**encoded)
        token_embeddings = outputs.last_hidden_state
    if freeze and was_training and hasattr(model, "train"):
        model.train(was_training)
    attention_mask = encoded.get("attention_mask")
    residue_batches: List[torch.Tensor] = []
    for row_idx, sequence in enumerate(sequences):
        if attention_mask is not None:
            valid_mask = attention_mask[row_idx].bool()
        else:
            valid_mask = torch.ones(token_embeddings.shape[1], device=token_embeddings.device, dtype=torch.bool)
        if special_tokens_mask is not None:
            residue_mask = valid_mask & (special_tokens_mask[row_idx] == 0)
            residues = token_embeddings[row_idx][residue_mask]
        else:
            valid_positions = torch.nonzero(valid_mask, as_tuple=False).flatten()
            if int(valid_positions.numel()) >= len(sequence) + 2:
                positions = valid_positions[1 : 1 + len(sequence)]
            else:
                positions = valid_positions[: len(sequence)]
            residues = token_embeddings[row_idx, positions, :]
        if int(residues.shape[0]) != len(sequence):
            raise ValueError(
                "residue embedding count {} does not match sequence length {}".format(
                    int(residues.shape[0]),
                    len(sequence),
                )
            )
        residue_batches.append(residues)
    return residue_batches


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False


class TinyProteinTokenizer:
    """Small local tokenizer used only for smoke-mode compatibility."""

    def __init__(self):
        alphabet = "ACDEFGHIKLMNPQRSTVWYXBZUO"
        self.vocab = {aa: index + 3 for index, aa in enumerate(alphabet)}
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2

    def __call__(
        self,
        sequences: Sequence[str],
        *,
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
    ) -> Mapping[str, torch.Tensor]:
        del return_tensors
        encoded: List[List[int]] = []
        for sequence in sequences:
            tokens = [self.vocab.get(residue.upper(), self.vocab["X"]) for residue in str(sequence)]
            if add_special_tokens:
                tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
            if truncation and max_length is not None:
                tokens = tokens[: int(max_length)]
                if add_special_tokens and tokens and tokens[-1] != self.eos_token_id:
                    tokens[-1] = self.eos_token_id
            encoded.append(tokens)
        width = max(len(tokens) for tokens in encoded)
        input_ids = torch.full((len(encoded), width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(encoded), width), dtype=torch.long)
        for row_idx, tokens in enumerate(encoded):
            input_ids[row_idx, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            attention_mask[row_idx, : len(tokens)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class TinyProteinEncoder(nn.Module):
    """Tiny encoder for local smoke paths when no formal checkpoint is loaded."""

    def __init__(self, hidden_size: int = 64, vocab_size: int = 64):
        super().__init__()
        self.config = type("TinyProteinConfig", (), {"hidden_size": int(hidden_size)})()
        self.embedding = nn.Embedding(int(vocab_size), int(hidden_size), padding_idx=0)
        self.proj = nn.Sequential(nn.LayerNorm(int(hidden_size)), nn.Linear(int(hidden_size), int(hidden_size)), nn.GELU())

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        hidden = self.proj(self.embedding(input_ids))
        return type("TinyProteinOutput", (), {"last_hidden_state": hidden})()


def candidate_from_proposal(proposal: DeletionProposal) -> DeletionCandidate:
    return DeletionCandidate(
        protein_id=proposal.protein_id,
        perturbation_id=proposal.proposal_id,
        plan_id=proposal.proposal_id,
        wildtype_sequence=proposal.wildtype_sequence,
        deleted_intervals=proposal.deleted_intervals,
        deleted_length=proposal.deleted_length,
        deletion_fraction=proposal.deletion_fraction,
        target_delete_fraction=proposal.target_delete_fraction,
        generation_policy=proposal.generation_policy or "structured_span_refiner",
    )


PROPOSAL_MASK_COLUMNS = (
    "protein_id",
    "proposal_id",
    "target_delete_fraction",
    "deleted_intervals",
    "num_intervals",
    "deleted_length",
    "deletion_fraction",
    "coordinate_system",
    "proposal_score",
    "proposal_logprob",
    "proposal_confidence",
    "proposal_mode",
    "proposal_rank",
    "proposal_status",
    "proposal_model",
    "proposal_checkpoint",
    "metadata_json",
)


def make_student_proposal_mask_row(
    proposal: DeletionProposal,
    *,
    proposal_model: str,
    proposal_checkpoint: str,
    proposal_mode: str,
    proposal_status: str = "ok",
    metadata: Optional[Mapping[str, object]] = None,
) -> Dict[str, str]:
    row_metadata = dict(metadata or {})
    if proposal.metadata:
        row_metadata.update(dict(proposal.metadata))
    row_metadata.setdefault("policy_arch", proposal.policy_arch)
    row_metadata.setdefault("generation_policy", proposal.generation_policy)
    row_metadata.setdefault("span_count", int(proposal.span_count or len(proposal.deleted_intervals)))
    row_metadata.setdefault("budget_error", float(proposal.budget_error))
    row_metadata.setdefault("fragmentation", float(proposal.fragmentation))
    row_metadata.setdefault("mean_span_length", float(proposal.mean_span_length))
    row_metadata.setdefault("max_span_length", int(proposal.max_span_length))
    return {
        "protein_id": proposal.protein_id,
        "proposal_id": proposal.proposal_id,
        "target_delete_fraction": "{:.10f}".format(float(proposal.target_delete_fraction)),
        "deleted_intervals": canonical_deleted_intervals(proposal.deleted_intervals),
        "num_intervals": str(len(proposal.deleted_intervals)),
        "deleted_length": str(int(proposal.deleted_length)),
        "deletion_fraction": "{:.10f}".format(float(proposal.deletion_fraction)),
        "coordinate_system": COORDINATE_SYSTEM,
        "proposal_score": "{:.8f}".format(float(proposal.proposal_score)),
        "proposal_logprob": "{:.8f}".format(float(proposal.proposal_logprob)),
        "proposal_confidence": "",
        "proposal_mode": proposal_mode,
        "proposal_rank": str(int(proposal.proposal_rank)),
        "proposal_status": proposal_status,
        "proposal_model": proposal_model,
        "proposal_checkpoint": proposal_checkpoint,
        "metadata_json": json.dumps(row_metadata, sort_keys=True, separators=(",", ":")),
    }


def write_student_proposal_masks(path: str, rows: Iterable[Mapping[str, object]]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROPOSAL_MASK_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in PROPOSAL_MASK_COLUMNS})
