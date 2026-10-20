"""Self-contained loader for the frozen local ESMC-600M checkpoint."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Optional

import torch
from torch import nn


class ESMCSequenceTokenizer:
    """Minimal character tokenizer for the fixed ESMC sequence vocabulary."""

    _TOKENS = (
        "<cls>",
        "<pad>",
        "<eos>",
        "<unk>",
        "L",
        "A",
        "G",
        "V",
        "S",
        "E",
        "R",
        "T",
        "I",
        "D",
        "P",
        "K",
        "Q",
        "N",
        "F",
        "Y",
        "M",
        "H",
        "W",
        "C",
        "X",
        "B",
        "U",
        "Z",
        "O",
        ".",
        "-",
        "|",
        "<mask>",
    )

    def __init__(self, *, max_length: int = 1022) -> None:
        self.max_length = int(max_length)
        self._token_to_id = {token: index for index, token in enumerate(self._TOKENS)}
        self.cls_token_id = self._token_to_id["<cls>"]
        self.pad_token_id = self._token_to_id["<pad>"]
        self.eos_token_id = self._token_to_id["<eos>"]
        self.unk_token_id = self._token_to_id["<unk>"]

    def __call__(
        self,
        sequences: str | list[str],
        *,
        return_tensors: str = "pt",
        padding: bool = True,
    ) -> dict[str, torch.Tensor]:
        if return_tensors != "pt":
            raise ValueError("ESMCSequenceTokenizer supports only return_tensors='pt'")
        batch = [sequences] if isinstance(sequences, str) else list(sequences)
        if not batch:
            raise ValueError("cannot tokenize an empty sequence batch")
        encoded: list[list[int]] = []
        for sequence in batch:
            if len(sequence) > self.max_length:
                raise ValueError(
                    f"ESMC sequence length {len(sequence)} exceeds maximum {self.max_length}"
                )
            residue_ids = [
                self._token_to_id.get(residue.upper(), self.unk_token_id)
                for residue in sequence
            ]
            encoded.append([self.cls_token_id, *residue_ids, self.eos_token_id])
        widths = {len(row) for row in encoded}
        if len(widths) > 1 and not padding:
            raise ValueError("variable-length sequence batches require padding=True")
        width = max(widths)
        padded = [row + [self.pad_token_id] * (width - len(row)) for row in encoded]
        input_ids = torch.tensor(padded, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.pad_token_id),
        }


class ESMCOfficialModelWrapper(nn.Module):
    """Expose official ESMC with the small interface used by EvoCompress."""

    backend_type = "esmc_official_safetensors"

    def __init__(self, esmc_model: nn.Module, *, max_length: int = 1022) -> None:
        super().__init__()
        self.esmc_model = esmc_model
        self.tokenizer = getattr(esmc_model, "tokenizer", None)
        if self.tokenizer is None:
            raise ValueError("official ESMC model wrapper requires model.tokenizer")
        self.max_length = int(max_length)


def normalize_esmc_state_key(key: str) -> Optional[str]:
    """Map BioHub safetensor keys to the current official ``esm`` package."""

    if key.endswith("._extra_state") or key.endswith("_extra_state"):
        return None
    if key.startswith("esmc."):
        key = key[len("esmc.") :]
    replacements = (
        (".attn.layernorm_qkv.layer_norm_weight", ".attn.layernorm_qkv.0.weight"),
        (".attn.layernorm_qkv.layer_norm_bias", ".attn.layernorm_qkv.0.bias"),
        (".attn.layernorm_qkv.weight", ".attn.layernorm_qkv.1.weight"),
        (".ffn.layer_norm_weight", ".ffn.0.weight"),
        (".ffn.layer_norm_bias", ".ffn.0.bias"),
        (".ffn.fc1_weight", ".ffn.1.weight"),
        (".ffn.fc2_weight", ".ffn.3.weight"),
    )
    for source, target in replacements:
        if source in key:
            return key.replace(source, target)
    if key.startswith("lm_head."):
        return "sequence_head." + key[len("lm_head.") :]
    return key


def load_esmc_official_safetensors_model(
    snapshot_path: str | Path,
    *,
    dtype: str = "bfloat16",
    device: object = "cpu",
    max_length: int = 1022,
    require_cuda: bool = True,
) -> ESMCOfficialModelWrapper:
    """Load ESMC-600M without depending on the former DelePath package."""

    from esm.models.esmc import ESMC

    from safetensors.torch import load_file

    checkpoint = Path(snapshot_path) / "model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"ESMC safetensors not found: {checkpoint}")
    torch_device = torch.device(device)
    if require_cuda and torch_device.type != "cuda":
        raise RuntimeError(f"ESMC-600M requires CUDA for this pipeline; got {torch_device}")

    kwargs = {
        "d_model": 1152,
        "n_heads": 18,
        "n_layers": 36,
        "tokenizer": ESMCSequenceTokenizer(max_length=max_length),
    }
    if "use_flash_attn" in inspect.signature(ESMC).parameters:
        kwargs["use_flash_attn"] = False
    model = ESMC(**kwargs).eval()
    state = load_file(str(checkpoint), device="cpu")
    mapped: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, tensor in state.items():
        target_key = normalize_esmc_state_key(key)
        if target_key is None:
            skipped.append(key)
        else:
            mapped[target_key] = tensor
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "ESMC state_dict mapping mismatch: "
            f"missing={list(missing)[:20]} unexpected={list(unexpected)[:20]} "
            f"skipped={len(skipped)}"
        )

    model.to(torch_device)
    normalized_dtype = dtype.lower()
    if normalized_dtype in {"bf16", "bfloat16"}:
        model.to(torch.bfloat16)
    elif normalized_dtype in {"fp16", "float16"}:
        model.to(torch.float16)
    elif normalized_dtype not in {"fp32", "float32"}:
        raise ValueError(f"unsupported ESMC dtype: {dtype}")
    model.eval()
    return ESMCOfficialModelWrapper(model, max_length=max_length)
