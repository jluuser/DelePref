"""Outcome-blind fixed-variant baseline utilities. No sequence generation."""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np


def load_original_teacher(root):
    path = Path(root) / "provenance/original_teacher_239942a/score_scisor_style_stage1_segments.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != "95ecf093753720b156432b10a163cbf48b85193f9a7dc80af9a82001963a3d7e":
        raise ValueError("original teacher source identity changed")
    spec = importlib.util.spec_from_file_location("delepref_v8_original_teacher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_utility(impact):
    if impact["scoring_status"] != "ok":
        raise ValueError("original teacher scoring failure")
    values = [float(impact[k]) for k in ("global_similarity", "retained_similarity", "junction_plausibility")]
    if not np.isfinite(values).all():
        raise ValueError("nonfinite original teacher views")
    return float(np.dot([.35, .45, .20], values))


def residue_deletion_probabilities(logits, sequence_length):
    logits = np.asarray(logits, dtype=float).reshape(-1)
    if logits.shape != (sequence_length + 2,):
        raise ValueError("SCISOR token logits must include BOS and EOS only")
    residues = logits[1:-1]
    if not np.isfinite(residues).all():
        raise ValueError("nonfinite SCISOR residue logits")
    # Exactly the distribution used by the native sampler after special-token
    # exclusion and temperature=1 renormalization. Never sample an edit here.
    probabilities = np.exp(residues - residues.max())
    return probabilities / probabilities.sum()


def scisor_mapping_views(links, probability_by_edit):
    """Primary common-map mean; sequence-marginal sum is a prespecified diagnostic.

    The paper's M=1 denoiser defines probabilities for deletion events. Repeated
    residues can give the same physical sequence from several events. The fixed
    common mapping evaluator averages legal maps for every method; we retain it
    for the primary comparison and also report the summed event probability as
    a separate sequence-level diagnostic, never select the better correlation.
    """
    required = {r["edit_key"] for r in links}
    if set(probability_by_edit) != required:
        raise ValueError("incomplete SCISOR mapping coverage")
    grouped = {}
    for link in links:
        grouped.setdefault(link["row_id"], []).append(link)
    primary = []; marginal = []
    for row_id, maps in grouped.items():
        positions = [r["deleted_position"] for r in maps]
        if len(set(positions)) != len(positions):
            raise ValueError("duplicate legal deletion map")
        values = [float(probability_by_edit[r["edit_key"]]) for r in maps]
        if not all(0 <= p <= 1 and np.isfinite(p) for p in values) or sum(values) > 1 + 1e-6:
            raise ValueError("invalid deletion probabilities")
        for link, p in zip(maps, values):
            primary.append(dict(row_id=row_id, deleted_position=link["deleted_position"], score=p))
            marginal.append(dict(row_id=row_id, deleted_position=link["deleted_position"], score=sum(values)))
    return primary, marginal


def paired_target_difference(x, y, seed=20261017):
    if set(x) != set(y) or not x:
        raise ValueError("paired comparison requires the identical fixed target roster")
    if any(x[k] is None or y[k] is None for k in x):
        return dict(status="UNDEFINED_ON_FIXED_ROSTER_NO_TARGET_DROPPING", targets=len(x),
                    undefined_targets=sorted(k for k in x if x[k] is None or y[k] is None), difference=None, CI95=None)
    deltas = np.array([x[k] - y[k] for k in sorted(x)])
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(deltas), size=(10000, len(deltas)))
    return dict(status="PAIRED_TARGET_COMPARISON", targets=len(x), difference=float(deltas.mean()),
                CI95=np.quantile(deltas[draws].mean(1), [.025, .975]).tolist(),
                per_target_difference={k: float(x[k] - y[k]) for k in sorted(x)},
                replicates=10000, seed=seed, uncertainty="conditional on the four eligible targets and frozen assay/mapping protocol")
