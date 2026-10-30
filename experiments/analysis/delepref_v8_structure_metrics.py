"""Fixed sequence correspondence metrics for frozen v8 outputs.

T/contact bounds keep the full native reference denominator. TM uses the
official USalign search with -I (fixed alignment), -u full WT length, and
protein d0. Missing-coordinate TM bounds are conditional on that known-CA fit.
No sequence or structural realignment is permitted.
"""
from pathlib import Path
import re
import subprocess
import tempfile
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from build_pairs import (reference_pairs, reference_preservation_labels,
                         preservation_bounds, PRESERVATION_VALUES)


def _coordinates(reference, prediction, kept):
    ref = np.asarray(reference, dtype=float)
    kept = np.asarray(kept, dtype=int)
    pred = np.asarray(prediction, dtype=float)
    if ref.ndim != 2 or ref.shape[1] != 3 or pred.shape != (len(kept), 3):
        raise ValueError("coordinates must preserve every input sequence position")
    if kept.ndim != 1 or np.any(kept < 0) or np.any(kept >= len(ref)) or np.any(np.diff(kept) <= 0):
        raise ValueError("invalid fixed WT-to-edited correspondence")
    return ref, pred, kept


def relationship_metrics(reference, prediction, kept):
    ref, pred, kept = _coordinates(reference, prediction, kept)
    i, j = reference_pairs(ref, cutoff=15., min_separation=3)
    labels = reference_preservation_labels(ref, pred, kept, i, j)
    result = preservation_bounds(labels, len(i))
    result["static_relationship_loss"] = 1. - result["C"] if len(i) else None
    retained_ids = np.flatnonzero(labels["retained_mask"])
    degree = np.zeros(len(ref), dtype=int)
    for end in (i[retained_ids], j[retained_ids]):
        np.add.at(degree, end, 1)
    sums = np.zeros(len(ref))
    unknown = np.zeros(len(ref), dtype=int)
    for end in (i[labels["pair_ids"]], j[labels["pair_ids"]]):
        np.add.at(sums, end, PRESERVATION_VALUES[labels["target"]])
    for end in (i[labels["unknown_pair_ids"]], j[labels["unknown_pair_ids"]]):
        np.add.at(unknown, end, 1)
    eligible = degree > 0
    result.update(retained_ca_lddt_lower=float(np.mean(sums[eligible] / degree[eligible])) if eligible.any() else None,
                  retained_ca_lddt_upper=float(np.mean((sums[eligible] + unknown[eligible]) / degree[eligible])) if eligible.any() else None,
                  retained_ca_lddt_residues=int(eligible.sum()),
                  retained_ca_lddt_isolated_residues=int(len(kept) - eligible.sum()))
    # Native contacts use their own immutable full-WT denominator, never a
    # denominator restricted to contacts that survive deletion.
    ci, cj = reference_pairs(ref, cutoff=8., min_separation=3)
    inv = np.full(len(ref), -1, dtype=int)
    inv[kept] = np.arange(len(kept))
    retained = (inv[ci] >= 0) & (inv[cj] >= 0)
    a, b = inv[ci[retained]], inv[cj[retained]]
    distances = np.linalg.norm(pred[a] - pred[b], axis=1)
    finite = np.isfinite(distances)
    recovered = int((distances[finite] < 8.).sum())
    n = len(ci)
    result.update(native_contacts=n, retained_native_contacts=int(retained.sum()),
                  known_retained_contacts=int(finite.sum()), unknown_retained_contacts=int((~finite).sum()),
                  contact_recovery_lower=recovered / n if n else None,
                  contact_recovery_upper=(recovered + int((~finite).sum())) / n if n else None,
                  static_contact_loss=1. - int(retained.sum()) / n if n else None)
    return result


def common_retained_diagnostic(reference, prediction_x, kept_x, prediction_y, kept_y):
    ref, px, kx = _coordinates(reference, prediction_x, kept_x)
    _, py, ky = _coordinates(reference, prediction_y, kept_y)
    i, j = reference_pairs(ref, cutoff=15., min_separation=3)
    common = np.intersect1d(kx, ky)
    use = np.isin(i, common) & np.isin(j, common)
    i, j = i[use], j[use]
    out = {"common_retained_residues": len(common), "common_native_pairs": len(i)}
    for name, p, k in (("x", px, kx), ("y", py, ky)):
        b = preservation_bounds(reference_preservation_labels(ref, p, k, i, j), len(i))
        out[name + "_lower"] = b["T_lower"] if len(i) else None
        out[name + "_upper"] = b["T_upper"] if len(i) else None
    if len(i):
        out.update(difference_lower=out["x_lower"] - out["y_upper"],
                   difference_upper=out["x_upper"] - out["y_lower"])
    return out


def write_ca_pdb(path, coordinates):
    lines = [f"ATOM  {i:5d}  CA  ALA A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
             for i, (x, y, z) in enumerate(coordinates, 1)]
    Path(path).write_text("\n".join(lines + ["TER", "END", ""]), encoding="ascii", newline="\n")


def fixed_full_wt_tm(reference, prediction, kept, binary):
    ref, pred, kept = _coordinates(reference, prediction, kept)
    length = len(ref)
    known = np.isfinite(ref[kept]).all(1) & np.isfinite(pred).all(1)
    count = int(known.sum())
    base = dict(full_wt_length=length, retained_residues=len(kept), known_correspondences=count,
                unknown_retained_residues=len(kept) - count,
                uncertainty="missing contributions in [0,1] conditional on the fixed known-CA TM fit; not a bound on an unknown reoptimized structure",
                implementation="official USalign -I fixed_alignment -u full_WT_length -mol prot; no -fast")
    if count < 3:
        return dict(base, lower=0., upper=len(kept) / length, status="INSUFFICIENT_KNOWN_CA_NO_FIT")
    with tempfile.TemporaryDirectory(prefix="v8_fixed_tm_") as tmp:
        tmp = Path(tmp)
        write_ca_pdb(tmp / "pred.pdb", pred[known])
        write_ca_pdb(tmp / "native.pdb", ref[kept][known])
        (tmp / "fixed.fa").write_text(">pred\n" + "A" * count + "\n>native\n" + "A" * count + "\n", encoding="ascii")
        command = [str(binary), str(tmp / "pred.pdb"), str(tmp / "native.pdb"), "-I", str(tmp / "fixed.fa"),
                   "-u", str(length), "-mol", "prot", "-m", str(tmp / "matrix.txt")]
        output = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120).stdout
        match = re.search(r"TM-score=\s*([\d.]+) \(normalized by user-specified L=([\d.]+) and d0=([\d.]+)\)", output)
        aligned = re.search(r"Aligned length=\s*(\d+)", output)
        if not match or not aligned or int(aligned[1]) != count or float(match[2]) != length:
            raise ValueError("USalign did not preserve the complete supplied correspondence/normalizer")
        matrix = []
        for line in (tmp / "matrix.txt").read_text().splitlines():
            fields = line.split()
            if len(fields) == 5 and fields[0] in ("0", "1", "2"):
                matrix.append([float(x) for x in fields[1:]])
        if len(matrix) != 3:
            raise ValueError("missing USalign transform")
        matrix = np.asarray(matrix)
        # USalign evaluates the PDB coordinates rounded to 0.001 A.
        p = np.round(pred[known], 3)
        r = np.round(ref[kept][known], 3)
        transformed = p @ matrix[:, 1:].T + matrix[:, 0]
        d0 = max(.5, 1.24 * np.cbrt(length - 15.) - 1.8) if length > 21 else .5
        contribution = float(np.sum(1. / (1. + np.sum((transformed - r) ** 2, axis=1) / d0 ** 2)))
        score = contribution / length
        if abs(score - float(match[1])) > 2e-5:
            raise ValueError("USalign TM score/returned transform mismatch")
        return dict(base, lower=score, upper=(contribution + len(kept) - count) / length,
                    status="FIXED_CORRESPONDENCE_TM_COMPUTED", d0=d0,
                    reported_USalign_score=float(match[1]), output_sha256=__import__("hashlib").sha256(output.encode()).hexdigest())
