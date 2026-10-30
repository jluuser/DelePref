"""Fixed WT relationship labels and honest missing-coordinate score bounds."""
import argparse
from pathlib import Path
import numpy as np
from common import AUTHORIZED_ROOT, confined_path, read_jsonl, sha_file, write_json, write_jsonl
from adapters.structure_adapter import coordinates_from_native_json, read_coordinates
from adapters.delepref_adapter import retained_mapping

ERROR_THRESHOLDS = np.array([.5, 1., 2., 4.], dtype=np.float32)
PRESERVATION_VALUES = np.array([1., .75, .5, .25, 0.], dtype=np.float32)


def reference_pairs(coords, cutoff=15., min_separation=3):
    coords = np.asarray(coords, dtype=float)
    i, j = np.triu_indices(len(coords), k=min_separation)
    distances = np.linalg.norm(coords[i] - coords[j], axis=1)
    valid = np.isfinite(distances) & (distances < cutoff)
    return i[valid], j[valid]


def retained_pairs(length, kept, pair_i, pair_j):
    inv = np.full(length, -1, dtype=int)
    inv[kept] = np.arange(len(kept))
    retained = (inv[pair_i] >= 0) & (inv[pair_j] >= 0)
    ids = np.flatnonzero(retained)
    return ids, inv[pair_i[ids]], inv[pair_j[ids]], retained


def reference_preservation_labels(wt_reference, mutant_prediction, kept, pair_i, pair_j):
    wt_reference = np.asarray(wt_reference, dtype=float)
    mutant_prediction = np.asarray(mutant_prediction, dtype=float)
    kept = np.asarray(kept, dtype=int)
    pair_i, pair_j = np.asarray(pair_i, dtype=int), np.asarray(pair_j, dtype=int)
    length = len(wt_reference)
    if wt_reference.shape != (length, 3) or mutant_prediction.shape != (len(kept), 3):
        raise ValueError("coordinates must preserve sequence positions")
    if kept.ndim != 1 or np.any(kept < 0) or np.any(kept >= length) or np.any(np.diff(kept) <= 0):
        raise ValueError("kept must be a strictly increasing exact deletion map")
    if pair_i.shape != pair_j.shape or pair_i.ndim != 1:
        raise ValueError("pairs must be one dimensional and equally long")
    if np.any(pair_i < 0) or np.any(pair_j >= length) or np.any(pair_i >= pair_j):
        raise ValueError("pairs must use valid unique-orientation WT indices")
    if len(set(zip(pair_i.tolist(), pair_j.tolist()))) != len(pair_i):
        raise ValueError("duplicate unordered relationships")
    ids, a, b, retained = retained_pairs(length, kept, pair_i, pair_j)
    i, j = pair_i[ids], pair_j[ids]
    dw = np.linalg.norm(wt_reference[i] - wt_reference[j], axis=1)
    dm = np.linalg.norm(mutant_prediction[a] - mutant_prediction[b], axis=1)
    finite = np.isfinite(dw) & np.isfinite(dm)
    error = np.abs(dm[finite] - dw[finite])
    return dict(pair_ids=ids[finite], wt_i=i[finite], wt_j=j[finite],
                edited_i=a[finite], edited_j=b[finite],
                target=np.searchsorted(ERROR_THRESHOLDS, error, side="right").astype(np.int64),
                error=error, retained_mask=retained, unknown_pair_ids=ids[~finite])


def preservation_bounds(labels, total_pairs):
    retained = int(labels["retained_mask"].sum())
    unknown = len(labels["unknown_pair_ids"])
    contribution = float(PRESERVATION_VALUES[labels["target"]].sum())
    return {"reference_pairs": int(total_pairs), "retained_pairs": retained,
            "known_pairs": len(labels["target"]), "unknown_pairs": unknown,
            "C": retained / total_pairs if total_pairs else None,
            "G_lower": contribution / retained if retained else None,
            "G_upper": (contribution + unknown) / retained if retained else None,
            "T_lower": contribution / total_pairs if total_pairs else 0.,
            "T_upper": (contribution + unknown) / total_pairs if total_pairs else 0.}


def legacy_lddt(labels):
    # Historical retained_ca_lddt is mean per-residue score, unlike new pair-weighted G.
    ids = np.union1d(labels["wt_i"], labels["wt_j"])
    score = np.mean(labels["error"][:, None] < ERROR_THRESHOLDS, axis=1)
    sums = np.zeros(len(ids)); counts = np.zeros(len(ids))
    for endpoint in [labels["wt_i"], labels["wt_j"]]:
        indices = np.searchsorted(ids, endpoint)
        np.add.at(sums, indices, score); np.add.at(counts, indices, 1)
    return float(np.mean(sums / counts)) if len(ids) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import json
    protocol = json.loads(Path(args.protocol).read_text())
    if Path(protocol["root"]).resolve() != AUTHORIZED_ROOT.resolve():
        raise ValueError("unauthorized protocol root")
    root = confined_path(args.output)
    if root.resolve() != (AUTHORIZED_ROOT / "labels/pairs").resolve():
        raise ValueError("labels output must be experiment labels/pairs")
    root.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.manifest); summaries = []; failures = []
    references = {}; metric_differences = []
    for row in rows:
        try:
            pid = row["protein_id"]
            if pid not in references:
                if sha_file(row["wt_native_json"]) != row["wt_native_json_sha256"] or sha_file(row["reference_path"]) != row["reference_sha256"]:
                    raise ValueError("WT reference changed after manifest freeze")
                coords, _ = coordinates_from_native_json(row["wt_native_json"], row["wt_sequence"])
                i, j = reference_pairs(coords, protocol["labels"]["ca_cutoff_angstrom"], protocol["labels"]["min_sequence_separation"])
                references[pid] = coords, i, j
            coords, i, j = references[pid]
            kept = retained_mapping(row["wt_sequence"], row["sequence"], row["mask"])
            if sha_file(row["prediction_path"]) != row["prediction_sha256"]:
                raise ValueError("variant structure changed after manifest freeze")
            mut, _ = read_coordinates(row["prediction_path"], row["sequence"])
            labels = reference_preservation_labels(coords, mut, kept, i, j)
            if not (np.array_equal(kept[labels["edited_i"]], labels["wt_i"]) and np.array_equal(kept[labels["edited_j"]], labels["wt_j"])):
                raise ValueError("end-to-end pair inverse mapping failed")
            output = confined_path(root / (row["candidate_id"] + ".npz"))
            np.savez_compressed(output, kept=kept, reference_i=i, reference_j=j, **labels)
            summary = {"candidate_id": row["candidate_id"], "protein_id": pid, "source": row["source"], "budget": row["budget"], "labels_path": str(output), "labels_sha256": sha_file(output), **preservation_bounds(labels, len(i)), "class_counts": np.bincount(labels["target"], minlength=5).tolist()}
            expected = row.get("legacy_metrics", {}).get("retained_ca_lddt")
            if expected is not None:
                actual = legacy_lddt(labels)
                delta = abs(actual - expected) if actual is not None else None
                metric_differences.append({"candidate_id": row["candidate_id"], "expected": expected, "recomputed": actual, "abs_difference": delta})
                if delta is None or delta > 1e-6:
                    raise ValueError(f"historical retained lDDT mismatch: {delta}")
            summaries.append(summary)
        except Exception as error:
            failures.append({"candidate_id": row["candidate_id"], "error": str(error)})
    write_jsonl(root.parent.parent / "reports/label_candidates.jsonl", summaries)
    write_jsonl(root.parent.parent / "reports/legacy_metric_reproduction.jsonl", metric_differences)
    report = {"status": "LABELS_VERIFIED" if not failures and summaries else "LABELS_FAILED", "manifest_sha256": sha_file(args.manifest), "protocol_sha256": sha_file(args.protocol), "candidates": len(summaries), "proteins": len(references), "failures": failures, "legacy_metrics_recomputed": len(metric_differences)}
    write_json(root.parent.parent / "reports/label_audit.json", report)
    print(json.dumps(report))
    if failures or not summaries:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
