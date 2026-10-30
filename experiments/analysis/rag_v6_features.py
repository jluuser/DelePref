"""Sequence-only homolog evidence. This module has no structure/label inputs."""
import hashlib
import json
from pathlib import Path

import numpy as np

AA = set("ACDEFGHIKLMNPQRSTVWY")
AA_ARRAY = np.array(sorted(AA))
PROFILE = dict(identity_min=.20, identity_max_exclusive=.95, known_coverage_min=.80,
               evalue_max=.001, cluster_identity=.90, cluster_common_min=.50,
               representatives_max=64, common_records_min=8, anchor_window=3,
               anchor_standard_min=2, require_immediate_retained_anchors=True,
               terminal_absence="unknown_no_double_anchor",
               unknown=".", internal_gap="-", no_gap_fraction_maximum=True)
S_NAMES = ["span_count", "longest_span_fraction", "span_length_dispersion",
           "deleted_center", "N_terminal_fraction", "C_terminal_fraction", "interspan_distance"]
M_NAMES = ["deleted_gap_mean", "span_gap_min", "span_gap_max", "span_gap_dispersion",
           "within_span_gap_run_mean", "deleted_query_match", "anchor_query_match"]
J_NAMES = ["H_joint"]
NAMES = S_NAMES + M_NAMES + J_NAMES


def digest(text):
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def spans(mask):
    m = np.asarray(mask)
    if m.ndim != 1 or not np.isin(m, [0, 1]).all():
        raise ValueError("mask must be binary and one dimensional")
    edges = np.diff(np.r_[0, m.astype(np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1).tolist(), (np.flatnonzero(edges == -1) - 1).tolist()))


def project_hit(fields, wt):
    if len(fields) != 16:
        raise ValueError("expected the frozen 16 alignment fields")
    qid, target, ident, _, qstart, qend, qlen, tstart, tend, tlen, evalue, bits, qcov, tcov, qaln, taln = fields
    qaln, taln = qaln.upper(), taln.upper()
    if int(qlen) != len(wt) or len(qaln) != len(taln):
        raise ValueError("alignment dimensions do not match exact WT")
    if not (1 <= int(tstart) <= int(tend) <= int(tlen)) or sum(c!="-" for c in taln)!=int(tend)-int(tstart)+1:
        raise ValueError("target alignment coordinate mismatch")
    ident, evalue, bits = float(ident), float(evalue), float(bits)
    if not (0 <= ident <= 1 and np.isfinite([ident, evalue, bits]).all()):
        raise ValueError("nonfinite alignment or wrong identity units")
    result = ["."] * len(wt)
    position = int(qstart) - 1
    for q, t in zip(qaln, taln):
        if q == "-":
            continue
        if position >= len(wt) or q != wt[position]:
            raise ValueError("alignment query does not reconstruct exact WT positions")
        result[position] = t if t in AA or t == "-" else "."
        position += 1
    if position != int(qend) or not (1 <= int(qstart) <= int(qend) <= len(wt)):
        raise ValueError("query alignment end mismatch")
    # Gaps outside observed target residues are not internal absence evidence.
    standard = [i for i, x in enumerate(result) if x in AA]
    if standard:
        for i in range(len(result)):
            if result[i] == "-" and not standard[0] < i < standard[-1]:
                result[i] = "."
    else:
        result = ["."] * len(wt)
    projected = "".join(result)
    return dict(query_id=qid, target=target, identity=ident, evalue=evalue, bits=bits,
                qcov=float(qcov), tcov=float(tcov), target_length=int(tlen),
                projected=projected, known_coverage=1-projected.count(".")/len(wt),
                substitutions_and_other_indels_not_removed=True)


def aligned_identity(a, b):
    a, b = np.asarray(list(a)), np.asarray(list(b))
    known = (a != ".") & (b != ".")
    if known.mean() < PROFILE["cluster_common_min"]:
        return -1.
    # Double gaps do not increase identity; discordant gaps count as differences.
    informative = known & ((a != "-") | (b != "-"))
    return float(((a == b) & (a != "-") & informative).sum()/max(1, informative.sum()))


def representatives(hits, wt, excluded_sequences):
    accepted, seen_targets, seen_rows = [], set(), set()
    counts = dict(raw=len(hits), filtered=0, duplicate_target=0, duplicate_projection=0,
                  redundant=0, selected=0)
    ordered = sorted(hits, key=lambda h: (-h["bits"], h["evalue"], h["target"], h["projected"]))
    visited = 0
    for h in ordered:
        visited += 1
        row = h["projected"]
        if not (PROFILE["identity_min"] <= h["identity"] < PROFILE["identity_max_exclusive"] and
                h["evalue"] <= PROFILE["evalue_max"] and
                h["known_coverage"] >= PROFILE["known_coverage_min"]):
            counts["filtered"] += 1
            continue
        if digest(row.replace("-", "").replace(".", "")) in excluded_sequences or row == wt:
            counts["filtered"] += 1
            continue
        if h["target"] in seen_targets:
            counts["duplicate_target"] += 1
            continue
        seen_targets.add(h["target"])
        if row in seen_rows:
            counts["duplicate_projection"] += 1
            continue
        seen_rows.add(row)
        if any(aligned_identity(row, old["projected"]) >= PROFILE["cluster_identity"] for old in accepted):
            counts["redundant"] += 1
            continue
        accepted.append(h)
        if len(accepted) == PROFILE["representatives_max"]:
            break
    counts["selected"] = len(accepted)
    counts["unprocessed_after_cap"] = len(ordered)-visited
    return accepted, counts


def build_profiles(query_map, hits_path, excluded_sequences):
    queries = {q["query_id"]: q for q in query_map}
    grouped = {qid: [] for qid in queries}
    with Path(hits_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split("\t")
            if fields[0] not in queries:
                raise ValueError(f"unknown query at line {line_number}")
            grouped[fields[0]].append(project_hit(fields, queries[fields[0]]["wt_sequence"]))
    profiles = {}
    for qid, q in queries.items():
        reps, audit = representatives(grouped[qid], q["wt_sequence"], excluded_sequences)
        profiles[q["protein_id"]] = dict(query=q, representatives=reps, audit=audit,
                                          weighting="equal_dereplicated_representatives_not_phylogenetically_independent")
    return profiles


def longest_run(values):
    longest = current = 0
    for v in values:
        current = current + 1 if v else 0
        longest = max(longest, current)
    return longest


def plan_features(wt, mask, profile):
    """Only WT sequence, binary mask, and frozen natural-homolog profile enter."""
    m = np.asarray(mask, dtype=bool)
    if len(m) != len(wt):
        raise ValueError("mask/WT length mismatch")
    intervals = spans(mask)
    if not intervals or len(intervals) > 3:
        raise ValueError("expected 1..3 maximal deletion spans")
    length, k = len(wt), int(m.sum())
    lens = np.array([b-a+1 for a, b in intervals])
    center = float(np.flatnonzero(m).mean()/max(1, length-1))
    prefix = intervals[0][1]+1 if intervals[0][0] == 0 else 0
    suffix = length-intervals[-1][0] if intervals[-1][1] == length-1 else 0
    distance = np.mean([intervals[t+1][0]-intervals[t][1]-1 for t in range(len(intervals)-1)]) if len(intervals)>1 else 0.
    static = [len(intervals), float(lens.max()/k), float(lens.std()/k), center,
              prefix/k, suffix/k, float(distance/length)]
    rows = profile["representatives"]
    common, anchors = [], sorted(set([x for a, b in intervals for x in (a-1, b+1) if 0 <= x < length]))
    if not prefix and not suffix:
        for h in rows:
            seq = h["projected"]
            eligible = True
            for a, b in intervals:
                left, right = seq[max(0, a-3):a], seq[b+1:min(length, b+4)]
                if (seq[a-1] not in AA or seq[b+1] not in AA or
                    sum(c in AA for c in left) < min(PROFILE["anchor_standard_min"], len(left)) or
                    sum(c in AA for c in right) < min(PROFILE["anchor_standard_min"], len(right)) or
                    "." in seq[a:b+1]):
                    eligible = False
                    break
            if eligible:
                common.append(seq)
    evidence = len(common) >= PROFILE["common_records_min"]
    marginal, joint, joint_valid = [0.] * len(M_NAMES), 0., False
    if evidence:
        array = np.array([list(s) for s in common])
        supports = np.array([[np.mean(array[h, a:b+1] == "-") for a, b in intervals] for h in range(len(common))])
        means = supports.mean(0)
        runmean = np.mean([longest_run(array[h, a:b+1] == "-")/(b-a+1) for h in range(len(common)) for a, b in intervals])
        deleted = array[:, m]
        standard = np.isin(deleted, AA_ARRAY)
        matches = (deleted == np.array(list(wt))[m]) & standard
        query_match = float(matches.sum()/max(1, standard.sum()))
        anchor_match = float(np.mean(array[:, anchors] == np.array(list(wt))[anchors]))
        marginal = [float(np.mean(deleted == "-")), float(means.min()), float(means.max()),
                    float(means.std()), float(runmean), query_match, anchor_match]
        joint_valid = len(intervals)>1 and bool(np.all(supports.std(0)>1e-12))
        joint = float(supports.min(1).mean()) if joint_valid else 0.
    values = np.array(static + marginal + [joint], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite sequence-only feature")
    return dict(values=values.tolist(), evidence=bool(evidence), joint_valid=bool(joint_valid),
                common_records=len(common), representatives=len(rows), q=len(intervals),
                fallback_reason=None if evidence else ("terminal_no_double_anchor" if prefix or suffix else "insufficient_common_observations"))


def pure_candidate(row):
    result = {k: row[k] for k in ("candidate_id", "protein_id", "budget", "wt_sequence", "sequence", "mask")}
    if "".join(c for c, deleted in zip(result["wt_sequence"], result["mask"]) if not deleted) != result["sequence"]:
        raise ValueError("candidate deletion map does not reconstruct sequence")
    if sum(result["mask"]) != max(1, min(len(result["mask"])-1, round(float(result["budget"])*len(result["mask"])))):
        raise ValueError("candidate violates exact budget")
    return result
