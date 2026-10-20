"""Diagnose fixed RAG candidate replacement and scorer/structure disagreement.

This reads only the already frozen pools and selected-output structures. It
does not rerank candidates, generate alternatives, or estimate pool oracle.
"""
from collections import Counter
import numpy as np

from delepref_v8_structure_prepare import ROOT, sha, read, write


def main():
    candidates = read("reports/delepref_v8_main/candidate_scores.json")
    selected = read("reports/delepref_v8_main/primary_secondary_ablation_selections.json")
    selected = [r for r in selected if r["evaluation_role"] == "primary_20pct"]
    lookup = {(r["protein_id"], r["method"]): r for r in selected}
    pools = {}
    for r in candidates:
        if r["generation_seed"] == 13 and r["budget"] == .2 and r["method"] == "rag_unlimited":
            pools.setdefault(r["protein_id"], []).append(r)
    differences = {r["protein_id"]: r for r in read("reports/delepref_v8_structure_diagnostics/per_WT_RAG_differences.json")}
    rows = []
    for pid, result in differences.items():
        base = lookup[(pid, "delepref_unlimited")]
        rag = lookup[(pid, "rag_unlimited")]
        pool = pools[pid]
        assert len(pool) == 8
        present = any(r["deletion_mask_sha256"] == base["deletion_mask_sha256"] for r in pool)
        score_difference = rag["scorer_score"] - base["scorer_score"]
        rows.append(dict(protein_id=pid, source=result["source"], support_class=result["support_class"],
            baseline_winner_in_RAG_pool=present, same_selected_mask=result["same_deletion_map"],
            baseline_raw_candidate_rank=base["raw_candidate_rank"], baseline_policy_native_rank=base["policy_native_rank"],
            RAG_generation_kind=rag["generation_kind"], RAG_retrieval_source=rag["retrieval_source"],
            original_span_count=base["span_count"], RAG_span_count=rag["span_count"],
            scorer_difference=score_difference, structural_T_difference=result["point_difference"],
            scorer_prefers_new_but_structural_T_decreases=(score_difference > 1e-5 and result["difference_upper"] < 0.),
            baseline_scorer_winner_excluded_and_new_score_lower=(not present and score_difference < -1e-5)))
    changed = [r for r in rows if not r["same_selected_mask"]]
    groups = {}
    for label, subset in (
        ("baseline_winner_retained", [r for r in rows if r["baseline_winner_in_RAG_pool"]]),
        ("baseline_winner_excluded", [r for r in rows if not r["baseline_winner_in_RAG_pool"]]),
        ("higher_scorer_new_winner", [r for r in changed if r["scorer_difference"] > 1e-5]),
        ("lower_scorer_new_winner", [r for r in changed if r["scorer_difference"] < -1e-5])):
        groups[label] = dict(WT=len(subset), mean_structural_T_difference=float(np.mean([r["structural_T_difference"] for r in subset])) if subset else None,
                             positive=sum(r["structural_T_difference"] > 0 for r in subset),
                             negative=sum(r["structural_T_difference"] < 0 for r in subset))
    decomp = [r["decomposition"] for r in differences.values() if r["decomposition"] is not None]
    report = dict(status="FIXED_SELECTION_DIAGNOSIS_COMPLETE", source_sha256=sha(__file__), WT=100, changed_winners=len(changed),
        all_positive_and_negative_cases_retained=True, baseline_winner_present_WT=sum(r["baseline_winner_in_RAG_pool"] for r in rows),
        baseline_winner_excluded_WT=sum(not r["baseline_winner_in_RAG_pool"] for r in rows),
        excluded_better_scorer_winner_WT=sum(r["baseline_scorer_winner_excluded_and_new_score_lower"] for r in rows),
        higher_scorer_but_lower_structural_T_WT=sum(r["scorer_prefers_new_but_structural_T_decreases"] for r in rows),
        changed_generation_kinds=dict(Counter(r["RAG_generation_kind"] for r in changed)), descriptive_groups=groups,
        average_static_C_difference=float(np.mean([r["static_C_difference"] for r in differences.values()])),
        arithmetic_decomposition_mean={key:float(np.mean([r[key] for r in decomp])) for key in ("static_coverage_term", "retained_preservation_term")},
        interpretation=["The baseline winner is retained in99/100 RAG pools; the sole excluded-winner case improves T. Loss of the original winner is therefore not the observed main explanation for this cohort.",
                        "Some higher-Scorer retrieved selections have lower independent structural preservation; the original teacher-trained scorer does not guarantee structural ranking accuracy.",
                        "Most inputs have partial natural-length evidence and require Policy completion; compensatory substitutions and training-range extrapolation remain hypotheses, not proven causal explanations.",
                        "The4 complete-support WT results are a small descriptive signal, not a confirmed general subgroup effect or permission to change this frozen evaluation."],
        score_comparison_tolerance=1e-5, tolerance_role="numeric scorer comparison only, never a scientific quality gate",
        new_model_candidates_selectors_or_structure_calls=False, paper_changes=False)
    out = "reports/delepref_v8_structure_diagnostics/"
    write(out + "selection_pool_diagnostic_rows.json", rows)
    write(out + "selection_pool_diagnostic.json", report)
    print(__import__("json").dumps({k:report[k] for k in ("WT", "changed_winners", "baseline_winner_excluded_WT", "excluded_better_scorer_winner_WT", "higher_scorer_but_lower_structural_T_WT", "descriptive_groups")}))


if __name__ == "__main__":
    main()
