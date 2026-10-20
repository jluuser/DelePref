"""Whole-cohort descriptive diagnostics; no model/selector changes or filtering."""
from collections import Counter
import json
import numpy as np

from delepref_v8_structure_prepare import ROOT, sha, read, write
from paired_statistics import paired_bootstrap


def mean_defined(values):
    values = [v for v in values if v is not None]
    return dict(mean=float(np.mean(values)) if values else None, defined_WT=len(values))


def main():
    complete = read("reports/delepref_v8_structure_evaluation/completion.json")
    assert complete["status"] == "ALL_FROZEN_OUTPUTS_EVALUATED" and complete["logical_outputs"] == 732
    rows = read("reports/delepref_v8_structure_evaluation/per_output_metrics.json")
    common = {r["protein_id"]: r for r in read("reports/delepref_v8_structure_evaluation/common_retained_diagnostics.json")}
    annotations = {r["request_id"]: r for r in read("reports/delepref_v8_native_annotations/per_output_deletion_annotations.json")}
    panel = {r["protein_id"]: r for r in read("reports/delepref_v8_native_panel/main_panel.json")}
    expected = {k: r["panel_component"] for k, r in panel.items()}
    lookup = {(r["predictor"], r["role"], r["protein_id"], r["method"]): r for r in rows}
    primary = [r for r in rows if r["predictor"] == "esmfold" and r["role"] == "primary_20pct"]
    assert len(primary) == 400
    pair_rows = [dict(wt_id=r["protein_id"], group_id=expected[r["protein_id"]], method=r["method"],
                      lower=r["metrics"]["T_lower"], upper=r["metrics"]["T_upper"]) for r in primary]
    comparisons = {}
    for other in ("delepref_unlimited", "msa_conservation", "terminal"):
        comparisons["rag_minus_" + other] = paired_bootstrap(pair_rows, "rag_unlimited", other, expected,
                                                          replicates=10000, seed=20261016)
    # The original frozen RAG-vs-original endpoint remains the sole primary
    # contrast. Other intervals are descriptive, with no adjusted p-value claim.
    differences = []
    for pid in panel:
        a = lookup[("esmfold", "primary_20pct", pid, "rag_unlimited")]
        b = lookup[("esmfold", "primary_20pct", pid, "delepref_unlimited")]
        am, bm = a["metrics"], b["metrics"]
        exact = am["T_lower"] == am["T_upper"] and bm["T_lower"] == bm["T_upper"]
        d = am["T_lower"] - bm["T_lower"] if exact else None
        decomposition = None
        if exact and am["G_lower"] is not None and bm["G_lower"] is not None:
            coverage = (am["C"] - bm["C"]) * bm["G_lower"]
            retained_quality = am["C"] * (am["G_lower"] - bm["G_lower"])
            assert abs(coverage + retained_quality - d) < 1e-6
            decomposition = dict(static_coverage_term=coverage, retained_preservation_term=retained_quality,
                                 meaning="arithmetic T=C*G decomposition with original reference, not a causal attribution; G sets differ")
        features = annotations[a["request_id"]]
        differences.append(dict(protein_id=pid, source=panel[pid]["stratum"], length_bin=features["length_bin"],
            support_class=features["retrieval_support_class"], same_deletion_map=a["kept"] == b["kept"],
            difference_lower=am["T_lower"] - bm["T_upper"], difference_upper=am["T_upper"] - bm["T_lower"],
            point_difference=d, static_C_difference=am["C"] - bm["C"],
            common_retained=common[pid], decomposition=decomposition,
            RAG_T=[am["T_lower"], am["T_upper"]], original_T=[bm["T_lower"], bm["T_upper"]]))
    strata = []
    for factor in ("source", "length_bin", "support_class"):
        for value in sorted({r[factor] for r in differences}):
            subset = [r for r in differences if r[factor] == value]
            strata.append(dict(factor=factor, value=value, WT=len(subset),
                mean_difference_lower=float(np.mean([r["difference_lower"] for r in subset])),
                mean_difference_upper=float(np.mean([r["difference_upper"] for r in subset])),
                mean_static_C_difference=float(np.mean([r["static_C_difference"] for r in subset])),
                mean_common_retained_difference=mean_defined([r["common_retained"].get("difference_lower") for r in subset]),
                scope="auxiliary descriptive; all members retained; not a confirmatory subgroup claim"))
    feature_summaries = []
    for method in sorted({r["method"] for r in primary}):
        selected = [annotations[r["request_id"]] for r in primary if r["method"] == method]
        feature_summaries.append(dict(method=method, WT=len(selected),
            **{k: mean_defined([r[k] for r in selected]) for k in ("annotated_helix_retention", "annotated_sheet_retention",
                "high_contact_fraction_deleted", "mean_native_contact_degree_deleted", "deletion_fraction_at_outer_10pct")}))
    ablation = [r for r in rows if r["predictor"] == "esmfold" and r["role"] == "completion_ablation"]
    assert len(ablation) == 20
    ablation_rows = []; ablation_expected = {}; same_ablation_original = 0
    for r in ablation:
        pid = r["protein_id"]; ablation_expected[pid] = expected[pid]
        original = lookup[("esmfold", "primary_20pct", pid, "delepref_unlimited")]
        same_ablation_original += r["kept"] == original["kept"]
        rag = lookup[("esmfold", "primary_20pct", pid, "rag_unlimited")]
        for method, result in (("full_RAG", rag), ("direct_only", r)):
            ablation_rows.append(dict(wt_id=pid, group_id=expected[pid], method=method,
                lower=result["metrics"]["T_lower"], upper=result["metrics"]["T_upper"]))
    ablation_result = paired_bootstrap(ablation_rows, "full_RAG", "direct_only", ablation_expected,
                                      replicates=10000, seed=20261016)
    second_rows = []; second_expected = {}; predictor_agreement = []
    second_pids = sorted({r["protein_id"] for r in rows if r["predictor"] == "boltz2"})
    assert len(second_pids) == 8
    for pid in second_pids:
        second_expected[pid] = expected[pid]
        for method in ("rag_unlimited", "delepref_unlimited"):
            r = lookup[("boltz2", "second_predictor", pid, method)]
            second_rows.append(dict(wt_id=pid, group_id=expected[pid], method=method,
                lower=r["metrics"]["T_lower"], upper=r["metrics"]["T_upper"]))
        a = lookup[("boltz2", "second_predictor", pid, "rag_unlimited")]["metrics"]
        b = lookup[("boltz2", "second_predictor", pid, "delepref_unlimited")]["metrics"]
        main = next(r for r in differences if r["protein_id"] == pid)
        predictor_agreement.append(dict(protein_id=pid, ESMFold_difference_lower=main["difference_lower"],
                                       Boltz2_difference_lower=a["T_lower"] - b["T_upper"],
                                       Boltz2_difference_upper=a["T_upper"] - b["T_lower"]))
    second_result = paired_bootstrap(second_rows, "rag_unlimited", "delepref_unlimited", second_expected,
                                    replicates=10000, seed=20261016)
    second_WT_controls = {
        predictor: float(np.mean([lookup[(predictor, "WT_controls" if predictor == "esmfold" else "second_predictor", pid, "WT")]["metrics"]["T_lower"]
                                  for pid in second_pids])) for predictor in ("esmfold", "boltz2")}
    secondary = [r for r in rows if r["predictor"] == "esmfold" and r["role"] == "secondary_10pct"]
    secondary_expected = {r["protein_id"]: expected[r["protein_id"]] for r in secondary}
    assert len(secondary) == 80 and len(secondary_expected) == 40
    secondary_pairs = [dict(wt_id=r["protein_id"], group_id=expected[r["protein_id"]], method=r["method"],
                            lower=r["metrics"]["T_lower"], upper=r["metrics"]["T_upper"]) for r in secondary]
    secondary_result = paired_bootstrap(secondary_pairs, "rag_unlimited", "delepref_unlimited", secondary_expected,
                                       replicates=10000, seed=20261016)
    # Illustrations follow the already frozen input-hash rule; no output is
    # excluded from the complete result table or statistical denominator.
    exact = [r for r in differences if r["point_difference"] is not None]
    illustrations = {}
    if exact:
        low, high = np.quantile([r["point_difference"] for r in exact], [.25, .75])
        for label, subset in (("lowest_quartile", [r for r in exact if r["point_difference"] <= low]),
                              ("highest_quartile", [r for r in exact if r["point_difference"] >= high])):
            illustrations[label] = [r["protein_id"] for r in sorted(subset,
                key=lambda r: panel[r["protein_id"]]["sequence_sha256"])[:2]]
    out = "reports/delepref_v8_structure_diagnostics/"
    write(out + "per_WT_RAG_differences.json", differences)
    write(out + "descriptive_strata.json", strata)
    write(out + "native_feature_deletion_summaries.json", feature_summaries)
    write(out + "paired_comparisons.json", comparisons)
    write(out + "completion_ablation.json", dict(result=ablation_result, direct_only_equal_original_WT=same_ablation_original,
                                                 total_WT=20, genotype_outputs_not_changed=True))
    write(out + "second_predictor.json", dict(result=second_result, per_WT=predictor_agreement, same8_WT_control_T=second_WT_controls,
                                              selection="fixed source-balanced8WT from input hashes, not reselected"))
    write(out + "secondary_10pct_statistics.json", dict(result=secondary_result, scope="prespecified secondary40WT; descriptive interval"))
    write(out + "illustration_manifest.json", illustrations)
    write(out + "completion.json", dict(status="WHOLE_COHORT_DESCRIPTIVE_DIAGNOSTICS_COMPLETE", source_sha256=sha(__file__),
        logical_outputs=732, WT=100, same_RAG_original_maps=sum(r["same_deletion_map"] for r in differences),
        strict_positive_WT=sum(r["difference_lower"] > 1e-12 for r in differences),
        strict_negative_WT=sum(r["difference_upper"] < -1e-12 for r in differences),
        exact_ties_WT=sum(r["point_difference"] == 0. for r in differences),
        primary_contrast="rag_minus_delepref_unlimited only; other contrasts/strata descriptive",
        all_positive_and_negative_cases_retained=True, no_quality_gate_or_model_change=True,
        no_additional_structure_predictions=True, paper_changes=False))
    print(json.dumps({"status": "WHOLE_COHORT_DESCRIPTIVE_DIAGNOSTICS_COMPLETE", "WT": 100, "outputs": 732}))


if __name__ == "__main__":
    main()
