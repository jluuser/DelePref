#!/usr/bin/env python3
"""Validate DelePref proposal and deletion-preference CSV contracts."""

import argparse
import csv
import json
import math
import os
import sys
import tempfile


REQUIRED_COLUMNS = [
    "protein_id",
    "sequence_length",
    "start",
    "end",
    "length",
    "stage1_proposal_confidence",
    "stage1_interval_rank",
    "stage1_score_mode",
]

FORBIDDEN_COLUMNS = [
    "stage1_utility_budget",
    "budget_conditioned_stage1",
    "risk_upper",
    "risk_lower",
    "risk_certificate_status",
    "final_delepref_score",
    "target_delete_ratio",
    "user_budget",
]

EXPECTED_SCORE_MODE = "no_budget_strength_grid"

COORDINATE_SYSTEM = "0_based_closed_inclusive"
PREFERENCE_TOLERANCE = 1e-6

DELETION_PERTURBATION_REQUIRED_COLUMNS = [
    "protein_id",
    "perturbation_id",
    "plan_id",
    "wildtype_length",
    "deleted_intervals",
    "num_intervals",
    "deleted_length",
    "deletion_fraction",
    "budget_bin",
    "generation_policy",
    "coordinate_system",
]

DELETION_GENERATION_POLICIES = set([
    "stage1_proposal",
    "random_interval",
    "uniform_random_interval",
    "single_residue",
    "short_span",
    "budget_matched_random",
    "boundary_challenge",
    "low_complexity_or_boundary",
    "high_risk_negative",
    "multi_interval_combination",
    "terminal_baseline",
    "toy",
])

TEACHER_IMPACT_REQUIRED_COLUMNS = [
    "protein_id",
    "perturbation_id",
    "plan_id",
    "teacher_model",
    "teacher_scoring_mode",
    "global_shift",
    "retained_region_shift",
    "junction_plausibility",
    "sequence_manifold_plausibility",
    "risk_overlap",
    "risk_confidence",
    "impact_confidence",
    "scoring_status",
]

TEACHER_SCORING_MODES = set(["mock", "fast", "fast_real_plm", "full"])
TEACHER_SCORING_STATUS = set(["ok", "partial", "failed"])
TEACHER_IMPACT_NUMERIC_COLUMNS = [
    "global_shift",
    "retained_region_shift",
    "junction_plausibility",
    "sequence_manifold_plausibility",
    "risk_overlap",
    "risk_confidence",
    "impact_confidence",
]
TEACHER_IMPACT_OK_REQUIRED = [
    "global_shift",
    "retained_region_shift",
    "risk_overlap",
    "risk_confidence",
    "impact_confidence",
]

TEACHER_PREFERENCE_REQUIRED_COLUMNS = [
    "protein_id",
    "budget_bin",
    "pair_id",
    "perturbation_id_a",
    "plan_id_a",
    "perturbation_id_b",
    "plan_id_b",
    "preference",
    "preference_margin",
    "preference_confidence",
    "comparison_basis_json",
    "construction_policy",
    "is_valid_pair",
]

PREFERENCE_VALUES = set(["a_better", "b_better", "tie", "invalid"])

STUDENT_PREFERENCE_REQUIRED_COLUMNS = [
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

STUDENT_PROPOSAL_REQUIRED_COLUMNS = [
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
]

STUDENT_OUTPUT_STATUSES = set(["ok", "skipped", "error"])
STUDENT_PLANNER_FORBIDDEN_COLUMNS = set([
    "selected",
    "planner_score",
    "planner_rank",
    "feasibility_status",
    "risk_adjusted_score",
    "risk_penalty",
    "risk_filtered",
    "planner_utility_score",
    "final_selection_score",
    "filter_reason",
    "planner_status",
    "final_selected",
])
STUDENT_PREFERENCE_FORBIDDEN_COLUMNS = STUDENT_PLANNER_FORBIDDEN_COLUMNS | set([
    "student_prediction_score",
    "student_prediction_confidence",
    "student_prediction_mode",
    "student_prediction_status",
    "student_prediction_rank",
])
STUDENT_PROPOSAL_FORBIDDEN_COLUMNS = STUDENT_PLANNER_FORBIDDEN_COLUMNS | set([
    "function_preserved",
    "biologically_safe",
    "safe_to_delete",
    "functional_label",
])


def parse_int(value, field, row_number):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError("row {}: {} must be an integer, got {!r}".format(row_number, field, value))


def parse_float(value, field, row_number):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError("row {}: {} must be numeric, got {!r}".format(row_number, field, value))


def validate(path):
    errors = []
    warnings = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        fieldset = set(fieldnames)
        missing = [field for field in REQUIRED_COLUMNS if field not in fieldset]
        forbidden = [field for field in FORBIDDEN_COLUMNS if field in fieldset]
        if missing:
            errors.append("missing required columns: {}".format(", ".join(missing)))
        if forbidden:
            errors.append("forbidden legacy columns present: {}".format(", ".join(forbidden)))
        rows = list(reader)

    if not rows:
        errors.append("CSV has no data rows")

    if errors:
        return errors, warnings, {"rows": len(rows), "columns": fieldnames}

    for idx, row in enumerate(rows, start=2):
        protein_id = (row.get("protein_id") or "").strip()
        if not protein_id:
            errors.append("row {}: protein_id is empty".format(idx))
        sequence_length = parse_int(row.get("sequence_length"), "sequence_length", idx)
        start = parse_int(row.get("start"), "start", idx)
        end = parse_int(row.get("end"), "end", idx)
        length = parse_int(row.get("length"), "length", idx)
        rank = parse_int(row.get("stage1_interval_rank"), "stage1_interval_rank", idx)
        confidence = parse_float(row.get("stage1_proposal_confidence"), "stage1_proposal_confidence", idx)
        if start < 0:
            errors.append("row {}: start must be nonnegative".format(idx))
        if end < start:
            errors.append("row {}: end must be >= start".format(idx))
        if length != end - start + 1:
            errors.append("row {}: length != end - start + 1".format(idx))
        if sequence_length > 0 and end >= sequence_length:
            errors.append("row {}: interval end is out of sequence bounds".format(idx))
        if rank <= 0:
            errors.append("row {}: stage1_interval_rank must be a positive integer".format(idx))
        if not (confidence == confidence):
            errors.append("row {}: stage1_proposal_confidence is NaN".format(idx))
        if (row.get("stage1_score_mode") or "").strip() != EXPECTED_SCORE_MODE:
            errors.append(
                "row {}: stage1_score_mode must be {}, got {!r}".format(
                    idx, EXPECTED_SCORE_MODE, row.get("stage1_score_mode")
                )
            )
        sequence = row.get("sequence")
        if sequence and len(sequence) != sequence_length:
            warnings.append("row {}: optional sequence length does not match sequence_length".format(idx))

    summary = {
        "rows": len(rows),
        "columns": fieldnames,
        "required_columns": REQUIRED_COLUMNS,
        "forbidden_columns_checked": FORBIDDEN_COLUMNS,
    }
    return errors, warnings, summary


def is_empty(value):
    return value is None or str(value).strip() == ""


def table_error(table_name, row_number, column, message):
    return "{} row {} column {}: {}".format(table_name, row_number, column, message)


def table_warning(table_name, row_number, column, message):
    return "{} row {} column {}: {}".format(table_name, row_number, column, message)


def read_csv_table(path, table_name, errors):
    try:
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            return list(reader), list(reader.fieldnames or [])
    except OSError as exc:
        errors.append("{}: cannot read {}: {}".format(table_name, path, exc))
        return [], []


def check_required_columns(table_name, fieldnames, required_columns, errors):
    fieldset = set(fieldnames)
    missing = [field for field in required_columns if field not in fieldset]
    if missing:
        errors.append("{}: missing required columns: {}".format(table_name, ", ".join(missing)))
    return not missing


def check_forbidden_columns(table_name, fieldnames, forbidden_columns, errors):
    fieldset = set(fieldnames)
    forbidden = sorted(field for field in forbidden_columns if field in fieldset)
    if forbidden:
        errors.append("{}: forbidden columns present: {}".format(table_name, ", ".join(forbidden)))
    return not forbidden


def parse_required_int_cell(table_name, row_number, row, column, errors):
    value = row.get(column)
    if is_empty(value):
        errors.append(table_error(table_name, row_number, column, "must be non-empty integer"))
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        errors.append(table_error(table_name, row_number, column, "must be integer, got {!r}".format(value)))
        return None


def parse_optional_float_cell(table_name, row_number, row, column, errors):
    value = row.get(column)
    if is_empty(value):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        errors.append(table_error(table_name, row_number, column, "must be numeric when non-empty, got {!r}".format(value)))
        return None


def parse_required_float_cell(table_name, row_number, row, column, errors):
    value = row.get(column)
    if is_empty(value):
        errors.append(table_error(table_name, row_number, column, "must be non-empty numeric"))
        return None
    return parse_optional_float_cell(table_name, row_number, row, column, errors)


def parse_required_finite_float_cell(table_name, row_number, row, column, errors):
    value = parse_required_float_cell(table_name, row_number, row, column, errors)
    if value is not None and not math.isfinite(value):
        errors.append(table_error(table_name, row_number, column, "must be a finite number"))
        return None
    return value


def parse_required_unit_float_cell(table_name, row_number, row, column, errors):
    value = parse_required_finite_float_cell(table_name, row_number, row, column, errors)
    if value is not None and not (0.0 <= value <= 1.0):
        errors.append(table_error(table_name, row_number, column, "must be in [0, 1]"))
    return value


def parse_bool_cell(table_name, row_number, row, column, errors):
    value = row.get(column)
    text = "" if value is None else str(value).strip().lower()
    if text in ("true", "1", "yes", "y"):
        return True
    if text in ("false", "0", "no", "n"):
        return False
    errors.append(table_error(table_name, row_number, column, "must be parseable bool true/false, got {!r}".format(value)))
    return None


def parse_json_cell(table_name, row_number, row, column, errors, required=False):
    value = row.get(column)
    if is_empty(value):
        if required:
            errors.append(table_error(table_name, row_number, column, "must be non-empty JSON"))
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError) as exc:
        errors.append(table_error(table_name, row_number, column, "must be parseable JSON: {}".format(exc)))
        return None


def parse_json_object_cell(table_name, row_number, row, column, errors, required=False):
    data = parse_json_cell(table_name, row_number, row, column, errors, required=required)
    if data is not None and not isinstance(data, dict):
        errors.append(table_error(table_name, row_number, column, "must be a JSON object"))
        return None
    return data


def parse_deleted_intervals(table_name, row_number, row, wildtype_length, errors):
    data = parse_json_cell(table_name, row_number, row, "deleted_intervals", errors, required=True)
    if data is None:
        return None
    if not isinstance(data, list):
        errors.append(table_error(table_name, row_number, "deleted_intervals", "must be a JSON list"))
        return None
    intervals = []
    for item_idx, item in enumerate(data):
        if not isinstance(item, list) or len(item) != 2:
            errors.append(table_error(table_name, row_number, "deleted_intervals", "interval {} must be [start, end]".format(item_idx)))
            continue
        start, end = item
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
            errors.append(table_error(table_name, row_number, "deleted_intervals", "interval {} start/end must be integers".format(item_idx)))
            continue
        if start < 0:
            errors.append(table_error(table_name, row_number, "deleted_intervals", "interval {} start must be nonnegative".format(item_idx)))
        if end < start:
            errors.append(table_error(table_name, row_number, "deleted_intervals", "interval {} end must be >= start".format(item_idx)))
        if wildtype_length is not None and end >= wildtype_length:
            errors.append(table_error(table_name, row_number, "deleted_intervals", "interval {} end must be < wildtype_length".format(item_idx)))
        intervals.append((start, end))
    previous_start = None
    previous_end = None
    for start, end in intervals:
        if previous_start is not None:
            if start < previous_start:
                errors.append(table_error(table_name, row_number, "deleted_intervals", "intervals must be sorted by ascending start"))
            if start <= previous_end:
                errors.append(table_error(table_name, row_number, "deleted_intervals", "intervals overlap under closed-inclusive coordinates"))
        previous_start = start
        previous_end = end
    return intervals


def normalize_intervals_from_cell(value):
    data = json.loads(value)
    return [(item[0], item[1]) for item in data]


def linked_int_value(table_name, row_number, column, linked_row, errors):
    try:
        return int(str(linked_row.get(column)).strip())
    except (TypeError, ValueError):
        errors.append(table_error(table_name, row_number, column, "linked deletion_perturbations.csv value is not integer"))
        return None


def linked_float_value(table_name, row_number, column, linked_row, errors):
    try:
        return float(str(linked_row.get(column)).strip())
    except (TypeError, ValueError):
        errors.append(table_error(table_name, row_number, column, "linked deletion_perturbations.csv value is not numeric"))
        return None


def mutant_from_intervals(wildtype_sequence, intervals):
    pieces = []
    cursor = 0
    for start, end in intervals:
        pieces.append(wildtype_sequence[cursor:start])
        cursor = end + 1
    pieces.append(wildtype_sequence[cursor:])
    return "".join(pieces)


def validate_deletion_perturbations(path):
    table_name = "deletion_perturbations.csv"
    errors = []
    warnings = []
    rows, fieldnames = read_csv_table(path, table_name, errors)
    if not check_required_columns(table_name, fieldnames, DELETION_PERTURBATION_REQUIRED_COLUMNS, errors):
        return errors, warnings, {}, {}, {"rows": len(rows), "columns": fieldnames}
    perturbations_by_id = {}
    plans_by_id = {}
    seen_perturbations = set()
    for row_number, row in enumerate(rows, start=2):
        protein_id = (row.get("protein_id") or "").strip()
        perturbation_id = (row.get("perturbation_id") or "").strip()
        plan_id = (row.get("plan_id") or "").strip()
        if not protein_id:
            errors.append(table_error(table_name, row_number, "protein_id", "must be non-empty"))
        if not perturbation_id:
            errors.append(table_error(table_name, row_number, "perturbation_id", "must be non-empty"))
        elif perturbation_id in seen_perturbations:
            errors.append(table_error(table_name, row_number, "perturbation_id", "must be unique"))
        else:
            seen_perturbations.add(perturbation_id)
            perturbations_by_id[perturbation_id] = row
        if not plan_id:
            errors.append(table_error(table_name, row_number, "plan_id", "must be non-empty"))
        else:
            plans_by_id[plan_id] = row
        wildtype_length = parse_required_int_cell(table_name, row_number, row, "wildtype_length", errors)
        num_intervals = parse_required_int_cell(table_name, row_number, row, "num_intervals", errors)
        deleted_length = parse_required_int_cell(table_name, row_number, row, "deleted_length", errors)
        deletion_fraction = parse_required_float_cell(table_name, row_number, row, "deletion_fraction", errors)
        if wildtype_length is not None and wildtype_length <= 0:
            errors.append(table_error(table_name, row_number, "wildtype_length", "must be positive"))
        if (row.get("coordinate_system") or "").strip() != COORDINATE_SYSTEM:
            errors.append(table_error(table_name, row_number, "coordinate_system", "must be {}".format(COORDINATE_SYSTEM)))
        policy = (row.get("generation_policy") or "").strip()
        if policy not in DELETION_GENERATION_POLICIES:
            warnings.append(table_warning(table_name, row_number, "generation_policy", "unknown policy {!r}".format(policy)))
        intervals = parse_deleted_intervals(table_name, row_number, row, wildtype_length, errors)
        if intervals is not None:
            computed_length = sum(end - start + 1 for start, end in intervals)
            if num_intervals is not None and num_intervals != len(intervals):
                errors.append(table_error(table_name, row_number, "num_intervals", "must equal len(deleted_intervals)"))
            if deleted_length is not None and deleted_length != computed_length:
                errors.append(table_error(table_name, row_number, "deleted_length", "must equal sum(end - start + 1)"))
            if wildtype_length is not None and deleted_length is not None and deletion_fraction is not None:
                expected = deleted_length / float(wildtype_length)
                if abs(deletion_fraction - expected) > PREFERENCE_TOLERANCE:
                    errors.append(table_error(table_name, row_number, "deletion_fraction", "must equal deleted_length / wildtype_length"))
            wildtype_sequence = row.get("wildtype_sequence")
            mutant_sequence = row.get("mutant_sequence")
            if not is_empty(wildtype_sequence) and wildtype_length is not None and len(wildtype_sequence) != wildtype_length:
                errors.append(table_error(table_name, row_number, "wildtype_sequence", "length must equal wildtype_length"))
            if not is_empty(wildtype_sequence) and not is_empty(mutant_sequence):
                expected_mutant = mutant_from_intervals(wildtype_sequence, intervals)
                if mutant_sequence != expected_mutant:
                    errors.append(table_error(table_name, row_number, "mutant_sequence", "does not match wildtype_sequence with deleted_intervals removed"))
        for json_column in ("junctions_json", "metadata_json"):
            if json_column in row and not is_empty(row.get(json_column)):
                parse_json_cell(table_name, row_number, row, json_column, errors)
    summary = {"rows": len(rows), "columns": fieldnames}
    return errors, warnings, perturbations_by_id, plans_by_id, summary


def validate_teacher_impacts(path, perturbations_by_id=None, plans_by_id=None):
    table_name = "teacher_impact_scores.csv"
    errors = []
    warnings = []
    perturbations_by_id = perturbations_by_id or {}
    plans_by_id = plans_by_id or {}
    rows, fieldnames = read_csv_table(path, table_name, errors)
    if not check_required_columns(table_name, fieldnames, TEACHER_IMPACT_REQUIRED_COLUMNS, errors):
        return errors, warnings, {"rows": len(rows), "columns": fieldnames}
    for row_number, row in enumerate(rows, start=2):
        for column in ("protein_id", "perturbation_id", "plan_id", "teacher_model"):
            if is_empty(row.get(column)):
                errors.append(table_error(table_name, row_number, column, "must be non-empty"))
        mode = (row.get("teacher_scoring_mode") or "").strip()
        status = (row.get("scoring_status") or "").strip()
        if mode not in TEACHER_SCORING_MODES:
            errors.append(table_error(table_name, row_number, "teacher_scoring_mode", "must be one of {}".format(", ".join(sorted(TEACHER_SCORING_MODES)))))
        if status not in TEACHER_SCORING_STATUS:
            errors.append(table_error(table_name, row_number, "scoring_status", "must be one of {}".format(", ".join(sorted(TEACHER_SCORING_STATUS)))))
        for column in TEACHER_IMPACT_NUMERIC_COLUMNS:
            parse_optional_float_cell(table_name, row_number, row, column, errors)
        for column in ("risk_confidence", "impact_confidence"):
            value = parse_optional_float_cell(table_name, row_number, row, column, errors)
            if value is not None and not (0.0 <= value <= 1.0):
                errors.append(table_error(table_name, row_number, column, "must be in [0, 1] when non-empty"))
        if status == "ok":
            for column in TEACHER_IMPACT_OK_REQUIRED:
                if is_empty(row.get(column)):
                    errors.append(table_error(table_name, row_number, column, "must be non-empty when scoring_status=ok"))
        if mode in ("fast", "fast_real_plm") and is_empty(row.get("sequence_manifold_plausibility")) and status == "ok" and is_empty(row.get("view_status_json")):
            errors.append(table_error(table_name, row_number, "sequence_manifold_plausibility", "fast mode ok rows with empty manifold plausibility need view_status_json"))
        for json_column in ("view_status_json", "risk_summary_json", "metadata_json"):
            if json_column in row and not is_empty(row.get(json_column)):
                parse_json_cell(table_name, row_number, row, json_column, errors)
        if perturbations_by_id:
            perturbation_id = (row.get("perturbation_id") or "").strip()
            plan_id = (row.get("plan_id") or "").strip()
            linked = perturbations_by_id.get(perturbation_id)
            if linked is None:
                errors.append(table_error(table_name, row_number, "perturbation_id", "not found in deletion_perturbations.csv"))
            elif (row.get("protein_id") or "").strip() != (linked.get("protein_id") or "").strip():
                errors.append(table_error(table_name, row_number, "protein_id", "does not match linked perturbation"))
            if plan_id not in plans_by_id:
                errors.append(table_error(table_name, row_number, "plan_id", "not found in deletion_perturbations.csv"))
    return errors, warnings, {"rows": len(rows), "columns": fieldnames}


def validate_teacher_preference_pairs(path, perturbations_by_id=None, plans_by_id=None):
    table_name = "teacher_preference_pairs.csv"
    errors = []
    warnings = []
    perturbations_by_id = perturbations_by_id or {}
    plans_by_id = plans_by_id or {}
    rows, fieldnames = read_csv_table(path, table_name, errors)
    if not check_required_columns(table_name, fieldnames, TEACHER_PREFERENCE_REQUIRED_COLUMNS, errors):
        return errors, warnings, {"rows": len(rows), "columns": fieldnames}
    seen_pairs = set()
    for row_number, row in enumerate(rows, start=2):
        pair_id = (row.get("pair_id") or "").strip()
        if not pair_id:
            errors.append(table_error(table_name, row_number, "pair_id", "must be non-empty"))
        elif pair_id in seen_pairs:
            errors.append(table_error(table_name, row_number, "pair_id", "must be unique"))
        else:
            seen_pairs.add(pair_id)
        for column in ("protein_id", "budget_bin", "perturbation_id_a", "plan_id_a", "perturbation_id_b", "plan_id_b", "construction_policy"):
            if is_empty(row.get(column)):
                errors.append(table_error(table_name, row_number, column, "must be non-empty"))
        preference = (row.get("preference") or "").strip()
        if preference not in PREFERENCE_VALUES:
            errors.append(table_error(table_name, row_number, "preference", "must be one of {}".format(", ".join(sorted(PREFERENCE_VALUES)))))
        parse_optional_float_cell(table_name, row_number, row, "preference_margin", errors)
        confidence = parse_optional_float_cell(table_name, row_number, row, "preference_confidence", errors)
        if confidence is not None and not (0.0 <= confidence <= 1.0):
            errors.append(table_error(table_name, row_number, "preference_confidence", "must be in [0, 1] when non-empty"))
        is_valid = parse_bool_cell(table_name, row_number, row, "is_valid_pair", errors)
        if is_valid is False and is_empty(row.get("invalid_reason")):
            errors.append(table_error(table_name, row_number, "invalid_reason", "must be non-empty when is_valid_pair=false"))
        if preference == "invalid" and is_valid is True:
            errors.append(table_error(table_name, row_number, "is_valid_pair", "must be false when preference=invalid"))
        parse_json_cell(table_name, row_number, row, "comparison_basis_json", errors, required=True)
        for json_column in ("metadata_json",):
            if json_column in row and not is_empty(row.get(json_column)):
                parse_json_cell(table_name, row_number, row, json_column, errors)
        if perturbations_by_id:
            pid_a = (row.get("perturbation_id_a") or "").strip()
            pid_b = (row.get("perturbation_id_b") or "").strip()
            plan_a = (row.get("plan_id_a") or "").strip()
            plan_b = (row.get("plan_id_b") or "").strip()
            linked_a = perturbations_by_id.get(pid_a)
            linked_b = perturbations_by_id.get(pid_b)
            if pid_a == pid_b and pid_a:
                errors.append(table_error(table_name, row_number, "perturbation_id_b", "A and B must be different perturbations"))
            if linked_a is None:
                errors.append(table_error(table_name, row_number, "perturbation_id_a", "not found in deletion_perturbations.csv"))
            if linked_b is None:
                errors.append(table_error(table_name, row_number, "perturbation_id_b", "not found in deletion_perturbations.csv"))
            if plan_a not in plans_by_id:
                errors.append(table_error(table_name, row_number, "plan_id_a", "not found in deletion_perturbations.csv"))
            if plan_b not in plans_by_id:
                errors.append(table_error(table_name, row_number, "plan_id_b", "not found in deletion_perturbations.csv"))
            if linked_a is not None and linked_b is not None:
                protein = (row.get("protein_id") or "").strip()
                protein_a = (linked_a.get("protein_id") or "").strip()
                protein_b = (linked_b.get("protein_id") or "").strip()
                if protein_a != protein_b:
                    errors.append(table_error(table_name, row_number, "protein_id", "A and B must belong to the same protein"))
                if protein != protein_a or protein != protein_b:
                    errors.append(table_error(table_name, row_number, "protein_id", "pair protein_id must match linked perturbations"))
                budget_a = (linked_a.get("budget_bin") or "").strip()
                budget_b = (linked_b.get("budget_bin") or "").strip()
                pair_budget = (row.get("budget_bin") or "").strip()
                policy = (row.get("construction_policy") or "").strip().lower()
                if (budget_a != budget_b or pair_budget not in (budget_a, budget_b)) and "near_budget" not in policy:
                    errors.append(table_error(table_name, row_number, "budget_bin", "A and B budget_bin must match unless construction_policy marks near_budget"))
    return errors, warnings, {"rows": len(rows), "columns": fieldnames}


def validate_student_preference_scores(path, deletion_perturbations_by_id=None):
    table_name = "student_preference_scores.csv"
    errors = []
    warnings = []
    deletion_perturbations_by_id = deletion_perturbations_by_id or {}
    rows, fieldnames = read_csv_table(path, table_name, errors)
    has_required = check_required_columns(table_name, fieldnames, STUDENT_PREFERENCE_REQUIRED_COLUMNS, errors)
    has_no_forbidden = check_forbidden_columns(table_name, fieldnames, STUDENT_PREFERENCE_FORBIDDEN_COLUMNS, errors)
    if not (has_required and has_no_forbidden):
        return errors, warnings, {"rows": len(rows), "columns": fieldnames}
    for row_number, row in enumerate(rows, start=2):
        for column in ("protein_id", "perturbation_id", "plan_id", "scorer_model", "scorer_mode"):
            if is_empty(row.get(column)):
                errors.append(table_error(table_name, row_number, column, "must be non-empty"))
        status = (row.get("scoring_status") or "").strip()
        if status not in STUDENT_OUTPUT_STATUSES:
            errors.append(table_error(table_name, row_number, "scoring_status", "must be one of {}".format(", ".join(sorted(STUDENT_OUTPUT_STATUSES)))))
        target_delete_fraction = parse_required_unit_float_cell(table_name, row_number, row, "target_delete_fraction", errors)
        deletion_fraction = parse_required_unit_float_cell(table_name, row_number, row, "deletion_fraction", errors)
        preference_score = parse_required_finite_float_cell(table_name, row_number, row, "preference_score", errors)
        if preference_score is not None:
            pass
        preference_confidence = parse_required_unit_float_cell(table_name, row_number, row, "preference_confidence", errors)
        if preference_confidence is not None:
            pass
        num_intervals = parse_required_int_cell(table_name, row_number, row, "num_intervals", errors)
        deleted_length = parse_required_int_cell(table_name, row_number, row, "deleted_length", errors)
        if num_intervals is not None and num_intervals <= 0:
            errors.append(table_error(table_name, row_number, "num_intervals", "must be positive"))
        if deleted_length is not None and deleted_length < 0:
            errors.append(table_error(table_name, row_number, "deleted_length", "must be nonnegative"))
        if (row.get("coordinate_system") or "").strip() != COORDINATE_SYSTEM:
            errors.append(table_error(table_name, row_number, "coordinate_system", "must be {}".format(COORDINATE_SYSTEM)))
        intervals = parse_deleted_intervals(table_name, row_number, row, None, errors)
        if intervals is not None:
            computed_length = sum(end - start + 1 for start, end in intervals)
            if num_intervals is not None and num_intervals != len(intervals):
                errors.append(table_error(table_name, row_number, "num_intervals", "must equal len(deleted_intervals)"))
            if deleted_length is not None and deleted_length != computed_length:
                errors.append(table_error(table_name, row_number, "deleted_length", "must equal sum(end - start + 1)"))
        parse_json_object_cell(table_name, row_number, row, "metadata_json", errors, required=True)
        if deletion_perturbations_by_id:
            perturbation_id = (row.get("perturbation_id") or "").strip()
            linked = deletion_perturbations_by_id.get(perturbation_id)
            if linked is None:
                errors.append(table_error(table_name, row_number, "perturbation_id", "not found in deletion_perturbations.csv"))
                continue
            if (row.get("protein_id") or "").strip() != (linked.get("protein_id") or "").strip():
                errors.append(table_error(table_name, row_number, "protein_id", "does not match linked deletion_perturbations.csv row"))
            if (row.get("plan_id") or "").strip() != (linked.get("plan_id") or "").strip():
                errors.append(table_error(table_name, row_number, "plan_id", "does not match linked deletion_perturbations.csv row"))
            if (row.get("deleted_intervals") or "").strip() != (linked.get("deleted_intervals") or "").strip():
                try:
                    row_intervals = normalize_intervals_from_cell(row.get("deleted_intervals"))
                    linked_intervals = normalize_intervals_from_cell(linked.get("deleted_intervals"))
                except (TypeError, ValueError):
                    row_intervals = None
                    linked_intervals = None
                if row_intervals is None or linked_intervals is None or row_intervals != linked_intervals:
                    errors.append(table_error(table_name, row_number, "deleted_intervals", "does not match linked deletion_perturbations.csv row"))
            if deleted_length is not None:
                linked_deleted_length = linked_int_value(table_name, row_number, "deleted_length", linked, errors)
                if linked_deleted_length is not None and deleted_length != linked_deleted_length:
                    errors.append(table_error(table_name, row_number, "deleted_length", "does not match linked deletion_perturbations.csv row"))
            if deletion_fraction is not None:
                linked_fraction = linked_float_value(table_name, row_number, "deletion_fraction", linked, errors)
                if linked_fraction is not None and abs(deletion_fraction - linked_fraction) > PREFERENCE_TOLERANCE:
                    errors.append(table_error(table_name, row_number, "deletion_fraction", "does not match linked deletion_perturbations.csv row"))
            if deleted_length is not None:
                linked_wildtype_length = linked_int_value(table_name, row_number, "wildtype_length", linked, errors)
                if linked_wildtype_length:
                    expected = deleted_length / float(linked_wildtype_length)
                else:
                    expected = None
                if deletion_fraction is not None and expected is not None and abs(deletion_fraction - expected) > PREFERENCE_TOLERANCE:
                    errors.append(table_error(table_name, row_number, "deletion_fraction", "must equal deleted_length / linked wildtype_length"))
        if target_delete_fraction is not None and deletion_fraction is not None:
            pass
    return errors, warnings, {"rows": len(rows), "columns": fieldnames}


def validate_student_proposal_masks(path):
    table_name = "student_proposal_masks.csv"
    errors = []
    warnings = []
    rows, fieldnames = read_csv_table(path, table_name, errors)
    has_required = check_required_columns(table_name, fieldnames, STUDENT_PROPOSAL_REQUIRED_COLUMNS, errors)
    has_no_forbidden = check_forbidden_columns(table_name, fieldnames, STUDENT_PROPOSAL_FORBIDDEN_COLUMNS, errors)
    if not (has_required and has_no_forbidden):
        return errors, warnings, {"rows": len(rows), "columns": fieldnames}
    seen_proposals = set()
    for row_number, row in enumerate(rows, start=2):
        protein_id = (row.get("protein_id") or "").strip()
        proposal_id = (row.get("proposal_id") or "").strip()
        if not protein_id:
            errors.append(table_error(table_name, row_number, "protein_id", "must be non-empty"))
        if not proposal_id:
            errors.append(table_error(table_name, row_number, "proposal_id", "must be non-empty"))
        elif proposal_id in seen_proposals:
            errors.append(table_error(table_name, row_number, "proposal_id", "must be unique"))
        else:
            seen_proposals.add(proposal_id)
        for column in ("proposal_mode", "proposal_model"):
            if is_empty(row.get(column)):
                errors.append(table_error(table_name, row_number, column, "must be non-empty"))
        status = (row.get("proposal_status") or "").strip()
        if status not in STUDENT_OUTPUT_STATUSES:
            errors.append(table_error(table_name, row_number, "proposal_status", "must be one of {}".format(", ".join(sorted(STUDENT_OUTPUT_STATUSES)))))
        parse_required_unit_float_cell(table_name, row_number, row, "target_delete_fraction", errors)
        parse_required_unit_float_cell(table_name, row_number, row, "deletion_fraction", errors)
        parse_required_finite_float_cell(table_name, row_number, row, "proposal_score", errors)
        parse_required_finite_float_cell(table_name, row_number, row, "proposal_logprob", errors)
        parse_required_unit_float_cell(table_name, row_number, row, "proposal_confidence", errors)
        proposal_rank = parse_required_int_cell(table_name, row_number, row, "proposal_rank", errors)
        if proposal_rank is not None and proposal_rank <= 0:
            errors.append(table_error(table_name, row_number, "proposal_rank", "must be positive"))
        num_intervals = parse_required_int_cell(table_name, row_number, row, "num_intervals", errors)
        deleted_length = parse_required_int_cell(table_name, row_number, row, "deleted_length", errors)
        if num_intervals is not None and num_intervals <= 0:
            errors.append(table_error(table_name, row_number, "num_intervals", "must be positive"))
        if deleted_length is not None and deleted_length < 0:
            errors.append(table_error(table_name, row_number, "deleted_length", "must be nonnegative"))
        if (row.get("coordinate_system") or "").strip() != COORDINATE_SYSTEM:
            errors.append(table_error(table_name, row_number, "coordinate_system", "must be {}".format(COORDINATE_SYSTEM)))
        intervals = parse_deleted_intervals(table_name, row_number, row, None, errors)
        if intervals is not None:
            computed_length = sum(end - start + 1 for start, end in intervals)
            if num_intervals is not None and num_intervals != len(intervals):
                errors.append(table_error(table_name, row_number, "num_intervals", "must equal len(deleted_intervals)"))
            if deleted_length is not None and deleted_length != computed_length:
                errors.append(table_error(table_name, row_number, "deleted_length", "must equal sum(end - start + 1)"))
        parse_json_object_cell(table_name, row_number, row, "metadata_json", errors, required=True)
    return errors, warnings, {"rows": len(rows), "columns": fieldnames}


def validate_student_output_tables(deletion_path=None, student_preference_path=None, student_proposal_path=None):
    errors = []
    warnings = []
    summary = {}
    perturbations_by_id = {}
    if deletion_path:
        table_errors, table_warnings, perturbations_by_id, _plans_by_id, table_summary = validate_deletion_perturbations(deletion_path)
        errors.extend(table_errors)
        warnings.extend(table_warnings)
        summary["deletion_perturbations.csv"] = table_summary
    if student_preference_path:
        table_errors, table_warnings, table_summary = validate_student_preference_scores(
            student_preference_path,
            perturbations_by_id if deletion_path else None,
        )
        errors.extend(table_errors)
        warnings.extend(table_warnings)
        summary["student_preference_scores.csv"] = table_summary
    if student_proposal_path:
        table_errors, table_warnings, table_summary = validate_student_proposal_masks(student_proposal_path)
        errors.extend(table_errors)
        warnings.extend(table_warnings)
        summary["student_proposal_masks.csv"] = table_summary
    if not any([student_preference_path, student_proposal_path]):
        errors.append("no student output tables were provided")
    return errors, warnings, summary


def validate_deletion_preference_tables(deletion_path=None, impact_path=None, pair_path=None):
    errors = []
    warnings = []
    summary = {}
    perturbations_by_id = {}
    plans_by_id = {}
    if not any([deletion_path, impact_path, pair_path]):
        errors.append("no deletion preference tables were provided")
        return errors, warnings, summary
    if deletion_path:
        table_errors, table_warnings, perturbations_by_id, plans_by_id, table_summary = validate_deletion_perturbations(deletion_path)
        errors.extend(table_errors)
        warnings.extend(table_warnings)
        summary["deletion_perturbations.csv"] = table_summary
    if impact_path:
        table_errors, table_warnings, table_summary = validate_teacher_impacts(impact_path, perturbations_by_id if deletion_path else None, plans_by_id if deletion_path else None)
        errors.extend(table_errors)
        warnings.extend(table_warnings)
        summary["teacher_impact_scores.csv"] = table_summary
    if pair_path:
        table_errors, table_warnings, table_summary = validate_teacher_preference_pairs(pair_path, perturbations_by_id if deletion_path else None, plans_by_id if deletion_path else None)
        errors.extend(table_errors)
        warnings.extend(table_warnings)
        summary["teacher_preference_pairs.csv"] = table_summary
    return errors, warnings, summary


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def toy_mutant(sequence, intervals):
    return mutant_from_intervals(sequence, intervals)


def positive_self_test_rows():
    wildtype = "ACDEFGHIKLMNPQRSTVWY"
    single_intervals = [[3, 5]]
    multi_intervals = [[2, 3], [10, 12]]
    deletion_rows = [
        {
            "protein_id": "toy_protein",
            "perturbation_id": "toy_p1_single",
            "plan_id": "toy_p1_single",
            "wildtype_length": "20",
            "deleted_intervals": json.dumps(single_intervals),
            "num_intervals": "1",
            "deleted_length": "3",
            "deletion_fraction": "{:.8f}".format(3.0 / 20.0),
            "budget_bin": "b20",
            "generation_policy": "toy",
            "coordinate_system": COORDINATE_SYSTEM,
            "wildtype_sequence": wildtype,
            "mutant_sequence": toy_mutant(wildtype, [(3, 5)]),
        },
        {
            "protein_id": "toy_protein",
            "perturbation_id": "toy_p2_multi",
            "plan_id": "toy_p2_multi",
            "wildtype_length": "20",
            "deleted_intervals": json.dumps(multi_intervals),
            "num_intervals": "2",
            "deleted_length": "5",
            "deletion_fraction": "{:.8f}".format(5.0 / 20.0),
            "budget_bin": "b20",
            "generation_policy": "toy",
            "coordinate_system": COORDINATE_SYSTEM,
            "wildtype_sequence": wildtype,
            "mutant_sequence": toy_mutant(wildtype, [(2, 3), (10, 12)]),
        },
    ]
    impact_rows = [
        {
            "protein_id": "toy_protein",
            "perturbation_id": row["perturbation_id"],
            "plan_id": row["plan_id"],
            "teacher_model": "mock_teacher",
            "teacher_scoring_mode": "mock",
            "global_shift": "0.10",
            "retained_region_shift": "0.05",
            "junction_plausibility": "0.80",
            "sequence_manifold_plausibility": "0.75",
            "risk_overlap": "0.00",
            "risk_confidence": "0.90",
            "impact_confidence": "0.85",
            "scoring_status": "ok",
            "view_status_json": json.dumps({"mock": True}),
        }
        for row in deletion_rows
    ]
    pair_rows = [
        {
            "protein_id": "toy_protein",
            "budget_bin": "b20",
            "pair_id": "toy_pair_1",
            "perturbation_id_a": "toy_p1_single",
            "plan_id_a": "toy_p1_single",
            "perturbation_id_b": "toy_p2_multi",
            "plan_id_b": "toy_p2_multi",
            "preference": "a_better",
            "preference_margin": "0.20",
            "preference_confidence": "0.80",
            "comparison_basis_json": json.dumps({"global": "mock", "risk": "auxiliary"}),
            "construction_policy": "toy_budget_matched_pairwise",
            "is_valid_pair": "true",
            "invalid_reason": "",
        }
    ]
    return deletion_rows, impact_rows, pair_rows


def run_deletion_preference_self_test():
    failures = []
    deletion_fields = list(DELETION_PERTURBATION_REQUIRED_COLUMNS) + ["wildtype_sequence", "mutant_sequence"]
    impact_fields = list(TEACHER_IMPACT_REQUIRED_COLUMNS) + ["view_status_json"]
    pair_fields = list(TEACHER_PREFERENCE_REQUIRED_COLUMNS) + ["invalid_reason"]
    with tempfile.TemporaryDirectory() as tempdir:
        deletion_rows, impact_rows, pair_rows = positive_self_test_rows()
        deletion_path = os.path.join(tempdir, "deletion_perturbations.csv")
        impact_path = os.path.join(tempdir, "teacher_impact_scores.csv")
        pair_path = os.path.join(tempdir, "teacher_preference_pairs.csv")
        write_csv(deletion_path, deletion_fields, deletion_rows)
        write_csv(impact_path, impact_fields, impact_rows)
        write_csv(pair_path, pair_fields, pair_rows)
        errors, _warnings, _summary = validate_deletion_preference_tables(deletion_path, impact_path, pair_path)
        if errors:
            failures.append("positive cases should pass, got: {}".format("; ".join(errors)))

        negative_cases = []
        rows = [dict(row) for row in deletion_rows]
        rows[0]["deleted_length"] = "2"
        negative_cases.append(("half-open length error", rows, pair_rows))
        rows = [dict(row) for row in deletion_rows]
        rows[0]["deleted_intervals"] = json.dumps([[5, 3]])
        negative_cases.append(("end before start", rows, pair_rows))
        rows = [dict(row) for row in deletion_rows]
        rows[1]["deleted_intervals"] = json.dumps([[2, 5], [5, 8]])
        rows[1]["deleted_length"] = "8"
        rows[1]["num_intervals"] = "2"
        rows[1]["deletion_fraction"] = "{:.8f}".format(8.0 / 20.0)
        negative_cases.append(("overlapping closed intervals", rows, pair_rows))
        rows = [dict(row) for row in deletion_rows]
        rows[1]["perturbation_id"] = rows[0]["perturbation_id"]
        negative_cases.append(("duplicate perturbation_id", rows, pair_rows))

        cross_rows = [dict(row) for row in deletion_rows]
        extra = dict(deletion_rows[0])
        extra["protein_id"] = "other_protein"
        extra["perturbation_id"] = "other_p"
        extra["plan_id"] = "other_p"
        cross_rows.append(extra)
        bad_pair = [dict(pair_rows[0])]
        bad_pair[0]["perturbation_id_b"] = "other_p"
        bad_pair[0]["plan_id_b"] = "other_p"
        negative_cases.append(("pair crosses protein", cross_rows, bad_pair))

        bad_pair = [dict(pair_rows[0])]
        bad_pair[0]["perturbation_id_b"] = bad_pair[0]["perturbation_id_a"]
        bad_pair[0]["plan_id_b"] = bad_pair[0]["plan_id_a"]
        negative_cases.append(("pair compares same perturbation", deletion_rows, bad_pair))

        bad_pair = [dict(pair_rows[0])]
        bad_pair[0]["preference"] = "invalid"
        bad_pair[0]["is_valid_pair"] = "true"
        negative_cases.append(("invalid preference marked valid", deletion_rows, bad_pair))

        bad_pair = [dict(pair_rows[0])]
        bad_pair[0]["is_valid_pair"] = "false"
        bad_pair[0]["invalid_reason"] = ""
        negative_cases.append(("invalid pair missing reason", deletion_rows, bad_pair))

        for case_name, case_deletions, case_pairs in negative_cases:
            write_csv(deletion_path, deletion_fields, case_deletions)
            write_csv(pair_path, pair_fields, case_pairs)
            errors, _warnings, _summary = validate_deletion_preference_tables(deletion_path, impact_path, pair_path)
            if not errors:
                failures.append("negative case did not fail: {}".format(case_name))
    return failures


def positive_student_output_self_test_rows():
    deletion_rows, _impact_rows, _pair_rows = positive_self_test_rows()
    preference_rows = [
        {
            "protein_id": "toy_protein",
            "perturbation_id": "toy_p1_single",
            "plan_id": "toy_p1_single",
            "target_delete_fraction": "0.15000000",
            "deleted_intervals": json.dumps([[3, 5]]),
            "num_intervals": "1",
            "deleted_length": "3",
            "deletion_fraction": "{:.8f}".format(3.0 / 20.0),
            "coordinate_system": COORDINATE_SYSTEM,
            "preference_score": "0.42",
            "preference_confidence": "0.70",
            "scorer_model": "toy_preference_scorer",
            "scorer_checkpoint": "",
            "scorer_mode": "smallreal_smoke",
            "scoring_status": "ok",
            "metadata_json": json.dumps({"smoke": True, "score_direction": "higher_is_preferred"}),
        },
        {
            "protein_id": "toy_protein",
            "perturbation_id": "toy_p2_multi",
            "plan_id": "toy_p2_multi",
            "target_delete_fraction": "0.25000000",
            "deleted_intervals": json.dumps([[2, 3], [10, 12]]),
            "num_intervals": "2",
            "deleted_length": "5",
            "deletion_fraction": "{:.8f}".format(5.0 / 20.0),
            "coordinate_system": COORDINATE_SYSTEM,
            "preference_score": "0.25",
            "preference_confidence": "0.65",
            "scorer_model": "toy_preference_scorer",
            "scorer_checkpoint": "",
            "scorer_mode": "smallreal_smoke",
            "scoring_status": "ok",
            "metadata_json": json.dumps({"smoke": True, "score_direction": "higher_is_preferred"}),
        },
    ]
    proposal_rows = [
        {
            "protein_id": "toy_protein",
            "proposal_id": "toy_prop_1",
            "target_delete_fraction": "0.15000000",
            "deleted_intervals": json.dumps([[3, 5]]),
            "num_intervals": "1",
            "deleted_length": "3",
            "deletion_fraction": "{:.8f}".format(3.0 / 20.0),
            "coordinate_system": COORDINATE_SYSTEM,
            "proposal_score": "0.90",
            "proposal_logprob": "-1.50",
            "proposal_confidence": "0.75",
            "proposal_mode": "smallreal_smoke",
            "proposal_rank": "1",
            "proposal_status": "ok",
            "proposal_model": "toy_proposal_policy",
            "proposal_checkpoint": "",
            "metadata_json": json.dumps({"smoke": True, "score_direction": "higher_is_preferred"}),
        },
        {
            "protein_id": "toy_protein",
            "proposal_id": "toy_prop_2",
            "target_delete_fraction": "0.25000000",
            "deleted_intervals": json.dumps([[2, 3], [10, 12]]),
            "num_intervals": "2",
            "deleted_length": "5",
            "deletion_fraction": "{:.8f}".format(5.0 / 20.0),
            "coordinate_system": COORDINATE_SYSTEM,
            "proposal_score": "0.55",
            "proposal_logprob": "-2.00",
            "proposal_confidence": "0.60",
            "proposal_mode": "budget_conditioned_mask",
            "proposal_rank": "2",
            "proposal_status": "ok",
            "proposal_model": "toy_proposal_policy",
            "proposal_checkpoint": "",
            "metadata_json": json.dumps({"smoke": True, "score_direction": "higher_is_preferred"}),
        },
    ]
    return deletion_rows, preference_rows, proposal_rows


def run_student_outputs_self_test():
    failures = []
    deletion_fields = list(DELETION_PERTURBATION_REQUIRED_COLUMNS) + ["wildtype_sequence", "mutant_sequence"]
    preference_fields = list(STUDENT_PREFERENCE_REQUIRED_COLUMNS)
    proposal_fields = list(STUDENT_PROPOSAL_REQUIRED_COLUMNS)
    with tempfile.TemporaryDirectory() as tempdir:
        deletion_rows, preference_rows, proposal_rows = positive_student_output_self_test_rows()
        deletion_path = os.path.join(tempdir, "deletion_perturbations.csv")
        preference_path = os.path.join(tempdir, "student_preference_scores.csv")
        proposal_path = os.path.join(tempdir, "student_proposal_masks.csv")
        write_csv(deletion_path, deletion_fields, deletion_rows)
        write_csv(preference_path, preference_fields, preference_rows)
        write_csv(proposal_path, proposal_fields, proposal_rows)
        errors, _warnings, _summary = validate_student_output_tables(
            deletion_path=deletion_path,
            student_preference_path=preference_path,
            student_proposal_path=proposal_path,
        )
        if errors:
            failures.append("positive cases should pass, got: {}".format("; ".join(errors)))

        bad_preference = [dict(row) for row in preference_rows]
        bad_preference[0]["coordinate_system"] = "half_open"
        write_csv(preference_path, preference_fields, bad_preference)
        write_csv(proposal_path, proposal_fields, proposal_rows)
        errors, _warnings, _summary = validate_student_output_tables(
            deletion_path=deletion_path,
            student_preference_path=preference_path,
            student_proposal_path=proposal_path,
        )
        if not errors:
            failures.append("negative case did not fail: bad coordinate_system")

        bad_preference = [dict(row) for row in preference_rows]
        bad_preference[0]["num_intervals"] = "2"
        write_csv(preference_path, preference_fields, bad_preference)
        errors, _warnings, _summary = validate_student_output_tables(
            deletion_path=deletion_path,
            student_preference_path=preference_path,
            student_proposal_path=proposal_path,
        )
        if not errors:
            failures.append("negative case did not fail: num_intervals mismatch")

        write_csv(preference_path, preference_fields, preference_rows)
        bad_proposal = [dict(row) for row in proposal_rows]
        bad_proposal[1]["proposal_id"] = bad_proposal[0]["proposal_id"]
        write_csv(proposal_path, proposal_fields, bad_proposal)
        errors, _warnings, _summary = validate_student_output_tables(
            deletion_path=deletion_path,
            student_preference_path=preference_path,
            student_proposal_path=proposal_path,
        )
        if not errors:
            failures.append("negative case did not fail: duplicate proposal_id")

        proposal_fields_with_forbidden = list(proposal_fields) + ["biologically_safe"]
        bad_proposal = [dict(row) for row in proposal_rows]
        bad_proposal[0]["biologically_safe"] = "true"
        bad_proposal[1]["biologically_safe"] = "false"
        write_csv(proposal_path, proposal_fields_with_forbidden, bad_proposal)
        errors, _warnings, _summary = validate_student_output_tables(
            deletion_path=deletion_path,
            student_preference_path=preference_path,
            student_proposal_path=proposal_path,
        )
        if not errors:
            failures.append("negative case did not fail: forbidden biological safety field")
    return failures


def main():
    parser = argparse.ArgumentParser(description="Validate the Stage1 proposal CSV contract.")
    parser.add_argument("--input_csv", default=None)
    parser.add_argument("--deletion-perturbations", default=None)
    parser.add_argument("--teacher-impact-scores", default=None)
    parser.add_argument("--teacher-preference-pairs", default=None)
    parser.add_argument("--student-preference-scores", default=None)
    parser.add_argument("--student-proposal-masks", default=None)
    parser.add_argument("--self-test-deletion-preference", action="store_true")
    parser.add_argument("--self-test-student-outputs", action="store_true")
    args = parser.parse_args()
    if args.self_test_deletion_preference:
        failures = run_deletion_preference_self_test()
        if failures:
            print("DELETION_PREFERENCE_CONTRACT_SELF_TEST_FAIL")
            for failure in failures:
                print("ERROR: {}".format(failure), file=sys.stderr)
            return 1
        print("DELETION_PREFERENCE_CONTRACT_SELF_TEST_PASS")
        return 0

    if args.self_test_student_outputs:
        failures = run_student_outputs_self_test()
        if failures:
            print("STUDENT_OUTPUT_TABLE_CONTRACT_SELF_TEST_FAIL")
            for failure in failures:
                print("ERROR: {}".format(failure), file=sys.stderr)
            return 1
        print("STUDENT_OUTPUT_TABLE_CONTRACT_SELF_TEST_PASS")
        return 0

    preference_paths = [
        args.deletion_perturbations,
        args.teacher_impact_scores,
        args.teacher_preference_pairs,
    ]
    teacher_preference_paths = [
        args.teacher_impact_scores,
        args.teacher_preference_pairs,
    ]
    student_paths = [
        args.student_preference_scores,
        args.student_proposal_masks,
    ]
    if any(student_paths) and not any(teacher_preference_paths):
        errors, warnings, summary = validate_student_output_tables(
            deletion_path=args.deletion_perturbations,
            student_preference_path=args.student_preference_scores,
            student_proposal_path=args.student_proposal_masks,
        )
        for warning in warnings:
            print("WARNING: {}".format(warning), file=sys.stderr)
        if errors:
            print("STUDENT_OUTPUT_TABLE_CONTRACT_FAIL")
            for error in errors:
                print("ERROR: {}".format(error), file=sys.stderr)
            return 1
        print("STUDENT_OUTPUT_TABLE_CONTRACT_PASS")
        for table_name in sorted(summary):
            print("{} rows: {}".format(table_name, summary[table_name]["rows"]))
        return 0

    if any(preference_paths):
        errors, warnings, summary = validate_deletion_preference_tables(
            deletion_path=args.deletion_perturbations,
            impact_path=args.teacher_impact_scores,
            pair_path=args.teacher_preference_pairs,
        )
        if any(student_paths):
            student_errors, student_warnings, student_summary = validate_student_output_tables(
                deletion_path=args.deletion_perturbations,
                student_preference_path=args.student_preference_scores,
                student_proposal_path=args.student_proposal_masks,
            )
            errors.extend(student_errors)
            warnings.extend(student_warnings)
            summary.update(student_summary)
        for warning in warnings:
            print("WARNING: {}".format(warning), file=sys.stderr)
        if errors:
            print("DELETION_PREFERENCE_TABLE_CONTRACT_FAIL")
            for error in errors:
                print("ERROR: {}".format(error), file=sys.stderr)
            return 1
        print("DELETION_PREFERENCE_TABLE_CONTRACT_PASS")
        for table_name in sorted(summary):
            print("{} rows: {}".format(table_name, summary[table_name]["rows"]))
        return 0

    if not args.input_csv:
        parser.error("--input_csv is required unless using deletion-preference or student-output validation options")
    errors, warnings, summary = validate(args.input_csv)
    for warning in warnings:
        print("WARNING: {}".format(warning), file=sys.stderr)
    if errors:
        print("STAGE1_PROPOSAL_CONTRACT_FAIL")
        for error in errors:
            print("ERROR: {}".format(error), file=sys.stderr)
        return 1
    print("STAGE1_PROPOSAL_CONTRACT_PASS")
    print("input_csv: {}".format(os.path.abspath(args.input_csv)))
    print("rows: {}".format(summary["rows"]))
    print("columns: {}".format(",".join(summary["columns"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
