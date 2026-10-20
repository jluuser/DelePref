"""Audit frozen structure identities/coverage without examining quality gains."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from delepref_v8_structure_prepare import ROOT, sha, read, write
sys.path.insert(0, str(ROOT / "code"))
from budget_ledger import BudgetLedger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    freeze_path = ROOT / "configs/delepref_v8_confirmation_freeze.json"
    freeze_sha = sha(freeze_path)
    freeze = read("configs/delepref_v8_confirmation_freeze.json")
    assert freeze_sha == "721032b75900b7936bd9705e2db3ab4fda61c5c381ad85bd7ab1846b6fe92061"
    for r in freeze["files"]:
        assert sha(r["path"]) == r["sha256"], r["path"]
    physical = {r["task_id"]: r for r in read("reports/delepref_v8_structure_preparation/physical_requests.json")}
    logical = read("reports/delepref_v8_structure_preparation/logical_requests.json")
    panel = {r["protein_id"]: r for r in read("reports/delepref_v8_native_panel/main_panel.json")}
    assert len(physical) == 641 and len(logical) == 732 and len(panel) == 100
    for r in logical:
        p = physical[r["physical_id"]]
        seq = panel[r["protein_id"]]["sequence"]
        kept = r["kept"]
        assert kept == sorted(set(kept)) and all(0 <= i < len(seq) for i in kept)
        assert "".join(seq[i] for i in kept) == p["sequence"]
        assert hashlib.sha256(p["sequence"].encode()).hexdigest() == r["sequence_sha256"] == p["sequence_sha256"]
    ledger = BudgetLedger(ROOT / "reports/delepref_v8_evaluation_preparation/statistical_protocol.json",
                          ROOT / "reports/delepref_v8_structures/budget_ledger.jsonl")
    snapshot = ledger.snapshot()
    states = {}
    for row in snapshot["attempts"]:
        request = physical[row["task_id"]]
        assert row["sequence_sha"] == request["sequence_sha256"]
        assert row["predictor_weights_sha"] == request["weights_sha256"]
        assert row["profile_sha"] == request["profile_sha256"]
        assert (row["seed"], row["sample"], row["wt_variant"]) == (request["seed"], request["sample"], request["wt_variant"])
        assert row["category"] == ("retry" if row["attempt"] else request["category"])
        if row["task_id"] not in states or row["attempt"] > states[row["task_id"]]["attempt"]:
            states[row["task_id"]] = row
    assert snapshot["consumed"] <= 800
    artifacts = []; records_pending_write = []
    for task, state in sorted(states.items()):
        request = physical[task]
        if state["state"] not in ("succeeded", "cache_hit"):
            continue
        path = Path(state["output_path"])
        assert path.resolve().is_relative_to(ROOT / "reports/delepref_v8_structures/artifacts")
        assert sha(path) == state["output_sha"]
        with np.load(path, allow_pickle=False) as artifact:
            assert artifact["sequence"].item() == request["sequence"]
            assert artifact["sequence_sha256"].item() == request["sequence_sha256"]
            coords = artifact["coordinates"]
            assert coords.shape == (len(request["sequence"]), 3)
            finite = int(np.isfinite(coords).all(1).sum())
        metadata_path = ROOT / "reports/delepref_v8_structures/records" / (task + ".json")
        try:
            record = json.loads(metadata_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            records_pending_write.append(task)
        else:
            assert record["confirmation_freeze_sha256"] == freeze_sha
            assert record["output_sha256"] == state["output_sha"]
            assert record["sequence_sha256"] == request["sequence_sha256"]
        artifacts.append(dict(task_id=task, predictor=request["predictor"], sequence_length=len(request["sequence"]),
                              finite_CA=finite, unknown_CA=len(request["sequence"]) - finite,
                              output_sha256=state["output_sha"], seconds=state["seconds"], max_memory_bytes=state["max_memory"]))
    unstarted = sorted(set(physical) - set(states))
    nonterminal = sorted(k for k, r in states.items() if r["state"] not in ("succeeded", "cache_hit", "failed"))
    if args.final and (unstarted or nonterminal or records_pending_write):
        raise RuntimeError("final identity audit requires every frozen physical request to be terminal with complete metadata")
    report = dict(status="FINAL_ARTIFACT_IDENTITIES_VERIFIED" if args.final else "INCREMENTAL_ARTIFACT_IDENTITIES_VERIFIED",
        confirmation_freeze_sha256=freeze_sha, audit_source_sha256=sha(__file__), observed_at_UTC=datetime.now(timezone.utc).isoformat(),
        frozen_logical_outputs=len(logical), frozen_physical_requests=len(physical), logical_maps_verified=len(logical),
        calls_charged=snapshot["consumed"], attempt_states=dict(Counter(r["state"] for r in snapshot["attempts"])),
        physical_latest_states=dict(Counter(r["state"] for r in states.values())), verified_artifacts=len(artifacts),
        unstarted_physical_requests=len(unstarted), nonterminal_requests=len(nonterminal),
        records_pending_write=records_pending_write, ledger_categories=snapshot["categories"],
        known_and_unknown_CA=dict(known=sum(r["finite_CA"] for r in artifacts), unknown=sum(r["unknown_CA"] for r in artifacts)),
        per_predictor={p: dict(Counter(r["state"] for k, r in states.items() if physical[k]["predictor"] == p))
                       for p in ("esmfold", "boltz2")},
        quality_metrics_used_for_filtering=False, failed_requests_retained=True, new_prediction_calls=0, artifacts=artifacts)
    out = "reports/delepref_v8_structure_audit/"
    suffix = "final" if args.final else "snapshot"
    write(out + suffix + ".json", report)
    print(json.dumps({k: report[k] for k in ("status", "calls_charged", "verified_artifacts", "physical_latest_states", "unstarted_physical_requests", "per_predictor")}))


if __name__ == "__main__":
    main()
