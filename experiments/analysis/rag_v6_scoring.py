"""Frozen CPU integration of v6 homolog evidence and the original DelePref U."""
import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

import rag_v6_features as features
import rag_v6_retrieval as retrieval

FREEZE = "configs/rag_v6_scoring_freeze.json"
BASE = "reports/rag_v6"
SOURCES = ["analysis/rag_v6_scoring.py", "analysis/rag_v6_features.py",
           "analysis/tests/test_rag_v6_scoring.py", "slurm/rag_v6_scoring_cpu.sbatch"]
ARMS = {"U": 0, "U_S": len(features.S_NAMES),
        "U_S_M": len(features.S_NAMES)+len(features.M_NAMES), "U_S_M_J": len(features.NAMES)}
CONTRACT = dict(version="v6_direct_integration", feature_names=features.NAMES, profile=features.PROFILE,
                arms=ARMS, ridge_lambda=1., intercept=False, fixed_U_coefficient=1.,
                pairs="all_unordered_same_WT_same_budget_including_ties_and_fallback",
                weights="equal_WT_then_budget_then_candidate_pair",
                target="cached_complete_T_bounds_equal", primary_budget=.2,
                primary_comparison="U_S_M_J-U_S_M", bootstrap_replicates=10000,
                bootstrap_seed=20261015, mixed_train_WT_min=5, mixed_train_pools_min=10,
                rank_relative_tolerance=1e-8, calibration="all_arms_rank_pair_design_equals_supported_feature_rank",
                insufficient_mixed_calibration="mixed_pools_whole_pool_fallback_U",
                q1_joint_zero="after_standardization", confirmation_used_for_fit=False,
                validation_is_reused_development=True, hyperparameter_search=False,
                GPU_jobs=0, new_structure_predictions=0, new_PLM_forwards=0, paper_changes=False)


def lines(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def groups(rows):
    result = {}
    for i, row in enumerate(rows):
        result.setdefault((row["protein_id"], float(row["budget"])), []).append(i)
    return result


def balanced_weights(rows):
    pools = groups(rows)
    per_wt = {}
    for p, b in pools:
        per_wt.setdefault(p, set()).add(b)
    w = np.zeros(len(rows))
    for (p, _), ids in pools.items():
        w[ids] = 1/len(per_wt)/len(per_wt[p])/len(ids)
    return w


def scaler(x, valid, weights):
    w = weights[:, None]*valid
    n = w.sum(0)
    mu = np.divide((x*w).sum(0), n, out=np.zeros(x.shape[1]), where=n>0)
    var = np.divide(((x-mu)**2*w).sum(0), n, out=np.zeros(x.shape[1]), where=n>0)
    sd = np.sqrt(var)
    sd[sd<1e-12] = 1.
    return mu, sd


def standardize(x, valid, mu, sd):
    # Mask AFTER centering: q=1/unknown joint must stay exactly zero.
    return np.where(valid, (x-mu)/sd, 0.)


def pair_design(z, gate, rows):
    ds, ids, weights = [], [], []
    pools = groups(rows)
    budgets = {}
    for p, b in pools:
        budgets.setdefault(p, set()).add(b)
    for (p, _), pool in pools.items():
        pairs = list(itertools.combinations(pool, 2))
        if not pairs:
            raise ValueError("training pool has fewer than two candidates")
        weight = 1/len(budgets)/len(budgets[p])/len(pairs)
        for i, j in pairs:
            ds.append(gate[i]*z[i]-gate[j]*z[j])
            ids.append((i, j)); weights.append(weight)
    return np.array(ds), np.array(ids), np.array(weights)


def numeric_rank(x):
    singular = np.linalg.svd(x, compute_uv=False)
    cutoff = CONTRACT["rank_relative_tolerance"]*max(1., float(singular[0]) if len(singular) else 0.)
    return int((singular>cutoff).sum()), singular.tolist()


def calibration_audit(z, evidence, rows):
    mixed = [ids for ids in groups(rows).values() if 0 < int(evidence[ids].sum()) < len(ids)]
    wt = {rows[ids[0]]["protein_id"] for ids in mixed}
    base, _, pair_weights = pair_design(z, evidence, rows)
    ranks = {}
    for arm, count in ARMS.items():
        if not count:
            continue
        masked = z.copy(); masked[:, count:] = 0.
        d = base.copy(); d[:, count:] = 0.
        dr, sv = numeric_rank(d*np.sqrt(pair_weights[:, None]))
        zr, _ = numeric_rank(masked*evidence[:, None])
        ranks[arm] = dict(pair_rank=dr, supported_feature_rank=zr, singular_values=sv, anchored=dr==zr)
    enough = len(wt)>=CONTRACT["mixed_train_WT_min"] and len(mixed)>=CONTRACT["mixed_train_pools_min"]
    return dict(allow_mixed_residual=bool(enough and all(r["anchored"] for r in ranks.values())),
                mixed_pools=len(mixed), mixed_WT=len(wt), ranks=ranks,
                mixed_pairs=sum(int(evidence[ids].sum())*(len(ids)-int(evidence[ids].sum())) for ids in mixed))


def final_gate(evidence, rows, allow_mixed):
    gate = evidence.astype(float).copy()
    if not allow_mixed:
        for ids in groups(rows).values():
            if 0 < int(evidence[ids].sum()) < len(ids):
                gate[ids] = 0.
    return gate


def fit_arms(z, gate, u, target, rows, s_u, s_t):
    d, pairs, weights = pair_design(z, gate, rows)
    y = (target[pairs[:, 0]]-target[pairs[:, 1]])/s_t - (u[pairs[:, 0]]-u[pairs[:, 1]])/s_u
    models = {}
    for arm, count in ARMS.items():
        design = d.copy(); design[:, count:] = 0.
        a = design.T@(weights[:, None]*design) + np.eye(z.shape[1])*CONTRACT["ridge_lambda"]
        beta = np.linalg.solve(a, design.T@(weights*y)) if count else np.zeros(z.shape[1])
        residual = design@beta-y
        models[arm] = dict(theta=beta.tolist(), pair_loss=float(np.sum(weights*residual**2)),
                           ridge_penalty=float(np.sum(beta**2)), pairs=len(pairs))
    return models


def predict(models, z, gate, utility, s_u):
    return {arm: utility+s_u*gate*(z@np.asarray(model["theta"])) for arm, model in models.items()}


def targets(path, requested_ids):
    result = {}
    for row in lines(path):
        if row["candidate_id"] not in requested_ids:
            continue
        if row["unknown_pairs"] or abs(row["T_upper"]-row["T_lower"])>1e-12:
            raise ValueError("frozen legal candidate has uncertain geometry; do not silently filter")
        result[row["candidate_id"]] = float(row["T_lower"])
    if set(result) != set(requested_ids):
        raise ValueError("missing pinned candidate target")
    return result


def legal_rows(root):
    rows = lines(root/retrieval.MANIFEST)
    selected = [features.pure_candidate(r) for r in rows if 1<=len(features.spans(r["mask"]))<=3]
    assignments = json.loads((root/retrieval.SPLITS).read_text())["assignment"]
    for row in selected:
        row["split"] = assignments[row["protein_id"]]
    selected.sort(key=lambda x: x["candidate_id"])
    if len(selected)!=1083 or sum(r["split"]=="train" for r in selected)!=908:
        raise ValueError("legal candidate roster differs from mask audit")
    return selected


def prepare(root):
    raw = json.loads((root/retrieval.FREEZE).read_text())
    retrieval.load_verified(root, retrieval.sha(root/retrieval.FREEZE))
    old = json.loads((root/"configs/execution_freeze.json").read_text())
    pin_names = [retrieval.MANIFEST, retrieval.SPLITS, "reports/label_candidates.jsonl"]
    pins = {p: old["input_hashes"][p] for p in pin_names}
    report = json.loads((root/"reports/utility_baseline/report.json").read_text())
    if report["weights"] != dict(global_similarity=.35,retained_similarity=.45,junction_plausibility=.20):
        raise ValueError("original utility weights differ")
    pins["reports/utility_baseline/candidates.jsonl"] = report["output_sha256"]
    pins["reports/utility_baseline/report.json"] = retrieval.sha(root/"reports/utility_baseline/report.json")
    conf_relative = "reports/rag_v6_confirmation/candidates.jsonl"
    if (root/conf_relative).exists():
        pins[conf_relative] = retrieval.sha(root/conf_relative)
    retrieval.verify_hashes(root, pins)
    rows = legal_rows(root)
    smoke_ids = sorted({r["protein_id"] for r in rows if r["split"]=="train"})[:6]
    checked = 0
    for row in rows:
        if row["protein_id"] in smoke_ids:
            f = features.plan_features(row["wt_sequence"], row["mask"], {"representatives": []})
            if f["evidence"] or f["joint_valid"] or f["values"][-1]!=0:
                raise ValueError("empty retrieval fallback failed")
            checked += 1
    out = root/BASE; out.mkdir(exist_ok=False)
    seal = dict(status="RAG_V6_SCORING_FROZEN", contract=CONTRACT, input_hashes=pins,
                source_hashes={p: retrieval.sha(root/p) for p in SOURCES},
                retrieval_freeze_sha256=retrieval.sha(root/retrieval.FREEZE),
                smoke=dict(train_WT=smoke_ids, candidates_checked=checked, labels_read=False),
                query_counts=dict(train=50, validation=10, confirmation=32))
    retrieval.write_json(root/FREEZE, seal, atomic=True)
    retrieval.write_json(out/"technical_smoke.json", seal["smoke"], atomic=True)
    return dict(freeze_sha256=retrieval.sha(root/FREEZE), smoke=seal["smoke"])


def verified(root, checksum):
    retrieval.verify_hashes(root, {FREEZE: checksum})
    config = json.loads((root/FREEZE).read_text())
    if config["contract"]!=CONTRACT:
        raise ValueError("scoring contract changed")
    for name in ("source_hashes", "input_hashes"):
        retrieval.verify_hashes(root, config[name])
    retrieval.load_verified(root, config["retrieval_freeze_sha256"])
    return config


def execute(root, checksum):
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("CUDA_VISIBLE_DEVICES")!="" or os.environ.get("SLURM_CPUS_PER_TASK")!="8":
        raise ValueError("CPU Slurm allocation required")
    if any(os.environ.get(k, "") not in ("", "N/A", "NoDevFiles") for k in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS")):
        raise ValueError("unexpected GPU allocation")
    config = verified(root, checksum)
    out = root/BASE
    retrieval.write_json(out/"execution_claim.json", dict(job_id=os.environ["SLURM_JOB_ID"], started=time.time()))
    started = time.time()
    complete = json.loads((root/retrieval.BASE/"completion.json").read_text())
    if complete["freeze_sha256"]!=config["retrieval_freeze_sha256"]:
        raise ValueError("wrong retrieval run")
    hit_path = root/retrieval.BASE/"search/hits.tsv"
    retrieval.verify_hashes(root, {str(hit_path): complete["hits_sha256"]})
    query_map = json.loads((root/retrieval.BASE/"prepared/query_map.json").read_text())
    rows = legal_rows(root)
    excluded = {features.digest(r["wt_sequence"]) for r in rows} | {features.digest(r["sequence"]) for r in rows}
    # Known designed confirmation variants are input identities, never outcomes.
    conf_path = root/"reports/rag_v6_confirmation/candidates.jsonl"
    if "reports/rag_v6_confirmation/candidates.jsonl" in config["input_hashes"]:
        excluded.update(features.digest(r["sequence"]) for r in lines(conf_path))
    excluded.update(q["wt_sequence_sha256"] for q in query_map)
    profiles = features.build_profiles(query_map, hit_path, excluded)
    retrieval.write_json(out/"profiles.json", profiles, atomic=True)
    fs = [features.plan_features(r["wt_sequence"], r["mask"], profiles[r["protein_id"]]) for r in rows]
    x = np.array([f["values"] for f in fs]); evidence = np.array([f["evidence"] for f in fs], dtype=bool)
    valid = np.repeat(evidence[:, None], len(features.NAMES), axis=1)
    valid[:, -1] &= np.array([f["joint_valid"] for f in fs])
    train_ix = np.array([i for i,r in enumerate(rows) if r["split"]=="train"])
    valid_ix = np.array([i for i,r in enumerate(rows) if r["split"]=="validation"])
    train_rows = [rows[i] for i in train_ix]
    weights = balanced_weights(train_rows)
    mu, sd = scaler(x[train_ix], valid[train_ix], weights)
    z = standardize(x, valid, mu, sd)
    audit = calibration_audit(z[train_ix], evidence[train_ix], train_rows)
    gate = final_gate(evidence, rows, audit["allow_mixed_residual"])
    utility = {r["candidate_id"]: float(r["utility"]) for r in lines(root/"reports/utility_baseline/candidates.jsonl")}
    u = np.array([utility[r["candidate_id"]] for r in rows])
    if not np.isfinite(u).all() or np.any((u<0)|(u>1)):
        raise ValueError("nonfinite or out-of-range original utility")
    train_targets = targets(root/"reports/label_candidates.jsonl", {r["candidate_id"] for r in train_rows})
    t = np.array([train_targets[r["candidate_id"]] for r in train_rows])
    s_u = float(np.sqrt(np.sum(weights*(u[train_ix]-np.sum(weights*u[train_ix]))**2)))
    s_t = float(np.sqrt(np.sum(weights*(t-np.sum(weights*t))**2)))
    if min(s_u, s_t)<1e-12:
        raise ValueError("training target scale is degenerate")
    models = fit_arms(z[train_ix], gate[train_ix], u[train_ix], t, train_rows, s_u, s_t)
    artifact = dict(models=models, mean=mu.tolist(), scale=sd.tolist(), s_U=s_u, s_T=s_t,
                    calibration=audit, contract=CONTRACT, freeze_sha256=checksum)
    retrieval.write_json(out/"models.json", artifact, atomic=True)
    predictions = predict(models, z, gate, u, s_u)
    scored = []
    for i, row in enumerate(rows):
        scored.append(dict(candidate_id=row["candidate_id"], protein_id=row["protein_id"], budget=row["budget"],
                           split=row["split"], features=fs[i], raw_evidence=bool(evidence[i]), gate=float(gate[i]),
                           scores={a:float(v[i]) for a,v in predictions.items()}))
    with (out/"scores.jsonl").open("x", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, sort_keys=True)+"\n")
    score_sha = retrieval.sha(out/"scores.jsonl")
    # Validation target access begins only after model and scores are saved.
    validation_rows = [rows[i] for i in valid_ix]
    truth = targets(root/"reports/label_candidates.jsonl", {r["candidate_id"] for r in validation_rows})
    decisions = []
    for (pid, budget), ids in groups(rows).items():
        if rows[ids[0]]["split"]!="validation":
            continue
        oracle = max(truth[rows[i]["candidate_id"]] for i in ids)
        selections = {}
        for arm, values in predictions.items():
            best = min(ids, key=lambda i: (-values[i], rows[i]["candidate_id"]))
            selected_t = truth[rows[best]["candidate_id"]]
            selections[arm] = dict(candidate_id=rows[best]["candidate_id"], T=selected_t, regret=oracle-selected_t)
        decisions.append(dict(protein_id=pid, budget=budget, pool_size=len(ids), selections=selections,
                              raw_supported=int(evidence[ids].sum()), final_supported=int(gate[ids].sum())))
    summary = {}
    for budget in (.2, .3, .4):
        part = [r for r in decisions if r["budget"]==budget]
        summary[str(budget)] = {arm: dict(selected_T=float(np.mean([r["selections"][arm]["T"] for r in part])),
                                        regret=float(np.mean([r["selections"][arm]["regret"] for r in part]))) for arm in ARMS}
    primary = [r for r in decisions if r["budget"]==.2]
    delta = np.array([r["selections"]["U_S_M_J"]["T"]-r["selections"]["U_S_M"]["T"] for r in primary])
    rng = np.random.default_rng(CONTRACT["bootstrap_seed"])
    boot = delta[rng.integers(len(delta), size=(CONTRACT["bootstrap_replicates"],len(delta)))].mean(1)
    report = dict(status="RAG_V6_DEVELOPMENT_COMPLETE", seconds=time.time()-started, freeze_sha256=checksum,
                  score_sha256=score_sha, models_sha256=retrieval.sha(out/"models.json"), calibration=audit,
                  retrieval_hits_sha256=complete["hits_sha256"], profiles_sha256=retrieval.sha(out/"profiles.json"),
                  counts=dict(train=908, validation=175, development_WT=60, confirmation_profiles=32,
                              supported_train=int(evidence[train_ix].sum()), supported_validation=int(evidence[valid_ix].sum()),
                              joint_valid_train=int(valid[train_ix,-1].sum()), joint_valid_validation=int(valid[valid_ix,-1].sum())),
                  primary=dict(WT=len(primary), joint_minus_marginal=float(delta.mean()),
                               conditional_CI95=np.quantile(boot,[.025,.975]).tolist(),
                               positive_WT=int((delta>0).sum()), negative_WT=int((delta<0).sum())),
                  summary=summary, decisions=decisions, GPU_used=False, new_PLM_forwards=0,
                  new_structure_predictions=0, paper_changes=False, independent_confirmation=False,
                  interpretation="reused historical development; null mechanisms not yet evaluated; no hyperparameter search")
    retrieval.write_json(out/"development_report.json", report, atomic=True)
    print(json.dumps({k:v for k,v in report.items() if k not in ("decisions", "calibration")}), flush=True)
    return report["status"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--freeze-sha256")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        print(json.dumps(prepare(root) if args.prepare else execute(root, args.freeze_sha256)), flush=True)
    except Exception as error:
        if not args.prepare and (root/BASE).exists() and not (root/BASE/"failure.json").exists():
            retrieval.write_json(root/BASE/"failure.json",dict(status="FAILED_NO_AUTOMATIC_RETRY",type=type(error).__name__,error=str(error)),atomic=True)
        raise


if __name__=="__main__":
    main()
