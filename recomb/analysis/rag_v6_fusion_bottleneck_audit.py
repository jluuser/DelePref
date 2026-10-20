"""Read-only v6 calibration, candidate-distribution and evidence audit. No fits or GPU calls."""

import json, hashlib
from pathlib import Path
from collections import Counter
import numpy as np
root=Path("/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013")
def read(p): return json.loads((root/p).read_text())
def rows(p): return [json.loads(x) for x in (root/p).read_text().splitlines() if x.strip()]
def sha(p): return hashlib.sha256((root/p).read_bytes()).hexdigest()
freeze=read("configs/rag_v6_scoring_freeze.json")
for p,h in freeze["input_hashes"].items():
    if sha(p)!=h: raise ValueError("changed input "+p)
conf=read("reports/rag_v6_confirmation/selector_freeze_report.json")
if sha("reports/rag_v6_confirmation/scores.jsonl")!=conf["score_sha256"]:raise ValueError("changed confirmation score")
if sha("reports/rag_v6/scores.jsonl")!=read("reports/rag_v6/development_report.json")["score_sha256"]:raise ValueError("changed development score")
model=read("reports/rag_v6/models.json")
manifest={x["candidate_id"]:x for x in rows("manifests/development_primary.jsonl")}
cr={x["candidate_id"]:x for x in rows("reports/rag_v6_confirmation/candidates.jsonl")}
labels={x["candidate_id"]:x for x in rows("reports/label_candidates.jsonl")}
def stats(v):
    return dict(n=len(v),mean=float(np.mean(v)),min_median_max=np.quantile(v,[0,.5,1]).tolist()) if v else dict(n=0)
out=dict(status="READ_ONLY_FIXED_V6_FUSION_AUDIT",no_refit=True,new_GPU_jobs=0,confirmation_structure_outcomes_read=False,
         s_U=model["s_U"],s_T=model["s_T"],joint_effect_per_standard_deviation=model["s_U"]*model["models"]["U_S_M_J"]["theta"][-1])
for name,path,meta in [("development","reports/rag_v6/scores.jsonl",manifest),("confirmation","reports/rag_v6_confirmation/scores.jsonl",cr)]:
    data=rows(path);parts={}
    splits=sorted({x.get("split","confirmation") for x in data})
    for split in splits:
        part=[x for x in data if x.get("split","confirmation")==split]
        terminal=[x for x in part if meta[x["candidate_id"]]["mask"][0] or meta[x["candidate_id"]]["mask"][-1]]
        supported=[x for x in part if x["raw_evidence"]]
        joint=[x for x in part if x["features"]["joint_valid"]]
        parts[split]=dict(n=len(part),terminal=len(terminal),terminal_supported=sum(x["raw_evidence"] for x in terminal),
                         fallback_reasons=dict(Counter(x["features"]["fallback_reason"] for x in part if not x["raw_evidence"])),
                         marginal_gap_mean=stats([x["features"]["values"][7] for x in supported]),
                         positive_gap_candidate_count=sum(x["features"]["values"][7]>0 for x in supported),
                         joint_raw=stats([x["features"]["values"][-1] for x in joint]),
                         joint_valid=len(joint),
                         joint_vs_marginal_score_delta=stats([abs(x["scores"]["U_S_M_J"]-x["scores"]["U_S_M"]) for x in part]))
        if name=="development":
            bypool={}
            for x in part:
                if abs(x["budget"]-.2)<1e-12:bypool.setdefault(x["protein_id"],[]).append(x)
            poolout=[]
            for pid,pool in sorted(bypool.items()):
                target=lambda x:labels[x["candidate_id"]]["T_lower"]
                best_u=min(pool,key=lambda x:(-x["scores"]["U"],x["candidate_id"]))
                terminal_pool=[x for x in pool if meta[x["candidate_id"]]["mask"][0] or meta[x["candidate_id"]]["mask"][-1]]
                oracle=max(map(target,pool))
                terminal_oracle=max(map(target,terminal_pool)) if terminal_pool else None
                poolout.append(dict(protein_id=pid,pool_size=len(pool),U_T=target(best_u),oracle_T=oracle,
                                    U_regret=oracle-target(best_u),terminal_oracle_T=terminal_oracle,
                                    oracle_is_terminal_any=any(target(x)==oracle for x in terminal_pool),
                                    U_winner_terminal=bool(meta[best_u["candidate_id"]]["mask"][0] or meta[best_u["candidate_id"]]["mask"][-1])))
            parts[split]["budget_0.2"]=dict(WT=len(poolout),U_regret=stats([x["U_regret"] for x in poolout]),
                                         oracle_terminal_WT=sum(x["oracle_is_terminal_any"] for x in poolout),
                                         U_terminal_WT=sum(x["U_winner_terminal"] for x in poolout),per_WT=poolout)
    out[name]=parts



import json
from pathlib import Path
from collections import Counter
import numpy as np
r=Path("/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013")
def rows(p): return [json.loads(x) for x in (r/p).read_text().splitlines() if x.strip()]
m={x["candidate_id"]:x for x in rows("manifests/development_primary.jsonl")}
c={x["candidate_id"]:x for x in rows("reports/rag_v6_confirmation/candidates.jsonl")}
o={}
for name,path,meta in [("train","reports/rag_v6/scores.jsonl",m),("confirmation","reports/rag_v6_confirmation/scores.jsonl",c)]:
    data=[x for x in rows(path) if name!="train" or x["split"]=="train"]
    t=[x for x in data if meta[x["candidate_id"]]["mask"][0] or meta[x["candidate_id"]]["mask"][-1]]
    a=np.array([x["features"]["values"][:7] for x in data]);ta=np.array([x["features"]["values"][:7] for x in t])
    o[name]=dict(terminal_WT=len({x["protein_id"] for x in t}),pure_terminal_count=sum(abs(x["features"]["values"][4]+x["features"]["values"][5]-1)<1e-12 for x in t),
                terminal_q=dict(Counter(x["features"]["q"] for x in t)),N_fraction_minmax=[float(ta[:,4].min()),float(ta[:,4].max())],
                C_fraction_minmax=[float(ta[:,5].min()),float(ta[:,5].max())],terminal_total_fraction_minmax=[float((ta[:,4]+ta[:,5]).min()),float((ta[:,4]+ta[:,5]).max())])
out["terminal_coverage_detail"]=o


import json
from pathlib import Path
from collections import Counter
import numpy as np
root=Path("/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013")
p=json.loads((root/"reports/rag_v6/profiles.json").read_text())
rows=[json.loads(x) for x in (root/"reports/rag_v6_confirmation/candidates.jsonl").read_text().splitlines() if x.strip()]
aa=set("ACDEFGHIKLMNPQRSTVWY")
column_counts=Counter()
for r in rows:
    m=np.array(r["mask"],dtype=bool)
    reps=p[r["protein_id"]]["representatives"]
    canonical=np.array([[c in aa for c in h["projected"]] for h in reps],dtype=bool)
    known=canonical.sum(0)>=8 if reps else np.zeros(len(m),dtype=bool)
    valid=known[m].mean()>=.8 and known[~m].mean()>=.8
    is_terminal=bool(m[0] or m[-1]);generator=r["generator"]
    column_counts[generator+"_n"]+=1;column_counts[generator+"_column_support_available"]+=int(valid)
    if is_terminal:
        column_counts["terminal_including_mixed_n"]+=1;column_counts["terminal_including_mixed_column_support_available"]+=int(valid)


out["illustrative_column_support_audit"]=dict(column_counts)
out["illustrative_column_support_contract"]=dict(minimum_canonical_observations_per_column=8,deleted_and_retained_observed_column_fraction_min=.8,known_residues_only=True,no_terminal_gap_inference=True,model_fit=False)
out["input_sha256"]={p:sha(p) for p in ["configs/rag_v6_scoring_freeze.json","reports/rag_v6/models.json","reports/rag_v6/profiles.json","reports/rag_v6/scores.jsonl","reports/rag_v6_confirmation/scores.jsonl","manifests/development_primary.jsonl","reports/label_candidates.jsonl","reports/rag_v6_confirmation/candidates.jsonl"]}
print(json.dumps(out,sort_keys=True))
