"""Pure-sequence confirmation candidates, using the inherited exact DP code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import rag_v6_features as features
import rag_v6_retrieval as retrieval

BASE = "reports/rag_v6_confirmation"
FREEZE = "configs/rag_v6_confirmation_freeze.json"
CONTRACT = dict(budget=.2, qmax=3, terminal_N_fractions=[1.,0.,.25,.5,.75],
                terminal_rounding="N=floor(a*k),C=k-N;k=round(beta*L)",
                DP_base="all_zero_logits", DP_requested=12, DP_max_attempts=96,
                DP_seed=20261015, DP_perturbation_scale=1., DP_keep_after_terminal_dedup=7,
                candidate_target=12, shortage="report_actual_K_no_WT_replacement",
                WT_structure_used=False, variant_labels_read=False, PLM_calls=0,
                structure_calls=0, original_checkpoint_used=False, paper_changes=False)


def terminal_masks(length, k):
    seen, result = set(), []
    for a in CONTRACT["terminal_N_fractions"]:
        n, mask = int(a*k), [0]*length
        c = k-n
        mask[:n] = [1]*n
        if c: mask[length-c:] = [1]*c
        key = tuple(mask)
        if key not in seen:
            result.append(mask); seen.add(key)
    return result


def prepare(root):
    conf = root/"manifests/confirmation_wt_frozen.jsonl"
    retrieval.verify_hashes(root,{str(conf):"7b7bfc84859aa4fb9a43390810b3d40e40d8bcdd640887f5442587240e91baa4"})
    protocol = json.loads((root/"configs/protocol_v2.json").read_text())
    source = Path(protocol["sources"]["delepref"])/"src/delepref"
    source_files = {str(p.relative_to(source)):retrieval.sha(p) for p in sorted(source.rglob("*.py"))}
    fingerprint = hashlib.sha256(json.dumps(source_files,sort_keys=True).encode()).hexdigest()
    snapshot = root/"provenance"/("rag_v6_delepref_"+fingerprint[:16])/"delepref"
    snapshot.mkdir(parents=True,exist_ok=False)
    for relative, checksum in source_files.items():
        output = snapshot/relative; output.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/relative,output)
        if retrieval.sha(output)!=checksum: raise ValueError("DP snapshot mismatch")
    out = root/BASE; out.mkdir(exist_ok=False)
    seal = dict(status="PURE_SEQUENCE_CANDIDATES_FROZEN",contract=CONTRACT,
                cohort_sha256=retrieval.sha(conf),source_directory=str(source),
                snapshot=str(snapshot),snapshot_hashes=source_files,
                code_hashes={p:retrieval.sha(root/p) for p in ["analysis/rag_v6_confirmation.py","analysis/rag_v6_features.py","slurm/rag_v6_confirmation_cpu.sbatch"]})
    retrieval.write_json(root/FREEZE,seal,atomic=True)
    return dict(freeze_sha256=retrieval.sha(root/FREEZE))


def execute(root, checksum):
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise ValueError("CPU Slurm allocation required")
    retrieval.verify_hashes(root,{FREEZE:checksum})
    config=json.loads((root/FREEZE).read_text())
    if config["contract"]!=CONTRACT: raise ValueError("candidate contract differs")
    retrieval.verify_hashes(root,config["code_hashes"])
    snapshot=Path(config["snapshot"])
    retrieval.verify_hashes(snapshot,config["snapshot_hashes"])
    cohort=root/"manifests/confirmation_wt_frozen.jsonl"
    retrieval.verify_hashes(root,{str(cohort):config["cohort_sha256"]})
    sys.path.insert(0,str(snapshot.parent))
    import torch
    from delepref.student_model import generate_strict_budget_plans,strict_delete_count
    torch.set_num_threads(1)
    out=root/BASE; start=time.time()
    retrieval.write_json(out/"execution_claim.json",dict(job_id=os.environ["SLURM_JOB_ID"],started=start))
    results,audit=[],[]
    for line in cohort.read_text().splitlines():
        original=json.loads(line)
        # Explicit whitelist: coordinates and structure selection fields are not accessed.
        pid,wt=original["protein_id"],original["wt_sequence"]
        length=len(wt); k=strict_delete_count(length,CONTRACT["budget"])
        masks=[("terminal",m) for m in terminal_masks(length,k)]
        seen={tuple(m) for _,m in masks}; terminals=len(masks)
        seed=CONTRACT["DP_seed"]+int(hashlib.sha256(pid.encode()).hexdigest()[:8],16)
        generated=generate_strict_budget_plans(torch.zeros(length,dtype=torch.float64),CONTRACT["budget"],
            num_candidates=CONTRACT["DP_requested"],qmax=3,seed=seed,
            perturbation_scale=1.,max_attempts=CONTRACT["DP_max_attempts"])
        for plan in generated.plans:
            mask=[0]*length
            for a,b in plan.deleted_intervals: mask[a:b+1]=[1]*(b-a+1)
            if tuple(mask) in seen: continue
            masks.append(("cold_DP",mask)); seen.add(tuple(mask))
            if len(masks)==CONTRACT["candidate_target"]: break
        for generator,mask in masks:
            sequence="".join(c for c,d in zip(wt,mask) if not d)
            cid=hashlib.sha256((pid+"\0"+features.digest(wt)+"\0"+"".join(map(str,mask))).encode()).hexdigest()
            row=dict(candidate_id=cid,protein_id=pid,budget=.2,wt_sequence=wt,wt_sequence_sha256=features.digest(wt),
                     sequence=sequence,sequence_sha256=features.digest(sequence),mask=mask,
                     generator=generator,num_spans=len(features.spans(mask)),deleted_count=sum(mask),
                     role="independent_confirmation",variant_labels_generated=False)
            features.pure_candidate(row)
            if not 1<=row["num_spans"]<=3: raise ValueError("inherited DP violated span cap")
            results.append(row)
        audit.append(dict(protein_id=pid,length=length,k=k,K=len(masks),terminal_unique=terminals,
                          DP_attempts=generated.attempts,DP_reason=generated.reason))
        print(json.dumps(audit[-1]),flush=True)
    with (out/"candidates.jsonl").open("x",encoding="utf-8") as handle:
        for row in results: handle.write(json.dumps(row,sort_keys=True)+"\n")
    report=dict(status="PURE_SEQUENCE_CONFIRMATION_CANDIDATES_COMPLETE",seconds=time.time()-start,
                freeze_sha256=checksum,candidates_sha256=retrieval.sha(out/"candidates.jsonl"),
                WT=len(audit),candidates=len(results),audit=audit,contract=CONTRACT)
    retrieval.write_json(out/"report.json",report,atomic=True)
    return {k:v for k,v in report.items() if k not in ("audit","contract")}


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--prepare",action="store_true"); parser.add_argument("--freeze-sha256")
    args=parser.parse_args()
    print(json.dumps(prepare(args.root) if args.prepare else execute(args.root,args.freeze_sha256)),flush=True)
