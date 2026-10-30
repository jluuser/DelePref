"""Freeze 2x2 diagnostic outputs and reuse exact existing ESMFold identities."""
import argparse
import hashlib
import json
from pathlib import Path
from recomb_confirm_v2_diagnostics import read,sha,write


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def main(root):
    out=root/"reports/recomb_confirm_v2_span_catalog"
    if (out/"completion.json").exists():
        raise FileExistsError("Diagnostic already released")
    planning=root/"reports/recomb_confirm_v2_span_diagnostic"
    done=read(planning/"planning_completion.json")
    for name,h in done["files"].items():
        if name.endswith(".log"):
            continue
        if sha(planning/name)!=h:
            raise ValueError("Diagnostic output changed: "+name)
    selected=read(planning/"q3_selected_outputs.json")
    panel={r["protein_id"]:r for r in read(root/"reports/recomb_benign_scope_20261017/general_panel.json")}
    profiles=read(root/"reports/recomb_benign_prediction_catalog_20261017/predictor_profiles.json")
    profile=profiles["esmfold"]
    oldphysical=read(root/"reports/recomb_benign_prediction_catalog_20261017/physical_requests.json")
    oldaudit={r["physical_id"]:r for r in read(root/"reports/recomb_benign_structure_evaluation_20261017/prediction_identity_audit.json")}
    old={p["task_id"]:p for p in oldphysical}
    weights=next(p["weights_sha256"] for p in oldphysical if p["predictor"]=="esmfold")
    requests={};logical=[];cache={}
    for row in selected:
        pid=row["protein_id"];p=panel[pid];seq=row["edited_sequence"]
        if hashlib.sha256(p["sequence"].encode()).hexdigest()!=row["sequence_sha256"]:
            raise ValueError("Unreviewed or changed diagnostic WT")
        spans=json.loads(row["deleted_intervals"])
        if len(spans)>3 or sum(b-a+1 for a,b in spans)!=round(.2*len(p["sequence"])):
            raise ValueError("Diagnostic constraints differ")
        deleted={i for a,b in spans for i in range(a,b+1)}
        mapping=[];j=0
        for i in range(len(p["sequence"])):
            mapping.append(None if i in deleted else j)
            if i not in deleted:j+=1
        rebuilt="".join(a for i,a in enumerate(p["sequence"]) if i not in deleted)
        if rebuilt!=seq:raise ValueError("Diagnostic edited sequence mismatch")
        signature=dict(sequence_sha256=hashlib.sha256(seq.encode()).hexdigest(),weights_sha256=weights,
            profile_sha256=digest(profile),seed=profile["seed"],sample=profile["sample"])
        task="esmfold_"+digest(signature);chains=[dict(id="A",sequence=seq)]
        request=dict(signature,task_id=task,predictor="esmfold",profile_name="esmfold",sequence=seq,
            chains=chains,canonical_chains_sha256=digest(chains),category="main_or_application_final_output",
            wt_variant="variant",requires_all_atom_output=False)
        requests[task]=request
        if task in old:
            if old[task]["chains"]!=chains:raise ValueError("Identity collision")
            artifact=oldaudit[task]
            if sha(artifact["artifact_path"])!=artifact["artifact_sha256"]:
                raise ValueError("Cache artifact changed")
            cache[task]=dict(artifact_path=artifact["artifact_path"],artifact_sha256=artifact["artifact_sha256"],
                             prior_prediction_identity=task)
        record=dict(protein_id=pid,cohort="general",method=row["method"],budget=.2,
            evaluation_role="development_cap_only_diagnostic",profile_name="esmfold",predictor="esmfold",
            physical_id=task,WT_sha256=p["sequence_sha256"],edited_sequence_sha256=signature["sequence_sha256"],
            correspondence=dict(wt_to_edit=mapping,position_states=["deleted" if i in deleted else "stable" for i in range(len(mapping))]),
            native_to_prediction_chains=[dict(native_chain_id=p.get("reference_chain","A"),prediction_chain_id="A",role="design")])
        logical.append(dict(record,request_id=digest(record)))
    if len(logical)!=40:raise ValueError("Incomplete q3 diagnostic release")
    write(out/"physical_requests.json",list(requests.values()))
    write(out/"logical_requests.json",logical)
    write(out/"compatible_cache.json",cache)
    write(out/"predictor_profiles.json",dict(esmfold=profile))
    protocol=read(root/"reports/recomb_benign_prediction_catalog_20261017/statistical_protocol.json")
    protocol.update(status="CAP_ONLY_DEVELOPMENT_NOT_CONFIRMATION",main_task_qmax=None)
    protocol["compute"]["structure_call_limit"]=None
    protocol["compute"]["structure_categories"]={k:None for k in protocol["compute"]["structure_categories"]}
    write(out/"statistical_protocol.json",protocol)
    files=["analysis/recomb_confirm_v2_span_release.py","analysis/recomb_confirm_v2_fold_worker.py",
        "analysis/recomb_benign_fold_worker_20261017.py","slurm/recomb_confirm_v2_span_fold.sbatch",
        "reports/recomb_confirm_v2_span_diagnostic/planning_completion.json",
        "reports/recomb_confirm_v2_span_diagnostic/q3_selected_outputs.json"]
    files += [str(p.relative_to(root)).replace("\\","/") for p in out.glob("*.json")]
    freeze=dict(status="DEVELOPMENT_SPAN_OUTPUTS_PREDICTION_RELEASED",fold_allowed=True,
        all_selected_outputs_frozen=True,main_task_qmax=None,new_training=False,
        files=[dict(path=str(root/p),sha256=sha(root/p)) for p in files])
    fp=root/"configs/recomb_confirm_v2_span_prediction_freeze.json"
    if fp.exists():raise FileExistsError(fp)
    write(fp,freeze)
    report=dict(status="DEVELOPMENT_SPAN_PREDICTIONS_FROZEN",logical=len(logical),physical=len(requests),
        verified_cache=len(cache),missing=len(requests)-len(cache),freeze_sha256=sha(fp))
    write(out/"completion.json",report);print(json.dumps(report))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    main(p.parse_args().root)
