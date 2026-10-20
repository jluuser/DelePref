"""Read-only recovery audit of the original DelePref assets retained under SurgEdit."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import yaml

BASE = Path("/public/home/zhangyangroup/chengshiz/keyuan.zhou/SurgEdit")
PUBLIC = BASE.parent / "DelePref"
EXPORT = BASE / "results/experiments/student_v3_pretraining_evaluation_readiness/final_freeze/student_v3_evaluation_step_00004000.pt"
PROTECTED = BASE / "results/experiments/student_v3_pretraining_evaluation_readiness/protected_training_asset_hashes_before.json"
TRAINING = BASE / "results/student_v3_esmc600m_same_count/final_readiness/corrected/training/v3_final_training"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def recover_bytes(relative):
    return subprocess.check_output(["git", "--git-dir=" + str(BASE / ".git"), "show", "239942a:" + relative])

def resolve(value):
    p = Path(str(value))
    return p if p.is_absolute() else BASE / p

torch.set_num_threads(1)
payload = torch.load(EXPORT, map_location="cpu", weights_only=True)
historical = json.loads((BASE / "results/experiments/student_v3_step004000_exp0_exp1_exp2/checkpoint_provenance.json").read_text())
assert sha(EXPORT) == historical["evaluation_checkpoint_sha256"]
assert payload["selected_step"] == historical["selected_step"] == 4000
assert sha(Path(payload["source_checkpoint_path"])) == payload["source_checkpoint_sha256"] == historical["source_checkpoint_sha256"]
assert sha(TRAINING / "checkpoint_selection.json") == payload["checkpoint_selection_sha256"]

protected = json.loads(PROTECTED.read_text())["files"]
source_checks = {}
for rel, expected in protected.items():
    try:
        raw = recover_bytes(rel)
        actual = hashlib.sha256(raw).hexdigest()
        source_checks[rel] = dict(git_commit="239942a", sha256=actual, expected_sha256=expected, match=actual == expected)
    except subprocess.CalledProcessError as error:
        source_checks[rel] = dict(error=str(error), match=False)

config_rel = "configs/final_readiness/deletion_preference_student_v3_final.yaml"
config_bytes = recover_bytes(config_rel)
config = yaml.safe_load(config_bytes)
canonical_config_sha = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
assert canonical_config_sha == payload["config_sha256"]
assert all(item["match"] for item in source_checks.values())

# Test all learned parameter names/shapes with the published architecture, without a PLM forward.
sys.path.insert(0, str(PUBLIC / "src"))
from delepref.student_model import FrozenWTEncoder, DeletionPreferenceStudent
encoder = FrozenWTEncoder(torch.nn.Identity(), None, hidden_size=1152, max_length=1022)
mc, pc = payload["model_config"], payload["policy_config"]
model = DeletionPreferenceStudent(
    encoder, hidden_size=1152, scorer_hidden_dim=mc["scorer_hidden_dim"],
    span_feature_dim=mc["span_feature_dim"], budget_hidden_dim=pc["budget_hidden_dim"],
    policy_hidden_dim=pc["policy_hidden_dim"], policy_span_hidden_dim=pc["span_hidden_dim"],
    qmax=pc["qmax"], min_span_len=pc["min_span_len"],
    max_span_len_fraction=pc["max_span_len_fraction"], dropout=mc["dropout"])
state = payload["student_head_state_dict"]
loaded = model.load_state_dict(state, strict=True)
assert not loaded.missing_keys and not loaded.unexpected_keys
assert all(torch.equal(model.state_dict()[key], value) for key, value in state.items())
assert all(torch.isfinite(value).all() for value in state.values())

data_checks = {}
for key, expected in payload["data_hashes"].items():
    value = config.get("data", {}).get(key)
    if value is None:
        value = config.get("integrity", {}).get(key)
    if isinstance(value, dict):
        value = value.get("path")
    if not value or not isinstance(value, str):
        data_checks[key] = dict(expected_sha256=expected, resolved_from_config=False)
        continue
    path = resolve(value)
    if path.is_file():
        actual = sha(path)
        data_checks[key] = dict(path=str(path), bytes=path.stat().st_size, sha256=actual,
                                expected_sha256=expected, match=actual == expected)
    else:
        data_checks[key] = dict(path=str(path), exists=False, expected_sha256=expected)

shards = {}
for key in ("base_pair_index", "b20_pair_index"):
    path = resolve(config["data"][key])
    with path.open(newline="") as handle:
        index = list(csv.DictReader(handle))
    missing, size_mismatches = [], []
    total_pairs = total_bytes = 0
    for row in index:
        target = resolve(row["output_csv"])
        total_pairs += int(row["num_pairs"])
        if not target.is_file():
            missing.append(str(target))
        else:
            size = target.stat().st_size
            total_bytes += size
            if size != int(row["output_size_bytes"]):
                size_mismatches.append(str(target))
    shards[key] = dict(index_path=str(path), shards=len(index),
                       indexed_pairs=total_pairs, existing_bytes=total_bytes,
                       missing=missing, size_mismatches=size_mismatches,
                       full_shard_content_hashes_recomputed=False)

assert all(item.get("match", False) for item in data_checks.values())
assert all(item["shards"] == 300 and not item["missing"] and not item["size_mismatches"] for item in shards.values())

report = dict(
    status="ORIGINAL_DELEPREF_ASSETS_LOCATED_AND_AUDITED",
    recovered_root=str(BASE), published_code=str(PUBLIC),
    evaluation_checkpoint=dict(path=str(EXPORT), sha256=sha(EXPORT), bytes=EXPORT.stat().st_size,
        selected_step=4000, matches_historical_provenance=True,
        source_checkpoint_path=payload["source_checkpoint_path"], source_checkpoint_sha256=payload["source_checkpoint_sha256"],
        schema_version=payload["schema_version"], student_version=payload["student_version"],
        policy_tensors=sum(k.startswith("policy.") for k in state),
        scorer_tensors=sum(k.startswith("scorer.") for k in state),
        parameters=sum(v.numel() for v in state.values())),
    published_head_compatibility=dict(strict_state_load=True, exact_tensor_equality=True,
        missing_keys=[], unexpected_keys=[], real_encoder_loaded=False, real_sequence_forward=False,
        note="Head compatibility only; published schema/config names differ from legacy export."),
    original_source_checks=source_checks,
    original_config=dict(git_commit="239942a", relative_path=config_rel,
        file_sha256=hashlib.sha256(config_bytes).hexdigest(), canonical_sha256=canonical_config_sha,
        expected_canonical_sha256=payload["config_sha256"], canonical_match=canonical_config_sha == payload["config_sha256"],
        model=config.get("model", {}), data=config.get("data", {}), integrity=config.get("integrity", {})),
    data_checks=data_checks, pair_shards=shards,
    training_run=json.loads((TRAINING / "run_summary.json").read_text()),
    gpu_jobs_started=0, plm_forwards=0, structure_predictions=0, original_assets_modified=False)
print(json.dumps(report, sort_keys=True))
