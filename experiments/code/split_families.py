"""Family/source/sequence components with explicit all-pairs cross-set verification."""
import argparse
import collections
import json
import subprocess
from pathlib import Path
import Bio
from Bio.Align import PairwiseAligner, substitution_matrices
import numpy as np
from common import AUTHORIZED_ROOT, confined_path, read_jsonl, sha_file, write_json, write_jsonl


class Components:
    def __init__(self, names):
        self.parent = {n: n for n in names}

    def find(self, name):
        while self.parent[name] != name:
            self.parent[name] = self.parent[self.parent[name]]
            name = self.parent[name]
        return name

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def make_aligner(config):
    aligner = PairwiseAligner()
    aligner.mode = config["mode"]
    aligner.substitution_matrix = substitution_matrices.load(config["substitution_matrix"])
    aligner.open_gap_score = config["open_gap_score"]
    aligner.extend_gap_score = config["extend_gap_score"]
    return aligner


def similarity(a, b, aligner):
    alignment = aligner.align(a, b)[0]
    indices = alignment.indices
    both = (indices[0] >= 0) & (indices[1] >= 0)
    matches = sum(a[i] == b[j] for i, j in zip(indices[0, both], indices[1, both]))
    return {"identity": matches / min(len(a), len(b)),
            "coverage_a": float(np.count_nonzero(indices[0] >= 0) / len(a)),
            "coverage_b": float(np.count_nonzero(indices[1] >= 0) / len(b)),
            "matches": int(matches), "alignment_score": float(alignment.score)}


def threshold_hit(stats, config):
    return stats["identity"] >= config["identity"] and min(stats["coverage_a"], stats["coverage_b"]) >= config["coverage_each"]


def unique_proteins(rows):
    proteins = {}
    for row in rows:
        pid = row["protein_id"]
        if pid in proteins and proteins[pid]["wt_sequence_sha256"] != row["wt_sequence_sha256"]:
            raise ValueError("protein ID has multiple WT sequences")
        proteins[pid] = row
    return proteins


def split(protocol_path, manifest, output):
    protocol = json.loads(Path(protocol_path).read_text())
    if Path(protocol["root"]).resolve() != AUTHORIZED_ROOT.resolve():
        raise ValueError("unauthorized protocol root")
    output = confined_path(output)
    config = protocol["split"]
    rows = read_jsonl(manifest); proteins = unique_proteins(rows); names = sorted(proteins)
    components = Components(names); initial_edges = []
    for key in ["cath_h_superfamily", "wt_sequence_sha256"]:
        groups = collections.defaultdict(list)
        for pid, row in proteins.items():
            groups[row[key]].append(pid)
        for value, members in groups.items():
            if not value:
                raise ValueError("missing family/source identity")
            for pid in members[1:]:
                components.union(members[0], pid)
                initial_edges.append({"a": members[0], "b": pid, "reason": key})
    sources = collections.defaultdict(list)
    for pid, row in proteins.items():
        sources[(row["pdb_id"], row["reference_auth_chain"])].append(pid)
    for members in sources.values():
        for pid in members[1:]:
            components.union(members[0], pid)
            initial_edges.append({"a": members[0], "b": pid, "reason": "original_pdb_chain"})
    work = AUTHORIZED_ROOT / "manifests/similarity"
    confined_path(work); work.mkdir(parents=True, exist_ok=True)
    fasta = confined_path(work / "development.fasta")
    fasta.write_text("".join(f">{pid}\n{proteins[pid]['wt_sequence']}\n" for pid in names))
    cdhit_out = confined_path(work / "development_cdhit40")
    confined_path(work / "development_cdhit40.clstr")
    command = [protocol["environments"]["cdhit"], "-i", str(fasta), "-o", str(cdhit_out), "-G", "1", "-c", str(config["identity"]), "-n", "2", "-aS", str(config["coverage_each"]), "-aL", str(config["coverage_each"]), "-d", "0", "-T", "4", "-M", "4000"]
    result = subprocess.run(command, text=True, capture_output=True)
    confined_path(work / "cdhit.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("CD-HIT failed; see similarity/cdhit.log")
    current = []
    for line in (work / "development_cdhit40.clstr").read_text().splitlines() + [">end"]:
        if line.startswith(">"):
            for pid in current[1:]:
                components.union(current[0], pid)
            current = []
        else:
            current.append(line.split(">")[1].split("...")[0])
    aligner = make_aligner(config["alignment"]); all_pairs = []; hits = []
    for n, a in enumerate(names):
        for b in names[n + 1:]:
            stats = similarity(proteins[a]["wt_sequence"], proteins[b]["wt_sequence"], aligner)
            item = {"a": a, "b": b, **stats}
            all_pairs.append(item)
            if threshold_hit(stats, config):
                components.union(a, b); hits.append(item)
    groups = collections.defaultdict(list)
    for pid in names:
        groups[components.find(pid)].append(pid)
    order = sorted(groups)
    rng = np.random.default_rng(protocol["seed"]); rng.shuffle(order)
    target = round(len(names) * config["train_fraction"]); train = []; validation = []
    for key in order:
        if len(train) < target:
            train.extend(groups[key])
        else:
            validation.extend(groups[key])
    if not train or not validation:
        raise ValueError("family components do not permit nonempty train/validation")
    assignment = {pid: "train" for pid in train} | {pid: "validation" for pid in validation}
    cross = [x for x in all_pairs if assignment[x["a"]] != assignment[x["b"]]]
    violations = [x for x in cross if threshold_hit(x, config)]
    if violations:
        raise ValueError("explicit train/validation similarity violations remain")
    write_jsonl(work / "all_development_pairs.jsonl", all_pairs)
    write_jsonl(work / "threshold_hits.jsonl", hits)
    write_jsonl(work / "train_validation_pairs.jsonl", cross)
    report = {"status": "SPLIT_FROZEN", "manifest_sha256": sha_file(manifest), "protocol_sha256": sha_file(protocol_path),
              "seed": protocol["seed"], "assignment": assignment, "component": {p: components.find(p) for p in names},
              "components": dict(groups), "train_count": len(train), "validation_count": len(validation),
              "cdhit_command": command, "cdhit_binary_sha256": sha_file(protocol["environments"]["cdhit"]),
              "alignment": config["alignment"], "biopython_version": Bio.__version__,
              "coverage_definition": "non-gap sequence residues within selected local alignment, including internal gap-opposed residues, divided by full sequence length",
              "identity_denominator": "shorter_full_sequence_length", "pairwise_comparisons": len(all_pairs),
              "cross_set_comparisons": len(cross), "cross_set_threshold_hits": violations,
              "highest_cross_set_pairs": sorted(cross, key=lambda x: -x["identity"])[:20],
              "legacy_policy_training_overlap": "not_assessed_optional_baseline_omitted_per_user",
              "confirmation_isolation": "pending_separate_WT_selection_and_all_pairs_check"}
    write_json(output, report)
    print(json.dumps({k: report[k] for k in ["status", "train_count", "validation_count", "pairwise_comparisons", "cross_set_comparisons"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", required=True)
    parser.add_argument("--manifest", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    if args.seed != 20261013:
        raise ValueError("v2 frozen seed is 20261013")
    split(args.protocol, args.manifest, args.output)
