"""CPU identity/import checks only: never execute a structure forward."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys


ROOT = Path("/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(paths):
    return [{"path": str(p.resolve()), "sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(paths)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor", choices=["esmfold", "boltz2"], required=True)
    args = parser.parse_args()
    out = ROOT / "reports/delepref_v8_predictors"
    out.mkdir(parents=True, exist_ok=True)
    report = {"predictor": args.predictor, "python": sys.executable,
              "structure_forward_executed": False, "weights_loaded": False,
              "new_structure_predictions": 0, "entry_sha256": sha(__file__)}
    import torch
    report["torch"] = torch.__version__
    if args.predictor == "esmfold":
        import transformers
        from transformers import AutoTokenizer, EsmForProteinFolding
        from transformers.models.esm.configuration_esm import EsmConfig
        directory = Path("/public/home/zhangyangroup/chengshiz/keyuan.zhou/delepath/checkpoints/esmfold_v1")
        files = rows(directory.iterdir())
        weight = next(r for r in files if r["path"].endswith("pytorch_model.bin"))
        assert weight["sha256"] == "2ee07356b125d1e3e57503c204111fd7323347fc4735d41d3caac57c2a78e116"
        config = EsmConfig.from_pretrained(directory, local_files_only=True)
        assert config.is_folding_model
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        ids = tokenizer(sequence, add_special_tokens=False)["input_ids"]
        assert len(ids) == len(sequence) and "".join(tokenizer.convert_ids_to_tokens(ids)) == sequence
        source_dir = Path(inspect.getfile(EsmForProteinFolding)).parent
        source = rows(source_dir.rglob("*.py"))
        report.update(version=transformers.__version__, files=files, implementation=source,
                      forward_signature=str(inspect.signature(EsmForProteinFolding.forward)),
                      tokenizer_no_special_token_roundtrip=True,
                      planned_profile={"num_recycles": 3, "chunk_size": 64, "esm_precision": "float16",
                                       "folding_head_precision": "float32", "batch_size": 1,
                                       "tf32_allowed": True, "seed": 20261017, "sample": 0,
                                       "MSA": "none_single_sequence", "device_class": "A100_80GB"})
    else:
        import boltz.main
        from boltz.model.models.boltz2 import Boltz2
        source_dir = Path(inspect.getfile(boltz.main)).parent
        weights = Path("/public/home/zhangyangroup/chengshiz/.boltz/boltz2_conf.ckpt")
        assert sha(weights) == "090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1"
        cache = weights.parent
        aa = "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
        assets = [cache / "ccd.pkl"] + [cache / "mols" / (x + ".pkl") for x in aa]
        report.update(version=importlib.metadata.version("boltz"), files=rows([weights]),
                      implementation=rows(source_dir.rglob("*.py")), standard_amino_acid_assets=rows(assets),
                      model_class=Boltz2.__module__ + "." + Boltz2.__name__,
                      planned_profile={"recycling_steps": 1, "sampling_steps": 50, "diffusion_samples": 1,
                                       "msa": "empty", "templates": "none", "num_workers": 0,
                                       "seed": 20261017, "sample": 0, "device_class": "A100_80GB",
                                       "affinity": False, "use_msa_server": False})
    report["status"] = "IDENTITY_AND_IMPORT_PASS_NO_FOLD"
    path = out / (args.predictor + "_preflight.json")
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "predictor": args.predictor,
                      "version": report["version"], "source_files": len(report["implementation"]), "report": str(path)}))


if __name__ == "__main__":
    main()
