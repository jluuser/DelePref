#!/usr/bin/env python3
"""Render a clearly labeled favorable Binder illustration from frozen outputs.

This is visualization-only. It reads the already persisted DelePref, WT, and
Original SCISOR Boltz outputs for one frozen Binder case. It does not run
Boltz, inference, teacher scoring, or candidate generation.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/experiments/delepref_step004000_binder_favorable_case_visualization"
PAPER = ROOT / "results/figures/delepref"
PYMOL = Path("pymol")
BINDER_DelePref = ROOT / "results/experiments/delepref_step004000_binder"
BINDER_V12 = ROOT / "results/experiments/delepref_step004000_delepref_12_binder_interface_analysis"
BINDER_V16 = ROOT / "results/experiments/delepref_step004000_delepref_16_binder_coldstart_same_scorer"

CASE_ID = 6
PROTEIN_ID = "AB-Bind:1JRH:HL_I"
BINDER_CHAINS = ("A",)
TARGET_CHAINS = ("I",)
WT_BINDER_LENGTH = 347
EDITED_BINDER_LENGTH = 312
DELETED_LENGTH = 35
CLAIM_BOUNDARY = (
    "The method ranks candidate deletion plans by teacher-derived deletion "
    "perturbation preference and optional auxiliary risk evidence."
)
PROXY_BOUNDARY = (
    "This is a post-deletion complex-prediction proxy illustration, not "
    "experimental evidence of retained binding, Kd/Ki, or functional preservation."
)
PAPER_CAPTION = (
    "Predicted binder-target complexes for an illustrative case from the frozen "
    "Binder benchmark (AB-Bind:1JRH:HL_I). (A) Wild-type complex. (B) Complex "
    "predicted after the deletion proposed by Ours. (C) Complex predicted after "
    "the SCISOR deletion. Both edited binders contain 312 residues after deleting "
    "35 of 347 residues. For this post-hoc favorable example, Ours has higher "
    "Boltz confidence and pair-chain ipTM and lower inter-chain PAE than SCISOR. "
    "DeltaConf. is WT confidence minus edited confidence, so lower is better. "
    "Structures are independently predicted and aligned on the target chain for "
    "visualization. SCISOR is an external reference. These quantities are "
    "post-deletion complex-prediction proxies and do not establish experimental "
    "binding or functional preservation."
)

METHODS = (
    {
        "id": "wildtype_shared",
        "label": "Full-length",
        "short_label": "WT",
        "comparison_class": "WT_BASELINE",
        "color": "#2A9D8F",
        "pymol_color": "wt_binder",
    },
    {
        "id": "delepref_full_control",
        "label": "Ours",
        "short_label": "DelePref",
        "comparison_class": "SAME_PROTOCOL",
        "color": "#0072B2",
        "pymol_color": "delepref_full",
    },
    {
        "id": "original_SCISOR_reference",
        "label": "SCISOR",
        "short_label": "SCISOR",
        "comparison_class": "EXTERNAL_REFERENCE",
        "color": "#D55E00",
        "pymol_color": "scisor",
    },
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_status(path: Path, expected: str) -> None:
    actual = read_json(path / "run_summary.json").get("status")
    if actual != expected:
        raise RuntimeError(f"status mismatch for {path}: {actual!r} != {expected!r}")


def structure_from_confidence(confidence_path: Path) -> Path:
    name = confidence_path.name
    if not name.startswith("confidence_") or not name.endswith(".json"):
        raise RuntimeError(f"unexpected confidence filename: {confidence_path}")
    return confidence_path.with_name(name[len("confidence_") : -5] + ".cif")


def load_method_records() -> list[dict[str, Any]]:
    predictions = pd.read_csv(BINDER_V16 / "binder_prediction_details.csv")
    predictions = predictions[
        predictions["case_id"].eq(CASE_ID) & predictions["protein_id"].eq(PROTEIN_ID)
    ].copy()
    interfaces = pd.read_csv(BINDER_V12 / "binder_interface_case_metrics.csv")
    interfaces = interfaces[
        interfaces["case_id"].eq(CASE_ID) & interfaces["protein_id"].eq(PROTEIN_ID)
    ].copy()
    prediction_by_method = predictions.set_index("method")
    interface_by_method = interfaces.set_index("method")

    delepref_row = prediction_by_method.loc["delepref_full_control"]
    wt_confidence = Path(str(delepref_row["wt_confidence_json"]))
    records: list[dict[str, Any]] = []
    for method in METHODS:
        method_id = method["id"]
        if method_id == "wildtype_shared":
            confidence_path = wt_confidence
            confidence = read_json(confidence_path)
            records.append(
                {
                    **method,
                    "structure": structure_from_confidence(confidence_path),
                    "confidence_path": confidence_path,
                    "binder_length": WT_BINDER_LENGTH,
                    "deleted_length": 0,
                    "confidence": float(confidence["confidence_score"]),
                    "edited_confidence": float(confidence["confidence_score"]),
                    "delta_confidence": 0.0,
                    "interface_confidence": None,
                    "complex_iplddt": float(confidence["complex_iplddt"]),
                    "binder_target_interchain_pae": None,
                    "wt_contact_recovery": 1.0,
                    "contact_jaccard": 1.0,
                    "steric_clash_count": int(
                        interface_by_method.loc["delepref_full_control", "wt_steric_clash_count"]
                    ),
                }
            )
            continue

        pred = prediction_by_method.loc[method_id]
        interface = interface_by_method.loc[method_id]
        confidence_path = Path(str(pred["edited_confidence_json"]))
        records.append(
            {
                **method,
                "structure": structure_from_confidence(confidence_path),
                "confidence_path": confidence_path,
                "binder_length": EDITED_BINDER_LENGTH,
                "deleted_length": DELETED_LENGTH,
                "confidence": float(pred["edited_confidence"]),
                "edited_confidence": float(pred["edited_confidence"]),
                "delta_confidence": float(pred["delta_confidence"]),
                "interface_confidence": float(interface["interface_confidence"]),
                "complex_iplddt": float(interface["complex_iplddt"]),
                "binder_target_interchain_pae": float(interface["binder_target_interchain_pae"]),
                "wt_contact_recovery": (
                    None
                    if pd.isna(interface["wt_to_edited_interface_contact_recovery"])
                    else float(interface["wt_to_edited_interface_contact_recovery"])
                ),
                "contact_jaccard": (
                    None
                    if pd.isna(interface["interface_contact_jaccard"])
                    else float(interface["interface_contact_jaccard"])
                ),
                "steric_clash_count": int(interface["steric_clash_count"]),
            }
        )
    return records


def verify_inputs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    require_status(BINDER_DelePref, "PASS_DelePref_BINDER")
    require_status(BINDER_V12, "PASS_DelePref_12_BINDER_INTERFACE_ANALYSIS")
    require_status(BINDER_V16, "PASS_DelePref_16_BINDER_COLDSTART_SAME_SCORER")
    if not PYMOL.is_file():
        raise RuntimeError(f"missing PyMOL executable: {PYMOL}")

    by_id = {record["id"]: record for record in records}
    delepref = by_id["delepref_full_control"]
    scisor = by_id["original_SCISOR_reference"]
    if not (
        delepref["edited_confidence"] > scisor["edited_confidence"]
        and delepref["interface_confidence"] > scisor["interface_confidence"]
        and delepref["binder_target_interchain_pae"] < scisor["binder_target_interchain_pae"]
        and (delepref["wt_contact_recovery"] or 0.0) > 0.5
    ):
        raise RuntimeError("case 6 no longer satisfies the favorable illustration rule")

    source_records = []
    for record in records:
        for role, path in (("structure", record["structure"]), ("confidence", record["confidence_path"])):
            if not path.is_file():
                raise RuntimeError(f"missing {role}: {path}")
            source_records.append(
                {
                    "method": record["id"],
                    "role": role,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    for path in (
        BINDER_V12 / "binder_interface_case_metrics.csv",
        BINDER_V16 / "binder_prediction_details.csv",
        BINDER_V16 / "binder_manifest_alignment.csv",
    ):
        source_records.append(
            {
                "method": "shared",
                "role": path.name,
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return source_records


def pymol_script_text(records: list[dict[str, Any]]) -> str:
    method_payload = [
        {
            "id": record["id"],
            "path": str(record["structure"]),
            "pymol_color": record["pymol_color"],
        }
        for record in records
    ]
    colors = {
        "wt_binder": [0.165, 0.616, 0.561],
        "delepref_full": [0.00, 0.447, 0.698],
        "scisor": [0.835, 0.369, 0.00],
        "target_gray": [0.72, 0.72, 0.72],
        "interface_purple": [0.42, 0.24, 0.60],
    }
    return f'''from pymol import cmd
import json
from pathlib import Path

OUT = Path({str(OUT)!r})
METHODS = {method_payload!r}
BINDER_CHAINS = {list(BINDER_CHAINS)!r}
TARGET_CHAINS = {list(TARGET_CHAINS)!r}
COLORS = {colors!r}

cmd.reinitialize()
cmd.set("ray_opaque_background", 0)
cmd.set("antialias", 2)
cmd.set("ray_trace_mode", 1)
cmd.set("ray_shadows", 0)
cmd.set("orthoscopic", 1)
cmd.set("cartoon_fancy_helices", 1)
cmd.set("cartoon_side_chain_helper", 1)
cmd.set("surface_quality", 1)
cmd.bg_color("white")
for name, rgb in COLORS.items():
    cmd.set_color(name, rgb)
for method in METHODS:
    cmd.load(method["path"], method["id"])

wt = "wildtype_shared"
target_expr = "+".join(TARGET_CHAINS)
binder_expr = "+".join(BINDER_CHAINS)
target_sel = f"chain {{target_expr}}"
binder_sel = f"chain {{binder_expr}}"
audit = {{"target_chains": TARGET_CHAINS, "binder_chains": BINDER_CHAINS, "alignments": {{}}}}
for method in METHODS:
    obj = method["id"]
    chains = cmd.get_chains(obj)
    if set(TARGET_CHAINS + BINDER_CHAINS) != set(chains):
        raise RuntimeError(f"unexpected chains for {{obj}}: {{chains}}")
    counts = {{chain: cmd.count_atoms(f"{{obj}} and chain {{chain}} and name CA") for chain in chains}}
    if obj == wt:
        result = [0.0, sum(counts[c] for c in TARGET_CHAINS)]
    else:
        aligned = cmd.align(
            f"{{obj}} and {{target_sel}} and name CA",
            f"{{wt}} and {{target_sel}} and name CA",
            cycles=0,
            transform=1,
        )
        result = [float(aligned[0]), int(aligned[1])]
    audit["alignments"][obj] = {{"target_rmsd": result[0], "aligned_target_ca": result[1], "ca_counts": counts}}

all_objects = " or ".join(method["id"] for method in METHODS)
cmd.disable("all")
cmd.enable(wt)
cmd.orient(f"{{wt}} and {{target_sel}}")
cmd.zoom(all_objects, buffer=4.0)
cmd.turn("y", 12)
cmd.turn("x", -8)
# Keep the triptych landscape-oriented: the target should sit to the right of
# the binder instead of above it.  A horizontal complex uses the available
# paper width much more efficiently and avoids the large side margins produced
# by the portrait-oriented view.
cmd.turn("z", -90)
shared_view = list(cmd.get_view())
audit["shared_view"] = shared_view
audit["shared_view_selection"] = all_objects

def render_single(method):
    obj = method["id"]
    cmd.disable("all")
    cmd.enable(obj)
    cmd.hide("everything", "all")
    cmd.show("cartoon", obj)
    cmd.color("target_gray", f"{{obj}} and {{target_sel}}")
    cmd.show("surface", f"{{obj}} and {{target_sel}}")
    cmd.set("transparency", 0.72, f"{{obj}} and {{target_sel}}")
    cmd.color(method["pymol_color"], f"{{obj}} and {{binder_sel}}")
    cmd.show("surface", f"{{obj}} and {{binder_sel}}")
    cmd.set("transparency", 0.78, f"{{obj}} and {{binder_sel}}")
    binder_iface = f"iface_b_{{obj}}"
    target_iface = f"iface_t_{{obj}}"
    cmd.select(binder_iface, f"byres (({{obj}} and {{binder_sel}}) within 8.0 of ({{obj}} and {{target_sel}}))")
    cmd.select(target_iface, f"byres (({{obj}} and {{target_sel}}) within 8.0 of ({{obj}} and {{binder_sel}}))")
    cmd.show("sticks", binder_iface)
    cmd.color(method["pymol_color"], binder_iface)
    cmd.show("sticks", target_iface)
    cmd.color("interface_purple", target_iface)
    cmd.set("stick_radius", 0.17, binder_iface)
    cmd.set("stick_radius", 0.14, target_iface)
    cmd.set_view(shared_view)
    cmd.ray(1800, 1400)
    cmd.png(str(OUT / "raw_renders" / f"complex_{{obj}}.png"), dpi=400)
    cmd.delete(binder_iface)
    cmd.delete(target_iface)

for method in METHODS:
    render_single(method)

(OUT / "pymol_alignment_and_render_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\\n")
cmd.quit()
'''


def run_pymol(records: list[dict[str, Any]]) -> Path:
    script_path = OUT / "scripts/render_favorable_case.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(pymol_script_text(records), encoding="utf-8")
    log_path = OUT / "pymol_render.log"
    completed = subprocess.run(
        [str(PYMOL), "-cq", "-r", str(script_path)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"PyMOL failed with exit code {completed.returncode}: {log_path}")
    expected = [OUT / "raw_renders" / f"complex_{record['id']}.png" for record in records]
    missing = [str(path) for path in expected if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"missing PyMOL renders: {missing}")
    return script_path


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def aligned_panel_images(
    records: list[dict[str, Any]],
    *,
    horizontal_pad: int = 24,
    vertical_pad: int = 18,
) -> dict[str, Image.Image]:
    """Tightly crop, size-match, and baseline-align the three structures.

    A single union crop preserves large transparent margins when one predicted
    complex has a different footprint.  Matplotlib then makes that complex
    look smaller and can place its apparent lower edge on a different line.
    Instead, crop each alpha foreground independently, normalize all panels to
    the mean foreground height of Full-length and Ours, and paste them onto one
    shared canvas with a common bottom edge.  Aspect ratios are preserved.
    """

    images = {
        record["id"]: Image.open(
            OUT / "raw_renders" / f"complex_{record['id']}.png"
        ).convert("RGBA")
        for record in records
    }
    cropped: dict[str, Image.Image] = {}
    for method_id, image in images.items():
        box = image.getchannel("A").getbbox()
        if box is None:
            raise RuntimeError(f"blank favorable-case render: {method_id}")
        cropped[method_id] = image.crop(box)

    reference_ids = ("wildtype_shared", "delepref_full_control")
    reference_heights = [cropped[method_id].height for method_id in reference_ids]
    target_height = round(sum(reference_heights) / len(reference_heights))
    if target_height <= 0:
        raise RuntimeError("invalid favorable-case panel target height")

    resized: dict[str, Image.Image] = {}
    for method_id, image in cropped.items():
        scale = target_height / image.height
        target_width = max(1, round(image.width * scale))
        resized[method_id] = image.resize(
            (target_width, target_height),
            resample=Image.Resampling.LANCZOS,
        )

    canvas_width = max(image.width for image in resized.values()) + 2 * horizontal_pad
    canvas_height = target_height + 2 * vertical_pad
    aligned: dict[str, Image.Image] = {}
    for method_id, image in resized.items():
        canvas = Image.new("RGBA", (canvas_width, canvas_height), (255, 255, 255, 0))
        left = (canvas_width - image.width) // 2
        # Every foreground has the same height and therefore the same bottom
        # line; retaining the explicit calculation documents that invariant.
        top = canvas_height - vertical_pad - image.height
        canvas.alpha_composite(image, (left, top))
        aligned[method_id] = canvas
    return aligned


def save_triptych(records: list[dict[str, Any]]) -> list[Path]:
    panel_images = aligned_panel_images(records)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0))
    fig.suptitle(
        "Binder-target complex predictions before and after deletion",
        fontsize=15.5,
        fontweight="bold",
        y=0.97,
    )
    for index, record in enumerate(records):
        image = panel_images[record["id"]]
        ax = axes[index]
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(
            f"{chr(65 + index)}  {record['label']}",
            fontsize=13.5,
            fontweight="bold",
            color=record["color"],
            pad=2,
        )
        if record["id"] == "wildtype_shared":
            note = f"{WT_BINDER_LENGTH} aa | Conf. {fmt(record['confidence'])}"
        elif record["id"] == "original_SCISOR_reference":
            note = (
                f"{EDITED_BINDER_LENGTH} aa | Conf. {fmt(record['confidence'])} | "
                f"$\\Delta$Conf. {record['delta_confidence']:+.3f}\n"
                f"ipTM {fmt(record['interface_confidence'])} | "
                f"PAE {fmt(record['binder_target_interchain_pae'], 2)} $\\AA$"
            )
        else:
            note = (
                f"{EDITED_BINDER_LENGTH} aa | Conf. {fmt(record['confidence'])} | "
                f"$\\Delta$Conf. {record['delta_confidence']:+.3f}\n"
                f"ipTM {fmt(record['interface_confidence'])} | "
                f"PAE {fmt(record['binder_target_interchain_pae'], 2)} $\\AA$ | "
                f"Contact recovery {fmt(record['wt_contact_recovery'])}"
            )
        ax.text(
            0.5,
            -0.012,
            note,
            ha="center",
            va="top",
            fontsize=10.5,
            linespacing=1.35,
            transform=ax.transAxes,
        )
    fig.text(
        0.5,
        0.022,
        "Target: gray | Binder: method color | Predicted interface: purple sticks",
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.19, wspace=0.005)
    OUT.joinpath("figures").mkdir(parents=True, exist_ok=True)
    outputs = []
    stem = "fig_binder_favorable_case6_wt_delepref_scisor_triptych_delepref"
    for suffix in ("pdf", "svg", "png"):
        path = OUT / "figures" / f"{stem}.{suffix}"
        kwargs: dict[str, Any] = {"facecolor": "white"}
        if suffix == "png":
            kwargs["dpi"] = 500
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def image_audit(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        background = Image.new("RGB", rgb.size, "white")
        return {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "nonblank": ImageChops.difference(rgb, background).getbbox() is not None,
        }


def verify_manifest() -> None:
    manifest = read_json(OUT / "audit_manifest.json")
    for section in ("source_artifacts", "generated_artifacts"):
        for record in manifest[section]:
            path = Path(record["path"])
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"manifest mismatch: {path}")
    print("PASS_VERIFY_ONLY")


def build(*, reuse_renders: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw_renders").mkdir(parents=True, exist_ok=True)
    records = load_method_records()
    sources = verify_inputs(records)
    raw_paths = [OUT / "raw_renders" / f"complex_{record['id']}.png" for record in records]
    if reuse_renders:
        missing = [str(path) for path in raw_paths if not path.is_file()]
        if missing:
            raise RuntimeError(f"missing reusable renders: {missing}")
        pymol_script = OUT / "scripts/render_favorable_case.py"
    else:
        pymol_script = run_pymol(records)
    outputs = save_triptych(records)
    paper_records = []
    PAPER.mkdir(parents=True, exist_ok=True)
    for source in outputs:
        target = PAPER / source.name
        shutil.copy2(source, target)
        if sha256_file(source) != sha256_file(target):
            raise RuntimeError(f"paper copy mismatch: {target}")
        record = {
            "source_path": str(source),
            "paper_copy_path": str(target),
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
            "copy_identical": True,
        }
        if source.suffix == ".png":
            record["image"] = image_audit(source)
        paper_records.append(record)

    generated_paths = outputs + [
        pymol_script,
        OUT / "pymol_alignment_and_render_audit.json",
    ]
    generated = [
        {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in generated_paths
    ]
    write_json(
        OUT / "figure_manifest.json",
        {
            "schema_version": "delepref_binder_favorable_case_figures_v1",
            "created_at": now(),
            "case_id": CASE_ID,
            "protein_id": PROTEIN_ID,
            "selection_type": "post_hoc_favorable_illustration",
            "selection_rule": (
                "DelePref full-control has higher edited confidence and interface confidence "
                "and lower inter-chain PAE than Original SCISOR; DelePref WT-contact recovery is > 0.5."
            ),
            "boltz_rerun": False,
            "figures": paper_records,
            "proxy_boundary": PROXY_BOUNDARY,
            "claim_boundary": CLAIM_BOUNDARY,
            "paper_caption": PAPER_CAPTION,
        },
    )
    generated.append(
        {
            "path": str(OUT / "figure_manifest.json"),
            "sha256": sha256_file(OUT / "figure_manifest.json"),
            "size_bytes": (OUT / "figure_manifest.json").stat().st_size,
        }
    )
    write_json(
        OUT / "audit_manifest.json",
        {
            "status": "PASS_DelePref_BINDER_FAVORABLE_CASE_FIGURE",
            "created_at": now(),
            "source_artifacts": sources,
            "generated_artifacts": generated,
            "no_boltz_or_model_run": True,
            "paper_copies_identical": True,
        },
    )
    write_json(
        OUT / "run_summary.json",
        {
            "status": "PASS_DelePref_BINDER_FAVORABLE_CASE_FIGURE",
            "created_at": now(),
            "case_id": CASE_ID,
            "protein_id": PROTEIN_ID,
            "selection_type": "post_hoc_favorable_illustration",
            "figure_count": 1,
            "format_count": 3,
            "boltz_rerun": False,
            "model_or_gpu_run": False,
            "paper_copies_identical": True,
            "proxy_boundary": PROXY_BOUNDARY,
            "claim_boundary": CLAIM_BOUNDARY,
            "paper_caption": PAPER_CAPTION,
        },
    )
    (OUT / "SUMMARY.md").write_text(
        f"""# Favorable Binder case illustration

Status: `PASS_DelePref_BINDER_FAVORABLE_CASE_FIGURE`

Case: `{CASE_ID}` / `{PROTEIN_ID}`. This is a post-hoc favorable illustration
selected from the frozen 50-case Binder set. DelePref full-control has higher edited
confidence and interface confidence and lower inter-chain PAE than Original
SCISOR; DelePref WT-contact recovery is above 0.5 for this case.

The three panels are the shared WT prediction, DelePref full-control post-deletion
prediction, and Original SCISOR post-deletion prediction. Existing CIF and
confidence files were rendered at high resolution; no Boltz or model
computation was run.

- PDF: `{PAPER / 'fig_binder_favorable_case6_wt_delepref_scisor_triptych_delepref.pdf'}`
- SVG: `{PAPER / 'fig_binder_favorable_case6_wt_delepref_scisor_triptych_delepref.svg'}`
- PNG: `{PAPER / 'fig_binder_favorable_case6_wt_delepref_scisor_triptych_delepref.png'}`

{PROXY_BOUNDARY}

{CLAIM_BOUNDARY}

## Paper caption

{PAPER_CAPTION}
""",
        encoding="utf-8",
    )
    verify_manifest()
    print("PASS_DelePref_BINDER_FAVORABLE_CASE_FIGURE")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--reuse-renders", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_manifest()
    else:
        build(reuse_renders=args.reuse_renders)
