#!/usr/bin/env python3
"""Render frozen WT and post-deletion Binder complex predictions.

This workflow is visualization-only. It reads existing Boltz CIF/confidence
outputs and frozen DelePref-12/DelePref-16 interface metrics. It does not run Boltz,
student inference, a teacher, training, or metric selection.
"""

from __future__ import annotations

import argparse
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
OUT = ROOT / "results/experiments/delepref_step004000_binder_postdeletion_visualization"
PAPER = ROOT / "results/figures/delepref"
PYMOL = Path("pymol")

BINDER = ROOT / "results/experiments/delepref_step004000_binder"
BINDER_V16 = ROOT / "results/experiments/delepref_step004000_delepref_16_binder_coldstart_same_scorer"
CASE_FIGURES = ROOT / "results/experiments/delepref_step004000_case_study_visualization"
PROTOCOL_SOURCE = Path(__file__).resolve()

CASE_ID = 2
PROTEIN_ID = "AB-Bind:1BJ1:HL_VW"
CLAIM_BOUNDARY = (
    "The method ranks candidate deletion plans by teacher-derived deletion "
    "perturbation preference and optional auxiliary risk evidence."
)
PROXY_BOUNDARY = (
    "These are post-deletion complex-prediction proxies, not experimental "
    "evidence of retained binding, Kd/Ki, or functional preservation."
)

METHODS = [
    {
        "id": "wildtype_shared",
        "label": "Unedited",
        "short_label": "Unedited",
        "comparison_class": "WT_BASELINE",
        "color": "#4D4D4D",
        "pymol_color": "wt_binder",
        "structure": ROOT
        / "results/experiments/binder_benchmark_v2_step3400000/binder_b2_boltz_full50"
        / "boltz_outputs/boltz_results_inputs/predictions"
        / "case002_AB-Bind_1BJ1_HL_VW_wt/case002_AB-Bind_1BJ1_HL_VW_wt_model_0.cif",
        "confidence": ROOT
        / "results/experiments/binder_benchmark_v2_step3400000/binder_b2_boltz_full50"
        / "boltz_outputs/boltz_results_inputs/predictions"
        / "case002_AB-Bind_1BJ1_HL_VW_wt/confidence_case002_AB-Bind_1BJ1_HL_VW_wt_model_0.json",
    },
    {
        "id": "delepref_full_control",
        "label": "Ours",
        "short_label": "DelePref full",
        "comparison_class": "SAME_PROTOCOL",
        "color": "#0072B2",
        "pymol_color": "delepref_full",
        "structure": BINDER
        / "boltz_outputs/boltz_results_boltz_run_inputs/predictions"
        / "case002_AB-Bind_1BJ1_HL_VW_deleprefedit_b310d8a5b1fc"
        / "case002_AB-Bind_1BJ1_HL_VW_deleprefedit_b310d8a5b1fc_model_0.cif",
        "confidence": BINDER
        / "boltz_outputs/boltz_results_boltz_run_inputs/predictions"
        / "case002_AB-Bind_1BJ1_HL_VW_deleprefedit_b310d8a5b1fc"
        / "confidence_case002_AB-Bind_1BJ1_HL_VW_deleprefedit_b310d8a5b1fc_model_0.json",
    },
    {
        "id": "delepref_no_scorer",
        "label": "DelePref no-scorer",
        "short_label": "DelePref no-scorer",
        "comparison_class": "SAME_PROTOCOL",
        "color": "#009E73",
        "pymol_color": "delepref_no_scorer",
        "structure": BINDER
        / "boltz_outputs/boltz_results_boltz_run_inputs/predictions"
        / "case002_AB-Bind_1BJ1_HL_VW_deleprefedit_dd00b2fb6567"
        / "case002_AB-Bind_1BJ1_HL_VW_deleprefedit_dd00b2fb6567_model_0.cif",
        "confidence": BINDER
        / "boltz_outputs/boltz_results_boltz_run_inputs/predictions"
        / "case002_AB-Bind_1BJ1_HL_VW_deleprefedit_dd00b2fb6567"
        / "confidence_case002_AB-Bind_1BJ1_HL_VW_deleprefedit_dd00b2fb6567_model_0.json",
    },
    {
        "id": "cold_start_method_v1_plus_same_scorer",
        "label": "Cold-start + same scorer",
        "short_label": "Cold-start",
        "comparison_class": "SAME_PROTOCOL",
        "color": "#CC79A7",
        "pymol_color": "cold_start",
        "structure": BINDER_V16
        / "boltz_outputs/boltz_results_boltz_run_inputs/predictions"
        / "case002_AB-Bind_1BJ1_HL_VW_delepref16cold_4823bee95dc1"
        / "case002_AB-Bind_1BJ1_HL_VW_delepref16cold_4823bee95dc1_model_0.cif",
        "confidence": BINDER_V16
        / "boltz_outputs/boltz_results_boltz_run_inputs/predictions"
        / "case002_AB-Bind_1BJ1_HL_VW_delepref16cold_4823bee95dc1"
        / "confidence_case002_AB-Bind_1BJ1_HL_VW_delepref16cold_4823bee95dc1_model_0.json",
    },
    {
        "id": "original_SCISOR_reference",
        "label": "SCISOR",
        "short_label": "SCISOR",
        "comparison_class": "EXTERNAL_REFERENCE",
        "color": "#D55E00",
        "pymol_color": "scisor",
        "structure": ROOT
        / "results/experiments/binder_benchmark_v2_step3400000/binder_b2_boltz_full50"
        / "boltz_outputs/boltz_results_inputs/predictions"
        / "case002_AB-Bind_1BJ1_HL_VW_original_SCISOR"
        / "case002_AB-Bind_1BJ1_HL_VW_original_SCISOR_model_0.cif",
        "confidence": ROOT
        / "results/experiments/binder_benchmark_v2_step3400000/binder_b2_boltz_full50"
        / "boltz_outputs/boltz_results_inputs/predictions"
        / "case002_AB-Bind_1BJ1_HL_VW_original_SCISOR"
        / "confidence_case002_AB-Bind_1BJ1_HL_VW_original_SCISOR_model_0.json",
    },
]


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


def verify_inputs() -> list[dict[str, Any]]:
    require_status(BINDER, "PASS_DelePref_BINDER")
    require_status(BINDER_V16, "PASS_DelePref_16_BINDER_COLDSTART_SAME_SCORER")
    selected = read_json(CASE_FIGURES / "selected_cases.json")
    selected_binder = selected.get("binder", {})
    if int(selected_binder.get("case_id", -1)) != CASE_ID:
        raise RuntimeError("frozen Binder representative case is not case 2")
    if selected_binder.get("protein_id") != PROTEIN_ID:
        raise RuntimeError("frozen Binder representative protein ID changed")
    if not PYMOL.is_file():
        raise RuntimeError(f"missing PyMOL executable: {PYMOL}")

    sources = []
    for method in METHODS:
        for role in ("structure", "confidence"):
            path = Path(method[role])
            if not path.is_file():
                raise RuntimeError(f"missing {role} for {method['id']}: {path}")
            sources.append(
                {
                    "method": method["id"],
                    "role": role,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    for role, path in (
        ("interface_metrics", BINDER_V16 / "binder_interface_case_metrics.csv"),
        ("prediction_details", BINDER_V16 / "binder_prediction_details.csv"),
        ("selected_cases", CASE_FIGURES / "selected_cases.json"),
    ):
        sources.append(
            {
                "method": "shared",
                "role": role,
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return sources


def finite_or_none(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def collect_metrics() -> pd.DataFrame:
    interface = pd.read_csv(BINDER_V16 / "binder_interface_case_metrics.csv")
    predictions = pd.read_csv(BINDER_V16 / "binder_prediction_details.csv")
    interface = interface[
        interface["case_id"].eq(CASE_ID) & interface["protein_id"].eq(PROTEIN_ID)
    ].copy()
    predictions = predictions[
        predictions["case_id"].eq(CASE_ID) & predictions["protein_id"].eq(PROTEIN_ID)
    ].copy()
    interface_by_method = interface.set_index("method")
    prediction_by_method = predictions.set_index("method")

    common = interface_by_method.loc["delepref_full_control"]
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        method_id = method["id"]
        confidence = read_json(Path(method["confidence"]))
        if method_id == "wildtype_shared":
            row = {
                "case_id": CASE_ID,
                "protein_id": PROTEIN_ID,
                "method": method_id,
                "method_label": method["label"],
                "comparison_class": method["comparison_class"],
                "binder_length": 431,
                "deleted_length": 0,
                "confidence": float(confidence["confidence_score"]),
                "delta_confidence": 0.0,
                "interface_confidence": None,
                "complex_iplddt": float(confidence["complex_iplddt"]),
                "binder_target_interchain_pae": None,
                "wt_contact_recovery": 1.0,
                "contact_jaccard": 1.0,
                "interface_contact_count": int(common["wt_contact_count"]),
                "binder_interface_residue_count": int(
                    common["wt_binder_interface_residue_count"]
                ),
                "target_interface_residue_count": int(
                    common["wt_target_interface_residue_count"]
                ),
                "steric_clash_count": int(common["wt_steric_clash_count"]),
            }
        else:
            metric = interface_by_method.loc[method_id]
            pred = prediction_by_method.loc[method_id]
            row = {
                "case_id": CASE_ID,
                "protein_id": PROTEIN_ID,
                "method": method_id,
                "method_label": method["label"],
                "comparison_class": method["comparison_class"],
                "binder_length": 388,
                "deleted_length": 43,
                "confidence": float(pred["edited_confidence"]),
                "delta_confidence": float(pred["delta_confidence"]),
                "interface_confidence": finite_or_none(metric["interface_confidence"]),
                "complex_iplddt": finite_or_none(metric["complex_iplddt"]),
                "binder_target_interchain_pae": finite_or_none(
                    metric["binder_target_interchain_pae"]
                ),
                "wt_contact_recovery": finite_or_none(
                    metric["wt_to_edited_interface_contact_recovery"]
                ),
                "contact_jaccard": finite_or_none(metric["interface_contact_jaccard"]),
                "interface_contact_count": int(metric["edited_contact_count"]),
                "binder_interface_residue_count": int(
                    metric["edited_binder_interface_residue_count"]
                ),
                "target_interface_residue_count": int(
                    metric["edited_target_interface_residue_count"]
                ),
                "steric_clash_count": int(metric["steric_clash_count"]),
            }
        row.update(
            {
                "structure_path": str(method["structure"]),
                "structure_sha256": sha256_file(Path(method["structure"])),
                "confidence_path": str(method["confidence"]),
                "confidence_sha256": sha256_file(Path(method["confidence"])),
                "proxy_boundary": PROXY_BOUNDARY,
            }
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "binder_postdeletion_case_metrics.csv", index=False)
    return frame


def pymol_script_text() -> str:
    method_payload = [
        {
            "id": method["id"],
            "path": str(method["structure"]),
            "pymol_color": method["pymol_color"],
        }
        for method in METHODS
    ]
    colors = {
        "wt_binder": [0.30, 0.30, 0.30],
        "delepref_full": [0.00, 0.447, 0.698],
        "delepref_no_scorer": [0.00, 0.620, 0.451],
        "cold_start": [0.80, 0.475, 0.655],
        "scisor": [0.835, 0.369, 0.00],
        "target_gray": [0.72, 0.72, 0.72],
        "interface_purple": [0.42, 0.24, 0.60],
        "wt_overlay": [0.42, 0.42, 0.42],
    }
    return f'''from pymol import cmd
import json
from pathlib import Path

OUT = Path({str(OUT)!r})
METHODS = {method_payload!r}
COLORS = {colors!r}

cmd.reinitialize()
cmd.set("ray_opaque_background", 0)
cmd.set("antialias", 2)
cmd.set("ray_trace_mode", 1)
cmd.set("ray_shadows", 0)
cmd.set("orthoscopic", 1)
cmd.set("cartoon_fancy_helices", 1)
cmd.set("cartoon_side_chain_helper", 1)
cmd.bg_color("white")
for name, rgb in COLORS.items():
    cmd.set_color(name, rgb)
for method in METHODS:
    cmd.load(method["path"], method["id"])

wt = "wildtype_shared"
audit = {{"target_alignment_selection": "chain V and name CA", "alignments": {{}}}}
for method in METHODS:
    obj = method["id"]
    chains = cmd.get_chains(obj)
    counts = {{chain: cmd.count_atoms(f"{{obj}} and chain {{chain}} and name CA") for chain in chains}}
    if set(chains) != {{"A", "V", "W"}}:
        raise RuntimeError(f"unexpected chains for {{obj}}: {{chains}}")
    if obj == wt:
        result = [0.0, counts["V"]]
    else:
        aligned = cmd.align(
            f"{{obj}} and chain V and name CA",
            f"{{wt}} and chain V and name CA",
            cycles=0,
            transform=1,
        )
        result = [float(aligned[0]), int(aligned[1])]
    audit["alignments"][obj] = {{
        "target_chain_v_rmsd": result[0],
        "target_chain_v_aligned_atoms": result[1],
        "ca_counts": counts,
    }}

cmd.disable("all")
cmd.enable(wt)
cmd.orient(wt)
cmd.zoom(wt, buffer=5.0)
cmd.turn("y", 16)
cmd.turn("x", -8)
shared_view = list(cmd.get_view())
audit["shared_view"] = shared_view

def render_single(method):
    obj = method["id"]
    cmd.disable("all")
    cmd.enable(obj)
    cmd.hide("everything", "all")
    cmd.show("cartoon", obj)
    cmd.color("target_gray", f"{{obj}} and chain V+W")
    cmd.show("surface", f"{{obj}} and chain V+W")
    cmd.set("transparency", 0.76, f"{{obj}} and chain V+W")
    cmd.color(method["pymol_color"], f"{{obj}} and chain A")
    cmd.show("surface", f"{{obj}} and chain A")
    cmd.set("transparency", 0.82, f"{{obj}} and chain A")
    binder_iface = f"iface_b_{{obj}}"
    target_iface = f"iface_t_{{obj}}"
    cmd.select(binder_iface, f"byres (({{obj}} and chain A) within 8.0 of ({{obj}} and chain V+W))")
    cmd.select(target_iface, f"byres (({{obj}} and chain V+W) within 8.0 of ({{obj}} and chain A))")
    cmd.show("sticks", binder_iface)
    cmd.color(method["pymol_color"], binder_iface)
    cmd.show("sticks", target_iface)
    cmd.color("interface_purple", target_iface)
    cmd.set("stick_radius", 0.16, binder_iface)
    cmd.set("stick_radius", 0.13, target_iface)
    cmd.set_view(shared_view)
    cmd.zoom(obj, buffer=4.0)
    cmd.ray(1600, 1150)
    cmd.png(str(OUT / "raw_renders" / f"complex_{{obj}}.png"), dpi=300)
    cmd.delete(binder_iface)
    cmd.delete(target_iface)

def render_overlay(method):
    obj = method["id"]
    cmd.disable("all")
    cmd.enable(wt)
    cmd.enable(obj)
    cmd.hide("everything", "all")
    cmd.show("cartoon", f"{{wt}} and chain V+W")
    cmd.color("target_gray", f"{{wt}} and chain V+W")
    cmd.show("surface", f"{{wt}} and chain V+W")
    cmd.set("transparency", 0.80, f"{{wt}} and chain V+W")
    cmd.show("cartoon", f"{{wt}} and chain A")
    cmd.color("wt_overlay", f"{{wt}} and chain A")
    cmd.set("cartoon_transparency", 0.48, f"{{wt}} and chain A")
    cmd.show("cartoon", f"{{obj}} and chain A")
    cmd.color(method["pymol_color"], f"{{obj}} and chain A")
    cmd.show("surface", f"{{obj}} and chain A")
    cmd.set("transparency", 0.84, f"{{obj}} and chain A")
    binder_iface = f"overlay_iface_{{obj}}"
    cmd.select(binder_iface, f"byres (({{obj}} and chain A) within 8.0 of ({{obj}} and chain V+W))")
    cmd.show("sticks", binder_iface)
    cmd.color(method["pymol_color"], binder_iface)
    cmd.set("stick_radius", 0.16, binder_iface)
    cmd.set_view(shared_view)
    cmd.zoom(f"{{wt}} or {{obj}}", buffer=4.0)
    cmd.ray(1600, 1150)
    cmd.png(str(OUT / "raw_renders" / f"overlay_{{obj}}.png"), dpi=300)
    cmd.delete(binder_iface)

for method in METHODS:
    render_single(method)
for method in METHODS[1:]:
    render_overlay(method)

with (OUT / "pymol_alignment_and_render_audit.json").open("w") as handle:
    json.dump(audit, handle, indent=2, sort_keys=True)
cmd.quit()
'''


def run_pymol() -> Path:
    script_path = OUT / "scripts/render_postdeletion_complexes.py"
    script_path.write_text(pymol_script_text(), encoding="utf-8")
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
    if "Traceback (most recent call last)" in completed.stdout or "Error:" in completed.stdout:
        raise RuntimeError(f"PyMOL reported a script error: {log_path}")
    expected = [
        OUT / "raw_renders" / f"complex_{method['id']}.png" for method in METHODS
    ] + [
        OUT / "raw_renders" / f"overlay_{method['id']}.png" for method in METHODS[1:]
    ]
    missing = [str(path) for path in expected if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"missing PyMOL renders: {missing}")
    return script_path


def nonblank_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    if alpha.getextrema()[0] < 255:
        return alpha.getbbox()
    rgb = rgba.convert("RGB")
    background = Image.new("RGB", rgb.size, "white")
    return ImageChops.difference(rgb, background).getbbox()


def crop_group(paths: list[Path]) -> None:
    images = [Image.open(path).convert("RGBA") for path in paths]
    boxes = [nonblank_bbox(image) for image in images]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        raise RuntimeError("all PyMOL renders are blank")
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes)
    bottom = max(box[3] for box in boxes)
    pad = 35
    width, height = images[0].size
    crop = (
        max(0, left - pad),
        max(0, top - pad),
        min(width, right + pad),
        min(height, bottom + pad),
    )
    for path, image in zip(paths, images):
        image.crop(crop).save(path)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def metric_text(row: pd.Series) -> str:
    if row["method"] == "wildtype_shared":
        return (
            f"Unedited binder length {int(row['binder_length'])} aa\n"
            f"Boltz confidence {fmt(row['confidence'])}\n"
            f"complex ipLDDT {fmt(row['complex_iplddt'])}\n"
            f"interface contacts {int(row['interface_contact_count'])} | "
            f"clashes {int(row['steric_clash_count'])}"
        )
    mapping_note = ""
    if row["comparison_class"] == "EXTERNAL_REFERENCE":
        mapping_note = "\ncontact recovery/Jaccard NA (external mapping)"
    else:
        mapping_note = (
            f"\ncontact recovery {fmt(row['wt_contact_recovery'])} | "
            f"Jaccard {fmt(row['contact_jaccard'])}"
        )
    return (
        f"edited binder {int(row['binder_length'])} aa (delete {int(row['deleted_length'])})\n"
        f"confidence {fmt(row['confidence'])} | delta {float(row['delta_confidence']):+.3f}\n"
        f"pair-chain ipTM {fmt(row['interface_confidence'])} | "
        f"PAE {fmt(row['binder_target_interchain_pae'], 2)} A"
        f"{mapping_note}\n"
        f"contacts {int(row['interface_contact_count'])} | "
        f"clashes {int(row['steric_clash_count'])}"
    )


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    outputs = []
    for suffix in ("pdf", "svg", "png"):
        path = OUT / "figures" / f"{stem}.{suffix}"
        kwargs = {"facecolor": "white"}
        if suffix == "png":
            kwargs["dpi"] = 240
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def render_complex_figure(metrics: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.7))
    fig.suptitle(
        "1BJ1: Unedited and post-deletion complex predictions",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    for index, method in enumerate(METHODS):
        row = metrics[metrics["method"].eq(method["id"])].iloc[0]
        image = Image.open(OUT / "raw_renders" / f"complex_{method['id']}.png")
        ax = axes.flat[index]
        ax.imshow(image)
        ax.axis("off")
        suffix = " (external)" if method["comparison_class"] == "EXTERNAL_REFERENCE" else ""
        ax.set_title(
            f"{chr(65 + index)}  {method['label']}{suffix}",
            fontsize=10,
            fontweight="bold",
            color=method["color"],
            pad=3,
        )
        if method["id"] == "wildtype_shared":
            note = f"431 aa | confidence {fmt(row['confidence'])}"
        elif method["comparison_class"] == "EXTERNAL_REFERENCE":
            note = (
                f"388 aa | delta conf. {float(row['delta_confidence']):+.3f}\n"
                "External reference"
            )
        else:
            note = (
                f"388 aa | delta conf. {float(row['delta_confidence']):+.3f}\n"
                f"Contact recovery {fmt(row['wt_contact_recovery'])}"
            )
        ax.text(
            0.5,
            -0.025,
            note,
            ha="center",
            va="top",
            fontsize=8.3,
            linespacing=1.25,
            transform=ax.transAxes,
        )
    legend = axes.flat[5]
    legend.axis("off")
    legend.text(
        0.05,
        0.80,
        "Color: binder\nLight gray: target\nPurple sticks: predicted interface\n\n"
        "Post hoc target-aligned;\npanels independently fitted.\n\n"
        "Boltz prediction proxy;\nnot binding evidence.",
        ha="left",
        va="top",
        fontsize=9.2,
        linespacing=1.4,
        transform=legend.transAxes,
    )
    fig.subplots_adjust(left=0.025, right=0.985, top=0.91, bottom=0.055, hspace=0.30, wspace=0.04)
    return save_figure(fig, "fig_binder_postdeletion_complexes_delepref")


def render_triptych_figure(metrics: pd.DataFrame) -> list[Path]:
    selected_methods = [METHODS[0], METHODS[1], METHODS[4]]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.7))
    fig.suptitle(
        "1BJ1: Unedited, Ours, and SCISOR complex predictions",
        fontsize=12.5,
        fontweight="bold",
        y=0.985,
    )
    for index, method in enumerate(selected_methods):
        row = metrics[metrics["method"].eq(method["id"])].iloc[0]
        image = Image.open(OUT / "raw_renders" / f"complex_{method['id']}.png")
        ax = axes[index]
        ax.imshow(image)
        ax.axis("off")
        suffix = " (external)" if method["comparison_class"] == "EXTERNAL_REFERENCE" else ""
        ax.set_title(
            f"{chr(65 + index)}  {method['label']}{suffix}",
            fontsize=10.5,
            fontweight="bold",
            color=method["color"],
            pad=4,
        )
        if method["id"] == "wildtype_shared":
            note = f"431 aa | confidence {fmt(row['confidence'])}"
        elif method["comparison_class"] == "EXTERNAL_REFERENCE":
            note = f"388 aa | delta conf. {float(row['delta_confidence']):+.3f}\nExternal reference"
        else:
            note = (
                f"388 aa | delta conf. {float(row['delta_confidence']):+.3f}\n"
                f"Contact recovery {fmt(row['wt_contact_recovery'])}"
            )
        ax.text(
            0.5,
            -0.025,
            note,
            ha="center",
            va="top",
            fontsize=8.2,
            linespacing=1.25,
            transform=ax.transAxes,
        )
    fig.text(
        0.5,
        0.055,
        "Gray: target | Binder color identifies method | Purple sticks: predicted interface",
        ha="center",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.020,
        "Post-deletion Boltz complex-prediction proxy; not experimental binding evidence.",
        ha="center",
        va="bottom",
        fontsize=8.0,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.18, wspace=0.04)
    return save_figure(fig, "fig_binder_wt_delepref_scisor_triptych_delepref")


def render_overlay_figure(metrics: pd.DataFrame) -> list[Path]:
    edited_methods = METHODS[1:]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    fig.suptitle(
        "1BJ1: target-aligned unedited vs edited binder predictions",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    for index, method in enumerate(edited_methods):
        row = metrics[metrics["method"].eq(method["id"])].iloc[0]
        image = Image.open(OUT / "raw_renders" / f"overlay_{method['id']}.png")
        ax = axes.flat[index]
        ax.imshow(image)
        ax.axis("off")
        suffix = " (external)" if method["comparison_class"] == "EXTERNAL_REFERENCE" else ""
        ax.set_title(
            f"{chr(65 + index)}  {method['label']}{suffix}",
            fontsize=10.5,
            fontweight="bold",
            color=method["color"],
            pad=3,
        )
        if method["comparison_class"] == "EXTERNAL_REFERENCE":
            note = f"delta conf. {float(row['delta_confidence']):+.3f} | contact recovery NA"
        else:
            note = (
                f"delta conf. {float(row['delta_confidence']):+.3f} | "
                f"contact recovery {fmt(row['wt_contact_recovery'])}"
            )
        ax.text(
            0.5,
            -0.015,
            note,
            ha="center",
            va="top",
            fontsize=8.8,
            transform=ax.transAxes,
        )
    fig.text(
        0.5,
        0.030,
        "Gray: unedited binder | Color: edited binder | Light gray: target | Post hoc aligned on target chain V (94 CA)",
        ha="center",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.010,
        "Boltz complex-prediction proxy; not experimental binding evidence.",
        ha="center",
        va="bottom",
        fontsize=8.2,
    )
    fig.subplots_adjust(left=0.025, right=0.985, top=0.91, bottom=0.105, hspace=0.22, wspace=0.03)
    return save_figure(fig, "fig_binder_postdeletion_overlays_delepref")


def image_audit(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        background = Image.new("RGB", rgb.size, "white")
        nonblank = ImageChops.difference(rgb, background).getbbox() is not None
        return {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "nonblank": nonblank,
        }


def copy_paper_outputs(outputs: list[Path]) -> list[dict[str, Any]]:
    PAPER.mkdir(parents=True, exist_ok=True)
    records = []
    for source in outputs:
        target = PAPER / source.name
        shutil.copy2(source, target)
        source_sha = sha256_file(source)
        target_sha = sha256_file(target)
        if source_sha != target_sha:
            raise RuntimeError(f"paper copy hash mismatch: {source} -> {target}")
        record = {
            "source_path": str(source),
            "paper_copy_path": str(target),
            "sha256": source_sha,
            "size_bytes": source.stat().st_size,
            "copy_identical": True,
        }
        if source.suffix == ".png":
            record["image"] = image_audit(source)
            if not record["image"]["nonblank"]:
                raise RuntimeError(f"blank final PNG: {source}")
        records.append(record)
    return records


def verify_manifest() -> None:
    manifest_path = OUT / "audit_manifest.json"
    manifest = read_json(manifest_path)
    for section in ("source_artifacts", "generated_artifacts"):
        for record in manifest[section]:
            path = Path(record["path"])
            if not path.is_file():
                raise RuntimeError(f"missing manifest artifact: {path}")
            actual = sha256_file(path)
            if actual != record["sha256"]:
                raise RuntimeError(f"hash mismatch: {path}: {actual} != {record['sha256']}")
    print("PASS_VERIFY_ONLY")


def build(*, reuse_renders: bool = False) -> None:
    master_before = sha256_file(PROTOCOL_SOURCE)
    for relative in ("figures", "raw_renders", "scripts"):
        (OUT / relative).mkdir(parents=True, exist_ok=True)
    sources = verify_inputs()
    metrics = collect_metrics()
    complex_raw = [OUT / "raw_renders" / f"complex_{method['id']}.png" for method in METHODS]
    overlay_raw = [OUT / "raw_renders" / f"overlay_{method['id']}.png" for method in METHODS[1:]]
    if reuse_renders:
        missing = [str(path) for path in complex_raw + overlay_raw if not path.is_file()]
        if missing:
            raise RuntimeError(f"missing reusable renders: {missing}")
        pymol_script = OUT / "scripts/render_postdeletion_complexes.py"
    else:
        pymol_script = run_pymol()
        crop_group(complex_raw)
        crop_group(overlay_raw)

    outputs = (
        render_triptych_figure(metrics)
        + render_complex_figure(metrics)
        + render_overlay_figure(metrics)
    )
    paper_records = copy_paper_outputs(outputs)
    master_after = sha256_file(PROTOCOL_SOURCE)
    if master_before != master_after:
        raise RuntimeError("protocol_source.md changed during visualization")

    alignment = read_json(OUT / "pymol_alignment_and_render_audit.json")
    generated_paths = outputs + complex_raw + overlay_raw + [
        pymol_script,
        OUT / "pymol_alignment_and_render_audit.json",
        OUT / "binder_postdeletion_case_metrics.csv",
    ]
    generated = [
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in generated_paths
    ]
    write_json(
        OUT / "figure_manifest.json",
        {
            "schema_version": "delepref_binder_postdeletion_figures_v1",
            "created_at": now(),
            "case_id": CASE_ID,
            "protein_id": PROTEIN_ID,
            "renderer": "PyMOL open-source headless plus Matplotlib assembly",
            "boltz_rerun": False,
            "figures": paper_records,
            "alignment": alignment,
            "proxy_boundary": PROXY_BOUNDARY,
            "claim_boundary": CLAIM_BOUNDARY,
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
            "status": "PASS_DelePref_BINDER_POSTDELETION_FIGURES",
            "created_at": now(),
            "source_artifacts": [
                {"path": record["path"], "sha256": record["sha256"]}
                for record in sources
            ],
            "generated_artifacts": generated,
            "protocol_source_sha256_before": master_before,
            "protocol_source_sha256_after": master_after,
            "no_boltz_or_model_run": True,
            "paper_copies_identical": all(record["copy_identical"] for record in paper_records),
        },
    )
    summary = f"""# DelePref Binder post-deletion complex-prediction figures

Status: `PASS_DelePref_BINDER_POSTDELETION_FIGURES`

The frozen representative Binder case is `{PROTEIN_ID}`. The figures compare
the shared WT Boltz prediction with frozen edited-complex predictions for DelePref
full-control, DelePref no-scorer, cold-start plus the same scorer, and Original
SCISOR as an external reference. All edited structures were aligned to the WT
prediction using target chain V before rendering.

No Boltz, GPU inference, teacher, scorer, proposal generation, training, or
metric selection was run. Existing CIF/confidence outputs and persisted
DelePref-12/DelePref-16 metrics were read only.

- Main predicted-complex figure: `{PAPER / 'fig_binder_postdeletion_complexes_delepref.pdf'}`
- WT/DelePref/SCISOR triptych: `{PAPER / 'fig_binder_wt_delepref_scisor_triptych_delepref.pdf'}`
- Target-aligned overlay figure: `{PAPER / 'fig_binder_postdeletion_overlays_delepref.pdf'}`

{PROXY_BOUNDARY}

{CLAIM_BOUNDARY}
"""
    (OUT / "SUMMARY.md").write_text(summary, encoding="utf-8")
    write_json(
        OUT / "run_summary.json",
        {
            "status": "PASS_DelePref_BINDER_POSTDELETION_FIGURES",
            "created_at": now(),
            "case_id": CASE_ID,
            "protein_id": PROTEIN_ID,
            "figure_count": 3,
            "format_count": 9,
            "boltz_rerun": False,
            "model_or_gpu_run": False,
            "paper_copies_identical": True,
            "protocol_source_modified": False,
            "proxy_boundary": PROXY_BOUNDARY,
            "claim_boundary": CLAIM_BOUNDARY,
        },
    )
    verify_manifest()
    print("PASS_DelePref_BINDER_POSTDELETION_FIGURES")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--reuse-renders", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_manifest()
    else:
        build(reuse_renders=args.reuse_renders)


if __name__ == "__main__":
    main()
