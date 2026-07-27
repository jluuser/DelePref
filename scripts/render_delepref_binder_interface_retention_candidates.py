#!/usr/bin/env python3
"""Render interface-focused Binder case-study candidates from frozen outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/experiments/delepref_step004000_binder_interface_retention_case_candidates"
PYMOL = Path("pymol")
BINDER = ROOT / "results/experiments/delepref_step004000_binder"
BINDER_V12 = ROOT / "results/experiments/delepref_step004000_delepref_12_binder_interface_analysis"
BINDER_V16 = ROOT / "results/experiments/delepref_step004000_delepref_16_binder_coldstart_same_scorer"

CLAIM_BOUNDARY = (
    "The method ranks candidate deletion plans by teacher-derived deletion "
    "perturbation preference and optional auxiliary risk evidence."
)
PROXY_BOUNDARY = (
    "Post-deletion complex-prediction proxy; not evidence of experimental "
    "binding, Kd/Ki, or functional preservation."
)

CASES = {
    49: {
        "protein_id": "SAbDab:5XJ3:GH:I",
        "target_chain": "I",
        "contact_recovery": 0.9285714285714286,
        "scisor_contact_recovery_display": "21.4%",
        "scisor_jaccard_display": "0.103",
        "scisor_embedding_count": 8,
        "scisor_mapping_note": "range across 8 valid embeddings",
        "binder_label_xy": (0.68, 0.48),
        "binder_label_text": (0.82, 0.72),
        "target_label_xy": (0.25, 0.55),
        "target_label_text": (0.26, 0.82),
    },
    50: {
        "protein_id": "SAbDab:5XJ3:JK:L",
        "target_chain": "L",
        "contact_recovery": 1.0,
        "scisor_contact_recovery_display": "10.7%",
        "scisor_jaccard_display": "0.058",
        "scisor_embedding_count": 48,
        "scisor_mapping_note": "mapping-invariant across 48 valid embeddings",
        "binder_label_xy": (0.31, 0.56),
        "binder_label_text": (0.08, 0.79),
        "target_label_xy": (0.69, 0.47),
        "target_label_text": (0.82, 0.72),
    },
}

METHODS = (
    ("wildtype_shared", "Full-length", "WT_BASELINE"),
    ("delepref_full_control", "Ours", "SAME_PROTOCOL"),
    ("original_SCISOR_reference", "SCISOR", "EXTERNAL_REFERENCE"),
)

COLORS = {
    "binder": "#1676A6",
    "target": "#B8BEC4",
    "interface": "#A23B8F",
    "deletion": "#E64B35",
    "text": "#20252A",
    "muted": "#606A73",
}


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_status(path: Path, expected: str) -> None:
    actual = read_json(path / "run_summary.json").get("status")
    if actual != expected:
        raise RuntimeError(f"status mismatch for {path}: {actual!r} != {expected!r}")


def structure_from_confidence(confidence_path: Path) -> Path:
    name = confidence_path.name
    if not name.startswith("confidence_") or not name.endswith(".json"):
        raise RuntimeError(f"unexpected confidence file: {confidence_path}")
    return confidence_path.with_name(name[len("confidence_") : -5] + ".cif")


def parse_intervals(value: str) -> list[list[int]]:
    intervals = json.loads(value)
    if not isinstance(intervals, list) or not intervals:
        raise RuntimeError(f"invalid deletion intervals: {value}")
    return [[int(start), int(end)] for start, end in intervals]


def load_case(case_id: int) -> dict[str, Any]:
    expected = CASES[case_id]
    predictions = read_csv(BINDER / "binder_prediction_details.csv")
    metrics = read_csv(BINDER_V12 / "binder_interface_case_metrics.csv")
    plans = read_csv(BINDER / "binder_selected_plans_full_control.csv")
    manifests = read_csv(BINDER / "binder_manifest.csv")

    pred_rows = {
        row["method"]: row
        for row in predictions
        if int(row["case_id"]) == case_id and row["protein_id"] == expected["protein_id"]
    }
    metric_rows = {
        row["method"]: row
        for row in metrics
        if int(row["case_id"]) == case_id and row["protein_id"] == expected["protein_id"]
    }
    plan = next(
        row
        for row in plans
        if int(row["case_id"]) == case_id
        and row["protein_id"] == expected["protein_id"]
        and row["method"] == "delepref_full_control"
    )
    manifest = next(
        row
        for row in manifests
        if int(row["case_id"]) == case_id and row["protein_id"] == expected["protein_id"]
    )

    required_methods = {"delepref_full_control", "original_SCISOR_reference"}
    if not required_methods.issubset(pred_rows) or not required_methods.issubset(metric_rows):
        raise RuntimeError(f"missing method rows for case {case_id}")
    if manifest["sequence_length"] != "225" or plan["deleted_length"] != "22":
        raise RuntimeError(f"unexpected length contract for case {case_id}")
    if plan["strict_plan_valid"].lower() != "true" or plan["fallback_used"].lower() != "false":
        raise RuntimeError(f"invalid selected plan for case {case_id}")

    ours_metrics = metric_rows["delepref_full_control"]
    if float(ours_metrics["deleted_wt_interface_overlap_fraction"]) != 0.0:
        raise RuntimeError(f"case {case_id} does not have zero interface deletion overlap")
    if abs(float(ours_metrics["wt_to_edited_interface_contact_recovery"]) - expected["contact_recovery"]) > 1e-12:
        raise RuntimeError(f"case {case_id} contact recovery changed")

    wt_confidence = Path(pred_rows["delepref_full_control"]["wt_confidence_json"])
    records = []
    for method_id, label, comparison_class in METHODS:
        if method_id == "wildtype_shared":
            confidence_path = wt_confidence
            record = {
                "id": method_id,
                "label": label,
                "comparison_class": comparison_class,
                "confidence": float(read_json(confidence_path)["confidence_score"]),
                "wt_confidence": float(read_json(confidence_path)["confidence_score"]),
                "delta_confidence": None,
                "relative_confidence_retention": None,
                "strict_retention": None,
                "relaxed_retention": None,
                "interface_confidence": None,
                "pae": None,
                "structure": structure_from_confidence(confidence_path),
                "confidence_path": confidence_path,
                "binder_length": 225,
            }
        else:
            prediction = pred_rows[method_id]
            metric = metric_rows[method_id]
            confidence_path = Path(prediction["edited_confidence_json"])
            record = {
                "id": method_id,
                "label": label,
                "comparison_class": comparison_class,
                "confidence": float(prediction["edited_confidence"]),
                "wt_confidence": float(prediction["wt_confidence"]),
                "delta_confidence": float(prediction["delta_confidence"]),
                "relative_confidence_retention": float(prediction["relative_confidence_retention"]),
                "strict_retention": prediction["strict_retention"].lower() == "true",
                "relaxed_retention": prediction["relaxed_retention"].lower() == "true",
                "interface_confidence": float(metric["interface_confidence"]),
                "pae": float(metric["binder_target_interchain_pae"]),
                "structure": structure_from_confidence(confidence_path),
                "confidence_path": confidence_path,
                "binder_length": 203,
            }
        records.append(record)

    intervals = parse_intervals(plan["deleted_intervals"])
    return {
        "case_id": case_id,
        "protein_id": expected["protein_id"],
        "target_chain": expected["target_chain"],
        "binder_chain": "A",
        "wt_length": 225,
        "edited_length": 203,
        "deleted_length": 22,
        "deleted_intervals_0based": intervals,
        "deleted_intervals_1based": [[start + 1, end + 1] for start, end in intervals],
        "records": records,
        "ours_metrics": ours_metrics,
        "scisor_metrics": metric_rows["original_SCISOR_reference"],
        "scisor_contact_recovery_display": expected["scisor_contact_recovery_display"],
        "scisor_jaccard_display": expected["scisor_jaccard_display"],
        "scisor_embedding_count": expected["scisor_embedding_count"],
        "scisor_mapping_note": expected["scisor_mapping_note"],
        "plan": plan,
        "manifest": manifest,
        "binder_label_xy": expected["binder_label_xy"],
        "binder_label_text": expected["binder_label_text"],
        "target_label_xy": expected["target_label_xy"],
        "target_label_text": expected["target_label_text"],
    }


def verify_sources(case: dict[str, Any]) -> list[dict[str, Any]]:
    require_status(BINDER, "PASS_DelePref_BINDER")
    require_status(BINDER_V12, "PASS_DelePref_12_BINDER_INTERFACE_ANALYSIS")
    require_status(BINDER_V16, "PASS_DelePref_16_BINDER_COLDSTART_SAME_SCORER")
    if not PYMOL.is_file():
        raise RuntimeError(f"missing PyMOL executable: {PYMOL}")

    sources = []
    for record in case["records"]:
        for role, path in (("structure", record["structure"]), ("confidence", record["confidence_path"])):
            if not path.is_file():
                raise RuntimeError(f"missing source {path}")
            sources.append(
                {
                    "case_id": case["case_id"],
                    "method": record["id"],
                    "role": role,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    for path in (
        BINDER / "binder_manifest.csv",
        BINDER / "binder_selected_plans_full_control.csv",
        BINDER / "binder_prediction_details.csv",
        BINDER_V12 / "binder_interface_case_metrics.csv",
    ):
        sources.append(
            {
                "case_id": case["case_id"],
                "method": "shared",
                "role": path.name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return sources


def pymol_script(case: dict[str, Any]) -> str:
    methods = [{"id": r["id"], "path": str(r["structure"])} for r in case["records"]]
    deletion_expr = "+".join(
        f"{start}-{end}" if start != end else str(start)
        for start, end in case["deleted_intervals_1based"]
    )
    palette = {
        "binder_blue": [0.086, 0.463, 0.651],
        "target_gray": [0.72, 0.745, 0.77],
        "interface_magenta": [0.635, 0.231, 0.561],
        "deletion_coral": [0.902, 0.294, 0.208],
    }
    return f'''from pymol import cmd, stored
import json
from pathlib import Path

OUT = Path({str(OUT)!r})
CASE_ID = {case['case_id']!r}
METHODS = {methods!r}
BINDER_CHAIN = {case['binder_chain']!r}
TARGET_CHAIN = {case['target_chain']!r}
DELETION_EXPR = {deletion_expr!r}
COLORS = {palette!r}

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
binder_sel = f"chain {{BINDER_CHAIN}}"
target_sel = f"chain {{TARGET_CHAIN}}"
audit = {{"case_id": CASE_ID, "binder_chain": BINDER_CHAIN, "target_chain": TARGET_CHAIN, "alignments": {{}}}}
for method in METHODS:
    obj = method["id"]
    chains = cmd.get_chains(obj)
    counts = {{chain: cmd.count_atoms(f"{{obj}} and chain {{chain}} and name CA") for chain in chains}}
    if BINDER_CHAIN not in chains or TARGET_CHAIN not in chains:
        raise RuntimeError(f"missing expected chains for {{obj}}: {{chains}}")
    if obj == wt:
        aligned = [0.0, counts[TARGET_CHAIN]]
    else:
        result = cmd.align(
            f"{{obj}} and {{target_sel}} and name CA",
            f"{{wt}} and {{target_sel}} and name CA",
            cycles=0,
            transform=1,
        )
        aligned = [float(result[0]), int(result[1])]
    audit["alignments"][obj] = {{"target_rmsd": aligned[0], "aligned_target_ca": aligned[1], "ca_counts": counts}}

cmd.disable("all")
cmd.enable(wt)
cmd.orient(wt)
cmd.zoom(wt, buffer=4.0)
cmd.turn("y", 16)
cmd.turn("x", -8)
cmd.turn("z", -18)
whole_view = list(cmd.get_view())
audit["whole_view"] = whole_view
interface_view = None

def style_object(obj):
    cmd.hide("everything", "all")
    cmd.show("cartoon", obj)
    cmd.color("binder_blue", f"{{obj}} and {{binder_sel}}")
    cmd.color("target_gray", f"{{obj}} and {{target_sel}}")
    cmd.show("surface", f"{{obj}} and {{binder_sel}}")
    cmd.set("transparency", 0.80, f"{{obj}} and {{binder_sel}}")
    cmd.show("surface", f"{{obj}} and {{target_sel}}")
    cmd.set("transparency", 0.68, f"{{obj}} and {{target_sel}}")
    binder_iface = f"iface_b_{{obj}}"
    target_iface = f"iface_t_{{obj}}"
    cmd.select(binder_iface, f"byres (({{obj}} and {{binder_sel}}) within 5.0 of ({{obj}} and {{target_sel}}))")
    cmd.select(target_iface, f"byres (({{obj}} and {{target_sel}}) within 5.0 of ({{obj}} and {{binder_sel}}))")
    cmd.color("interface_magenta", binder_iface)
    cmd.color("interface_magenta", target_iface)
    cmd.show("sticks", binder_iface)
    cmd.show("sticks", target_iface)
    cmd.set("stick_radius", 0.15, binder_iface)
    cmd.set("stick_radius", 0.15, target_iface)
    return binder_iface, target_iface

case_dir = OUT / f"case{{CASE_ID:03d}}"
(case_dir / "raw_renders").mkdir(parents=True, exist_ok=True)
for method in METHODS:
    obj = method["id"]
    cmd.disable("all")
    cmd.enable(obj)
    binder_iface, target_iface = style_object(obj)
    if obj == wt:
        deleted = f"{{obj}} and {{binder_sel}} and resi {{DELETION_EXPR}}"
        cmd.color("deletion_coral", deleted)
        cmd.show("sticks", deleted)
        cmd.set("stick_radius", 0.18, deleted)
        cmd.select("deleted_interface_overlap", f"({{deleted}}) and ({{binder_iface}})")
        audit["deleted_interface_overlap_ca"] = cmd.count_atoms("deleted_interface_overlap and name CA")
        stored.interface_residues = []
        cmd.iterate(f"{{binder_iface}} and name CA", "stored.interface_residues.append(int(resi))")
        audit["wt_binder_interface_residues_1based"] = sorted(set(stored.interface_residues))
    cmd.set_view(whole_view)
    cmd.ray(1800, 1350)
    cmd.png(str(case_dir / "raw_renders" / f"whole_{{obj}}.png"), dpi=400)

    if interface_view is None:
        cmd.set_view(whole_view)
        cmd.zoom(f"{{binder_iface}} or {{target_iface}}", buffer=5.5)
        interface_view = list(cmd.get_view())
    else:
        cmd.set_view(interface_view)
    cmd.ray(1300, 950)
    cmd.png(str(case_dir / "raw_renders" / f"interface_{{obj}}.png"), dpi=400)
    cmd.delete(binder_iface)
    cmd.delete(target_iface)
    if obj == wt:
        cmd.delete("deleted_interface_overlap")

audit["interface_view"] = interface_view
(case_dir / "pymol_alignment_and_render_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\\n")
cmd.quit()
'''


def run_pymol(case: dict[str, Any]) -> Path:
    case_dir = OUT / f"case{case['case_id']:03d}"
    script_path = case_dir / "scripts" / "render_case.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(pymol_script(case), encoding="utf-8")
    completed = subprocess.run(
        [str(PYMOL), "-cq", "-r", str(script_path)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=900,
    )
    (case_dir / "pymol_render.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"PyMOL failed for case {case['case_id']}: {case_dir / 'pymol_render.log'}")
    return script_path


def crop_image(path: Path, pad: int = 20) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    alpha_box = image.getchannel("A").getbbox()
    if alpha_box is None:
        raise RuntimeError(f"blank render: {path}")
    left, top, right, bottom = alpha_box
    return image.crop(
        (
            max(0, left - pad),
            max(0, top - pad),
            min(image.width, right + pad),
            min(image.height, bottom + pad),
        )
    )


def normalized_images(
    paths: dict[str, Path],
    target_height: int,
    *,
    bottom_pad: int,
    crop: bool = True,
) -> dict[str, Image.Image]:
    cropped = {
        key: (crop_image(path) if crop else Image.open(path).convert("RGBA"))
        for key, path in paths.items()
    }
    resized = {}
    for key, image in cropped.items():
        width = max(1, round(image.width * target_height / image.height))
        resized[key] = image.resize((width, target_height), Image.Resampling.LANCZOS)
    max_width = max(image.width for image in resized.values())
    output = {}
    for key, image in resized.items():
        canvas = Image.new(
            "RGBA",
            (max_width + 40, target_height + 20 + bottom_pad),
            (255, 255, 255, 0),
        )
        canvas.alpha_composite(image, ((canvas.width - image.width) // 2, 10))
        output[key] = canvas
    return output


def add_sequence_track(ax: plt.Axes, case: dict[str, Any], interface_residues: list[int]) -> None:
    length = case["wt_length"]
    ax.set_xlim(0, length + 1)
    ax.set_ylim(0, 1.30)
    ax.axis("off")
    ax.add_patch(Rectangle((1, 0.40), length, 0.20, facecolor=COLORS["binder"], edgecolor="none", alpha=0.78))
    for residue in interface_residues:
        ax.add_patch(Rectangle((residue, 0.32), 1.0, 0.36, facecolor=COLORS["interface"], edgecolor="none"))
    previous_end = None
    crowded_label = False
    for start, end in case["deleted_intervals_1based"]:
        ax.add_patch(
            Rectangle(
                (start, 0.25),
                end - start + 1,
                0.50,
                facecolor=COLORS["deletion"],
                edgecolor="white",
                linewidth=0.8,
            )
        )
        if previous_end is not None and start - previous_end < 12:
            crowded_label = not crowded_label
        else:
            crowded_label = False
        label_y = 1.05 if crowded_label else 0.82
        ax.text(
            (start + end) / 2,
            label_y,
            f"{start}-{end}",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color=COLORS["deletion"],
            fontweight="bold",
        )
        previous_end = end
    ax.text(1, 0.08, "1", ha="center", va="top", fontsize=8, color=COLORS["muted"])
    ax.text(length, 0.08, str(length), ha="center", va="top", fontsize=8, color=COLORS["muted"])
    ax.text(
        length / 2,
        1.22,
        "Ours deletion map (binder residues): "
        + ", ".join(f"{start}-{end}" for start, end in case["deleted_intervals_1based"])
        + " | red = deleted, purple = predicted interface",
        ha="center",
        va="center",
        fontsize=10.3,
        fontweight="bold",
        color=COLORS["text"],
    )


def make_figure(case: dict[str, Any]) -> list[Path]:
    case_dir = OUT / f"case{case['case_id']:03d}"
    raw = case_dir / "raw_renders"
    ids = [method_id for method_id, _, _ in METHODS]
    whole = normalized_images(
        {method_id: raw / f"whole_{method_id}.png" for method_id in ids},
        920,
        bottom_pad=10,
    )
    interface = normalized_images(
        {method_id: raw / f"interface_{method_id}.png" for method_id in ids},
        650,
        bottom_pad=95,
        crop=False,
    )
    audit = read_json(case_dir / "pymol_alignment_and_render_audit.json")
    if audit.get("deleted_interface_overlap_ca") != 0:
        raise RuntimeError(f"render audit found deletion-interface overlap for case {case['case_id']}")

    fig = plt.figure(figsize=(12.6, 7.6), facecolor="white")
    grid = fig.add_gridspec(
        4,
        3,
        height_ratios=[3.55, 0.72, 2.15, 0.82],
        hspace=0.055,
        wspace=0.025,
    )
    records = {record["id"]: record for record in case["records"]}
    titles = {
        "wildtype_shared": "A  Full-length",
        "delepref_full_control": "B  Ours",
        "original_SCISOR_reference": "C  SCISOR",
    }

    legend = [
        Patch(facecolor=COLORS["binder"], label="Binder"),
        Patch(facecolor=COLORS["target"], label="Target (unchanged)"),
        Patch(facecolor=COLORS["interface"], label="Predicted interface"),
        Patch(facecolor=COLORS["deletion"], label="Ours deletions (shown on full-length)"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=False, fontsize=10.5, handlelength=1.25, columnspacing=2.0)

    for column, method_id in enumerate(ids):
        record = records[method_id]
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(whole[method_id])
        ax.axis("off")
        ax.set_title(titles[method_id], fontsize=15, fontweight="bold", color=COLORS["text"], pad=0)
        if method_id == "wildtype_shared":
            note = (
                f"Binder 225 aa | confidence {record['confidence']:.3f}\n"
                "Red segments removed by Ours"
            )
        else:
            note = (
                f"Binder 203 aa (-10%)\n"
                f"confidence {record['confidence']:.3f} | delta confidence {record['delta_confidence']:+.3f}\n"
                f"retention {100 * record['relative_confidence_retention']:.1f}%\n"
                f"ipTM {record['interface_confidence']:.3f} | PAE {record['pae']:.2f} A"
            )
        metric_ax = fig.add_subplot(grid[1, column])
        metric_ax.axis("off")
        metric_ax.text(
            0.5,
            0.52,
            note,
            transform=metric_ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.0 if method_id != "wildtype_shared" else 10.0,
            color=COLORS["text"],
            linespacing=1.23,
        )
        if method_id == "wildtype_shared":
            ax.annotate(
                "Binder",
                xy=case["binder_label_xy"],
                xytext=case["binder_label_text"],
                xycoords="axes fraction",
                textcoords="axes fraction",
                fontsize=9.5,
                fontweight="bold",
                color=COLORS["binder"],
                ha="center",
                arrowprops={"arrowstyle": "->", "color": COLORS["binder"], "lw": 1.4},
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.86},
            )
            ax.annotate(
                "Target (unchanged)",
                xy=case["target_label_xy"],
                xytext=case["target_label_text"],
                xycoords="axes fraction",
                textcoords="axes fraction",
                fontsize=9.2,
                fontweight="bold",
                color="#707880",
                ha="center",
                arrowprops={"arrowstyle": "->", "color": "#707880", "lw": 1.4},
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.86},
            )

        zoom = fig.add_subplot(grid[2, column])
        zoom.imshow(interface[method_id])
        zoom.axis("off")
        if method_id == "wildtype_shared":
            zoom_note = f"Predicted interface before deletion\n{case['ours_metrics']['wt_contact_count']} reference contacts"
        elif method_id == "delepref_full_control":
            recovery = 100 * float(case["ours_metrics"]["wt_to_edited_interface_contact_recovery"])
            jaccard = float(case["ours_metrics"]["interface_contact_jaccard"])
            zoom_note = (
                f"Contact recovery {recovery:.1f}% | Jaccard {jaccard:.2f}"
            )
        else:
            zoom_note = (
                f"Contact recovery {case['scisor_contact_recovery_display']} | "
                f"Jaccard {case['scisor_jaccard_display']}"
            )
        zoom.text(
            0.5,
            0.02,
            zoom_note,
            transform=zoom.transAxes,
            ha="center",
            va="bottom",
            fontsize=8.4 if method_id == "original_SCISOR_reference" else 9.2,
            color=COLORS["text"],
            linespacing=1.2,
        )

    track = fig.add_subplot(grid[3, :])
    add_sequence_track(track, case, audit["wt_binder_interface_residues_1based"])
    fig.text(0.01, 0.016, f"Illustrative frozen case {case['case_id']}: {case['protein_id']}", ha="left", va="bottom", fontsize=8.8, color=COLORS["muted"])
    fig.text(
        0.5,
        0.016,
        "confidence / retention / ipTM: higher is better | delta / PAE: lower is better",
        ha="center",
        va="bottom",
        fontsize=8.1,
        color=COLORS["muted"],
    )
    fig.text(0.99, 0.016, "Post-deletion complex-prediction proxy", ha="right", va="bottom", fontsize=8.8, color=COLORS["muted"])
    fig.subplots_adjust(left=0.015, right=0.985, top=0.925, bottom=0.07)

    figure_dir = case_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    stem = f"fig_binder_interface_retention_case{case['case_id']}"
    for suffix in ("pdf", "svg", "png"):
        path = figure_dir / f"{stem}.{suffix}"
        kwargs: dict[str, Any] = {"facecolor": "white"}
        if suffix == "png":
            kwargs["dpi"] = 420
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def image_audit(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        white = Image.new("RGB", rgb.size, "white")
        return {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "nonblank": ImageChops.difference(rgb, white).getbbox() is not None,
        }


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_sources = []
    generated = []
    summaries = []
    for case_id in sorted(CASES):
        case = load_case(case_id)
        records_by_id = {record["id"]: record for record in case["records"]}
        all_sources.extend(verify_sources(case))
        script_path = run_pymol(case)
        outputs = make_figure(case)
        generated.append(
            {
                "case_id": case_id,
                "role": "pymol_script",
                "path": str(script_path),
                "sha256": sha256_file(script_path),
                "size_bytes": script_path.stat().st_size,
            }
        )
        for path in outputs:
            item = {
                "case_id": case_id,
                "role": f"figure_{path.suffix[1:]}",
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            if path.suffix == ".png":
                item["image"] = image_audit(path)
            generated.append(item)
        summaries.append(
            {
                "case_id": case_id,
                "protein_id": case["protein_id"],
                "deleted_intervals_1based": case["deleted_intervals_1based"],
                "contact_recovery": float(case["ours_metrics"]["wt_to_edited_interface_contact_recovery"]),
                "contact_jaccard": float(case["ours_metrics"]["interface_contact_jaccard"]),
                "binder_interface_residue_recovery": float(case["ours_metrics"]["binder_interface_residue_recovery"]),
                "deleted_interface_overlap_count": int(case["ours_metrics"]["deleted_wt_interface_overlap_count"]),
                "full_length_confidence": float(records_by_id["wildtype_shared"]["confidence"]),
                "ours_confidence": float(records_by_id["delepref_full_control"]["confidence"]),
                "ours_delta_confidence": float(records_by_id["delepref_full_control"]["delta_confidence"]),
                "ours_relative_confidence_retention": float(
                    records_by_id["delepref_full_control"]["relative_confidence_retention"]
                ),
                "ours_strict_retention": bool(records_by_id["delepref_full_control"]["strict_retention"]),
                "ours_relaxed_retention": bool(records_by_id["delepref_full_control"]["relaxed_retention"]),
                "scisor_confidence": float(records_by_id["original_SCISOR_reference"]["confidence"]),
                "scisor_delta_confidence": float(
                    records_by_id["original_SCISOR_reference"]["delta_confidence"]
                ),
                "scisor_relative_confidence_retention": float(
                    records_by_id["original_SCISOR_reference"]["relative_confidence_retention"]
                ),
                "scisor_strict_retention": bool(
                    records_by_id["original_SCISOR_reference"]["strict_retention"]
                ),
                "scisor_relaxed_retention": bool(
                    records_by_id["original_SCISOR_reference"]["relaxed_retention"]
                ),
                "scisor_contact_recovery_display": case["scisor_contact_recovery_display"],
                "scisor_jaccard_display": case["scisor_jaccard_display"],
                "scisor_embedding_count": case["scisor_embedding_count"],
                "scisor_mapping_note": case["scisor_mapping_note"],
                "ours_iptm": float(case["ours_metrics"]["interface_confidence"]),
                "ours_pae": float(case["ours_metrics"]["binder_target_interchain_pae"]),
                "scisor_iptm": float(case["scisor_metrics"]["interface_confidence"]),
                "scisor_pae": float(case["scisor_metrics"]["binder_target_interchain_pae"]),
            }
        )

    manifest = {
        "schema_version": "delepref_binder_interface_retention_case_candidates_v2",
        "created_at": now(),
        "status": "PASS_DelePref_BINDER_INTERFACE_RETENTION_CASE_CANDIDATES",
        "cases": summaries,
        "source_artifacts": all_sources,
        "generated_artifacts": generated,
        "boltz_or_model_rerun": False,
        "selection_role": "post_hoc_illustrative_candidates; not representative aggregate evidence",
        "claim_boundary": CLAIM_BOUNDARY,
        "proxy_boundary": PROXY_BOUNDARY,
    }
    write_json(OUT / "figure_manifest.json", manifest)
    write_json(
        OUT / "run_summary.json",
        {
            "schema_version": "delepref_binder_interface_retention_case_candidates_summary_v2",
            "created_at": now(),
            "status": manifest["status"],
            "cases": summaries,
            "figure_formats_per_case": 3,
            "boltz_or_model_rerun": False,
            "claim_boundary": CLAIM_BOUNDARY,
            "proxy_boundary": PROXY_BOUNDARY,
        },
    )
    summary_lines = [
        "# DelePref Binder interface-retention case candidates",
        "",
        "Status: `PASS_DelePref_BINDER_INTERFACE_RETENTION_CASE_CANDIDATES`",
        "",
        "These visualization-only candidates reuse frozen WT, Ours, and SCISOR Boltz outputs.",
        "No Boltz, GPU inference, teacher, scorer, candidate generation, or training was run.",
        "",
    ]
    for record in summaries:
        summary_lines.extend(
            [
                f"## Case {record['case_id']}: `{record['protein_id']}`",
                "",
                f"- Ours deletion intervals (1-based): `{record['deleted_intervals_1based']}`",
                f"- Predicted contact recovery: `{record['contact_recovery']:.6f}`",
                f"- Contact Jaccard: `{record['contact_jaccard']:.6f}`",
                f"- Binder interface-residue recovery: `{record['binder_interface_residue_recovery']:.6f}`",
                f"- Deleted predicted-interface residues: `{record['deleted_interface_overlap_count']}`",
                f"- Full-length confidence: `{record['full_length_confidence']:.6f}`",
                f"- Ours confidence / delta / retention: `{record['ours_confidence']:.6f}` / "
                f"`{record['ours_delta_confidence']:.6f}` / "
                f"`{record['ours_relative_confidence_retention']:.6f}`",
                f"- Ours strict / relaxed retention: `{record['ours_strict_retention']}` / "
                f"`{record['ours_relaxed_retention']}`",
                f"- SCISOR confidence / delta / retention: `{record['scisor_confidence']:.6f}` / "
                f"`{record['scisor_delta_confidence']:.6f}` / "
                f"`{record['scisor_relative_confidence_retention']:.6f}`",
                f"- SCISOR strict / relaxed retention: `{record['scisor_strict_retention']}` / "
                f"`{record['scisor_relaxed_retention']}`",
                f"- SCISOR contact recovery / Jaccard (mapping sensitivity): `"
                f"{record['scisor_contact_recovery_display']}` / `"
                f"{record['scisor_jaccard_display']}` ({record['scisor_mapping_note']})",
                f"- Ours / SCISOR ipTM: `{record['ours_iptm']:.6f}` / `{record['scisor_iptm']:.6f}`",
                f"- Ours / SCISOR inter-chain PAE: `{record['ours_pae']:.6f}` / `{record['scisor_pae']:.6f}`",
                "",
            ]
        )
    summary_lines.extend([PROXY_BOUNDARY, "", CLAIM_BOUNDARY, ""])
    (OUT / "SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")

    for record in all_sources + generated:
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"artifact verification failed: {path}")
    (OUT / "PASS_DelePref_BINDER_INTERFACE_RETENTION_CASE_CANDIDATES").write_text(
        "PASS_DelePref_BINDER_INTERFACE_RETENTION_CASE_CANDIDATES\n", encoding="utf-8"
    )


def verify() -> None:
    manifest = read_json(OUT / "figure_manifest.json")
    for section in ("source_artifacts", "generated_artifacts"):
        for record in manifest[section]:
            path = Path(record["path"])
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"manifest mismatch: {path}")
    print("PASS_VERIFY_ONLY")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify()
    else:
        build()


if __name__ == "__main__":
    main()
