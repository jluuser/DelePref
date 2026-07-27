#!/usr/bin/env python3
"""Render the DelePref-22 20% Binder case study from persisted predictions."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[1]
V22 = ROOT / "results/experiments/delepref_step004000_delepref_22_binder_budget_case_search"
OUT = V22 / "case_study_case012"
PYMOL = Path("pymol")
CASE_ID = "12"
PROTEIN_ID = "AB-Bind:2NY7:G_HL"
BINDER_CHAIN = "G"
TARGET_CHAINS = ("H", "L")
WT_LENGTH = 290
EDITED_LENGTH = 232
DELETED_LENGTH = 58
CLAIM_BOUNDARY = (
    "The method ranks candidate deletion plans by teacher-derived deletion "
    "perturbation preference and optional auxiliary risk evidence."
)
PROXY_BOUNDARY = (
    "Post-deletion complex-prediction proxy only; not evidence of experimental "
    "binding, Kd/Ki, or functional preservation."
)

COLORS = {
    "binder": "#1676A6",
    "target": "#B8BEC4",
    "interface": "#9B3A8E",
    "ours_deletion": "#E64B35",
    "scisor_deletion": "#D89B2B",
    "text": "#20252A",
    "muted": "#606A73",
    "track": "#D9DEE2",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def structure_from_confidence(path: Path) -> Path:
    if not path.name.startswith("confidence_") or path.suffix != ".json":
        raise ValueError(path)
    return path.with_name(path.name[len("confidence_") : -5] + ".cif")


def intervals(value: str) -> list[tuple[int, int]]:
    return [(int(start), int(end)) for start, end in json.loads(value)]


def one_based(values: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(start + 1, end + 1) for start, end in values]


def deletion_expression(values: Iterable[tuple[int, int]]) -> str:
    return "+".join(str(start) if start == end else f"{start}-{end}" for start, end in values)


def load_case() -> dict[str, Any]:
    summary = read_json(V22 / "run_summary.json")
    validation = read_json(V22 / "independent_validation.json")
    if summary.get("status") != "PASS_DelePref_22_BINDER_B20_CASE_FOUND" or validation.get("status") != "PASS":
        raise RuntimeError("DelePref-22 package is not validated")
    if summary.get("eligible_case_ids") != [12]:
        raise RuntimeError("eligible case identity changed")

    plans = [row for row in read_csv(V22 / "selected_plans.csv") if row["case_id"] == CASE_ID]
    metrics = [row for row in read_csv(V22 / "case_metrics.csv") if row["case_id"] == CASE_ID]
    if len(plans) != 2 or len(metrics) != 2:
        raise RuntimeError("case 12 does not have two paired methods")
    plan_by_method = {row["method"]: row for row in plans}
    metric_by_method = {row["method"]: row for row in metrics}
    if set(plan_by_method) != {"ours", "scisor"} or set(metric_by_method) != {"ours", "scisor"}:
        raise RuntimeError("method identity mismatch")

    ours = metric_by_method["ours"]
    scisor = metric_by_method["scisor"]
    if float(ours["contact_recovery"]) < 0.85:
        raise RuntimeError("case 12 no longer meets the visualization gate")
    if not (
        float(ours["contact_recovery"]) > float(scisor["contact_recovery"])
        and float(ours["iptm"]) > float(scisor["iptm"])
        and float(ours["interchain_pae"]) < float(scisor["interchain_pae"])
    ):
        raise RuntimeError("paired favorable directions changed")

    wt_confidence = Path(plan_by_method["ours"]["wt_confidence_json"])
    records = [
        {
            "id": "full_length",
            "label": "Full-length",
            "structure": structure_from_confidence(wt_confidence),
            "confidence_path": wt_confidence,
            "confidence": float(read_json(wt_confidence)["confidence_score"]),
            "length": WT_LENGTH,
            "intervals": [],
        }
    ]
    for method in ("ours", "scisor"):
        plan = plan_by_method[method]
        metric = metric_by_method[method]
        confidence_path = Path(metric["confidence_json"])
        parsed_intervals = intervals(plan["deleted_intervals"])
        if int(plan["deleted_length"]) != DELETED_LENGTH or len(plan["edited_sequence"]) != EDITED_LENGTH:
            raise RuntimeError(f"unexpected {method} deletion contract")
        records.append(
            {
                "id": method,
                "label": "Ours" if method == "ours" else "SCISOR",
                "structure": structure_from_confidence(confidence_path),
                "confidence_path": confidence_path,
                "confidence": float(metric["confidence"]),
                "delta_confidence": float(metric["delta_confidence"]),
                "contact_recovery": float(metric["contact_recovery"]),
                "contact_jaccard": float(metric["contact_jaccard"]),
                "interface_residue_recovery": float(metric["interface_residue_recovery"]),
                "iptm": float(metric["iptm"]),
                "pae": float(metric["interchain_pae"]),
                "clashes": int(metric["steric_clash_count"]),
                "length": EDITED_LENGTH,
                "intervals": one_based(parsed_intervals),
                "span_count": len(parsed_intervals),
            }
        )
    for record in records:
        for key in ("structure", "confidence_path"):
            if not record[key].is_file():
                raise FileNotFoundError(record[key])
    return {"records": records, "plans": plan_by_method, "metrics": metric_by_method}


def pymol_script(case: dict[str, Any]) -> str:
    methods = [{"id": row["id"], "path": str(row["structure"])} for row in case["records"]]
    ours_intervals = next(row["intervals"] for row in case["records"] if row["id"] == "ours")
    target_expression = "+".join(TARGET_CHAINS)
    palette = {
        "binder_blue": [0.086, 0.463, 0.651],
        "target_gray": [0.72, 0.745, 0.77],
        "interface_magenta": [0.608, 0.227, 0.557],
        "deletion_coral": [0.902, 0.294, 0.208],
    }
    return f'''from pymol import cmd, stored
import json
from pathlib import Path

OUT = Path({str(OUT)!r})
METHODS = {methods!r}
BINDER_CHAIN = {BINDER_CHAIN!r}
TARGET_CHAINS = {TARGET_CHAINS!r}
TARGET_EXPRESSION = {target_expression!r}
OURS_DELETION_EXPRESSION = {deletion_expression(ours_intervals)!r}
COLORS = {palette!r}

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

wt = "full_length"
binder_sel = f"chain {{BINDER_CHAIN}}"
target_sel = f"chain {{TARGET_EXPRESSION}}"
audit = {{"binder_chain": BINDER_CHAIN, "target_chains": TARGET_CHAINS, "alignments": {{}}}}
for method in METHODS:
    obj = method["id"]
    chains = cmd.get_chains(obj)
    if BINDER_CHAIN not in chains or any(chain not in chains for chain in TARGET_CHAINS):
        raise RuntimeError(f"missing expected chains for {{obj}}: {{chains}}")
    counts = {{chain: cmd.count_atoms(f"{{obj}} and chain {{chain}} and name CA") for chain in chains}}
    if obj == wt:
        result = [0.0, sum(counts[chain] for chain in TARGET_CHAINS)]
    else:
        aligned = cmd.align(
            f"{{obj}} and ({{target_sel}}) and name CA",
            f"{{wt}} and ({{target_sel}}) and name CA",
            cycles=0,
            transform=1,
        )
        result = [float(aligned[0]), int(aligned[1])]
    audit["alignments"][obj] = {{"target_rmsd": result[0], "aligned_target_ca": result[1], "ca_counts": counts}}

cmd.disable("all")
cmd.enable(wt)
cmd.orient(wt)
cmd.zoom(wt, buffer=2.0)
cmd.turn("y", 12)
cmd.turn("x", -10)
cmd.turn("z", -10)
whole_view = list(cmd.get_view())
audit["whole_view_scope"] = "complete binder-target complex; no target cropping"

def style(obj):
    cmd.hide("everything", "all")
    cmd.show("cartoon", obj)
    cmd.color("binder_blue", f"{{obj}} and {{binder_sel}}")
    cmd.color("target_gray", f"{{obj}} and ({{target_sel}})")
    cmd.show("surface", f"{{obj}} and {{binder_sel}}")
    cmd.set("transparency", 0.74, f"{{obj}} and {{binder_sel}}")
    cmd.show("surface", f"{{obj}} and ({{target_sel}})")
    cmd.set("transparency", 0.86, f"{{obj}} and ({{target_sel}})")
    binder_iface = f"iface_b_{{obj}}"
    target_iface = f"iface_t_{{obj}}"
    cmd.select(binder_iface, f"byres (({{obj}} and {{binder_sel}}) within 5.0 of ({{obj}} and ({{target_sel}})))")
    cmd.select(target_iface, f"byres (({{obj}} and ({{target_sel}})) within 5.0 of ({{obj}} and {{binder_sel}}))")
    cmd.color("interface_magenta", binder_iface)
    cmd.color("interface_magenta", target_iface)
    cmd.show("sticks", binder_iface)
    cmd.show("sticks", target_iface)
    cmd.set("stick_radius", 0.16, binder_iface)
    cmd.set("stick_radius", 0.16, target_iface)
    return binder_iface, target_iface

(OUT / "raw_renders").mkdir(parents=True, exist_ok=True)
for method in METHODS:
    obj = method["id"]
    cmd.disable("all")
    cmd.enable(obj)
    binder_iface, target_iface = style(obj)
    if obj == wt:
        deleted = f"{{obj}} and {{binder_sel}} and resi {{OURS_DELETION_EXPRESSION}}"
        cmd.color("deletion_coral", deleted)
        cmd.show("sticks", deleted)
        cmd.set("stick_radius", 0.18, deleted)
        stored.interface_residues = []
        cmd.iterate(f"{{binder_iface}} and name CA", "stored.interface_residues.append(int(resi))")
        audit["full_length_interface_residues_1based"] = sorted(set(stored.interface_residues))
        cmd.select("ours_deleted_interface", f"({{deleted}}) and ({{binder_iface}})")
        audit["ours_deleted_interface_ca"] = cmd.count_atoms("ours_deleted_interface and name CA")
    cmd.set_view(whole_view)
    cmd.ray(1900, 1450)
    cmd.png(str(OUT / "raw_renders" / f"whole_{{obj}}.png"), dpi=420)
    cmd.delete(binder_iface)
    cmd.delete(target_iface)
    if obj == wt:
        cmd.delete("ours_deleted_interface")

audit["whole_view"] = whole_view
(OUT / "pymol_alignment_and_render_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\\n")
cmd.quit()
'''


def render_pymol(case: dict[str, Any]) -> Path:
    script = OUT / "scripts/render_case12.py"
    write_text(script, pymol_script(case))
    completed = subprocess.run(
        [str(PYMOL), "-cq", "-r", str(script)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=1200,
    )
    write_text(OUT / "pymol_render.log", completed.stdout)
    required = [OUT / "raw_renders" / f"whole_{method}.png" for method in ("full_length", "ours", "scisor")]
    if completed.returncode != 0 or "Traceback (most recent call last)" in completed.stdout:
        raise RuntimeError(f"PyMOL failed: {OUT / 'pymol_render.log'}")
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"PyMOL omitted required renders: {missing}")
    return script


def common_crop(paths: dict[str, Path], pad: int = 18) -> dict[str, Image.Image]:
    images = {key: Image.open(path).convert("RGBA") for key, path in paths.items()}
    if len({image.size for image in images.values()}) != 1:
        raise RuntimeError("render dimensions differ")
    union: tuple[int, int, int, int] | None = None
    for image in images.values():
        box = image.getchannel("A").getbbox()
        if box is None:
            raise RuntimeError("blank PyMOL render")
        if union is None:
            union = box
        else:
            union = (min(union[0], box[0]), min(union[1], box[1]), max(union[2], box[2]), max(union[3], box[3]))
    assert union is not None
    width, height = next(iter(images.values())).size
    crop = (max(0, union[0] - pad), max(0, union[1] - pad), min(width, union[2] + pad), min(height, union[3] + pad))
    return {key: image.crop(crop) for key, image in images.items()}


def add_deletion_tracks(ax: plt.Axes, case: dict[str, Any], interface_residues: list[int]) -> None:
    records = {row["id"]: row for row in case["records"]}
    ax.set_xlim(-28, WT_LENGTH + 2)
    ax.set_ylim(0.0, 1.85)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    for y, method, color, label in (
        (1.25, "ours", COLORS["ours_deletion"], "Ours"),
        (0.35, "scisor", COLORS["scisor_deletion"], "SCISOR"),
    ):
        ax.add_patch(Rectangle((1, y - 0.11), WT_LENGTH, 0.22, facecolor=COLORS["track"], edgecolor="none"))
        for residue in interface_residues:
            ax.add_patch(Rectangle((residue, y - 0.18), 1, 0.36, facecolor=COLORS["interface"], edgecolor="none", alpha=0.78))
        for start, end in records[method]["intervals"]:
            ax.add_patch(Rectangle((start, y - 0.23), end - start + 1, 0.46, facecolor=color, edgecolor="white", linewidth=0.35))
        ax.text(-2, y, label, ha="right", va="center", fontsize=9.4, fontweight="bold", color=COLORS["text"])
        ax.text(
            WT_LENGTH + 1.5,
            y,
            f"58 aa / {records[method]['span_count']} spans",
            ha="left",
            va="center",
            fontsize=8.0,
            color=COLORS["muted"],
        )
    ours_ranges = ", ".join(f"{a}-{b}" for a, b in records["ours"]["intervals"])
    ax.text(1, 1.72, f"Ours deletions: {ours_ranges}", ha="left", va="center", fontsize=8.4, color=COLORS["ours_deletion"], fontweight="bold")


def make_figure(case: dict[str, Any]) -> list[Path]:
    ids = [row["id"] for row in case["records"]]
    records = {row["id"]: row for row in case["records"]}
    whole = common_crop({key: OUT / "raw_renders" / f"whole_{key}.png" for key in ids})
    audit = read_json(OUT / "pymol_alignment_and_render_audit.json")
    if audit.get("ours_deleted_interface_ca") != 0:
        raise RuntimeError("render audit found deleted full-length interface residues")

    fig = plt.figure(figsize=(13.2, 4.8), facecolor="white")
    grid = fig.add_gridspec(3, 3, height_ratios=[3.45, 0.72, 0.82], hspace=0.0, wspace=0.02)
    titles = {"full_length": "A  Full-length", "ours": "B  Ours", "scisor": "C  SCISOR"}
    legend = [
        Patch(facecolor=COLORS["binder"], label="Binder"),
        Patch(facecolor=COLORS["target"], label="Target"),
        Patch(facecolor=COLORS["interface"], label="Predicted interface"),
        Patch(facecolor=COLORS["ours_deletion"], label="Ours deletion"),
        Patch(facecolor=COLORS["scisor_deletion"], label="SCISOR deletion"),
    ]
    fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(0.985, 0.995), ncol=5, frameon=False, fontsize=9.8, handlelength=1.2, columnspacing=1.3)

    for column, method in enumerate(ids):
        record = records[method]
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(whole[method])
        ax.axis("off")
        ax.set_title(titles[method], fontsize=15.5, fontweight="bold", color=COLORS["text"], pad=0)
        metric = fig.add_subplot(grid[1, column])
        metric.axis("off")
        if method == "full_length":
            text = (
                f"290 aa  |  confidence {record['confidence']:.3f}  |  "
                f"{record.get('wt_contact_count', 56)} reference contacts"
            )
        else:
            text = (
                f"232 aa (-20%)  |  confidence {record['confidence']:.3f}  |  iPTM {record['iptm']:.3f}\n"
                f"PAE {record['pae']:.2f} \N{LATIN CAPITAL LETTER A WITH RING ABOVE}  |  "
                f"recovery {100 * record['contact_recovery']:.1f}%  |  "
                f"Jaccard {record['contact_jaccard']:.3f}"
            )
        metric.text(0.5, 0.70, text, transform=metric.transAxes, ha="center", va="center", fontsize=9.2, color=COLORS["text"], linespacing=1.22)

    track = fig.add_subplot(grid[2, :])
    add_deletion_tracks(track, case, audit["full_length_interface_residues_1based"])
    fig.subplots_adjust(left=0.018, right=0.965, top=0.86, bottom=0.035)

    figure_dir = OUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("pdf", "svg", "png"):
        path = figure_dir / f"fig_binder_case12_b20_interface_retention.{suffix}"
        kwargs: dict[str, Any] = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.03}
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
        return {"width": image.width, "height": image.height, "nonblank": ImageChops.difference(rgb, white).getbbox() is not None}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    case = load_case()
    script = render_pymol(case)
    figures = make_figure(case)
    source_paths = [V22 / name for name in ("protocol.json", "run_summary.json", "independent_validation.json", "selected_plans.csv", "case_metrics.csv")]
    for record in case["records"]:
        source_paths.extend([record["structure"], record["confidence_path"]])
    source_rows = [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in source_paths]
    generated_paths = [script, OUT / "pymol_alignment_and_render_audit.json", OUT / "pymol_render.log", *figures]
    generated_rows = [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in generated_paths]
    png = next(path for path in figures if path.suffix == ".png")
    audit = image_audit(png)
    if not audit["nonblank"] or audit["width"] < 4000 or audit["height"] < 1700:
        raise RuntimeError(f"figure image audit failed: {audit}")
    caption = (
        "Predicted binder-target interface retention after a 20% binder deletion for the prespecified "
        "visualization-eligible case AB-Bind:2NY7:G_HL. (A) Full-length predicted complex, with the "
        "regions removed by Ours shown in red. (B) Complex predicted after the Ours deletion. (C) "
        "Complex predicted after the SCISOR deletion at the same exact budget. All panels use the same "
        "Boltz2 configuration and are aligned on the unchanged target. The complete binder-target complex "
        "is shown in every panel; the target is rendered translucently to emphasize the binder and interface. "
        "Ours retains 89.3% of full-length "
        "interface contacts, compared with 69.6% for SCISOR. The bottom tracks show all deleted binder "
        "positions and the full-length predicted interface. These are post-deletion complex-prediction "
        "proxies, not experimental evidence of binding or functional preservation."
    )
    write_text(OUT / "paper_caption.txt", caption + "\n")
    manifest = {
        "schema_version": "delepref_22_binder_case12_figure_v1",
        "status": "PASS_DelePref_22_BINDER_CASE12_FIGURE",
        "case_id": 12,
        "protein_id": PROTEIN_ID,
        "budget": 0.20,
        "selection_role": "prespecified visualization-eligible subset; illustrative case, not aggregate evidence",
        "image_audit": audit,
        "source_artifacts": source_rows,
        "generated_artifacts": generated_rows,
        "caption": caption,
        "claim_boundary": CLAIM_BOUNDARY,
        "proxy_boundary": PROXY_BOUNDARY,
    }
    write_json(OUT / "figure_manifest.json", manifest)
    write_text(OUT / "PASS_DelePref_22_BINDER_CASE12_FIGURE", "PASS_DelePref_22_BINDER_CASE12_FIGURE\n")
    print(json.dumps({"status": manifest["status"], "figure": str(png), "image": audit}, sort_keys=True))


if __name__ == "__main__":
    main()
