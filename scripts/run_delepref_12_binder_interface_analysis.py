#!/usr/bin/env python3
"""CPU-only retrospective interface analysis of existing DelePref Binder outputs.

This program reads archived prediction artifacts only.  It never invokes Boltz,
model inference, CUDA, a scheduler, or a structure-prediction executable.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import io
import itertools
import json
import math
import os
import py_compile
import statistics
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import gemmi
import numpy as np
import yaml
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "results/experiments/delepref_step004000_binder"
DEFAULT_OUTPUT = REPO_ROOT / "results/experiments/delepref_step004000_delepref_12_binder_interface_analysis"
EXPECTED_PROTOCOL_SHA256 = "16006f186f28c516c06e5fb6648e157cd5ff9bbd550c3ba128f5577837712c22"
FINAL_STATUS = "PASS_DelePref_12_BINDER_INTERFACE_ANALYSIS"
MARKER_START = "DelePref_SUPP_BINDER_INTERFACE_START"
MARKER_END = "DelePref_SUPP_BINDER_INTERFACE_END"
METHODS = ("delepref_full_control", "delepref_no_scorer", "original_SCISOR_reference")
SAME_PROTOCOL_METHODS = ("delepref_full_control", "delepref_no_scorer")
EXTERNAL_METHOD = "original_SCISOR_reference"
BOOTSTRAP_SEED = 20260715
BOOTSTRAP_REPLICATES = 20000
CONTACT_CUTOFF = 4.5
INTERFACE_CUTOFF = 5.0
CLASH_CUTOFF = 2.0
TIE_TOLERANCE = 1e-12


METRIC_DIRECTIONS: dict[str, str] = {
    "directed_pair_chains_iptm_mean": "higher_is_better",
    "interface_confidence": "higher_is_better",
    "iptm": "higher_is_better",
    "protein_iptm": "higher_is_better",
    "complex_iplddt": "higher_is_better",
    "binder_target_interchain_pae": "lower_is_better",
    "wt_to_edited_interface_contact_recovery": "higher_is_better",
    "interface_contact_jaccard": "higher_is_better",
    "contact_count_retention": "descriptive",
    "binder_interface_residue_recovery": "higher_is_better",
    "binder_interface_residue_count_retention": "descriptive",
    "edited_contact_count": "descriptive",
    "edited_binder_interface_residue_count": "descriptive",
    "edited_target_interface_residue_count": "descriptive",
    "steric_clash_count": "lower_is_better",
    "steric_clash_rate": "lower_is_better",
    "deleted_wt_interface_overlap_count": "lower_is_better",
    "deleted_wt_interface_overlap_fraction": "lower_is_better",
    "prediction_available": "higher_is_better",
    "confidence_available": "higher_is_better",
    "structure_available": "higher_is_better",
    "pae_available": "higher_is_better",
    "chain_mapping_available": "higher_is_better",
    "residue_mapping_available": "higher_is_better",
    "contact_metrics_available": "higher_is_better",
    "analysis_failure": "lower_is_better",
}


CASE_BASE_FIELDS = [
    "case_id",
    "protein_id",
    "method",
    "comparison_class",
    "prediction_status",
    "external_reference_eligible",
    "plan_hash",
    "edited_sequence_sha256",
    "deleted_length",
    "prediction_available",
    "confidence_available",
    "structure_available",
    "pae_available",
    "chain_mapping_available",
    "residue_mapping_available",
    "contact_metrics_available",
    "analysis_failure",
    "failure_reason",
    "wt_contact_count",
    "wt_binder_interface_residue_count",
    "wt_target_interface_residue_count",
    "wt_steric_clash_count",
    "wt_steric_clash_rate",
]
CASE_METRIC_FIELDS = list(METRIC_DIRECTIONS)
CASE_FIELDS = CASE_BASE_FIELDS + [x for x in CASE_METRIC_FIELDS if x not in CASE_BASE_FIELDS]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        normalized: dict[str, Any] = {}
        for field in fields:
            value = row.get(field)
            if value is None:
                normalized[field] = ""
            elif isinstance(value, float):
                if not math.isfinite(value):
                    normalized[field] = ""
                else:
                    normalized[field] = repr(float(value))
            elif isinstance(value, bool):
                normalized[field] = str(value).lower()
            else:
                normalized[field] = value
        writer.writerow(normalized)
    return handle.getvalue().encode("utf-8")


def write_bytes(path: Path, data: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": str(path), "sha256": sha256_bytes(data), "size_bytes": len(data)}


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def safe_ratio(numerator: float | int, denominator: float | int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def parse_intervals(raw: str) -> list[tuple[int, int]]:
    data = json.loads(raw)
    intervals = [(int(start), int(end)) for start, end in data]
    return intervals


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def find_prediction_sidecars(confidence_path: Path) -> tuple[Path, Path, Path]:
    name = confidence_path.name
    if not name.startswith("confidence_") or not name.endswith(".json"):
        raise ValueError(f"Unexpected confidence filename: {confidence_path}")
    stem = name[len("confidence_") : -len(".json")]
    return (
        confidence_path.parent / f"{stem}.cif",
        confidence_path.parent / f"pae_{stem}.npz",
        confidence_path.parent / f"plddt_{stem}.npz",
    )


def yaml_entities(path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    entities: list[dict[str, Any]] = []
    protein_group_index = -1
    for item in payload.get("sequences", []):
        if "protein" not in item:
            continue
        protein_group_index += 1
        protein = item["protein"]
        ids = protein["id"] if isinstance(protein["id"], list) else [protein["id"]]
        sequence = str(protein["sequence"])
        for copy_index, chain_id in enumerate(ids):
            entities.append(
                {
                    "input_chain_id": str(chain_id),
                    "sequence": sequence,
                    "protein_group_index": protein_group_index,
                    "copy_index": copy_index,
                    "role": "binder" if protein_group_index == 0 else "target",
                }
            )
    if not entities or not any(x["role"] == "binder" for x in entities):
        raise ValueError(f"No binder protein entity in {path}")
    return entities


@dataclasses.dataclass
class ChainData:
    name: str
    sequence: str
    residues: list[gemmi.Residue]
    atom_coords: np.ndarray
    atom_residue_positions: np.ndarray


@dataclasses.dataclass
class StructureData:
    path: Path
    chains: list[ChainData]


@dataclasses.dataclass
class ChainMapping:
    binder_chain_indices: list[int]
    target_slot_to_chain_index: dict[int, int]
    target_identity_ambiguous: bool
    yaml_target_sequences: list[str]


@dataclasses.dataclass
class Geometry:
    contacts: set[tuple[int, int, int]]
    binder_interface_residues: set[int]
    target_interface_residues: set[tuple[int, int]]
    clash_count: int
    possible_heavy_atom_pairs: int
    binder_heavy_atom_count: int
    target_heavy_atom_count: int

    @property
    def clash_rate(self) -> float | None:
        return safe_ratio(self.clash_count, self.possible_heavy_atom_pairs)


@dataclasses.dataclass
class PredictionSpec:
    case_id: str
    protein_id: str
    method: str
    comparison_class: str
    yaml_path: Path
    confidence_path: Path
    plan_hash: str
    edited_sequence_sha256: str
    prediction_status: str
    intervals: list[tuple[int, int]] | None = None
    deleted_length: int | None = None
    inventory_record: dict[str, Any] | None = None
    predictor_config_sha256: str = ""


@dataclasses.dataclass
class ParsedPrediction:
    spec: PredictionSpec
    entities: list[dict[str, Any]]
    confidence: dict[str, Any] | None
    structure: StructureData | None
    mapping: ChainMapping | None
    pae_path: Path
    plddt_path: Path
    cif_path: Path
    pae: np.ndarray | None
    confidence_metrics: dict[str, float | None]
    pae_metric: float | None
    geometry: Geometry | None
    failure_reasons: list[str]


def residue_one_letter(residue: gemmi.Residue) -> str:
    info = gemmi.find_tabulated_residue(residue.name)
    code = info.one_letter_code
    return code if code and code != " " else "X"


def parse_structure(path: Path) -> StructureData:
    structure = gemmi.read_structure(str(path))
    if len(structure) != 1:
        raise ValueError(f"Expected exactly one model in {path}; observed {len(structure)}")
    chains: list[ChainData] = []
    for chain in structure[0]:
        residues = [residue for residue in chain if residue.entity_type == gemmi.EntityType.Polymer]
        if not residues:
            continue
        sequence = "".join(residue_one_letter(residue) for residue in residues)
        coords: list[tuple[float, float, float]] = []
        atom_residue_positions: list[int] = []
        for residue_position, residue in enumerate(residues):
            chosen: dict[str, tuple[float, str, gemmi.Atom]] = {}
            for atom in residue:
                element = atom.element.name.upper()
                if element in {"H", "D"}:
                    continue
                atom_name = atom.name.strip()
                occupancy = float(atom.occ)
                altloc = str(atom.altloc)
                candidate = (occupancy, altloc, atom)
                previous = chosen.get(atom_name)
                if previous is None or occupancy > previous[0] or (
                    occupancy == previous[0] and altloc < previous[1]
                ):
                    chosen[atom_name] = candidate
            for atom_name in sorted(chosen):
                atom = chosen[atom_name][2]
                coords.append((float(atom.pos.x), float(atom.pos.y), float(atom.pos.z)))
                atom_residue_positions.append(residue_position)
        chains.append(
            ChainData(
                name=chain.name,
                sequence=sequence,
                residues=residues,
                atom_coords=np.asarray(coords, dtype=np.float64).reshape((-1, 3)),
                atom_residue_positions=np.asarray(atom_residue_positions, dtype=np.int64),
            )
        )
    if not chains:
        raise ValueError(f"No polymer chains in {path}")
    return StructureData(path=path, chains=chains)


def map_entities_to_structure(
    entities: Sequence[Mapping[str, Any]],
    structure: StructureData,
    canonical_target_sequences: Sequence[str] | None,
) -> ChainMapping:
    binder_entities = [x for x in entities if x["role"] == "binder"]
    target_entities = [x for x in entities if x["role"] == "target"]
    if len(binder_entities) != 1:
        raise ValueError(f"Expected one archived binder entity, observed {len(binder_entities)}")
    binder_matches = [i for i, chain in enumerate(structure.chains) if chain.sequence == binder_entities[0]["sequence"]]
    if len(binder_matches) != 1:
        raise ValueError(f"Binder sequence has {len(binder_matches)} CIF matches")
    binder_index = binder_matches[0]
    unused = set(range(len(structure.chains))) - {binder_index}
    yaml_target_to_chain: list[int] = []
    ambiguous = False
    for entity in target_entities:
        matches = [i for i in sorted(unused) if structure.chains[i].sequence == entity["sequence"]]
        if not matches:
            raise ValueError(f"Target input entity {entity['input_chain_id']} has no CIF sequence match")
        if len(matches) > 1:
            ambiguous = True
        selected = matches[0]
        yaml_target_to_chain.append(selected)
        unused.remove(selected)
    if unused:
        raise ValueError(f"Unmapped CIF polymer chains: {sorted(unused)}")
    yaml_target_sequences = [str(x["sequence"]) for x in target_entities]
    if canonical_target_sequences is None:
        target_slot_to_chain = {slot: chain_index for slot, chain_index in enumerate(yaml_target_to_chain)}
    else:
        available_slots = set(range(len(canonical_target_sequences)))
        target_slot_to_chain: dict[int, int] = {}
        for sequence, chain_index in zip(yaml_target_sequences, yaml_target_to_chain):
            matches = [slot for slot in sorted(available_slots) if canonical_target_sequences[slot] == sequence]
            if not matches:
                raise ValueError("Edited target entity has no exact WT target-slot sequence match")
            if len(matches) > 1:
                ambiguous = True
            slot = matches[0]
            target_slot_to_chain[slot] = chain_index
            available_slots.remove(slot)
        if available_slots:
            raise ValueError(f"Missing edited target slots: {sorted(available_slots)}")
    return ChainMapping(
        binder_chain_indices=[binder_index],
        target_slot_to_chain_index=target_slot_to_chain,
        target_identity_ambiguous=ambiguous,
        yaml_target_sequences=yaml_target_sequences,
    )


def validate_inventory_mapping(parsed: ParsedPrediction) -> None:
    record = parsed.spec.inventory_record
    if record is None or parsed.structure is None or parsed.mapping is None:
        return
    declared = record["prediction"]["raw_fields"]["chain_mapping"]
    observed_order = [chain.name for chain in parsed.structure.chains]
    if observed_order != list(declared["chain_order"]):
        raise ValueError(f"Inventory chain order mismatch: {observed_order} != {declared['chain_order']}")
    binder_names = {parsed.structure.chains[i].name for i in parsed.mapping.binder_chain_indices}
    target_names = {
        parsed.structure.chains[i].name for i in parsed.mapping.target_slot_to_chain_index.values()
    }
    if binder_names != set(declared["binder_chain_ids_boltz"]):
        raise ValueError("Inventory binder-chain mapping does not match sequence-derived mapping")
    if target_names != set(declared["target_chain_ids_boltz"]):
        raise ValueError("Inventory target-chain mapping does not match sequence-derived mapping")


def geometry_arrays(
    structure: StructureData, mapping: ChainMapping
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    binder_coords: list[np.ndarray] = []
    binder_residue_ids: list[np.ndarray] = []
    binder_offset = 0
    for chain_index in mapping.binder_chain_indices:
        chain = structure.chains[chain_index]
        binder_coords.append(chain.atom_coords)
        binder_residue_ids.append(chain.atom_residue_positions + binder_offset)
        binder_offset += len(chain.residues)
    target_coords: list[np.ndarray] = []
    target_residue_ids: list[tuple[int, int]] = []
    for slot, chain_index in sorted(mapping.target_slot_to_chain_index.items()):
        chain = structure.chains[chain_index]
        target_coords.append(chain.atom_coords)
        target_residue_ids.extend((slot, int(pos)) for pos in chain.atom_residue_positions)
    return (
        np.concatenate(binder_coords, axis=0),
        np.concatenate(binder_residue_ids, axis=0),
        np.concatenate(target_coords, axis=0),
        target_residue_ids,
    )


def compute_geometry_tree(structure: StructureData, mapping: ChainMapping) -> Geometry:
    binder_coords, binder_residue_ids, target_coords, target_residue_ids = geometry_arrays(structure, mapping)
    neighbor_lists = cKDTree(binder_coords).query_ball_tree(cKDTree(target_coords), INTERFACE_CUTOFF)
    contacts: set[tuple[int, int, int]] = set()
    binder_interface: set[int] = set()
    target_interface: set[tuple[int, int]] = set()
    clash_count = 0
    contact_squared = CONTACT_CUTOFF**2
    interface_squared = INTERFACE_CUTOFF**2
    clash_squared = CLASH_CUTOFF**2
    for binder_atom_index, target_neighbors in enumerate(neighbor_lists):
        b_res = int(binder_residue_ids[binder_atom_index])
        b_coord = binder_coords[binder_atom_index]
        for target_atom_index in target_neighbors:
            delta = b_coord - target_coords[target_atom_index]
            distance_squared = float(np.dot(delta, delta))
            target_identity = target_residue_ids[target_atom_index]
            if distance_squared <= interface_squared:
                binder_interface.add(b_res)
                target_interface.add(target_identity)
            if distance_squared <= contact_squared:
                contacts.add((b_res, target_identity[0], target_identity[1]))
            if distance_squared < clash_squared:
                clash_count += 1
    return Geometry(
        contacts=contacts,
        binder_interface_residues=binder_interface,
        target_interface_residues=target_interface,
        clash_count=clash_count,
        possible_heavy_atom_pairs=len(binder_coords) * len(target_coords),
        binder_heavy_atom_count=len(binder_coords),
        target_heavy_atom_count=len(target_coords),
    )


def compute_geometry_grid(structure: StructureData, mapping: ChainMapping) -> Geometry:
    """Independent contact implementation used only by final validation."""
    binder_coords, binder_residue_ids, target_coords, target_residue_ids = geometry_arrays(structure, mapping)
    cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for target_index, coord in enumerate(target_coords):
        cell = tuple(int(x) for x in np.floor(coord / INTERFACE_CUTOFF))
        cells[cell].append(target_index)
    contacts: set[tuple[int, int, int]] = set()
    binder_interface: set[int] = set()
    target_interface: set[tuple[int, int]] = set()
    clash_count = 0
    for binder_index, coord in enumerate(binder_coords):
        base = tuple(int(x) for x in np.floor(coord / INTERFACE_CUTOFF))
        b_res = int(binder_residue_ids[binder_index])
        for offsets in itertools.product((-1, 0, 1), repeat=3):
            cell = tuple(base[i] + offsets[i] for i in range(3))
            for target_index in cells.get(cell, []):
                delta = coord - target_coords[target_index]
                distance_squared = float(delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2)
                target_identity = target_residue_ids[target_index]
                if distance_squared <= INTERFACE_CUTOFF**2:
                    binder_interface.add(b_res)
                    target_interface.add(target_identity)
                if distance_squared <= CONTACT_CUTOFF**2:
                    contacts.add((b_res, target_identity[0], target_identity[1]))
                if distance_squared < CLASH_CUTOFF**2:
                    clash_count += 1
    return Geometry(
        contacts=contacts,
        binder_interface_residues=binder_interface,
        target_interface_residues=target_interface,
        clash_count=clash_count,
        possible_heavy_atom_pairs=len(binder_coords) * len(target_coords),
        binder_heavy_atom_count=len(binder_coords),
        target_heavy_atom_count=len(target_coords),
    )


def confidence_metrics(
    confidence: Mapping[str, Any], mapping: ChainMapping, chain_count: int
) -> dict[str, float | None]:
    pair = confidence.get("pair_chains_iptm")
    directed: list[float] = []
    if isinstance(pair, Mapping) and len(pair) == chain_count:
        for binder_index in mapping.binder_chain_indices:
            for target_index in mapping.target_slot_to_chain_index.values():
                try:
                    directed.append(float(pair[str(binder_index)][str(target_index)]))
                    directed.append(float(pair[str(target_index)][str(binder_index)]))
                except (KeyError, TypeError, ValueError):
                    directed = []
                    break
    directed_mean = float(np.mean(np.asarray(directed, dtype=np.float64))) if directed else None
    result: dict[str, float | None] = {
        "directed_pair_chains_iptm_mean": directed_mean,
        "interface_confidence": directed_mean,
    }
    for output_name, source_name in (
        ("iptm", "iptm"),
        ("protein_iptm", "protein_iptm"),
        ("complex_iplddt", "complex_iplddt"),
    ):
        try:
            value = float(confidence[source_name])
            result[output_name] = value if math.isfinite(value) else None
        except (KeyError, TypeError, ValueError):
            result[output_name] = None
    return result


def aggregate_pae(
    pae: np.ndarray, structure: StructureData, mapping: ChainMapping
) -> float:
    residue_indices_by_chain: dict[int, np.ndarray] = {}
    offset = 0
    for chain_index, chain in enumerate(structure.chains):
        residue_indices_by_chain[chain_index] = np.arange(offset, offset + len(chain.residues), dtype=np.int64)
        offset += len(chain.residues)
    if pae.ndim != 2 or pae.shape != (offset, offset):
        raise ValueError(f"PAE shape {pae.shape} does not match {(offset, offset)}")
    binder_indices = np.concatenate([residue_indices_by_chain[i] for i in mapping.binder_chain_indices])
    target_indices = np.concatenate(
        [residue_indices_by_chain[i] for _, i in sorted(mapping.target_slot_to_chain_index.items())]
    )
    selected = np.concatenate(
        [pae[np.ix_(binder_indices, target_indices)].ravel(), pae[np.ix_(target_indices, binder_indices)].ravel()]
    )
    if selected.size == 0 or not np.isfinite(selected).all():
        raise ValueError("PAE inter-chain selection is empty or non-finite")
    return float(np.mean(selected, dtype=np.float64))


def aggregate_pae_independent(
    pae: np.ndarray, structure: StructureData, mapping: ChainMapping
) -> float:
    chain_indices: dict[int, list[int]] = {}
    offset = 0
    for chain_index, chain in enumerate(structure.chains):
        chain_indices[chain_index] = list(range(offset, offset + len(chain.residues)))
        offset += len(chain.residues)
    if pae.shape != (offset, offset):
        raise ValueError("Independent PAE shape check failed")
    binder = [x for chain_index in mapping.binder_chain_indices for x in chain_indices[chain_index]]
    targets = [
        x
        for _, chain_index in sorted(mapping.target_slot_to_chain_index.items())
        for x in chain_indices[chain_index]
    ]
    total = 0.0
    count = 0
    for i in binder:
        for j in targets:
            a = float(pae[i, j])
            b = float(pae[j, i])
            if not (math.isfinite(a) and math.isfinite(b)):
                raise ValueError("Independent PAE encountered a non-finite value")
            total += a + b
            count += 2
    if count == 0:
        raise ValueError("Independent PAE selection empty")
    return total / count


def load_prediction(
    spec: PredictionSpec,
    canonical_target_sequences: Sequence[str] | None,
) -> ParsedPrediction:
    failures: list[str] = []
    entities: list[dict[str, Any]] = []
    confidence: dict[str, Any] | None = None
    structure: StructureData | None = None
    mapping: ChainMapping | None = None
    pae: np.ndarray | None = None
    confidence_values = {name: None for name in (
        "directed_pair_chains_iptm_mean", "interface_confidence", "iptm", "protein_iptm", "complex_iplddt"
    )}
    pae_value: float | None = None
    geometry: Geometry | None = None
    try:
        entities = yaml_entities(spec.yaml_path)
    except Exception as exc:  # audit must retain failures
        failures.append(f"yaml:{type(exc).__name__}:{exc}")
    try:
        cif_path, pae_path, plddt_path = find_prediction_sidecars(spec.confidence_path)
    except Exception as exc:
        failures.append(f"sidecars:{type(exc).__name__}:{exc}")
        cif_path = spec.confidence_path.with_suffix(".cif")
        pae_path = spec.confidence_path.with_suffix(".npz")
        plddt_path = spec.confidence_path.with_suffix(".plddt.npz")
    try:
        confidence = json.loads(spec.confidence_path.read_text(encoding="utf-8"))
    except Exception as exc:
        failures.append(f"confidence:{type(exc).__name__}:{exc}")
    try:
        structure = parse_structure(cif_path)
    except Exception as exc:
        failures.append(f"structure:{type(exc).__name__}:{exc}")
    if entities and structure is not None:
        try:
            mapping = map_entities_to_structure(entities, structure, canonical_target_sequences)
        except Exception as exc:
            failures.append(f"chain_mapping:{type(exc).__name__}:{exc}")
    if mapping is not None and structure is not None:
        try:
            temporary = ParsedPrediction(
                spec, entities, confidence, structure, mapping, pae_path, plddt_path, cif_path,
                None, confidence_values, None, None, failures,
            )
            validate_inventory_mapping(temporary)
        except Exception as exc:
            failures.append(f"inventory_chain_mapping:{type(exc).__name__}:{exc}")
            mapping = None
    if confidence is not None and structure is not None and mapping is not None:
        confidence_values = confidence_metrics(confidence, mapping, len(structure.chains))
    try:
        with np.load(pae_path) as archive:
            pae = np.asarray(archive["pae"], dtype=np.float64)
    except Exception as exc:
        failures.append(f"pae:{type(exc).__name__}:{exc}")
    if pae is not None and structure is not None and mapping is not None:
        try:
            pae_value = aggregate_pae(pae, structure, mapping)
        except Exception as exc:
            failures.append(f"pae_aggregation:{type(exc).__name__}:{exc}")
    if structure is not None and mapping is not None:
        try:
            geometry = compute_geometry_tree(structure, mapping)
        except Exception as exc:
            failures.append(f"geometry:{type(exc).__name__}:{exc}")
    return ParsedPrediction(
        spec=spec,
        entities=entities,
        confidence=confidence,
        structure=structure,
        mapping=mapping,
        pae_path=pae_path,
        plddt_path=plddt_path,
        cif_path=cif_path,
        pae=pae,
        confidence_metrics=confidence_values,
        pae_metric=pae_value,
        geometry=geometry,
        failure_reasons=failures,
    )


def delepref_residue_mapping(
    wt_sequence: str,
    edited_sequence: str,
    intervals: Sequence[tuple[int, int]],
    declared_deleted_length: int,
) -> tuple[list[int], set[int]]:
    previous_end = -1
    deleted: set[int] = set()
    for start, end in intervals:
        if start <= previous_end or start < 0 or end < start or end >= len(wt_sequence):
            raise ValueError(f"Invalid or overlapping deletion interval {(start, end)}")
        deleted.update(range(start, end + 1))
        previous_end = end
    if len(deleted) != declared_deleted_length:
        raise ValueError("deleted_length does not match interval union")
    mapping = [index for index in range(len(wt_sequence)) if index not in deleted]
    reconstructed = "".join(wt_sequence[index] for index in mapping)
    if reconstructed != edited_sequence:
        raise ValueError("Selected intervals do not reconstruct edited binder sequence")
    return mapping, deleted


def external_residue_mapping(
    wt_sequence: str, edited_sequence: str
) -> tuple[list[int] | None, list[int], list[int]]:
    left: list[int] = []
    cursor = 0
    for residue in edited_sequence:
        position = wt_sequence.find(residue, cursor)
        if position < 0:
            raise ValueError("Edited external-reference sequence is not a WT subsequence")
        left.append(position)
        cursor = position + 1
    right_reversed: list[int] = []
    cursor = len(wt_sequence) - 1
    for residue in reversed(edited_sequence):
        position = wt_sequence.rfind(residue, 0, cursor + 1)
        if position < 0:
            raise ValueError("Edited external-reference sequence is not a WT subsequence")
        right_reversed.append(position)
        cursor = position - 1
    right = list(reversed(right_reversed))
    return (left if left == right else None), left, right


def remap_geometry(geometry: Geometry, edited_to_wt: Sequence[int]) -> Geometry:
    return Geometry(
        contacts={(edited_to_wt[b], slot, target) for b, slot, target in geometry.contacts},
        binder_interface_residues={edited_to_wt[b] for b in geometry.binder_interface_residues},
        target_interface_residues=set(geometry.target_interface_residues),
        clash_count=geometry.clash_count,
        possible_heavy_atom_pairs=geometry.possible_heavy_atom_pairs,
        binder_heavy_atom_count=geometry.binder_heavy_atom_count,
        target_heavy_atom_count=geometry.target_heavy_atom_count,
    )


def build_specs(input_root: Path) -> tuple[
    list[dict[str, str]],
    dict[str, PredictionSpec],
    dict[tuple[str, str], PredictionSpec],
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    manifest = read_csv(input_root / "binder_manifest.csv")
    details = read_csv(input_root / "binder_prediction_details.csv")
    plans = {
        "delepref_full_control": read_csv(input_root / "binder_selected_plans_full_control.csv"),
        "delepref_no_scorer": read_csv(input_root / "binder_selected_plans_no_scorer.csv"),
    }
    run_summary = json.loads((input_root / "run_summary.json").read_text(encoding="utf-8"))
    audit_manifest = json.loads((input_root / "audit_manifest.json").read_text(encoding="utf-8"))
    inventory = json.loads((input_root / "audit/source_cache_inventory.json").read_text(encoding="utf-8"))
    manifest_by_case = {row["case_id"]: row for row in manifest}
    detail_by_key = {(row["case_id"], row["method"]): row for row in details}
    plan_by_key = {
        (row["case_id"], method): row
        for method, rows in plans.items()
        for row in rows
    }
    inventory_by_key = {
        (str(row["case_id"]), row["method"]): row
        for row in inventory["source_audit"]
        if row["method"] in {"wildtype_shared", EXTERNAL_METHOD}
    }
    gate_errors: list[str] = []
    if run_summary.get("status") != "PASS_DelePref_BINDER" or audit_manifest.get("status") != "PASS_DelePref_BINDER":
        gate_errors.append("input_status_not_PASS_DelePref_BINDER")
    if len(manifest) != 50 or len(manifest_by_case) != 50 or run_summary.get("binder_case_count") != 50:
        gate_errors.append("case_count_not_50")
    if run_summary.get("new_prediction_count_unique") != 90:
        gate_errors.append("input_run_summary_unique_prediction_count_not_90")
    if run_summary.get("zero_fill_used") is not False or audit_manifest.get("zero_fill_used") is not False:
        gate_errors.append("zero_fill_not_zero")
    delepref_details = [row for row in details if row["method"] in SAME_PROTOCOL_METHODS]
    if len(delepref_details) != 100 or len({row["edited_confidence_json"] for row in delepref_details}) != 90:
        gate_errors.append("delepref_assignment_or_unique_prediction_count_mismatch")
    if any(row["prediction_status"] != "ok" for row in delepref_details):
        gate_errors.append("non_ok_delepref_prediction_status")
    if any(parse_bool(row["student_score_used_as_boltz_confidence"]) for row in delepref_details):
        gate_errors.append("student_score_substitution_detected")
    for method in SAME_PROTOCOL_METHODS:
        method_details = [row for row in details if row["method"] == method]
        if len(method_details) != 50 or len({row["case_id"] for row in method_details}) != 50:
            gate_errors.append(f"{method}_assignment_not_50_unique_cases")
        if len(plans[method]) != 50:
            gate_errors.append(f"{method}_plan_count_not_50")
    for case_id in manifest_by_case:
        for method in ("wildtype_shared", EXTERNAL_METHOD):
            if (case_id, method) not in inventory_by_key:
                gate_errors.append(f"missing_inventory:{case_id}:{method}")
        for method in METHODS:
            if (case_id, method) not in detail_by_key:
                gate_errors.append(f"missing_detail:{case_id}:{method}")
    if gate_errors:
        raise RuntimeError("Input acceptance gate failed: " + ";".join(gate_errors))

    wt_specs: dict[str, PredictionSpec] = {}
    method_specs: dict[tuple[str, str], PredictionSpec] = {}
    for case_id, manifest_row in manifest_by_case.items():
        wt_record = inventory_by_key[(case_id, "wildtype_shared")]
        wt_specs[case_id] = PredictionSpec(
            case_id=case_id,
            protein_id=manifest_row["protein_id"],
            method="wildtype_shared",
            comparison_class="WT_BASELINE",
            yaml_path=Path(wt_record["source_yaml_path"]),
            confidence_path=Path(wt_record["prediction"]["confidence_json"]),
            plan_hash=wt_record["source_plan_hash"],
            edited_sequence_sha256=wt_record["edited_sequence_sha256"],
            prediction_status="ok" if wt_record["prediction"]["success"] else "failed",
            inventory_record=wt_record,
            predictor_config_sha256=wt_record["predictor_config_sha256"],
        )
        for method in SAME_PROTOCOL_METHODS:
            detail = detail_by_key[(case_id, method)]
            plan = plan_by_key[(case_id, method)]
            if detail["plan_hash"] != plan["plan_hash"] or detail["edited_sequence_sha256"] != plan["edited_sequence_sha256"]:
                raise RuntimeError(f"Plan/detail mismatch for {case_id} {method}")
            yaml_candidates = sorted((input_root / "boltz_inputs" / method).glob(f"case{int(case_id):03d}_*_{method}.yaml"))
            if len(yaml_candidates) != 1:
                raise RuntimeError(f"Expected one archived YAML for {case_id} {method}; observed {len(yaml_candidates)}")
            method_specs[(case_id, method)] = PredictionSpec(
                case_id=case_id,
                protein_id=manifest_row["protein_id"],
                method=method,
                comparison_class="SAME_PROTOCOL",
                yaml_path=yaml_candidates[0],
                confidence_path=Path(detail["edited_confidence_json"]),
                plan_hash=detail["plan_hash"],
                edited_sequence_sha256=detail["edited_sequence_sha256"],
                prediction_status=detail["prediction_status"],
                intervals=parse_intervals(plan["deleted_intervals"]),
                deleted_length=int(plan["deleted_length"]),
                predictor_config_sha256=inventory["predictor_config_sha256"],
            )
        detail = detail_by_key[(case_id, EXTERNAL_METHOD)]
        record = inventory_by_key[(case_id, EXTERNAL_METHOD)]
        method_specs[(case_id, EXTERNAL_METHOD)] = PredictionSpec(
            case_id=case_id,
            protein_id=manifest_row["protein_id"],
            method=EXTERNAL_METHOD,
            comparison_class="EXTERNAL_REFERENCE",
            yaml_path=Path(record["source_yaml_path"]),
            confidence_path=Path(detail["edited_confidence_json"]),
            plan_hash=detail["plan_hash"],
            edited_sequence_sha256=detail["edited_sequence_sha256"],
            prediction_status=detail["prediction_status"],
            inventory_record=record,
            predictor_config_sha256=record["predictor_config_sha256"],
        )
    return manifest, wt_specs, method_specs, manifest_by_case, inventory, run_summary


def source_hash_or_empty(path: Path) -> str:
    return sha256_file(path) if path.is_file() else ""


def prediction_audit_row(
    parsed: ParsedPrediction,
    manifest_row: Mapping[str, str],
    reuse_count: int,
    external_eligible: bool | None,
) -> dict[str, Any]:
    spec = parsed.spec
    inventory = spec.inventory_record or {}
    declared_mapping = inventory.get("prediction", {}).get("raw_fields", {}).get("chain_mapping")
    mapping_json = None
    chain_order_json = None
    target_ambiguous: bool | None = None
    if parsed.structure is not None:
        chain_order_json = json.dumps([chain.name for chain in parsed.structure.chains], separators=(",", ":"))
    if parsed.mapping is not None and parsed.structure is not None:
        mapping_json = json.dumps(
            {
                "binder_cif_chains": [parsed.structure.chains[i].name for i in parsed.mapping.binder_chain_indices],
                "target_slot_to_cif_chain": {
                    str(slot): parsed.structure.chains[index].name
                    for slot, index in sorted(parsed.mapping.target_slot_to_chain_index.items())
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        target_ambiguous = parsed.mapping.target_identity_ambiguous
    yaml_expected_hash = inventory.get("source_yaml_sha256", "")
    yaml_actual_hash = source_hash_or_empty(spec.yaml_path)
    return {
        "case_id": spec.case_id,
        "protein_id": spec.protein_id,
        "method": spec.method,
        "comparison_class": spec.comparison_class,
        "assignment_prediction_reuse_count": reuse_count,
        "prediction_status": spec.prediction_status,
        "external_reference_eligible": external_eligible,
        "predictor_config_sha256": spec.predictor_config_sha256,
        "yaml_path": str(spec.yaml_path),
        "yaml_sha256": yaml_actual_hash,
        "yaml_expected_sha256": yaml_expected_hash,
        "yaml_hash_matches_inventory": (yaml_actual_hash == yaml_expected_hash) if yaml_expected_hash else None,
        "confidence_json_path": str(spec.confidence_path),
        "confidence_json_sha256": source_hash_or_empty(spec.confidence_path),
        "cif_path": str(parsed.cif_path),
        "cif_sha256": source_hash_or_empty(parsed.cif_path),
        "pae_path": str(parsed.pae_path),
        "pae_sha256": source_hash_or_empty(parsed.pae_path),
        "plddt_path": str(parsed.plddt_path),
        "plddt_sha256": source_hash_or_empty(parsed.plddt_path),
        "manifest_structure_path": manifest_row["structure_path"],
        "manifest_structure_sha256_expected": manifest_row["structure_sha256"],
        "manifest_structure_sha256_observed": source_hash_or_empty(Path(manifest_row["structure_path"])),
        "semantic_input_match": inventory.get("semantic_input_match"),
        "ordered_semantic_input_match": inventory.get("ordered_semantic_input_match"),
        "target_permutation_external_reference_match": inventory.get("target_permutation_external_reference_match"),
        "declared_chain_mapping_json": json.dumps(declared_mapping, sort_keys=True, separators=(",", ":")) if declared_mapping else None,
        "observed_cif_chain_order_json": chain_order_json,
        "resolved_chain_mapping_json": mapping_json,
        "target_identity_ambiguous": target_ambiguous,
        "prediction_available": int(spec.confidence_path.is_file()),
        "confidence_available": int(parsed.confidence is not None),
        "structure_available": int(parsed.structure is not None),
        "pae_available": int(parsed.pae is not None and parsed.pae_metric is not None),
        "chain_mapping_available": int(parsed.mapping is not None),
        "failure_reason": ";".join(parsed.failure_reasons),
    }


PREDICTION_AUDIT_FIELDS = [
    "case_id", "protein_id", "method", "comparison_class", "assignment_prediction_reuse_count",
    "prediction_status", "external_reference_eligible", "predictor_config_sha256", "yaml_path", "yaml_sha256",
    "yaml_expected_sha256", "yaml_hash_matches_inventory", "confidence_json_path", "confidence_json_sha256",
    "cif_path", "cif_sha256", "pae_path", "pae_sha256", "plddt_path", "plddt_sha256",
    "manifest_structure_path", "manifest_structure_sha256_expected", "manifest_structure_sha256_observed",
    "semantic_input_match", "ordered_semantic_input_match", "target_permutation_external_reference_match",
    "declared_chain_mapping_json", "observed_cif_chain_order_json", "resolved_chain_mapping_json",
    "target_identity_ambiguous", "prediction_available", "confidence_available", "structure_available",
    "pae_available", "chain_mapping_available", "failure_reason",
]


def analyze(input_root: Path) -> dict[str, Any]:
    manifest, wt_specs, method_specs, manifest_by_case, inventory, input_run_summary = build_specs(input_root)
    reuse_counts = Counter(str(spec.confidence_path) for spec in method_specs.values() if spec.method in SAME_PROTOCOL_METHODS)
    prediction_audit: list[dict[str, Any]] = []
    mapping_audit: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    parsed_records: dict[tuple[str, str], ParsedPrediction] = {}
    geometry_mappings: dict[tuple[str, str], list[int] | None] = {}
    deleted_sets: dict[tuple[str, str], set[int] | None] = {}
    external_eligibility: dict[str, bool] = {}

    for manifest_row in sorted(manifest, key=lambda row: int(row["case_id"])):
        case_id = manifest_row["case_id"]
        wt_sequence = manifest_row["wildtype_sequence"]
        wt_parsed = load_prediction(wt_specs[case_id], canonical_target_sequences=None)
        parsed_records[(case_id, "wildtype_shared")] = wt_parsed
        wt_entities = wt_parsed.entities
        wt_target_sequences = [x["sequence"] for x in wt_entities if x["role"] == "target"]
        wt_binder_sequences = [x["sequence"] for x in wt_entities if x["role"] == "binder"]
        if wt_binder_sequences != [wt_sequence]:
            wt_parsed.failure_reasons.append("wt_yaml_binder_sequence_does_not_match_manifest")
            wt_parsed.mapping = None
            wt_parsed.geometry = None
        prediction_audit.append(prediction_audit_row(wt_parsed, manifest_row, 1, None))

        for method in METHODS:
            spec = method_specs[(case_id, method)]
            parsed = load_prediction(spec, canonical_target_sequences=wt_target_sequences)
            parsed_records[(case_id, method)] = parsed
            edited_entities = [x for x in parsed.entities if x["role"] == "binder"]
            edited_sequence = edited_entities[0]["sequence"] if len(edited_entities) == 1 else ""
            if edited_sequence and hashlib.sha256(edited_sequence.encode("utf-8")).hexdigest() != spec.edited_sequence_sha256:
                parsed.failure_reasons.append("edited_sequence_sha256_mismatch")
            edited_to_wt: list[int] | None = None
            deleted: set[int] | None = None
            left_mapping: list[int] = []
            right_mapping: list[int] = []
            mapping_failure = ""
            try:
                if method in SAME_PROTOCOL_METHODS:
                    assert spec.intervals is not None and spec.deleted_length is not None
                    edited_to_wt, deleted = delepref_residue_mapping(
                        wt_sequence, edited_sequence, spec.intervals, spec.deleted_length
                    )
                    left_mapping = list(edited_to_wt)
                    right_mapping = list(edited_to_wt)
                else:
                    edited_to_wt, left_mapping, right_mapping = external_residue_mapping(wt_sequence, edited_sequence)
                    if edited_to_wt is not None:
                        deleted = set(range(len(wt_sequence))) - set(edited_to_wt)
            except Exception as exc:
                mapping_failure = f"{type(exc).__name__}:{exc}"
                parsed.failure_reasons.append(f"residue_mapping:{mapping_failure}")
            geometry_mappings[(case_id, method)] = edited_to_wt
            deleted_sets[(case_id, method)] = deleted
            for edited_index, residue in enumerate(edited_sequence):
                left_index = left_mapping[edited_index] if edited_index < len(left_mapping) else None
                right_index = right_mapping[edited_index] if edited_index < len(right_mapping) else None
                accepted_index = edited_to_wt[edited_index] if edited_to_wt is not None else None
                mapping_audit.append(
                    {
                        "case_id": case_id,
                        "protein_id": manifest_row["protein_id"],
                        "method": method,
                        "comparison_class": spec.comparison_class,
                        "edited_residue_index_0based": edited_index,
                        "edited_residue": residue,
                        "wt_residue_index_0based": accepted_index,
                        "wt_residue": wt_sequence[accepted_index] if accepted_index is not None else None,
                        "leftmost_wt_index_0based": left_index,
                        "rightmost_wt_index_0based": right_index,
                        "mapping_accepted": int(edited_to_wt is not None),
                        "mapping_rule": "selected_closed_intervals" if method in SAME_PROTOCOL_METHODS else "unique_deletion_only_embedding",
                        "deleted_intervals_json": json.dumps(spec.intervals, separators=(",", ":")) if spec.intervals is not None else None,
                        "failure_reason": mapping_failure,
                    }
                )

            record = spec.inventory_record or {}
            semantic_ok = bool(record.get("semantic_input_match", True))
            ordered_or_verified_permutation = bool(
                record.get("ordered_semantic_input_match", True)
                or record.get("target_permutation_external_reference_match", False)
            )
            external_eligible = True
            if method == EXTERNAL_METHOD:
                external_eligible = bool(
                    record.get("prediction", {}).get("success")
                    and semantic_ok
                    and ordered_or_verified_permutation
                    and record.get("prediction", {}).get("raw_fields", {}).get("chain_mapping")
                    and parsed.confidence is not None
                    and parsed.structure is not None
                    and parsed.mapping is not None
                    and parsed.pae is not None
                )
                external_eligibility[case_id] = external_eligible
            prediction_audit.append(
                prediction_audit_row(
                    parsed,
                    manifest_row,
                    reuse_counts[str(spec.confidence_path)] if method in SAME_PROTOCOL_METHODS else 1,
                    external_eligible if method == EXTERNAL_METHOD else None,
                )
            )

            eligible_for_metrics = external_eligible if method == EXTERNAL_METHOD else True
            prediction_available = int(spec.confidence_path.is_file() and spec.prediction_status == "ok")
            confidence_available = int(parsed.confidence is not None)
            structure_available = int(parsed.structure is not None)
            pae_available = int(parsed.pae_metric is not None)
            chain_mapping_available = int(parsed.mapping is not None)
            residue_mapping_available = int(edited_to_wt is not None)
            contact_metrics_available = int(
                eligible_for_metrics
                and wt_parsed.geometry is not None
                and parsed.geometry is not None
                and edited_to_wt is not None
            )
            analysis_failure = int(
                not eligible_for_metrics
                or not all((prediction_available, confidence_available, structure_available, pae_available, chain_mapping_available))
            )
            row: dict[str, Any] = {
                "case_id": case_id,
                "protein_id": manifest_row["protein_id"],
                "method": method,
                "comparison_class": spec.comparison_class,
                "prediction_status": spec.prediction_status,
                "external_reference_eligible": int(external_eligible) if method == EXTERNAL_METHOD else None,
                "plan_hash": spec.plan_hash,
                "edited_sequence_sha256": spec.edited_sequence_sha256,
                "deleted_length": len(deleted) if deleted is not None else spec.deleted_length,
                "prediction_available": prediction_available,
                "confidence_available": confidence_available,
                "structure_available": structure_available,
                "pae_available": pae_available,
                "chain_mapping_available": chain_mapping_available,
                "residue_mapping_available": residue_mapping_available,
                "contact_metrics_available": contact_metrics_available,
                "analysis_failure": analysis_failure,
                "failure_reason": ";".join(parsed.failure_reasons),
                "wt_contact_count": len(wt_parsed.geometry.contacts) if wt_parsed.geometry is not None else None,
                "wt_binder_interface_residue_count": len(wt_parsed.geometry.binder_interface_residues) if wt_parsed.geometry is not None else None,
                "wt_target_interface_residue_count": len(wt_parsed.geometry.target_interface_residues) if wt_parsed.geometry is not None else None,
                "wt_steric_clash_count": wt_parsed.geometry.clash_count if wt_parsed.geometry is not None else None,
                "wt_steric_clash_rate": wt_parsed.geometry.clash_rate if wt_parsed.geometry is not None else None,
            }
            for field in CASE_METRIC_FIELDS:
                row.setdefault(field, None)
            if eligible_for_metrics:
                row.update(parsed.confidence_metrics)
                row["binder_target_interchain_pae"] = parsed.pae_metric
                if parsed.geometry is not None:
                    row["edited_contact_count"] = len(parsed.geometry.contacts)
                    row["edited_binder_interface_residue_count"] = len(parsed.geometry.binder_interface_residues)
                    row["edited_target_interface_residue_count"] = len(parsed.geometry.target_interface_residues)
                    row["steric_clash_count"] = parsed.geometry.clash_count
                    row["steric_clash_rate"] = parsed.geometry.clash_rate
                if parsed.geometry is not None and wt_parsed.geometry is not None:
                    row["contact_count_retention"] = safe_ratio(
                        len(parsed.geometry.contacts), len(wt_parsed.geometry.contacts)
                    )
                    row["binder_interface_residue_count_retention"] = safe_ratio(
                        len(parsed.geometry.binder_interface_residues),
                        len(wt_parsed.geometry.binder_interface_residues),
                    )
                if contact_metrics_available:
                    assert parsed.geometry is not None and wt_parsed.geometry is not None and edited_to_wt is not None
                    mapped_geometry = remap_geometry(parsed.geometry, edited_to_wt)
                    intersection = wt_parsed.geometry.contacts & mapped_geometry.contacts
                    union = wt_parsed.geometry.contacts | mapped_geometry.contacts
                    row["wt_to_edited_interface_contact_recovery"] = safe_ratio(
                        len(intersection), len(wt_parsed.geometry.contacts)
                    )
                    row["interface_contact_jaccard"] = safe_ratio(len(intersection), len(union))
                    row["binder_interface_residue_recovery"] = safe_ratio(
                        len(wt_parsed.geometry.binder_interface_residues & mapped_geometry.binder_interface_residues),
                        len(wt_parsed.geometry.binder_interface_residues),
                    )
                if deleted is not None and wt_parsed.geometry is not None:
                    overlap = deleted & wt_parsed.geometry.binder_interface_residues
                    row["deleted_wt_interface_overlap_count"] = len(overlap)
                    row["deleted_wt_interface_overlap_fraction"] = safe_ratio(
                        len(overlap), len(wt_parsed.geometry.binder_interface_residues)
                    )
            case_rows.append(row)

    return {
        "manifest": manifest,
        "inventory": inventory,
        "input_run_summary": input_run_summary,
        "prediction_audit": prediction_audit,
        "mapping_audit": mapping_audit,
        "case_rows": case_rows,
        "parsed_records": parsed_records,
        "geometry_mappings": geometry_mappings,
        "deleted_sets": deleted_sets,
        "external_eligibility": external_eligibility,
    }


MAPPING_AUDIT_FIELDS = [
    "case_id", "protein_id", "method", "comparison_class", "edited_residue_index_0based", "edited_residue",
    "wt_residue_index_0based", "wt_residue", "leftmost_wt_index_0based", "rightmost_wt_index_0based",
    "mapping_accepted", "mapping_rule", "deleted_intervals_json", "failure_reason",
]


def numeric_values(rows: Sequence[Mapping[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(metric)
        if value is None or value == "":
            continue
        number = float(value)
        if math.isfinite(number):
            values.append(number)
    return values


def summarize_methods(case_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in METHODS:
        rows = [row for row in case_rows if row["method"] == method]
        comparison_class = "SAME_PROTOCOL" if method in SAME_PROTOCOL_METHODS else "EXTERNAL_REFERENCE"
        for metric, direction in METRIC_DIRECTIONS.items():
            values = numeric_values(rows, metric)
            output.append(
                {
                    "method": method,
                    "comparison_class": comparison_class,
                    "metric": metric,
                    "direction": direction,
                    "n_expected": 50,
                    "n_available": len(values),
                    "coverage": len(values) / 50.0,
                    "mean": statistics.fmean(values) if values else None,
                    "median": statistics.median(values) if values else None,
                    "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                    "minimum": min(values) if values else None,
                    "maximum": max(values) if values else None,
                }
            )
    return output


SUMMARY_FIELDS = [
    "method", "comparison_class", "metric", "direction", "n_expected", "n_available", "coverage",
    "mean", "median", "sample_sd", "minimum", "maximum",
]


def bootstrap_distribution(differences: np.ndarray, replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.empty(replicates, dtype=np.float64)
    chunk = 1000
    for start in range(0, replicates, chunk):
        size = min(chunk, replicates - start)
        indices = rng.integers(0, len(differences), size=(size, len(differences)))
        output[start : start + size] = differences[indices].mean(axis=1)
    return output


def paired_bootstrap(case_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(str(row["case_id"]), str(row["method"])): row for row in case_rows}
    output: list[dict[str, Any]] = []
    for metric, direction in METRIC_DIRECTIONS.items():
        pairs: list[tuple[float, float]] = []
        for case_id in map(str, range(1, 51)):
            full_value = by_key[(case_id, "delepref_full_control")].get(metric)
            no_value = by_key[(case_id, "delepref_no_scorer")].get(metric)
            if full_value is None or no_value is None or full_value == "" or no_value == "":
                continue
            full_number = float(full_value)
            no_number = float(no_value)
            if math.isfinite(full_number) and math.isfinite(no_number):
                pairs.append((full_number, no_number))
        differences = np.asarray([full - no for full, no in pairs], dtype=np.float64)
        point = float(differences.mean()) if len(differences) else None
        ci_low = ci_high = None
        if len(differences):
            distribution = bootstrap_distribution(differences, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
            ci_low, ci_high = [float(x) for x in np.quantile(distribution, [0.025, 0.975])]
        oriented = None
        oriented_low = None
        oriented_high = None
        if point is not None and direction in {"higher_is_better", "lower_is_better"}:
            sign = 1.0 if direction == "higher_is_better" else -1.0
            oriented = sign * point
            if ci_low is not None and ci_high is not None:
                oriented_low = min(sign * ci_low, sign * ci_high)
                oriented_high = max(sign * ci_low, sign * ci_high)
        full_higher = sum(full - no > TIE_TOLERANCE for full, no in pairs)
        full_lower = sum(no - full > TIE_TOLERANCE for full, no in pairs)
        ties = len(pairs) - full_higher - full_lower
        full_better = full_worse = None
        if direction == "higher_is_better":
            full_better, full_worse = full_higher, full_lower
        elif direction == "lower_is_better":
            full_better, full_worse = full_lower, full_higher
        output.append(
            {
                "metric": metric,
                "direction": direction,
                "contrast": "delepref_full_control_minus_delepref_no_scorer",
                "n_pairs_expected": 50,
                "n_pairs_available": len(pairs),
                "paired_coverage": len(pairs) / 50.0,
                "point_estimate_full_minus_no_scorer": point,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "oriented_effect_positive_favors_full": oriented,
                "oriented_ci95_low": oriented_low,
                "oriented_ci95_high": oriented_high,
                "full_higher_count": full_higher,
                "tie_count": ties,
                "full_lower_count": full_lower,
                "full_better_count": full_better,
                "full_worse_count": full_worse,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "ci_method": "paired_case_percentile_95",
            }
        )
    return output


BOOTSTRAP_FIELDS = [
    "metric", "direction", "contrast", "n_pairs_expected", "n_pairs_available", "paired_coverage",
    "point_estimate_full_minus_no_scorer", "ci95_low", "ci95_high",
    "oriented_effect_positive_favors_full", "oriented_ci95_low", "oriented_ci95_high",
    "full_higher_count", "tie_count", "full_lower_count", "full_better_count", "full_worse_count",
    "bootstrap_seed", "bootstrap_replicates", "ci_method",
]


def missingness_rows(case_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in METHODS:
        rows = [row for row in case_rows if row["method"] == method]
        for field in METRIC_DIRECTIONS:
            available = len(numeric_values(rows, field))
            output.append(
                {
                    "method": method,
                    "comparison_class": "SAME_PROTOCOL" if method in SAME_PROTOCOL_METHODS else "EXTERNAL_REFERENCE",
                    "field_or_metric": field,
                    "n_expected": 50,
                    "n_available": available,
                    "n_missing": 50 - available,
                    "coverage": available / 50.0,
                    "failure_rate": (50 - available) / 50.0,
                    "missing_encoding": "CSV empty field / JSON null; no zero-fill, substitution, or rerun",
                }
            )
    return output


MISSINGNESS_FIELDS = [
    "method", "comparison_class", "field_or_metric", "n_expected", "n_available", "n_missing", "coverage",
    "failure_rate", "missing_encoding",
]


def scisor_audit_rows(
    analysis: Mapping[str, Any], inventory: Mapping[str, Any]
) -> list[dict[str, Any]]:
    case_by_key = {(row["case_id"], row["method"]): row for row in analysis["case_rows"]}
    prediction_by_key = {(row["case_id"], row["method"]): row for row in analysis["prediction_audit"]}
    differences = {
        key: {"delepref": inventory["predictor_config"].get(key), "original_SCISOR": inventory["original_SCISOR_predictor_config"].get(key)}
        for key in sorted(set(inventory["predictor_config"]) | set(inventory["original_SCISOR_predictor_config"]))
        if inventory["predictor_config"].get(key) != inventory["original_SCISOR_predictor_config"].get(key)
    }
    output: list[dict[str, Any]] = []
    for case_id in map(str, range(1, 51)):
        source = analysis["parsed_records"][(case_id, EXTERNAL_METHOD)].spec.inventory_record or {}
        audit = prediction_by_key[(case_id, EXTERNAL_METHOD)]
        metric = case_by_key[(case_id, EXTERNAL_METHOD)]
        output.append(
            {
                "case_id": case_id,
                "protein_id": metric["protein_id"],
                "comparison_class": "EXTERNAL_REFERENCE",
                "same_protocol": 0,
                "eligible": int(analysis["external_eligibility"][case_id]),
                "prediction_success": int(bool(source.get("prediction", {}).get("success"))),
                "semantic_input_match": source.get("semantic_input_match"),
                "ordered_semantic_input_match": source.get("ordered_semantic_input_match"),
                "target_permutation_external_reference_match": source.get("target_permutation_external_reference_match"),
                "chain_mapping_available": metric["chain_mapping_available"],
                "structure_available": metric["structure_available"],
                "pae_available": metric["pae_available"],
                "residue_mapping_available": metric["residue_mapping_available"],
                "delepref_predictor_config_sha256": inventory["predictor_config_sha256"],
                "scisor_predictor_config_sha256": inventory["original_SCISOR_predictor_config_sha256"],
                "delepref_predictor_config_json": json.dumps(inventory["predictor_config"], sort_keys=True, separators=(",", ":")),
                "scisor_predictor_config_json": json.dumps(inventory["original_SCISOR_predictor_config"], sort_keys=True, separators=(",", ":")),
                "predictor_config_differences_json": json.dumps(differences, sort_keys=True, separators=(",", ":")),
                "yaml_path": audit["yaml_path"],
                "confidence_json_path": audit["confidence_json_path"],
                "cif_path": audit["cif_path"],
                "pae_path": audit["pae_path"],
                "failure_reason": metric["failure_reason"],
                "boundary": "Post-deletion complex-prediction proxy; not a same-protocol comparison and not evidence of retained binding or function.",
            }
        )
    return output


SCISOR_FIELDS = [
    "case_id", "protein_id", "comparison_class", "same_protocol", "eligible", "prediction_success",
    "semantic_input_match", "ordered_semantic_input_match", "target_permutation_external_reference_match",
    "chain_mapping_available", "structure_available", "pae_available", "residue_mapping_available",
    "delepref_predictor_config_sha256", "scisor_predictor_config_sha256", "delepref_predictor_config_json",
    "scisor_predictor_config_json", "predictor_config_differences_json", "yaml_path", "confidence_json_path",
    "cif_path", "pae_path", "failure_reason", "boundary",
]


def process_boltz_like() -> list[dict[str, Any]]:
    detected: list[dict[str, Any]] = []
    own_pid = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            args = [x.decode("utf-8", errors="replace") for x in (entry / "cmdline").read_bytes().split(b"\0") if x]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not args:
            continue
        executable = Path(args[0]).name.lower()
        arg_basenames = {Path(arg).name.lower() for arg in args[1:]}
        is_boltz_executable = executable == "boltz" or executable.startswith("boltz-") or "boltz" in arg_basenames
        is_boltz_predict = is_boltz_executable and any(arg.lower() == "predict" for arg in args[1:])
        if is_boltz_executable or is_boltz_predict:
            detected.append({"pid": int(entry.name), "argv": args})
    return detected


def tree_snapshot(root: Path) -> dict[str, Any]:
    records: list[str] = []
    total_size = 0
    for path in sorted(x for x in root.rglob("*") if x.is_file()):
        stat = path.stat()
        rel = path.relative_to(root).as_posix()
        digest = sha256_file(path)
        records.append(f"{rel}\t{stat.st_size}\t{stat.st_mtime_ns}\t{digest}")
        total_size += stat.st_size
    payload = "\n".join(records).encode("utf-8")
    return {
        "root": str(root),
        "file_count": len(records),
        "total_size_bytes": total_size,
        "inventory_sha256": sha256_bytes(payload),
    }


def independent_validate(
    analysis: Mapping[str, Any],
    output_root: Path,
    core_artifacts: Mapping[str, Mapping[str, Any]],
    boltz_snapshot_before: Mapping[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, details: Any) -> None:
        checks.append({"name": name, "pass": bool(passed), "details": details})

    audit_rows = read_csv(output_root / "prediction_source_audit.csv")
    hash_mismatches: list[dict[str, str]] = []
    hash_fields = [
        ("yaml_path", "yaml_sha256"),
        ("confidence_json_path", "confidence_json_sha256"),
        ("cif_path", "cif_sha256"),
        ("pae_path", "pae_sha256"),
        ("plddt_path", "plddt_sha256"),
        ("manifest_structure_path", "manifest_structure_sha256_observed"),
    ]
    for row in audit_rows:
        for path_field, hash_field in hash_fields:
            path = Path(row[path_field])
            observed = sha256_file(path) if path.is_file() else ""
            if observed != row[hash_field]:
                hash_mismatches.append({"case_id": row["case_id"], "method": row["method"], "path": str(path)})
    add("source_paths_and_hashes_recomputed", not hash_mismatches, {"rows": len(audit_rows), "mismatches": hash_mismatches})

    case_rows_disk = read_csv(output_root / "binder_interface_case_metrics.csv")
    assignments = Counter(row["method"] for row in case_rows_disk)
    case_sets = {method: sorted({row["case_id"] for row in case_rows_disk if row["method"] == method}, key=int) for method in METHODS}
    assignment_ok = all(assignments[method] == 50 and case_sets[method] == list(map(str, range(1, 51))) for method in METHODS)
    add("fifty_case_assignment_recomputed", assignment_ok, {"assignment_counts": dict(assignments)})

    chain_errors: list[str] = []
    contact_errors: list[str] = []
    pae_errors: list[str] = []
    case_row_by_key = {(row["case_id"], row["method"]): row for row in analysis["case_rows"]}
    for key, parsed in sorted(analysis["parsed_records"].items(), key=lambda item: (int(item[0][0]), item[0][1])):
        if parsed.structure is None or parsed.mapping is None:
            continue
        try:
            fresh_structure = parse_structure(parsed.cif_path)
            fresh_entities = yaml_entities(parsed.spec.yaml_path)
            case_id, method = key
            if method == "wildtype_shared":
                canonical = None
            else:
                wt_entities = yaml_entities(analysis["parsed_records"][(case_id, "wildtype_shared")].spec.yaml_path)
                canonical = [x["sequence"] for x in wt_entities if x["role"] == "target"]
            fresh_mapping = map_entities_to_structure(fresh_entities, fresh_structure, canonical)
            observed = (
                [fresh_structure.chains[i].name for i in fresh_mapping.binder_chain_indices],
                {slot: fresh_structure.chains[i].name for slot, i in fresh_mapping.target_slot_to_chain_index.items()},
            )
            expected = (
                [parsed.structure.chains[i].name for i in parsed.mapping.binder_chain_indices],
                {slot: parsed.structure.chains[i].name for slot, i in parsed.mapping.target_slot_to_chain_index.items()},
            )
            if observed != expected:
                chain_errors.append(f"{case_id}:{method}:{observed}!={expected}")
            validation_geometry = compute_geometry_grid(fresh_structure, fresh_mapping)
            expected_geometry = parsed.geometry
            if expected_geometry is None or (
                validation_geometry.contacts != expected_geometry.contacts
                or validation_geometry.binder_interface_residues != expected_geometry.binder_interface_residues
                or validation_geometry.target_interface_residues != expected_geometry.target_interface_residues
                or validation_geometry.clash_count != expected_geometry.clash_count
                or validation_geometry.possible_heavy_atom_pairs != expected_geometry.possible_heavy_atom_pairs
            ):
                contact_errors.append(f"{case_id}:{method}")
            if parsed.pae is not None and parsed.pae_metric is not None:
                independent_pae = aggregate_pae_independent(parsed.pae, fresh_structure, fresh_mapping)
                if not math.isclose(independent_pae, parsed.pae_metric, rel_tol=0.0, abs_tol=1e-10):
                    pae_errors.append(f"{case_id}:{method}:{independent_pae}:{parsed.pae_metric}")
        except Exception as exc:
            chain_errors.append(f"{key[0]}:{key[1]}:{type(exc).__name__}:{exc}")
    add("chain_mapping_independently_recomputed", not chain_errors, {"records_checked": len(analysis["parsed_records"]), "errors": chain_errors})
    add("contacts_independently_recomputed", not contact_errors, {"records_checked": len(analysis["parsed_records"]), "errors": contact_errors})
    add("pae_aggregation_independently_recomputed", not pae_errors, {"records_checked": len(analysis["parsed_records"]), "errors": pae_errors})

    missing_disk = read_csv(output_root / "missingness_and_coverage.csv")
    coverage_errors: list[str] = []
    for row in missing_disk:
        method_rows = [x for x in case_rows_disk if x["method"] == row["method"]]
        available = sum(x[row["field_or_metric"]] != "" for x in method_rows)
        if available != int(row["n_available"]):
            coverage_errors.append(f"{row['method']}:{row['field_or_metric']}")
    add("coverage_independently_recomputed", not coverage_errors, {"rows_checked": len(missing_disk), "errors": coverage_errors})

    bootstrap_disk = read_csv(output_root / "binder_interface_paired_bootstrap_ci.csv")
    bootstrap_errors: list[str] = []
    disk_by_key = {(row["case_id"], row["method"]): row for row in case_rows_disk}
    for row in bootstrap_disk:
        metric = row["metric"]
        differences: list[float] = []
        for case_id in map(str, range(1, 51)):
            full = disk_by_key[(case_id, "delepref_full_control")][metric]
            no = disk_by_key[(case_id, "delepref_no_scorer")][metric]
            if full != "" and no != "":
                differences.append(float(full) - float(no))
        values = np.asarray(differences, dtype=np.float64)
        if len(values) != int(row["n_pairs_available"]):
            bootstrap_errors.append(f"{metric}:coverage")
            continue
        expected_point = float(np.mean(values)) if len(values) else None
        if expected_point is not None:
            distribution = bootstrap_distribution(values, int(row["bootstrap_replicates"]), int(row["bootstrap_seed"]))
            expected_ci = np.quantile(distribution, [0.025, 0.975])
            observed = [float(row["point_estimate_full_minus_no_scorer"]), float(row["ci95_low"]), float(row["ci95_high"])]
            expected = [expected_point, float(expected_ci[0]), float(expected_ci[1])]
            if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in zip(observed, expected)):
                bootstrap_errors.append(f"{metric}:values:{observed}:{expected}")
    add("paired_bootstrap_independently_recomputed", not bootstrap_errors, {"metrics_checked": len(bootstrap_disk), "errors": bootstrap_errors})

    parse_errors: list[str] = []
    for name in core_artifacts:
        path = output_root / name
        try:
            if path.suffix == ".csv":
                read_csv(path)
            elif path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            parse_errors.append(f"{name}:{type(exc).__name__}:{exc}")
    add("json_csv_parse", not parse_errors, {"artifacts_checked": len(core_artifacts), "errors": parse_errors})

    artifact_hash_errors: list[str] = []
    for name, record in core_artifacts.items():
        observed = sha256_file(output_root / name)
        if observed != record["sha256"]:
            artifact_hash_errors.append(name)
    add("artifact_sha256_recomputed", not artifact_hash_errors, {"artifacts_checked": len(core_artifacts), "errors": artifact_hash_errors})

    try:
        with tempfile.TemporaryDirectory(prefix="delepref_12_pycompile_") as tempdir:
            py_compile.compile(str(Path(__file__).resolve()), cfile=str(Path(tempdir) / "analysis.pyc"), doraise=True)
        add("py_compile", True, {"script": str(Path(__file__).resolve())})
    except Exception as exc:
        add("py_compile", False, {"error": f"{type(exc).__name__}:{exc}"})

    boltz_snapshot_after = tree_snapshot(Path(boltz_snapshot_before["root"]))
    add(
        "no_new_or_modified_boltz_outputs",
        boltz_snapshot_after == dict(boltz_snapshot_before),
        {"before": dict(boltz_snapshot_before), "after": boltz_snapshot_after},
    )
    detected_processes = process_boltz_like()
    add("no_boltz_process_detected", not detected_processes, {"detected": detected_processes})

    git_result = subprocess.run(
        ["git", "diff", "--check"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    add("git_diff_check", git_result.returncode == 0, {"returncode": git_result.returncode, "output": git_result.stdout})
    return {
        "schema_version": "delepref_12_independent_validation_v1",
        "validated_at_utc": utc_now(),
        "independent_algorithms": {
            "contacts": "27-cell spatial grid, independent of formal cKDTree implementation",
            "pae": "explicit directed nested summation, independent of formal block concatenation",
            "bootstrap": "CSV reparse and deterministic distribution regeneration",
        },
        "checks": checks,
        "all_checks_pass": all(check["pass"] for check in checks),
        "status": "PASS" if all(check["pass"] for check in checks) else "FAIL",
    }


def fmt(value: Any, digits: int = 6) -> str:
    if value is None or value == "":
        return "NA"
    return f"{float(value):.{digits}g}"


def build_markdown(
    title: str,
    method_summary: Sequence[Mapping[str, Any]],
    bootstrap: Sequence[Mapping[str, Any]],
    external_eligible_count: int,
    status: str,
) -> str:
    summary_lookup = {(row["method"], row["metric"]): row for row in method_summary}
    lines = [
        f"# {title}",
        "",
        f"Status: `{status}`",
        "",
        "This is a strict CPU-only retrospective analysis of existing predictions. All values are post-deletion complex-prediction proxies. They do not establish retained binding, real Kd/Ki, functional preservation, or absence of function in a deleted region.",
        "",
        "## Input and protocol", "",
        "- Authoritative Binder package status: `PASS_DelePref_BINDER`.",
        "- Cases: 50; DelePref assignments: 50 full-control + 50 no-scorer; unique real DelePref edited predictions: 90; zero-fill: 0.",
        f"- Frozen protocol SHA256: `{EXPECTED_PROTOCOL_SHA256}`.",
        "- Protocol schema v2 records that an unaccepted v1 attempt was invalidated after the input schema proved `deleted_intervals` are zero-based closed intervals (100/100 deleted-length and edited-sequence reconstructions); no cutoff, metric, missingness, comparison, or statistic changed.",
        f"- Paired bootstrap: case-level, seed {BOOTSTRAP_SEED}, {BOOTSTRAP_REPLICATES:,} replicates.",
        "- The only SAME_PROTOCOL contrast is DelePref full-control minus DelePref no-scorer.",
        f"- Original SCISOR eligible cases: {external_eligible_count}/50 and always labeled EXTERNAL_REFERENCE; its predictor used 20 sampling steps versus 50 for DelePref, so it is not a fully same-protocol comparison.",
        "",
        "## SAME_PROTOCOL method means", "",
        "| Metric | Direction | Full control (n/50, mean) | No scorer (n/50, mean) |",
        "|---|---:|---:|---:|",
    ]
    for metric, direction in METRIC_DIRECTIONS.items():
        full = summary_lookup[("delepref_full_control", metric)]
        no = summary_lookup[("delepref_no_scorer", metric)]
        lines.append(
            f"| {metric} | {direction} | {full['n_available']}/50, {fmt(full['mean'])} | {no['n_available']}/50, {fmt(no['mean'])} |"
        )
    lines.extend(
        [
            "", "## Paired full-control minus no-scorer bootstrap", "",
            "Every prespecified metric is retained, including negative, null, or uncertain results.", "",
            "| Metric | Direction | Coverage | Point estimate | 95% CI | Oriented effect (positive favors full) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in bootstrap:
        lines.append(
            f"| {row['metric']} | {row['direction']} | {row['n_pairs_available']}/50 | {fmt(row['point_estimate_full_minus_no_scorer'])} | [{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] | {fmt(row['oriented_effect_positive_favors_full'])} |"
        )
    lines.extend(
        [
            "", "## Interpretation boundary", "",
            "The analysis measures predictor confidence, predicted inter-chain error, predicted interface contacts, deletion/interface overlap, and predicted steric geometry. Teacher-derived preference is not a ground-truth functional label, and no result proves that a deleted region lacks function.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    protocol_path = output_root / "protocol.json"
    if not protocol_path.is_file() or sha256_file(protocol_path) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Frozen protocol is missing or its SHA256 changed; formal computation refused")
    if process_boltz_like():
        raise RuntimeError("A Boltz process is present before analysis; refusing to continue")

    boltz_snapshot_before = tree_snapshot(input_root / "boltz_outputs")
    started_at = utc_now()
    analysis = analyze(input_root)
    method_summary = summarize_methods(analysis["case_rows"])
    bootstrap = paired_bootstrap(analysis["case_rows"])
    missingness = missingness_rows(analysis["case_rows"])
    scisor = scisor_audit_rows(analysis, analysis["inventory"])

    core_payloads: dict[str, bytes] = {
        "prediction_source_audit.csv": csv_bytes(analysis["prediction_audit"], PREDICTION_AUDIT_FIELDS),
        "residue_mapping_audit.csv": csv_bytes(analysis["mapping_audit"], MAPPING_AUDIT_FIELDS),
        "binder_interface_case_metrics.csv": csv_bytes(analysis["case_rows"], CASE_FIELDS),
        "binder_interface_method_summary.csv": csv_bytes(method_summary, SUMMARY_FIELDS),
        "binder_interface_paired_bootstrap_ci.csv": csv_bytes(bootstrap, BOOTSTRAP_FIELDS),
        "missingness_and_coverage.csv": csv_bytes(missingness, MISSINGNESS_FIELDS),
        "scisor_external_reference_audit.csv": csv_bytes(scisor, SCISOR_FIELDS),
    }
    core_artifacts: dict[str, dict[str, Any]] = {}
    for name, data in core_payloads.items():
        core_artifacts[name] = write_bytes(output_root / name, data)

    validation = independent_validate(analysis, output_root, core_artifacts, boltz_snapshot_before)
    if not validation["all_checks_pass"]:
        write_bytes(output_root / "independent_validation.json", canonical_json_bytes(validation))
        raise RuntimeError("Independent validation failed; PASS status and final payloads were not written")

    external_eligible_count = sum(analysis["external_eligibility"].values())
    summary_md = build_markdown(
        "DelePref-12 DelePref Binder Interface-Specific Retrospective Analysis",
        method_summary,
        bootstrap,
        external_eligible_count,
        FINAL_STATUS,
    )
    payload_body = build_markdown(
        "Supplementary Binder interface analysis payload",
        method_summary,
        bootstrap,
        external_eligible_count,
        FINAL_STATUS,
    )
    payload_md = f"<!-- {MARKER_START} -->\n{payload_body}\n<!-- {MARKER_END} -->\n"
    payload_json_obj = {
        "schema_version": "delepref_12_result_summary_v1",
        "marker_start": MARKER_START,
        "marker_end": MARKER_END,
        "status": FINAL_STATUS,
        "protocol_schema_version": "delepref_12_binder_interface_protocol_v2",
        "protocol_revision_note": "An unaccepted v1 attempt was invalidated; v2 corrects only selected-plan interval semantics to zero-based closed endpoints, as fixed by 100/100 input reconstructions.",
        "comparison_boundary": {
            "same_protocol": list(SAME_PROTOCOL_METHODS),
            "external_reference": [EXTERNAL_METHOD],
        },
        "claim_boundary": "Post-deletion complex-prediction proxy only; no retained-binding, Kd/Ki, or functional-preservation claim.",
        "method_summary": method_summary,
        "paired_bootstrap": bootstrap,
        "external_reference_eligible_cases": external_eligible_count,
    }
    run_summary_obj = {
        "schema_version": "delepref_12_binder_interface_run_summary_v1",
        "analysis_id": "DelePref-12",
        "status": FINAL_STATUS,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "cpu_only": True,
        "boltz_or_model_inference_run": False,
        "new_structure_predictions": 0,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "input_status": analysis["input_run_summary"]["status"],
        "case_count": 50,
        "delepref_assignment_count": 100,
        "delepref_full_control_assignment_count": 50,
        "delepref_no_scorer_assignment_count": 50,
        "unique_real_delepref_edited_prediction_count": 90,
        "zero_fill_count": 0,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "protocol_schema_version": "delepref_12_binder_interface_protocol_v2",
        "protocol_revision_note": "An unaccepted v1 attempt was invalidated; v2 corrects only selected-plan interval semantics to zero-based closed endpoints, with all other prespecified rules unchanged.",
        "same_protocol_methods": list(SAME_PROTOCOL_METHODS),
        "external_reference_method": EXTERNAL_METHOD,
        "external_reference_eligible_case_count": external_eligible_count,
        "bootstrap": {"unit": "case", "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "ci": "percentile_95"},
        "coverage": {
            method: {
                field: sum(int(row[field]) for row in analysis["case_rows"] if row["method"] == method)
                for field in ("prediction_available", "structure_available", "pae_available", "contact_metrics_available")
            }
            for method in METHODS
        },
        "independent_validation_status": validation["status"],
        "input_boltz_output_snapshot": boltz_snapshot_before,
        "claim_boundary": "All metrics are post-deletion complex-prediction proxies; they do not establish retained binding, real Kd/Ki, or functional preservation.",
    }

    final_payloads = {
        "result_summary.md": payload_md.encode("utf-8"),
        "result_summary.json": canonical_json_bytes(payload_json_obj),
        "SUMMARY.md": (summary_md + "\n").encode("utf-8"),
        "run_summary.json": canonical_json_bytes(run_summary_obj),
    }
    prospective_hashes = {
        **{name: record["sha256"] for name, record in core_artifacts.items()},
        **{name: sha256_bytes(data) for name, data in final_payloads.items()},
        "protocol.json": sha256_file(protocol_path),
        "scripts/run_delepref_12_binder_interface_analysis.py": sha256_file(Path(__file__).resolve()),
    }
    validation["artifact_sha256"] = prospective_hashes
    validation["status"] = "PASS"
    validation_bytes = canonical_json_bytes(validation)
    final_payloads["independent_validation.json"] = validation_bytes
    final_records: dict[str, dict[str, Any]] = {}
    for name, data in final_payloads.items():
        final_records[name] = write_bytes(output_root / name, data)

    artifacts: list[dict[str, Any]] = []
    for name in sorted(set(core_artifacts) | set(final_records) | {"protocol.json"}):
        path = output_root / name
        artifacts.append({"path": name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    script_path = Path(__file__).resolve()
    artifacts.append(
        {
            "path": str(script_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(script_path),
            "size_bytes": script_path.stat().st_size,
            "outside_output_root": True,
        }
    )
    audit_manifest_obj = {
        "schema_version": "delepref_12_binder_interface_audit_manifest_v1",
        "status": FINAL_STATUS,
        "created_at_utc": utc_now(),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "audit_manifest_self_hash_excluded": True,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "checks": validation["checks"],
        "all_checks_pass": validation["all_checks_pass"],
        "zero_fill_used": False,
        "new_boltz_outputs_or_predictions": False,
        "claim_boundary": run_summary_obj["claim_boundary"],
    }
    write_bytes(output_root / "audit_manifest.json", canonical_json_bytes(audit_manifest_obj))

    # Final non-mutating parse/hash sanity check. No status file is changed after this point.
    json_names = ["protocol.json", "independent_validation.json", "result_summary.json", "run_summary.json", "audit_manifest.json"]
    for name in json_names:
        json.loads((output_root / name).read_text(encoding="utf-8"))
    for artifact in artifacts:
        path = REPO_ROOT / artifact["path"] if artifact.get("outside_output_root") else output_root / artifact["path"]
        if sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"Final artifact hash mismatch: {path}")
    print(FINAL_STATUS)
    print(f"Output: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
