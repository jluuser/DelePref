"""Sequence-position preserving structure mapping; no compaction of missing CA."""
import json
from pathlib import Path
import gemmi
import numpy as np
from common import sha_file, sha_text


def coordinates_from_native_json(path, sequence):
    data = json.loads(Path(path).read_text())
    if data["wt_sequence_sha256"] != sha_text(sequence):
        raise ValueError("native JSON WT identity mismatch")
    coords = np.full((len(sequence), 3), np.nan)
    for key, value in data["coordinates"].items():
        i = int(key)
        if not 0 <= i < len(sequence) or np.asarray(value).shape != (3,):
            raise ValueError("native JSON index/coordinate invalid")
        coords[i] = value
    return coords, data


def read_coordinates(path, sequence, chain_id="A", native=False):
    structure = gemmi.read_structure(str(path))
    chain = structure[0].find_chain(chain_id)
    if chain is None:
        raise ValueError("missing chain " + chain_id)
    offset = 0
    if native:
        entity = structure.get_entity_of(chain.get_polymer())
        if entity is None or not entity.full_sequence:
            raise ValueError("reference lacks full polymer entity mapping")
        full = "".join(gemmi.find_tabulated_residue(x).one_letter_code.upper() for x in entity.full_sequence)
        starts = [i for i in range(len(full) - len(sequence) + 1) if full[i:i + len(sequence)] == sequence]
        if len(starts) != 1:
            raise ValueError("reference sequence is not a unique exact entity segment")
        offset = starts[0]
    coords = np.full((len(sequence), 3), np.nan)
    residue_map = []
    seen = set()
    for residue in chain:
        if native and residue.entity_type == gemmi.EntityType.NonPolymer:
            continue
        ca = [a for a in residue if a.name == "CA" and a.element.name == "C"]
        if not ca:
            continue
        position = residue.label_seq if native else residue.seqid.num
        if position is None:
            raise ValueError("missing sequence position")
        i = int(position) - 1 - offset
        if native and not 0 <= i < len(sequence):
            continue
        if not 0 <= i < len(sequence):
            raise ValueError("predictor position outside input sequence")
        aa = gemmi.find_tabulated_residue(residue.name).one_letter_code.upper()
        if aa != sequence[i] and sequence[i] != "X":
            raise ValueError(f"coordinate amino acid mismatch at {i}: {aa} vs {sequence[i]}")
        if i in seen:
            raise ValueError("multiple residues map to one sequence position")
        seen.add(i)
        # Keep the frozen native-export conformer: first CA in original file order.
        # Choosing highest occupancy can silently change the supplied WT reference.
        atom = ca[0]
        coords[i] = [atom.pos.x, atom.pos.y, atom.pos.z]
        residue_map.append({"sequence_index": i, "chain": chain.name,
                            "auth_number": residue.seqid.num, "insertion_code": residue.seqid.icode,
                            "label_seq": residue.label_seq, "residue_name": residue.name,
                            "altloc": atom.altloc, "occupancy": atom.occ})
    return coords, {"path": str(path), "sha256": sha_file(path), "chain": chain_id,
                    "entity_offset": offset, "altloc_policy": "first_CA_in_file_order_frozen_reference", "residue_map": residue_map,
                    "missing_positions": np.flatnonzero(~np.isfinite(coords).all(1)).tolist()}


def audited_wt_reference(native_json, sequence, reference_path, chain_id):
    exported, meta = coordinates_from_native_json(native_json, sequence)
    if sha_file(reference_path) != meta["reference_sha256"]:
        raise ValueError("WT reference hash differs from native export")
    remapped, mapping = read_coordinates(reference_path, sequence, chain_id, native=True)
    if not np.array_equal(np.isfinite(exported), np.isfinite(remapped)) or not np.allclose(exported, remapped, atol=1e-6, equal_nan=True):
        raise ValueError("native export differs from independent sequence-coordinate mapping")
    return exported, mapping


def read_cath_reference(path, sequence, chain_id):
    """Map an ATOM-only CATH domain by a unique exact subsequence; preserve missing WT positions."""
    structure = gemmi.read_structure(str(path), format=gemmi.CoorFormat.Pdb)
    chain = structure[0].find_chain(chain_id)
    if chain is None:
        raise ValueError("CATH reference chain missing")
    residues = [(r, [a for a in r if a.name == "CA" and a.element.name == "C"])
                for r in chain]
    residues = [(r, ca[0]) for r, ca in residues if ca]
    observed = "".join(gemmi.find_tabulated_residue(r.name).one_letter_code.upper() for r, _ in residues)
    if not observed:
        raise ValueError("CATH reference has no CA residues")
    earliest = []; position = 0
    for aa in observed:
        index = sequence.find(aa, position)
        if index < 0:
            raise ValueError("CATH coordinates not an exact WT subsequence")
        earliest.append(index); position = index + 1
    latest = []; position = len(sequence)
    for aa in observed[::-1]:
        index = sequence.rfind(aa, 0, position)
        if index < 0:
            raise ValueError("CATH reverse exact mapping failed")
        latest.append(index); position = index
    if earliest != latest[::-1]:
        raise ValueError("CATH missing-residue mapping is ambiguous")
    coords = np.full((len(sequence), 3), np.nan); residue_map = []
    for i, (residue, atom) in zip(earliest, residues):
        coords[i] = [atom.pos.x, atom.pos.y, atom.pos.z]
        residue_map.append({"sequence_index": i, "chain": chain_id, "auth_number": residue.seqid.num,
                            "insertion_code": residue.seqid.icode, "residue_name": residue.name,
                            "altloc": atom.altloc, "occupancy": atom.occ})
    return coords, {"path": str(path), "sha256": sha_file(path), "chain": chain_id,
                    "mapping_mode": "unique_exact_subsequence_atom_only_CATH",
                    "altloc_policy": "first_CA_in_file_order_frozen_reference", "residue_map": residue_map,
                    "missing_positions": np.flatnonzero(~np.isfinite(coords).all(1)).tolist()}
