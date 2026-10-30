import numpy as np
import pytest
from adapters.delepref_adapter import retained_mapping, deletion_mask_from_intervals, intervals_from_deletion_mask, strict_delete_count
from adapters.structure_adapter import read_coordinates, read_cath_reference
from build_pairs import reference_preservation_labels


def test_internal_double_deletion_end_to_end():
    wt = "ACDEFGHIKL"
    mask = deletion_mask_from_intervals(len(wt), [(2, 3), (6, 6)])
    edited = "".join(a for a, d in zip(wt, mask) if not d)
    kept = retained_mapping(wt, edited, mask)
    xyz = np.column_stack([np.arange(len(wt)) * 1., np.zeros((len(wt), 2))])
    labels = reference_preservation_labels(xyz, xyz[kept], kept, [0, 1, 4], [5, 8, 9])
    assert np.array_equal(kept[labels["edited_i"]], labels["wt_i"])
    assert np.array_equal(kept[labels["edited_j"]], labels["wt_j"])
    embeddings = np.repeat(np.arange(len(wt))[:, None], 1152, axis=1)
    edited_embeddings = embeddings[kept]
    assert np.array_equal(edited_embeddings[labels["edited_i"]], embeddings[labels["wt_i"]])
    assert np.array_equal(edited_embeddings[labels["edited_j"]], embeddings[labels["wt_j"]])
    assert all(wt[i] == edited[a] for i, a in zip(labels["wt_i"], labels["edited_i"]))
    assert labels["target"].tolist() == [0, 0, 0]
    assert intervals_from_deletion_mask(mask) == [(2, 3), (6, 6)]


def test_mask_contract():
    with pytest.raises(ValueError):
        retained_mapping("ACD", "AD", [0, 2, 0])
    with pytest.raises(ValueError):
        retained_mapping("ACD", "AC", [0, 1, 0])
    with pytest.raises(ValueError):
        deletion_mask_from_intervals(10, [(1, 3), (4, 5)])
    assert strict_delete_count(179, .2) == 36
    assert retained_mapping("ACD", "ACD", [0, 0, 0]).tolist() == [0, 1, 2]


def test_pdb_missing_position_and_insertion_code(tmp_path):
    p = tmp_path / "variant.pdb"
    p.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
                 "ATOM      2  CA  ASP A   3A      2.000   0.000   0.000  1.00 20.00           C\nEND\n")
    xyz, audit = read_coordinates(p, "ACD")
    assert xyz.shape == (3, 3)
    assert np.isnan(xyz[1]).all()
    assert xyz[2].tolist() == [2., 0., 0.]
    assert audit["residue_map"][1]["insertion_code"] == "A"
    assert audit["missing_positions"] == [1]
    xyz_cath, cath_audit = read_cath_reference(p, "ACD", "A")
    assert np.array_equal(np.isfinite(xyz_cath), np.isfinite(xyz))
    assert cath_audit["missing_positions"] == [1]
    with pytest.raises(ValueError):
        read_cath_reference(p, "AACD", "A")
    extensionless = tmp_path / "1abcA00"
    extensionless.write_text(p.read_text())
    xyz_extensionless, _ = read_cath_reference(extensionless, "ACD", "A")
    assert np.allclose(xyz_extensionless, xyz_cath, equal_nan=True)


def test_mmcif_label_auth_and_missing_mapping(tmp_path):
    import gemmi
    structure = gemmi.Structure()
    structure.name = "mapping"
    entity = gemmi.Entity("1")
    entity.entity_type = gemmi.EntityType.Polymer
    entity.polymer_type = gemmi.PolymerType.PeptideL
    entity.full_sequence = ["ALA", "CYS", "ASP", "GLU"]
    entity.subchains = ["Axp"]
    structure.entities.append(entity)
    model = gemmi.Model("1"); chain = gemmi.Chain("B")
    for label, auth, insertion, residue_name in [(1, 10, " ", "ALA"), (3, 11, "A", "ASP"), (4, 12, " ", "GLU")]:
        residue = gemmi.Residue(); residue.name = residue_name
        residue.seqid = gemmi.SeqId(auth, insertion); residue.label_seq = label
        residue.subchain = "Axp"; residue.entity_type = gemmi.EntityType.Polymer
        atom = gemmi.Atom(); atom.name = "CA"; atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(label, 0, 0); atom.occ = 1
        residue.add_atom(atom); chain.add_residue(residue)
    model.add_chain(chain); structure.add_model(model)
    p = tmp_path / "reference.cif"; structure.make_mmcif_document().write_file(str(p))
    xyz, audit = read_coordinates(p, "ACDE", "B", native=True)
    assert np.isnan(xyz[1]).all()
    assert xyz[2, 0] == 3
    assert audit["residue_map"][1]["auth_number"] == 11
    assert audit["residue_map"][1]["insertion_code"] == "A"
    with pytest.raises(ValueError):
        read_coordinates(p, "ACDE", "A", native=True)
