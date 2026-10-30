from split_families import Components, similarity, make_aligner, threshold_hit

CONFIG = {"identity": .4, "coverage_each": .8, "alignment": {"mode": "local", "substitution_matrix": "BLOSUM62", "open_gap_score": -10., "extend_gap_score": -.5}}


def test_similarity_identity_and_bidirectional_coverage():
    aligner = make_aligner(CONFIG["alignment"])
    seq = "ACDEFGHIKLMNPQRSTVWY" * 3
    stats = similarity(seq, seq, aligner)
    assert stats["identity"] == stats["coverage_a"] == stats["coverage_b"] == 1
    assert threshold_hit(stats, CONFIG)
    partial = similarity(seq, seq + "A" * 100, aligner)
    assert partial["identity"] == 1 and not threshold_hit(partial, CONFIG)
    reverse = similarity(seq + "A" * 100, seq, aligner)
    assert reverse["identity"] == partial["identity"]
    assert reverse["coverage_a"] == partial["coverage_b"]


def test_family_sequence_union_is_transitive():
    groups = Components(["a", "b", "c", "d"])
    groups.union("a", "b"); groups.union("b", "c")
    assert groups.find("a") == groups.find("c")
    assert groups.find("a") != groups.find("d")
