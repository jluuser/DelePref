from cache_plm import candidates, pilot_rows, verified_cache


def test_pilot_measures_full_microbatches_and_disjoint_shards():
    rows = [{"sequence_sha256": f"{length}_{i}", "length": length} for length in [150, 250, 550, 900] for i in range(8)]
    selected = pilot_rows(rows, 4)
    assert len(selected) == 12
    assert {row["length"] for _, row in selected} == {150, 550, 900}
    for shard in [0, 1]:
        shard_rows = [row for assigned, row in selected if assigned == shard]
        assert len(shard_rows) % 4 == 0
    assert len({row["sequence_sha256"] for _, row in selected}) == 12


def test_cache_plan_deduplicates_wt_and_variants():
    rows = [{"candidate_id": "a", "wt_sequence": "ACDE", "sequence": "ADE"},
            {"candidate_id": "b", "wt_sequence": "ACDE", "sequence": "ADE"}]
    assert len(candidates(rows)) == 2


def test_cache_reuse_requires_profile_and_output_integrity(tmp_path):
    import json
    import numpy as np
    from common import sha_file
    path = tmp_path / "embedding.npy"; meta = path.with_suffix(".json")
    np.save(path, np.zeros((3, 1152), dtype=np.float32))
    expected = {"length": 3, "sequence_sha256": "example", "weight_sha256": "weights"}
    meta.write_text(json.dumps({**expected, "output_sha256": sha_file(path)}))
    assert verified_cache(path, meta, expected)
    assert not verified_cache(path, meta, {**expected, "weight_sha256": "other"})
    np.save(path, np.ones((3, 1152), dtype=np.float32))
    assert not verified_cache(path, meta, expected)
