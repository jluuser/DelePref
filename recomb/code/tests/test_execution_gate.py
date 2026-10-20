import json
from pathlib import Path
import numpy as np
import pytest
from common import sha_file
from execution_gate import pilot_evidence


def fixture_pilot(root, failure=False, batch=4):
    directory = root / "caches/embeddings"; directory.mkdir(parents=True)
    expected_rows = []
    for shard, count in [(0, 8), (1, 4)]:
        rows = []
        for index in range(count):
            path = directory / f"{shard}_{index}.npy"
            np.save(path, np.zeros((3, 1152), dtype=np.float32))
            expected = {"sequence_sha256": f"{shard}_{index}", "length": 3, "weight_sha256": "weights", "protocol_sha256": "protocol"}
            expected_rows.append(expected)
            path.with_suffix(".json").write_text(json.dumps({**expected, "output_sha256": sha_file(path)}))
            rows.append({**expected, "path": str(path), "key": f"{shard}_{index}", "reused": False})
        (directory / f"pilot_shard{shard}_index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (directory / f"pilot_shard{shard}_report.json").write_text(json.dumps({"profile": {"weight_sha256": "weights"},
            "timings": [{"seconds": 1., "length": 3, "batch": batch, "peak_bytes": 1024}], "failures": ["failed"] if failure else []}))
    return expected_rows


def test_pilot_gate_requires_complete_verified_batch_timings(tmp_path):
    expected = fixture_pilot(tmp_path)
    reports, timings, peak = pilot_evidence(tmp_path, expected, "protocol", {"weight_sha256": "weights"})
    assert len(reports) == 2 and peak == 1024
    file = next((tmp_path / "caches/embeddings").glob("*.npy"))
    np.save(file, np.ones((3, 1152), dtype=np.float32))
    with pytest.raises(ValueError, match="integrity"):
        pilot_evidence(tmp_path, expected, "protocol", {"weight_sha256": "weights"})


@pytest.mark.parametrize("failure,batch", [(True, 4), (False, 1)])
def test_pilot_gate_stops_failures_and_unmeasured_full_microbatches(tmp_path, failure, batch):
    expected = fixture_pilot(tmp_path, failure, batch)
    with pytest.raises(ValueError):
        pilot_evidence(tmp_path, expected, "protocol", {"weight_sha256": "weights"})


def test_gate_rejects_stale_or_unrelated_pilot(tmp_path):
    expected = fixture_pilot(tmp_path)
    with pytest.raises(ValueError, match="authorized cohort"):
        pilot_evidence(tmp_path, expected[:-1], "protocol", {"weight_sha256": "weights"})
    with pytest.raises(ValueError, match="authorized current computation"):
        pilot_evidence(tmp_path, expected, "protocol", {"weight_sha256": "changed"})
    with pytest.raises(ValueError, match="protocol/profile"):
        pilot_evidence(tmp_path, expected, "other_protocol", {"weight_sha256": "weights"})
