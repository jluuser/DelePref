"""CPU-only, append-only structure-call accounting. No predictor is invoked here.

The JSONL journal is authoritative; its sibling lock serializes readers/writers.
Reservations are never reclaimed, even if a worker dies before starting. A torn
journal fails closed and requires manual audit. Use a filesystem whose advisory
locks are shared by all workers (verify this on the deployment filesystem).
"""
import hashlib
import json
import math
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class BudgetExceeded(RuntimeError):
    pass


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@contextmanager
def _lock(path):
    with open(path, "a+b") as f:
        if os.name == "nt":
            import msvcrt
            if not f.seek(0, 2):
                f.write(b"\0")
                f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f, fcntl.LOCK_UN)


class BudgetLedger:
    """Both paths are required: there is deliberately no historical-root default.

    `reserve` is idempotent, but only its `new_reservation=True` result grants
    dispatch ownership. Call `start` immediately before the actual predictor
    call. Failed actual calls remain charged. Retry with attempt=1 only after
    attempt=0 failed; retries charge the dedicated retry category.

    An identical pending cache identity returns the owning task's record plus
    requested_task_id; the caller must wait, not dispatch. Memory is recorded in
    bytes and elapsed runtime in seconds. Cache reuse requires a prior successful
    journal entry plus a currently matching output digest, not file existence.
    """
    def __init__(self, protocol_path, ledger_path):
        self.path = Path(ledger_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw = Path(protocol_path).read_bytes()
        self.protocol_sha = hashlib.sha256(raw).hexdigest()
        config = json.loads(raw)["compute"]
        self.limit = config["structure_call_limit"]
        self.caps = config["structure_categories"]
        self.retries = config["structure_retries_max"]
        if (not isinstance(self.limit, int) or not 0 < self.limit <= 800
                or self.retries not in (0, 1)
                or any(not isinstance(v, int) or v < 0 for v in self.caps.values())):
            raise ValueError("invalid structure budget protocol")

    def _read(self):
        if not self.path.exists():
            return {}
        raw = self.path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise ValueError("incomplete ledger; manual audit required")
        latest = {}
        for line in raw.splitlines():
            row = json.loads(line)
            if row["protocol_sha"] != self.protocol_sha:
                raise ValueError("ledger protocol identity mismatch")
            latest[(row["task_id"], row["attempt"])] = row
        return latest

    def _append(self, row):
        row = dict(row, event_time=datetime.now(timezone.utc).isoformat())
        with self.path.open("ab") as f:
            f.write((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode())
            f.flush()
            os.fsync(f.fileno())
        return row

    @staticmethod
    def _fingerprint(row):
        return tuple(row[k] for k in ("sequence_sha", "predictor_weights_sha",
                                     "profile_sha", "seed", "sample"))

    def snapshot(self):
        with _lock(str(self.path) + ".lock"):
            rows = list(self._read().values())
            return {"consumed": sum(r["reserved_call"] for r in rows),
                    "categories": {c: sum(r["reserved_call"] for r in rows
                                          if r["category"] == c) for c in self.caps},
                    "attempts": rows}

    def reserve(self, *, task_id, sequence_sha, wt_variant, predictor_weights_sha,
                profile_sha, seed, category, job_id=None, sample=0, attempt=0):
        if not task_id or wt_variant not in ("WT", "variant"):
            raise ValueError("task_id and WT/variant identity required")
        for digest in (sequence_sha, predictor_weights_sha, profile_sha):
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("identities must be lowercase SHA256 digests")
        if (type(seed) is not int or type(sample) is not int or sample < 0
                or type(attempt) is not int or not 0 <= attempt <= self.retries):
            raise ValueError("invalid seed, sample, or attempt")
        if category not in self.caps or (attempt == 0 and category == "retry"):
            raise ValueError("invalid primary category")
        row = dict(task_id=task_id, sequence_sha=sequence_sha, wt_variant=wt_variant,
                   predictor_weights_sha=predictor_weights_sha, profile_sha=profile_sha,
                   seed=seed, sample=sample, requested_time=datetime.now(timezone.utc).isoformat(),
                   job_id=job_id, state="reserved", attempt=attempt, reserved_call=1,
                   seconds=None, max_memory=None, output_sha=None, output_path=None,
                   cache_hit=False, category="retry" if attempt else category,
                   protocol_sha=self.protocol_sha)
        with _lock(str(self.path) + ".lock"):
            latest = self._read()
            old = latest.get((task_id, attempt))
            if old:
                if (self._fingerprint(old) != self._fingerprint(row)
                        or old["wt_variant"] != wt_variant or old["category"] != row["category"]):
                    raise ValueError("task id reused with conflicting request")
                if old["state"] in ("succeeded", "cache_hit"):
                    try:
                        valid = _sha(old["output_path"]) == old["output_sha"]
                    except OSError:
                        valid = False
                    if not valid:
                        raise ValueError("previous successful output no longer verifies; manual audit required")
                return dict(old, new_reservation=False)
            if attempt:
                prior = latest.get((task_id, attempt - 1))
                if (not prior or prior["state"] != "failed"
                        or self._fingerprint(prior) != self._fingerprint(row)
                        or prior["wt_variant"] != wt_variant):
                    raise ValueError("retry requires matching failed actual call")
            for pending in latest.values():
                if (pending["state"] in ("reserved", "inflight")
                        and self._fingerprint(pending) == self._fingerprint(row)):
                    return dict(pending, new_reservation=False, requested_task_id=task_id)
            for cached in latest.values():
                if cached["state"] not in ("succeeded", "cache_hit") or self._fingerprint(cached) != self._fingerprint(row):
                    continue
                try:
                    valid = _sha(cached["output_path"]) == cached["output_sha"]
                except (OSError, TypeError):
                    valid = False
                if valid:
                    row.update(state="cache_hit", reserved_call=0, cache_hit=True,
                               output_sha=cached["output_sha"], output_path=cached["output_path"])
                    return dict(self._append(row), new_reservation=False)
            if attempt == 0 and any(
                    previous["state"] == "failed"
                    and self._fingerprint(previous) == self._fingerprint(row)
                    for previous in latest.values()):
                raise ValueError("failed cache identity requires the original task's authorized retry")
            if sum(r["reserved_call"] for r in latest.values()) >= self.limit:
                raise BudgetExceeded("global structure cap reached")
            if sum(r["reserved_call"] for r in latest.values() if r["category"] == row["category"]) >= self.caps[row["category"]]:
                raise BudgetExceeded("category structure cap reached")
            return dict(self._append(row), new_reservation=True)

    def start(self, task_id, attempt=0):
        with _lock(str(self.path) + ".lock"):
            row = self._read()[(task_id, attempt)]
            if row["state"] != "reserved":
                raise ValueError("only reserved calls may start; duplicate dispatch rejected")
            return self._append(dict(row, state="inflight"))

    def finish(self, task_id, *, succeeded, seconds, max_memory, output_path=None, attempt=0):
        if type(succeeded) is not bool or any(not math.isfinite(v) or v < 0 for v in (seconds, max_memory)):
            raise ValueError("valid status, elapsed seconds and peak memory required")
        with _lock(str(self.path) + ".lock"):
            row = self._read()[(task_id, attempt)]
            if row["state"] != "inflight":
                raise ValueError("only actual inflight calls may finish")
            row = dict(row, state="succeeded" if succeeded else "failed",
                       seconds=seconds, max_memory=max_memory)
            if succeeded:
                row.update(output_path=str(Path(output_path).resolve()), output_sha=_sha(output_path))
            return self._append(row)
