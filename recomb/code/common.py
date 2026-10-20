"""Content identities and atomic output helpers."""
import hashlib
import json
import os
from pathlib import Path

AUTHORIZED_ROOT = Path("/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013")


def confined_path(path):
    path = Path(path)
    resolved = path.resolve()
    if resolved != AUTHORIZED_ROOT.resolve() and AUTHORIZED_ROOT.resolve() not in resolved.parents:
        raise ValueError("output escapes the authorized new experiment root")
    return path


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def identity(value):
    return sha_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def write_json(path, value):
    path = confined_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = confined_path(path.with_name(path.name + f".{os.getpid()}.tmp"))
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def write_jsonl(path, rows):
    path = confined_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = confined_path(path.with_name(path.name + f".{os.getpid()}.tmp"))
    with temp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]
