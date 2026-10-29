"""Validate safe handling of incomplete and untrusted training files."""
import hashlib
import json
from pathlib import Path

import pytest
from scripts.copy_release_via_ssh import paths, valid
import scripts.upload_training_release as uploader


def test_copy_paths_reject_traversal_and_check_hash(tmp_path):
    root = tmp_path.resolve()
    item = dict(path='pairs/base/shard.csv.gz', bytes=4,
                sha256=hashlib.sha256(b'data').hexdigest())
    target, partial, remote = paths(item, '/remote/release', root)
    target.parent.mkdir(parents=True)
    target.write_bytes(b'data')
    assert valid(target, item)
    target.write_bytes(b'bad!')
    assert not valid(target, item)
    with pytest.raises(ValueError, match='Unsafe'):
        paths(dict(item, path='../outside'), '/remote/release', root)


def test_uploader_never_uploads_incomplete_or_bad_hash_file(tmp_path, monkeypatch):
    class FakeApi:
        def list_repo_files(self, *args, **kwargs):
            return []
        def upload_folder(self, **kwargs):
            raise AssertionError('Incomplete file must not be uploaded')
    monkeypatch.setattr(uploader, 'HfApi', FakeApi)
    item = dict(path='pairs/base/incomplete.csv.gz', bytes=4,
                sha256=hashlib.sha256(b'good').hexdigest())
    path = tmp_path / item['path']; path.parent.mkdir(parents=True)
    path.write_bytes(b'bad!')
    manifest = dict(schema='delepref-training-release-v1', files=[item] * 600, metadata=[])
    (tmp_path / 'training_manifest.json').write_text(json.dumps(manifest), encoding='utf8')
    with pytest.raises(FileNotFoundError):
        uploader.upload(tmp_path, 'example/dataset', watch=False)
