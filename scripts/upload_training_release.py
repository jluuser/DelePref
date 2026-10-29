"""Upload checksum-verified training files in resumable, bounded batches.

The input directory follows `build_training_release.py`. When files are still
being copied, --watch uploads only files whose size and SHA256 already match
the frozen manifest. Authentication is obtained from the Hugging Face CLI or
environment; this script never asks for or prints a token.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from huggingface_hub import HfApi


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha(path):
    content = path.read_bytes()
    return hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()


def check_remote(api, repo, root, entries):
    wanted = ['training/' + r['path'] for r in entries]
    remote = {r.path: r for r in api.get_paths_info(repo, wanted, repo_type='dataset', expand=True)}
    for entry, name in zip(entries, wanted):
        item = remote.get(name)
        if item is None or item.size != entry['bytes']:
            raise ValueError('Remote file missing or size differs: ' + name)
        lfs = getattr(item, 'lfs', None)
        if lfs is not None and getattr(lfs, 'sha256', None):
            if lfs.sha256 != entry['sha256']:
                raise ValueError('Remote SHA256 differs: ' + name)
        elif item.blob_id != git_blob_sha(root / entry['path']):
            raise ValueError('Remote Git blob hash differs: ' + name)


def upload(root, repo, watch=False, batch_size=20):
    root = Path(root).resolve()
    manifest_path = root / 'training_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    entries = manifest['files'] + manifest['metadata']
    if manifest.get('schema') != 'delepref-training-release-v1' or len(manifest['files']) != 600:
        raise ValueError('Unexpected source training manifest')
    api = HfApi()
    present = set(api.list_repo_files(repo, repo_type='dataset'))
    done = [r for r in entries if 'training/' + r['path'] in present]
    for start in range(0, len(done), 100):
        check_remote(api, repo, root, done[start:start+100])
    pending = [r for r in entries if 'training/' + r['path'] not in present]
    print('Verified {} of {} existing remote files'.format(len(done), len(entries)), flush=True)
    while pending:
        ready = []
        for item in pending:
            path = (root / item['path']).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Unsafe source path')
            if not path.is_file() or path.stat().st_size != item['bytes']:
                continue
            if sha256(path) != item['sha256']:
                # An in-flight copy can temporarily reach its final size before
                # all bytes are visible. Recheck it on the next pass.
                continue
            ready.append(item)
            if len(ready) >= batch_size:
                break
        if not ready:
            if not watch:
                raise FileNotFoundError('{} files are missing or incomplete'.format(len(pending)))
            time.sleep(10)
            continue
        api.upload_folder(repo_id=repo, repo_type='dataset', folder_path=root,
                          path_in_repo='training', allow_patterns=[r['path'] for r in ready],
                          commit_message='Publish verified DelePref training files')
        check_remote(api, repo, root, ready)
        selected = {r['path'] for r in ready}
        pending = [r for r in pending if r['path'] not in selected]
        print('Uploaded and verified {}/{} files'.format(len(entries)-len(pending), len(entries)), flush=True)
    api.upload_file(path_or_fileobj=manifest_path, path_in_repo='training/training_manifest.json',
                    repo_id=repo, repo_type='dataset', commit_message='Publish training dataset manifest')
    print('Training dataset upload complete: {} files and manifest'.format(len(entries)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--repo', required=True, help='Hugging Face dataset repo, owner/name')
    p.add_argument('--watch', action='store_true')
    p.add_argument('--batch-size', type=int, default=20)
    args = p.parse_args()
    if args.batch_size < 1:
        raise ValueError('Batch size must be positive')
    upload(args.source, args.repo, watch=args.watch, batch_size=args.batch_size)
