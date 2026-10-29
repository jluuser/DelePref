"""Copy frozen compressed training files over SSH with resumable checksums."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def copy_item(item, host, remote_root, local_root):
    relative = PurePosixPath(item['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe manifest path: ' + item['path'])
    target = (local_root / Path(*relative.parts)).resolve()
    if not target.is_relative_to(local_root):
        raise ValueError('Output escapes selected local directory')
    if target.is_file() and target.stat().st_size == item['bytes'] and sha(target) == item['sha256']:
        return 'already verified'
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.transfer-part')
    if not partial.resolve().is_relative_to(local_root):
        raise ValueError('Temporary output escapes selected local directory')
    remote = host + ':' + str(PurePosixPath(remote_root) / relative)
    for attempt in range(3):
        result = subprocess.run(['scp', '-q', '-o', 'ClearAllForwardings=yes', '-o',
                                 'BatchMode=yes', remote, str(partial)], capture_output=True)
        if result.returncode == 0 and partial.is_file() and partial.stat().st_size == item['bytes'] and sha(partial) == item['sha256']:
            partial.replace(target)
            return 'copied'
        time.sleep(2 ** attempt)
    raise RuntimeError('Could not transfer and verify ' + item['path'])


def copy_release(local_root, host, remote_root, workers):
    local_root = Path(local_root).resolve()
    manifest = json.loads((local_root / 'training_manifest.json').read_text(encoding='utf8'))
    entries = manifest['files'] + manifest['metadata']
    if manifest.get('schema') != 'delepref-training-release-v1' or len(manifest['files']) != 600:
        raise ValueError('Unexpected training manifest')
    counts = {'copied': 0, 'already verified': 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(copy_item, item, host, remote_root, local_root) for item in entries]
        for index, future in enumerate(as_completed(futures), 1):
            counts[future.result()] += 1
            if index % 25 == 0 or index == len(entries):
                print('Verified {}/{} files: {}'.format(index, len(entries), counts), flush=True)
    return counts


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--local-root', required=True)
    p.add_argument('--host', required=True)
    p.add_argument('--remote-root', required=True)
    p.add_argument('--workers', type=int, default=6)
    args = p.parse_args()
    if not 1 <= args.workers <= 12:
        raise ValueError('workers must be between 1 and 12')
    copy_release(args.local_root, args.host, args.remote_root, args.workers)
