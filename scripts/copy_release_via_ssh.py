"""Copy frozen training files over one resumable SFTP connection per batch."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import time


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def paths(item, remote_root, local_root):
    relative = PurePosixPath(item['path'])
    if relative.is_absolute() or '..' in relative.parts or '"' in str(relative):
        raise ValueError('Unsafe manifest path: ' + item['path'])
    target = (local_root / Path(*relative.parts)).resolve()
    if not target.is_relative_to(local_root):
        raise ValueError('Output escapes selected local directory')
    partial = target.with_name(target.name + '.transfer-part')
    if not partial.resolve().is_relative_to(local_root):
        raise ValueError('Temporary output escapes selected local directory')
    remote = str(PurePosixPath(remote_root) / relative)
    return target, partial, remote


def valid(path, item):
    return path.is_file() and path.stat().st_size == item['bytes'] and sha(path) == item['sha256']


def copy_release(local_root, host, remote_root, batch_size):
    local_root = Path(local_root).resolve()
    manifest = json.loads((local_root / 'training_manifest.json').read_text(encoding='utf8'))
    entries = manifest['files'] + manifest['metadata']
    if manifest.get('schema') != 'delepref-training-release-v1' or len(manifest['files']) != 600:
        raise ValueError('Unexpected training manifest')
    if '"' in host or '"' in remote_root:
        raise ValueError('Unsafe remote argument')
    pending = []
    for item in entries:
        target, partial, remote = paths(item, remote_root, local_root)
        if not valid(target, item):
            target.parent.mkdir(parents=True, exist_ok=True)
            pending.append(item)
    print('Already verified {}/{} files'.format(len(entries)-len(pending), len(entries)), flush=True)
    delay = 5
    consecutive_failures = 0
    while pending:
        batch = pending[:batch_size]
        commands = []
        for item in batch:
            target, partial, remote = paths(item, remote_root, local_root)
            if partial.exists() and partial.stat().st_size > item['bytes']:
                # Only a single task-owned temporary file is removed; its
                # resolved parent was checked to remain under local_root.
                partial.unlink()
            commands.append('get -a "{}" "{}"'.format(remote, partial.as_posix()))
        process = subprocess.run(['sftp', '-b', '-', '-B', '262144', '-R', '64',
                                  '-o', 'ClearAllForwardings=yes', '-o', 'BatchMode=yes',
                                  '-o', 'ConnectTimeout=15', host],
                                 input='\n'.join(commands) + '\n', text=True,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        completed = set()
        for item in batch:
            target, partial, _ = paths(item, remote_root, local_root)
            if valid(partial, item):
                partial.replace(target)
                completed.add(item['path'])
            elif partial.exists() and partial.stat().st_size == item['bytes']:
                # A complete-size file with the wrong digest must be fetched anew.
                partial.unlink()
        pending = [item for item in pending if item['path'] not in completed]
        print('Verified {}/{} files'.format(len(entries)-len(pending), len(entries)), flush=True)
        if process.returncode and not completed:
            consecutive_failures += 1
            if consecutive_failures >= 8:
                raise RuntimeError('SFTP remains unavailable; rerun the command to resume')
            print('SFTP unavailable; retrying after {} seconds'.format(delay), flush=True)
            time.sleep(delay)
            delay = min(60, delay * 2)
        else:
            delay = 5
            consecutive_failures = 0
    print('Training transfer complete; all {} checksums verified'.format(len(entries)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--local-root', required=True)
    p.add_argument('--host', required=True)
    p.add_argument('--remote-root', required=True)
    p.add_argument('--batch-size', type=int, default=20)
    args = p.parse_args()
    if not 1 <= args.batch_size <= 100:
        raise ValueError('batch size must be between 1 and 100')
    copy_release(args.local_root, args.host, args.remote_root, args.batch_size)
