"""Verify a DelePref primary-structure archive without extracting its files."""
import argparse
import hashlib
import json
from pathlib import PurePosixPath
import tarfile


def verify(path):
    seen = {}
    manifest = None
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or '..' in name.parts:
                raise ValueError('Unsafe archive member: ' + member.name)
            if member.name in seen:
                raise ValueError('Duplicate archive member: ' + member.name)
            digest = hashlib.sha256()
            payload = [] if member.name == 'manifest.json' else None
            with archive.extractfile(member) as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
                    if payload is not None:
                        payload.append(block)
            seen[member.name] = dict(sha256=digest.hexdigest(), bytes=member.size)
            if payload is not None:
                manifest = json.loads(b''.join(payload))
    if manifest is None or manifest.get('schema') != 'delepref-primary-structures-v1':
        raise ValueError('Missing or unsupported structure manifest')
    required = {'manifest.json'}
    for prediction in manifest['predictions']:
        name = prediction['record']; required.add(name)
        if seen.get(name, {}).get('sha256') != prediction['normalized_record_sha256']:
            raise ValueError('Prediction record checksum mismatch: ' + name)
        for item in prediction['files']:
            name = item['path']; required.add(name)
            if seen.get(name) != {'sha256': item['sha256'], 'bytes': item['bytes']}:
                raise ValueError('Prediction artifact checksum/size mismatch: ' + name)
    for reference in manifest['native_references']:
        name = reference['path']; required.add(name)
        if seen.get(name, {}).get('sha256') != reference['normalized_sha256']:
            raise ValueError('Native reference checksum mismatch: ' + name)
    if set(seen) != required:
        raise ValueError('Archive has missing or undocumented members')
    if len(manifest['predictions']) != manifest['physical_predictions']:
        raise ValueError('Prediction inventory length mismatch')
    result = dict(status='VERIFIED', logical_rows=manifest['logical_rows'],
                  physical_predictions=manifest['physical_predictions'],
                  native_references=len(manifest['native_references']),
                  prediction_statuses=manifest['prediction_statuses'], files=len(seen))
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive')
    verify(parser.parse_args().archive)
