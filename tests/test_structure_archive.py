"""Check structural archive integrity without extracting untrusted paths."""
import hashlib
import io
import json
import tarfile

import pytest
from scripts.verify_structure_archive import verify


def make_archive(path, *, corrupt=False, unsafe=False):
    data = {
        'predictions/general/example.npz': b'predicted-coordinates',
        'records/general/example.json': b'{"task_id":"example"}\n',
        'references/general/native.json': b'{"coordinates":{}}\n',
    }
    sha = lambda b: hashlib.sha256(b).hexdigest()
    artifact = 'predictions/general/example.npz'
    record = 'records/general/example.json'
    native = 'references/general/native.json'
    manifest = dict(schema='delepref-primary-structures-v1', logical_rows=1,
        physical_predictions=1, prediction_statuses={'SUCCEEDED':1},
        predictions=[dict(record=record, normalized_record_sha256=sha(data[record]),
                          files=[dict(path=artifact, bytes=len(data[artifact]), sha256=sha(data[artifact]))])],
        native_references=[dict(path=native, normalized_sha256=sha(data[native]))])
    data['manifest.json'] = json.dumps(manifest).encode()
    if corrupt:
        data[artifact] = b'wrong-coordinates'
    if unsafe:
        data['../outside'] = b'unsafe'
    with tarfile.open(path, 'w:gz') as archive:
        for name, payload in data.items():
            info = tarfile.TarInfo(name); info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_structure_archive_checks_manifest_and_contents(tmp_path):
    good = tmp_path / 'good.tar.gz'; make_archive(good)
    assert verify(good)['physical_predictions'] == 1
    bad = tmp_path / 'bad.tar.gz'; make_archive(bad, corrupt=True)
    with pytest.raises(ValueError, match='checksum'):
        verify(bad)


def test_structure_archive_rejects_traversal(tmp_path):
    bad = tmp_path / 'traversal.tar.gz'; make_archive(bad, unsafe=True)
    with pytest.raises(ValueError, match='Unsafe'):
        verify(bad)


def test_cas9_structure_archive_verifies_all_manifest_members(tmp_path):
    files = {'records/SpCas9/x.json':b'{}', 'predictions/SpCas9/x.cif':b'coordinates',
             'references/SpCas9/native.json':b'{}', 'analysis/summary.csv':b'case\nSpCas9\n'}
    digest = lambda b: hashlib.sha256(b).hexdigest()
    record, prediction = 'records/SpCas9/x.json', 'predictions/SpCas9/x.cif'
    reference, analysis = 'references/SpCas9/native.json', 'analysis/summary.csv'
    manifest = dict(schema='delepref-cas9-structures-v1',cases=['SpCas9'],
        prediction_records=[dict(record=record,normalized_record_sha256=digest(files[record]),
            structure=dict(path=prediction,sha256=digest(files[prediction]),bytes=len(files[prediction])))],
        native_references=[dict(path=reference,sha256=digest(files[reference]))],
        analysis_files=[dict(path=analysis,sha256=digest(files[analysis]),bytes=len(files[analysis]))])
    files['manifest.json'] = json.dumps(manifest).encode()
    target = tmp_path/'cas9.tar.gz'
    with tarfile.open(target,'w:gz') as archive:
        for name,payload in files.items():
            info=tarfile.TarInfo(name);info.size=len(payload);archive.addfile(info,io.BytesIO(payload))
    assert verify(target)['cases'] == 1
