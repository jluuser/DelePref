"""Regression tests against independently frozen manuscript readouts."""
import json
from pathlib import Path
import zipfile

import pytest

from delepref.paper_evidence import validate
from delepref.downloads import catalog, download
from delepref.reproduction import reproduce


EVIDENCE = Path(__file__).resolve().parents[1] / 'benchmarks/assets/paper_evidence_v2.zip'


def test_five_draw_bootstrap_and_exact_identity_reproduce_paper_results():
    result = validate(EVIDENCE)
    assert result['completion_bootstrap_seed'] == 20261025
    assert len(result['paired_records']) == 500
    by_endpoint = {r['endpoint']: r['lower_effect'] for r in result['completion']}
    local = by_endpoint['structure_mapped_CA_lddt']
    contacts = by_endpoint['structure_mapped_native_contact_recovery']
    assert local['N'] == contacts['N'] == 20
    assert local['mean_difference'] == pytest.approx(0.017485651455214503)
    assert contacts['mean_difference'] == pytest.approx(0.02158646254886865)
    assert local['joint_CI97_5'] == pytest.approx([0.0022995829669376754, 0.0370013458453249])
    assert contacts['joint_CI97_5'] == pytest.approx([0.0018823529411764704, 0.04939563530660445])
    assert local['unchanged'] == contacts['unchanged'] == 15
    strict = {(r['cohort'], r['endpoint']): r['strict_canonical'] for r in result['raygun_strict']}
    assert strict['Enzyme80', 'identity_aware_catalytic_neighbor_contact_recovery'] == pytest.approx(0.01836976508741419)
    assert strict['Binder60', 'interface_contact_recovery'] == pytest.approx(0.046264170626951394)
    assert strict['Binder60', 'interface_F1'] == pytest.approx(0.0477948618430446)


def test_recomputed_results_detect_modified_per_target_measurement(tmp_path):
    changed = tmp_path / 'changed.zip'
    with zipfile.ZipFile(EVIDENCE) as original, zipfile.ZipFile(changed, 'w') as output:
        for item in original.infolist():
            data = original.read(item.filename)
            if item.filename == 'completion/paired_per_WT_and_repeat.json':
                rows = json.loads(data)
                rows[0]['difference_lower'] += 0.1
                data = json.dumps(rows).encode('utf8')
            output.writestr(item.filename, data)
    with pytest.raises(ValueError, match='disagrees with frozen record'):
        validate(changed)


def test_paper_reproduction_distinguishes_current_and_historical_readouts(tmp_path):
    import csv
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / 'data'
    for name, entry in catalog()['assets'].items():
        if entry['status'] == 'included':
            download(name, data, source_dir=root)
    out = tmp_path / 'out'
    reproduce(data, out)
    def csv_rows(name):
        with (out / name).open(encoding='utf8', newline='') as stream:
            return list(csv.DictReader(stream))
    current = {r['endpoint']: r for r in csv_rows('table_completion_morphology.csv')}
    old = {r['endpoint']: r for r in csv_rows('table_completion_morphology_single_draw.csv')}
    key = 'structure_mapped_CA_lddt'
    assert float(current[key]['mean_difference']) == pytest.approx(0.017485651455214503)
    assert float(old[key]['mean_difference']) == pytest.approx(0.007806374619694034)
    assert len(csv_rows('table_completion_morphology_per_WT_draw.csv')) == 500
    strict = csv_rows('table_raygun_strict_identity.csv')
    assert len(strict) == 4
    assert float(next(r['strict_canonical'] for r in strict if r['endpoint'] == 'interface_F1')) == pytest.approx(0.0477948618430446)
