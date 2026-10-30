"""Release material must preserve full cohorts, values and safe extraction."""
import csv
import gzip
import json
from pathlib import Path
import zipfile
import pytest
from delepref.downloads import catalog, download, extract_case
from delepref.reproduction import save_csv

REPO=Path(__file__).resolve().parents[1]


def read_gz(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)


def test_offline_assets_cover_full_cohorts_and_all_methods(tmp_path):
    for name,entry in catalog()['assets'].items():
        if entry['status']=='included':download(name,tmp_path,source_dir=REPO)
    rows=read_gz(tmp_path/'benchmark_manifests.json.gz')
    assert {k:len(v) for k,v in rows['cohorts'].items()}=={'General100':100,'Enzyme80':80,'Binder60':60}
    assert len(rows['per_output_measurements'])==3474
    tables=read_gz(tmp_path/'paper_tables.json.gz')
    assert {r['method'] for r in tables['general100']}=={
        'WT','rag_unlimited','delepref_unlimited','gap_frequency','terminal','scisor_L_native','raygun_native'}
    assert len(tables['structural_case_and_mechanism']['changed'])==23
    assert (tmp_path/'sec6_case/data/case_data.json').is_file()


def test_csv_preserves_frozen_numeric_values_and_intervals(tmp_path):
    tables=read_gz(REPO/'benchmarks/assets/paper_tables.json.gz')
    source=tables['general100'];path=tmp_path/'table.csv';save_csv(path,source)
    with path.open(newline='') as f:actual=list(csv.DictReader(f))
    assert len(actual)==len(source)
    for before,after in zip(source,actual):
        for key,value in before.items():
            if isinstance(value,(list,dict)):assert json.loads(after[key])==value
            elif isinstance(value,(int,float)):assert float(after[key])==value


def test_asset_mismatch_does_not_overwrite_existing_file(tmp_path):
    target=tmp_path/'paper_tables.json.gz';target.write_bytes(b'wrong')
    with pytest.raises(ValueError,match='different hash'):download('paper_tables',tmp_path,source_dir=REPO)
    assert target.read_bytes()==b'wrong'


def test_zip_traversal_is_rejected(tmp_path):
    bad=tmp_path/'bad.zip'
    with zipfile.ZipFile(bad,'w') as z:z.writestr('../escaped','invalid')
    with pytest.raises(ValueError,match='Unsafe'):extract_case(bad,tmp_path/'out')
    assert not (tmp_path/'escaped').exists()


def test_training_manifest_is_optional_and_identifies_the_corpus():
    entry=catalog()['assets']['training_preferences']
    assert entry['status']=='external_optional'
    assert entry['filename']=='training_manifest.json'
    assert entry['pairs']==24386017
    assert 'manifest only' in entry['note']
