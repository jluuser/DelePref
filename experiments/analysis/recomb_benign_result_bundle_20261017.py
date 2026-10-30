"""Package only the approved final outputs/references, with every member hashed."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from recomb_scope_screen_20261017 import read,sha,write


def build(root):
    report=root/'reports/recomb_benign_final_report_20261017';d=read(report/'completion.json')
    for p,h in d['files'].items():
        if sha(report/p)!=h:raise ValueError('final report member changed')
    cat=root/'reports/recomb_benign_prediction_catalog_20261017'
    physical=read(cat/'physical_requests.json');reuse=read(cat/'compatible_cache.json')
    if len(physical)!=651:raise ValueError('physical roster differs')
    target=root/'caches/recomb_benign_complete_results_20261017.zip'
    if target.exists():raise FileExistsError('result bundle already exists')
    files={};artifact_index=[]
    folders=['recomb_benign_final_report_20261017','recomb_benign_structure_evaluation_20261017',
        'recomb_benign_prediction_integrity_20261017','recomb_benign_final_quality_20261017',
        'recomb_benign_general_readout_20261017','recomb_benign_primary_readout_20261017',
        'recomb_benign_mechanism_contrasts_20261017','recomb_benign_catalytic_identity_20261017',
        'recomb_benign_final_diagnostics_v2_20261017','recomb_benign_efficiency_20261017',
        'recomb_benign_final_outputs_20261017','recomb_benign_final_subsets_20261017',
        'recomb_benign_prediction_catalog_20261017','recomb_benign_binding_native_20261017',
        'recomb_benign_enzyme_native_v2_20261017','recomb_benign_input_coverage_20261017']
    for folder in folders:
        for p in (root/'reports'/folder).rglob('*'):
            if p.is_file():files[str(p.relative_to(root))]=p
    for p in read(root/'reports/recomb_benign_scope_20261017/general_panel.json'):
        path=Path(p['wt_native_json'])
        if sha(path)!=p['wt_native_json_sha256']:raise ValueError('native reference differs')
        files['native/general/'+path.name]=path
    for path in [root/'reports/recomb_benign_scope_20261017/general_panel.json',
        root/'reports/recomb_benign_application_constructs_20261017/panel.json',
        root/'retrieval/recomb_benign_general_20261017/profiles.json',
        root/'retrieval/recomb_benign_app_profiles_20261017/profiles.json',
        root/'configs/recomb_benign_confirmation_freeze_20261017.json',
        root/'configs/recomb_benign_structure_dispatch_20261017.json',
        root/'configs/delepref_original_restored.yaml',root/'provenance/original_delepref_assets_20261016.json',
        root/'provenance/usalign_20261017/USalign']:
        files[str(path.relative_to(root))]=path
    for pattern in ['recomb_benign_*.py','recomb_edit_correspondence_20261017.py','recomb_scope_screen_20261017.py',
        'recomb_general100_common_20261017.py','recomb_general100_baselines_20261017.py','recomb_general100_mechanisms_20261017.py',
        'delepref_v8_runtime.py','delepref_v8_main.py','delepref_v8_rag_unlimited.py','delepref_unlimited_*.py','delepref_v8_structure_metrics.py',
        'delepref_v8_scisor_*py']:
        for p in (root/'analysis').glob(pattern):files[str(p.relative_to(root))]=p
    for p in (root/'code').rglob('*.py'):files[str(p.relative_to(root))]=p
    for p in (root/'analysis/tests').glob('test_recomb_*.py'):files[str(p.relative_to(root))]=p
    for p in (root/'slurm').glob('recomb_benign_*.sbatch'):files[str(p.relative_to(root))]=p
    struct=root/'reports/recomb_benign_structures_20261017'
    for path in struct.glob('*.log'):files[str(path.relative_to(root))]=path
    files[str((struct/'prediction_journal.jsonl').relative_to(root))]=struct/'prediction_journal.jsonl'
    for p in physical:
        task=p['task_id']
        if task in reuse:
            item=reuse[task];artifact=Path(item['artifact_path']);expected=item['artifact_sha256'];record=Path(item['record_path']);old=True
        else:
            record=struct/'records'/(task+'.json');item=read(record)
            if item['status']!='SUCCEEDED':raise ValueError('not all artifacts successful')
            artifact=Path(item['output_path']);expected=item['output_sha256'];old=False
            raw=Path(item['raw_structure_path']) if item.get('raw_structure_path') else None
            if raw is not None:
                if sha(raw)!=item['raw_structure_sha256']:raise ValueError('raw CIF differs')
                files['artifacts/'+task+'.cif']=raw
        if sha(artifact)!=expected:raise ValueError('structure artifact differs')
        relative='artifacts/'+task+'.npz';files[relative]=artifact
        files['records/'+task+'.json']=record
        artifact_index.append(dict(physical_id=task,predictor=p['predictor'],profile_name=p['profile_name'],
            artifact_path=relative,artifact_sha256=expected,record_path='records/'+task+'.json',reused_cache=old))
    note=('RECOMB fixed benign-scope result bundle\n\n'
        'Start with reports/recomb_benign_final_report_20261017/README_zh.md.\n'
        '867 logical outputs,651 unique prediction identities,419 new successful calls,232 verified old caches.\n'
        'All included prediction artifacts and native references belong to the approved44 general/23 application constructs.\n'
        'bundle_manifest.json hashes every archived file; artifact_index.json maps canonical prediction identities to portable artifact paths.\n'
        'Reports preserve all benefits, failures of scientific hypotheses, uncertainty and predictor calibration limits.\n'
        'No model weights/training corpus are copied. Original asset locations/checksums and source/config are recorded for the existing cu9 environment.\n'
        'The USalign executable is the fixed Linux build; this archive is data/source evidence, not a Windows one-click model installation.\n'
        'Old metadata-helper failures and source versions are retained;419 actual new structural predictions had0 failures and0 retries.\n'
        'Paper unchanged; no publication or code release was performed.\n')
    contents={'BUNDLE_README.txt':note.encode(),'artifact_index.json':(json.dumps(artifact_index,indent=2,sort_keys=True)+'\n').encode()}
    manifest={name:sha(path) for name,path in files.items()}
    manifest.update({name:hashlib.sha256(data).hexdigest() for name,data in contents.items()})
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,path in sorted(files.items()):z.write(path,name)
        for name,data in contents.items():z.writestr(name,data)
        z.writestr('bundle_manifest.json',json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    summary=dict(status='COMPLETE_RESULT_BUNDLE_READY',archive_path=str(target),archive_sha256=sha(target),archive_bytes=target.stat().st_size,
        hashed_members=len(manifest),physical_artifacts=651,new_calls=419,verified_cache=232,models_or_training_corpus_included=False,
        source_script_sha256=sha(Path(__file__)),paper_modified=False)
    write(root/'reports/recomb_benign_final_bundle_20261017.json',summary);print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
    build(p.parse_args().root)
