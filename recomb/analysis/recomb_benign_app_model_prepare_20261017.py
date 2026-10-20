"""Freeze reviewed application constructs and filtered evidence before model use."""
from pathlib import Path
import json
from recomb_scope_screen_20261017 import read,sha,write
from recomb_benign_profiles_20261017 import filtered_profile


def prepare(root):
    panel_path=root/'reports/recomb_benign_application_constructs_20261017/panel.json'
    construct=read(panel_path.parent/'completion.json')
    if sha(panel_path)!=construct['panel_sha256']:raise ValueError('corrected constructs changed')
    panel=read(panel_path)
    approved={r['protein_id'] for r in read(root/'reports/recomb_benign_application_scope_20261017/audit.json') if r['status']=='RETAIN'}
    if {r['protein_id'] for r in panel}!=approved or len(panel)!=23:raise ValueError('reviewed scope mismatch')
    native=root/'reports/recomb_benign_enzyme_native_v2_20261017/completion.json'
    if read(native)['WT']!=11 or read(native)['panel_sha256']!=sha(panel_path):raise ValueError('enzyme native mapping mismatch')
    source=root/'retrieval/recomb_benign_applications_20261017/profiles_unreviewed_evidence.json'
    metadata_root=root/'reports/recomb_benign_app_evidence_20261017'
    inventory=read(metadata_root/'inventory.json')
    if sha(source)!=inventory['parent_profiles_sha256'] or sha(panel_path)!=inventory['reviewed_application_panel_sha256']:raise ValueError('metadata input changed')
    previous=read(source);reviewed=read(metadata_root/'target_metadata_audit.json')
    profiles={r['protein_id']:filtered_profile(previous[r['protein_id']],r,reviewed) for r in panel}
    retrieval=root/'retrieval/recomb_benign_app_profiles_20261017';retrieval.mkdir(exist_ok=False)
    write(retrieval/'profiles.json',profiles)
    report=dict(status='REVIEWED_APPLICATION_PROFILES_FROZEN',WT=23,
        with_evidence=sum(bool(p['representatives']) for p in profiles.values()),
        representative_rows=sum(len(p['representatives']) for p in profiles.values()),
        profiles_sha256=sha(retrieval/'profiles.json'),parent_profiles_sha256=sha(source),
        metadata_sha256=sha(metadata_root/'target_metadata_audit.json'),empty_profiles_preserved=True,
        new_model_calls=0,new_structure_predictions=0)
    write(retrieval/'completion.json',report)
    out=root/'reports/recomb_benign_app_model_20261017';out.mkdir(exist_ok=False);write(out/'panel.json',panel)
    paths=[panel_path,panel_path.parent/'completion.json',native,native.parent/'native_references.json',
        root/'reports/recomb_benign_application_scope_20261017/audit.json',
        root/'configs/recomb_benign_scope_20261017.json',metadata_root/'inventory.json',metadata_root/'target_metadata_audit.json',
        retrieval/'profiles.json',retrieval/'completion.json',Path(__file__),
        root/'analysis/recomb_benign_app_model_20261017.py',root/'analysis/recomb_benign_profiles_20261017.py']
    freeze=root/'configs/recomb_benign_app_inputs_freeze_20261017.json'
    write(freeze,dict(status='REVIEWED_APPLICATION_ENCODING_AND_PLANNING_INPUTS_FROZEN',WT=23,Binder=12,Enzyme=11,
        panel_sha256=sha(out/'panel.json'),files={str(p.relative_to(root)):sha(p) for p in paths},
        corrected_Binder_cases=5,models_unchanged=True,structure_prediction_allowed=False))
    write(out/'preparation_completion.json',dict(status='REVIEWED_APPLICATION_MODEL_INPUTS_READY',
        freeze_sha256=sha(freeze),WT=23,profiles=report,new_model_calls=0))
    print(json.dumps(read(out/'preparation_completion.json')))


if __name__=='__main__':prepare(Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
