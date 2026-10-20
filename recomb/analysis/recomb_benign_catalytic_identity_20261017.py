"""Verify catalytic sequence identity separately from mapped position/CA geometry."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from recomb_scope_screen_20261017 import read,sha,write


def retained_identity(sites,sequence,correspondence):
    verified=[s for s in sites if s['status']=='PASS']
    deleted=[];substituted=[];unknown=[];retained=[]
    for site in verified:
        i=int(site['wt_index']);state=correspondence['position_states'][i]
        if state=='deleted':deleted.append(i)
        elif state!='stable':unknown.append(i)
        else:
            j=correspondence['wt_to_edit'][i]
            if j is None or not 0<=j<len(sequence):raise ValueError('invalid catalytic fixed correspondence')
            if sequence[j]!=site['wt_residue']:substituted.append(i)
            else:retained.append(i)
    return dict(verified_catalytic_sites=len(verified),annotation_sites=len(sites),
        retained_identical_sites=retained,certain_deleted_sites=deleted,certain_substituted_sites=substituted,unresolved_mapping_sites=unknown,
        all_verified_identities_retained_lower=(not deleted and not substituted and not unknown) if verified else None,
        all_verified_identities_retained_upper=(not deleted and not substituted) if verified else None,
        site_identity_retention_lower=len(retained)/len(verified) if verified else None,
        site_identity_retention_upper=(len(retained)+len(unknown))/len(verified) if verified else None,
        sequence_identity_only_not_geometry_or_activity=True)


def build(root):
    cat=root/'reports/recomb_benign_prediction_catalog_20261017';out=root/'reports/recomb_benign_catalytic_identity_20261017'
    if out.exists():raise FileExistsError('catalytic identity audit exists')
    panel={p['protein_id']:p for p in read(root/'reports/recomb_benign_application_constructs_20261017/panel.json') if p['dataset']=='Enzyme20'}
    native={p['protein_id']:p for p in read(root/'reports/recomb_benign_enzyme_native_v2_20261017/native_references.json')}
    if len(panel)!=11 or set(panel)!=set(native):raise ValueError('reviewed enzyme scope differs')
    physical={p['task_id']:p for p in read(cat/'physical_requests.json')}
    records=[]
    for r in read(cat/'logical_requests.json'):
        if r['cohort']!='enzyme':continue
        pid=r['protein_id'];parent=panel[pid];ref=native[pid]
        if r['WT_sha256']!=parent['sequence_sha256'] or ref['WT_sha256']!=parent['sequence_sha256']:raise ValueError('canonical enzyme WT identity differs')
        for site in ref['catalytic']:
            if site['status']=='PASS' and parent['sequence'][int(site['wt_index'])]!=site['wt_residue']:raise ValueError('catalytic annotation identity differs')
        sequence=physical[r['physical_id']]['chains'][0]['sequence']
        records.append(dict(protein_id=pid,method=r['method'],physical_id=r['physical_id'],
            **retained_identity(ref['catalytic'],sequence,r['correspondence'])))
    if len(records)!=55:raise ValueError('full11enzyme four-method-plusWT roster differs')
    groups=defaultdict(list)
    for r in records:groups[r['method']].append(r)
    summaries=[]
    for method,rows in sorted(groups.items()):
        if len(rows)!=11:raise ValueError('unequal enzyme method cohort')
        summaries.append(dict(method=method,N=11,verified_annotation_sites=sum(r['verified_catalytic_sites'] for r in rows),
            all_sites_identity_retained_case_fraction_lower=sum(r['all_verified_identities_retained_lower'] for r in rows)/11,
            all_sites_identity_retained_case_fraction_upper=sum(r['all_verified_identities_retained_upper'] for r in rows)/11,
            macro_site_identity_retention_lower=sum(r['site_identity_retention_lower'] for r in rows)/11,
            macro_site_identity_retention_upper=sum(r['site_identity_retention_upper'] for r in rows)/11,
            cases_with_known_substitution=sum(bool(r['certain_substituted_sites']) for r in rows),
            cases_with_known_deletion=sum(bool(r['certain_deleted_sites']) for r in rows)))
    out.mkdir();write(out/'per_output_identity.json',records);write(out/'summary.json',summaries)
    sources=['reports/recomb_benign_prediction_catalog_20261017/logical_requests.json',
        'reports/recomb_benign_prediction_catalog_20261017/physical_requests.json',
        'reports/recomb_benign_application_constructs_20261017/panel.json',
        'reports/recomb_benign_enzyme_native_v2_20261017/native_references.json']
    write(out/'completion.json',dict(status='ALL_REVIEWED_ENZYME_CATALYTIC_IDENTITIES_CHECKED',WT=11,logical_outputs=55,
        clarification='frozen geometric all_catalytic_sites_retained is mapped-position/coordinate retention; sequence identity is separately verified here',
        frozen_geometry_implementation_and_predictions_unchanged=True,new_GPU_calls=0,new_structure_predictions=0,
        input_sha256={p:sha(root/p) for p in sources},source_script_sha256=sha(Path(__file__)),
        outputs={n:sha(out/n) for n in ['per_output_identity.json','summary.json']}))
    print(json.dumps(summaries))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
    build(p.parse_args().root)
