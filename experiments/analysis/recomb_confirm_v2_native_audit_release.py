"""Exclude all functional holds and development relatives before native auditing."""
import argparse
from collections import Counter
from pathlib import Path
from recomb_confirm_v2_diagnostics import read,write,sha


def main(root):
    out=root/'reports/recomb_confirm_v2_native_audit_release'
    if (out/'completion.json').exists():raise FileExistsError('Native audit eligibility already released')
    scopefile=root/'reports/recomb_confirm_v2_design_function_review/per_candidate_function_review_v2.json'
    scope={r['candidate_id']:r for r in read(scopefile)}
    homfile=root/'reports/recomb_confirm_v2_candidate_homology/per_candidate_audit.json'
    homology={r['candidate_id']:r for r in read(homfile)}
    source=root/'reports/recomb_confirm_v2_WT_candidate_audit_inputs/WT_candidates.json'
    allowed=[];held=[]
    for row in read(source):
        cid=row['candidate_id']
        if cid not in scope or cid not in homology:raise ValueError('Candidate lacks explicit current review')
        decision=scope[cid];h=homology[cid]
        if decision['status']!='FUNCTIONAL_SCOPE_PASS_NATIVE_AND_FAMILY_PENDING':
            held.append(dict(candidate_id=cid,reason=decision['status']));continue
        if h['development_related_hits']>0:
            held.append(dict(candidate_id=cid,reason='development_related_input_component'));continue
        if h['training_exact_sequence_ids']:
            held.append(dict(candidate_id=cid,reason='exact_student_training_WT'));continue
        allowed.append(dict(row,native_audit_allowed=True,model_use_allowed=False,
            deletion_generation_allowed=False,PLM_encoding_allowed=False,structure_prediction_allowed=False,
            evaluation_sequence_component=h['evaluation_sequence_component'],
            training_domain_like_exposure=h['training_high_query_low_target_domain_like_hits']>0,
            native_and_final_cohort_review_pending=True))
    write(out/'eligible_WT_native_audit_inputs.json',allowed)
    write(out/'not_released.json',held)
    report=dict(status='NATIVE_INPUT_AUDIT_ONLY_RELEASED_NOT_MODELS',candidates=len(allowed),
        roles=dict(Counter(r['role'] for r in allowed)),held=len(held),
        source_sha256=sha(source),functional_review_sha256=sha(scopefile),homology_review_sha256=sha(homfile),
        no_candidate_model_use_allowed=True,new_model_calls=0,
        files={p.name:sha(p) for p in out.glob('*.json')})
    write(out/'completion.json',report);print(report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
