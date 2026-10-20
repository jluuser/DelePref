"""Apply the unchanged WT-calibration source/identity filter to new staged WT MSAs."""
from pathlib import Path
import recomb_confirm_v2_audit_msa as parent
from recomb_confirm_v2_diagnostics import read, sha, write


def main():
    root=Path(__file__).resolve().parents[1]
    base=root/'external/recomb_confirm_v2_application_WT_MSA_staging'
    out=root/'external/recomb_confirm_v2_application_WT_MSA_reviewed'
    if (out/'execution_claim.json').exists():
        raise FileExistsError('Application WT MSA review claimed; inspect progress/cache before resuming')
    staged=read(base/'completion.json')
    if staged['unique_queries']!=180 or staged['prediction_release_allowed']:
        raise ValueError('WT-only MSA stage incomplete or unexpected release state')
    old=root/'external/recomb_confirm_v2_WT_MSA_reviewed/review_policy.json'
    policy=read(old)
    if policy['source_sha256']!=sha(root/'analysis/recomb_confirm_v2_audit_msa.py'):
        raise ValueError('Existing calibration MSA filter implementation changed')
    if policy['prefix_rows']!=parent.PREFIX:
        raise ValueError('MSA depth policy differs from WT calibration')
    write(out/'execution_claim.json',dict(status='WT_SOURCE_AND_DONOR_IDENTITY_REVIEW_STARTED',
        staged_manifest_sha256=sha(base/'request_manifest.json'),staged_completion_sha256=sha(base/'completion.json'),
        unchanged_parent_implementation_sha256=sha(root/'analysis/recomb_confirm_v2_audit_msa.py'),
        wrapper_sha256=sha(Path(__file__)),parent_policy_sha256=sha(old),
        no_model_calls=True,no_new_sequences_submitted_to_MSA_provider=True,
        old_metadata_not_used_as_new_fragment_identity_proof=True))
    parent.BASE=base;parent.OUT=out
    parent.main()
    result=read(out/'completion.json')
    write(out/'application_review_completion.json',dict(status=result['status'],
        parent_completion_sha256=sha(out/'completion.json'),
        row_audit_sha256=sha(out/'row_source_and_identity_audit.json'),
        frozen_policy_sha256=sha(out/'review_policy.json'),unique_queries=180,
        rows_retained=result['rows_retained'],query_only=result['query_only'],
        source_disposition=result['source_disposition'],new_model_calls=0,new_structure_predictions=0,
        variant_donor_realignments_require_separate_uniform_protocol=True))
    print({k:v for k,v in result.items() if k!='files'},flush=True)


if __name__=='__main__':main()
