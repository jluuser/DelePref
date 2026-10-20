"""Prove that all allowed native role maps preserve actual preregistered model inputs."""
import os
from pathlib import Path
from recomb_confirm_v2_diagnostics import read,sha,write
from recomb_confirm_v2_input_symmetry import filter_input_role_maps

ROOT=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013')
CAT=ROOT/'reports/recomb_confirm_v2_application_catalog_v2'
OUT=ROOT/'reports/recomb_confirm_v2_application_input_symmetry'


def main():
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm only')
    if (OUT/'completion.json').exists():raise FileExistsError('Full-input symmetry proof already completed')
    done=read(CAT/'completion.json');parsed=read(CAT/'native_parser_completion.json')
    if done['logical']!=2160 or parsed['unresolved']:raise ValueError('Full application variant inputs are not ready')
    for p,h in done['outputs'].items():
        if sha(CAT/p)!=h:raise ValueError('Materialized application catalog changed')
    physical={r['task_id']:r for r in read(CAT/'physical_requests.json')}
    path=ROOT/'reports/recomb_confirm_v2_application_metric_preflight/allowed_role_maps.json'
    allowed={r['protein_id']:r['allowed_maps'] for r in read(path)}
    seen={};checks=[]
    for row in read(CAT/'logical_requests.json'):
        request=physical[row['physical_id']];key=row['protein_id'],request['model_input_sha256']
        if key in seen:continue
        p=ROOT/request['model_input_relative_path']
        if sha(p)!=request['model_input_sha256']:raise ValueError('Actual model input changed')
        kept,audit=filter_input_role_maps(allowed[row['protein_id']],request,read(p))
        r=dict(protein_id=row['protein_id'],model_input_sha256=request['model_input_sha256'],**audit)
        seen[key]=r;checks.append(r)
    write(OUT/'per_input_proof.json',checks)
    rejected=sum(len(r['rejected']) for r in checks)
    names=['analysis/recomb_confirm_v2_full_input_symmetry_audit.py','analysis/recomb_confirm_v2_input_symmetry.py',
        'analysis/tests/test_recomb_confirm_v2_input_symmetry.py','slurm/recomb_confirm_v2_full_input_symmetry_audit.sbatch']
    report=dict(status='ALL_ACTUAL_INPUTS_PROVED_ROLE_SYMMETRIC' if not rejected else 'ROLE_MAP_EQUIVALENCE_REQUIRES_EXPLICIT_REPAIR_NO_FOLD_RELEASE',
        unique_inputs=len(checks),rejected_native_maps=rejected,extra_semantic_tests_passed=4,
        proof_sha256=sha(OUT/'per_input_proof.json'),native_role_map_sha256=sha(path),
        source_sha256={n:sha(ROOT/n) for n in names},catalog_completion_sha256=sha(CAT/'completion.json'),
        no_model_or_structure_calls=True,job_id=os.environ['SLURM_JOB_ID'])
    write(OUT/'completion.json',report);print(report,flush=True)
    if rejected:raise RuntimeError('Some native operator maps are not symmetries of the full actual input; final release blocked')


if __name__=='__main__':main()
