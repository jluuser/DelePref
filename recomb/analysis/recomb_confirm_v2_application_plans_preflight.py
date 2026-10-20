"""Verify the application pool audit against existing real General100 outputs."""
from pathlib import Path
import os
from recomb_confirm_v2_diagnostics import read,sha,write
from recomb_confirm_v2_application_plans import audit

root=Path(__file__).resolve().parents[1]
if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
out=root/'reports/recomb_confirm_v2_application_heads_preflight'
if (out/'completion.json').exists():raise FileExistsError('Application audit preflight already complete')
base=root/'reports/recomb_confirm_v2_General100_plans';done=read(base/'completion.json')
for p,h in done['outputs'].items():
    if sha(base/p)!=h:raise ValueError('Real General100 output identity changed')
scores=read(base/'candidate_scores.json');selected=read(base/'all_selections.json');diags=read(base/'generation_diagnostics.json')
profiles=read(root/'retrieval/recomb_confirm_v2_General100/profiles.json')
result=audit(scores,selected,diags,profiles,320)
bad=[dict(selected[0],edited_sequence=selected[0]['edited_sequence']+'A')]+selected[1:]
try:audit(scores,bad,diags,profiles,320)
except ValueError as e:
    if 'reconstruction' not in str(e):raise
else:raise ValueError('Corrupted edited sequence was not rejected')
write(out/'completion.json',dict(status='REAL_GENERAL100_REGRESSION_AND_CORRUPTED_EDIT_REJECTION_PASS',
    audit=result,new_model_calls=0,new_structure_predictions=0,job_id=os.environ['SLURM_JOB_ID'],
    source_sha256=sha(root/'analysis/recomb_confirm_v2_application_plans.py'),
    input_completion_sha256=sha(base/'completion.json')))
print(result,flush=True)
