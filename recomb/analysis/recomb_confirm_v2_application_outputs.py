"""Audit and assemble the six fixed application methods without new generation."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from recomb_confirm_v2_diagnostics import read, sha, write
from recomb_confirm_v2_general_outputs import ray_mapping

BASE = Path('reports/recomb_confirm_v2_application_outputs')
FZ = Path('configs/recomb_confirm_v2_application_output_assembly_freeze.json')
METHODS = ['delepref_unlimited','rag_unlimited','gap_frequency','terminal','scisor_L_native','raygun_native']


def canonical_output(row, method, panel, source, mapping=None):
    pid = row['protein_id']; p = panel[pid]; wt = p['sequence']; beta = float(row.get('budget', 0.))
    if row.get('wildtype_sequence', row.get('sequence', wt)) != wt:
        raise ValueError('Application output WT differs from frozen ordinary roster')
    if mapping is None:
        raw = row.get('deleted_intervals', [])
        spans = json.loads(raw) if isinstance(raw, str) else raw
        deleted = set()
        for a, b in spans:
            if not 0 <= a <= b < len(wt) or deleted.intersection(range(a, b+1)):
                raise ValueError('Invalid explicit deletion intervals')
            deleted.update(range(a, b+1))
        edited = ''.join(aa for i, aa in enumerate(wt) if i not in deleted)
        positions = []; j = 0
        for i in range(len(wt)):
            positions.append(None if i in deleted else j)
            if i not in deleted: j += 1
        mapping = dict(method='explicit_native_deletion_plan' if beta else 'identity_WT',
            wt_to_edit=positions, position_states=['deleted' if i in deleted else 'stable' for i in range(len(wt))],
            stable_positions=len(edited), certain_deleted_positions=len(deleted), ambiguous_positions=0)
        if row.get('edited_sequence', edited) != edited:
            raise ValueError('Application explicit edit reconstruction differs')
    else:
        edited = row['edited_sequence']
    count = 0 if beta == 0 else max(1, min(len(wt)-1, round(len(wt)*beta)))
    digest = hashlib.sha256(edited.encode()).hexdigest()
    if len(edited) != len(wt)-count or row.get('edited_sequence_sha256', digest) != digest:
        raise ValueError('Application output exact budget/SHA differs')
    if hashlib.sha256(wt.encode()).hexdigest() != p['WT_sha256']:
        raise ValueError('Application frozen WT identity differs')
    return dict(protein_id=pid, cohort=p['cohort'], method=method, budget=beta,
        evaluation_role='WT_reference' if beta==0 else 'primary20pct' if beta==.2 else 'budget_curve',
        wildtype_sequence=wt, WT_sha256=p['WT_sha256'], edited_sequence=edited,
        edited_sequence_sha256=digest, correspondence=mapping, source_path=source,
        native_assembly_reference=p['native_assembly_reference'],
        native_assembly_reference_sha256=p['native_assembly_reference_sha256'],
        selection_metadata={k:row.get(k) for k in ['generation_seed','candidate_index','raw_candidate_rank',
            'scorer_score','scorer_rank','retrieval_source','retrieval_seed_spans','generation_kind',
            'raw_candidate_count','PLL_ranked_count','num_insertions','num_deletions','num_substitutions']})


def panel_and_completed(root):
    panel = {}; files = {}
    for cohort, n in [('Binder60',60),('Enzyme80',80)]:
        base = Path('reports/recomb_confirm_v2_'+cohort+'_constructs')
        path = base/(cohort+'_panel.json'); done = read(root/base/'completion.json')
        if sha(root/path) != done['outputs'][cohort+'_panel.json']:
            raise ValueError('Application panel SHA changed')
        rows = read(root/path)
        if len(rows) != n: raise ValueError('Application roster size differs')
        for r in rows:
            if not r['WT_encoding_allowed'] or not r['protein_assembly_complete']:
                raise ValueError('Application input was not released')
            panel[r['protein_id']] = dict(r, cohort=cohort)
            ref = Path(r['native_assembly_reference'])
            if sha(ref) != r['native_assembly_reference_sha256']:
                raise ValueError('Native application assembly changed')
        files[str(path)] = sha(root/path)
        files[str(base/'completion.json')] = sha(root/base/'completion.json')
        subset = 'reports/recomb_confirm_v2_'+cohort+'_subsets/subsets.json'
        files[subset] = sha(root/subset)
    selections = []
    for cohort in ['Binder60','Enzyme80']:
        base = Path('reports/recomb_confirm_v2_'+cohort+'_plans')
        done = read(root/base/'completion.json')
        for p, h in done['outputs'].items():
            if sha(root/base/p) != h: raise ValueError('Completed original/RAG output changed')
        for row in read(root/base/'all_selections.json'):
            if row['generation_seed'] == 13:
                selections.append(canonical_output(row,row['method'],panel,str(base/'all_selections.json')))
        for p in ['completion.json','all_selections.json','semantic_audit.json']:
            files[str(base/p)] = sha(root/base/p)
    if len(selections) != 880: raise ValueError('Four-method application matrix differs')
    base = Path('reports/recomb_confirm_v2_application_baselines/scisor_L')
    done = read(root/base/'completion.json')
    if sha(root/base/'all_conditions.json') != done['outputs_sha256'] or done['conditions'] != 220:
        raise ValueError('Completed native L output differs')
    tasks = read(root/'reports/recomb_confirm_v2_application_baselines/raygun_tasks.json')
    rows = read(root/base/'all_conditions.json')
    if len(rows) != 220: raise ValueError('Native L roster differs')
    for row, task in zip(rows,tasks):
        if any(row[k] != task[k] for k in ['protein_id','WT_sha256','sequence','budget','target_length','cohort']):
            raise ValueError('Native L task/coordinate result differs')
        selections.append(canonical_output(row,'scisor_L_native',panel,str(base/'all_conditions.json')))
    files[str(base/'completion.json')] = sha(root/base/'completion.json')
    files[str(base/'all_conditions.json')] = sha(root/base/'all_conditions.json')
    return panel, selections, files


def preflight(root):
    panel, selections, files = panel_and_completed(root)
    write(root/BASE/'five_method_preflight.json',dict(status='FIVE_COMPLETED_APPLICATION_METHODS_COORDINATE_AND_BUDGET_AUDITED',
        WT=len(panel), selected_outputs=len(selections), method_counts=dict(Counter(r['method'] for r in selections)),
        files=files, new_model_calls=0,new_structure_predictions=0,fold_allowed=False))
    print('FIVE_METHOD_PREFLIGHT_PASS '+str(len(selections)),flush=True)


def prepare(root):
    if (root/FZ).exists(): raise FileExistsError('Application full output assembly already frozen')
    panel, selections, files = panel_and_completed(root)
    audit_path = Path('reports/recomb_confirm_v2_application_Raygun_native_audit/completion.json')
    audit = read(root/audit_path)
    if audit['conditions']!=220 or audit['raw_candidates_audited']!=110000:
        raise ValueError('Native application Raygun audit incomplete')
    names = ['analysis/recomb_confirm_v2_application_outputs.py','analysis/recomb_confirm_v2_general_outputs.py',
        'analysis/recomb_confirm_v2_native_raygun_audit.py','analysis/recomb_edit_correspondence_20261017.py',
        'analysis/recomb_confirm_v2_diagnostics.py','slurm/recomb_confirm_v2_application_outputs.sbatch',str(audit_path)]
    files.update({p:sha(root/p) for p in names})
    write(root/FZ,dict(status='APPLICATION_ALL_SIX_METHOD_OUTPUT_ASSEMBLY_FROZEN_BEFORE_STRUCTURES',
        files=files,Raygun_source_sha256=audit['source_sha256'],methods=METHODS,
        WT=140,core_nonWT=1320,logical_with_WT=1460,no_new_generation_or_selection=True,
        biological_identity_not_inferred_from_structure=True,fold_allowed=False))
    print(sha(root/FZ),flush=True)


def run(root):
    f = read(root/FZ)
    for p,h in {**f['files'],**f['Raygun_source_sha256']}.items():
        path = Path(p) if Path(p).is_absolute() else root/p
        if sha(path)!=h: raise ValueError('Frozen application output input changed: '+p)
    out = root/BASE
    if (out/'execution_claim.json').exists(): raise FileExistsError('Application output assembly already claimed')
    write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID']))
    panel, result, _ = panel_and_completed(root)
    base = Path('reports/recomb_confirm_v2_application_baselines/raygun')
    paths = [str(base/('condition_%03d.json'%i)) for i in range(220)]
    rows = [read(root/p) for p in paths]
    with ProcessPoolExecutor(max_workers=8) as executor:
        for (row,mapping),source in zip(executor.map(ray_mapping,rows),paths):
            result.append(canonical_output(row,'raygun_native',panel,source,mapping))
    if len(result)!=1320: raise ValueError('Six-method application matrix incomplete')
    core = list(result)
    for pid,p in panel.items():
        result.append(canonical_output(dict(protein_id=pid,budget=0.),'WT',panel,'frozen_application_panel'))
    keys=[(r['protein_id'],r['method'],r['budget']) for r in result]
    if len(result)!=1460 or len(keys)!=len(set(keys)): raise ValueError('Duplicated or lost application logical output')
    counts=Counter((r['cohort'],r['method'],r['evaluation_role']) for r in result)
    for cohort,n,extra in [('Binder60',60,20),('Enzyme80',80,60)]:
        for method in METHODS:
            if counts[cohort,method,'primary20pct']!=n or counts[cohort,method,'budget_curve']!=extra:
                raise ValueError('Application method/condition roster differs')
    write(out/'core_selected_outputs.json',core);write(out/'all_logical_outputs.json',result)
    write(out/'completion.json',dict(status='APPLICATION_FULL_SIX_METHOD_OUTPUTS_AUDITED_NOT_STRUCTURE_EVALUATED',
        logical_outputs=len(result),core_nonWT=len(core),WT=140,
        primary_Binder60=360,primary_Enzyme80=480,additional_budget_outputs=480,
        Raygun_ambiguous_WT_positions=sum(r['correspondence']['ambiguous_positions'] for r in core if r['method']=='raygun_native'),
        outputs={p:sha(out/p) for p in ['core_selected_outputs.json','all_logical_outputs.json']},
        job_id=os.environ['SLURM_JOB_ID'],new_model_calls=0,new_structure_predictions=0,fold_allowed=False))
    print('APPLICATION_1460_LOGICAL_OUTPUTS_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--preflight',action='store_true');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('CPU Slurm required')
    if a.preflight:preflight(a.root)
    elif a.prepare:prepare(a.root)
    else:run(a.root)
