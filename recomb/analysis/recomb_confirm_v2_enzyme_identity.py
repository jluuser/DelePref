"""Predeclared Enzyme80 catalytic AA identity endpoint; no structural predictions."""
import argparse
from collections import defaultdict
import os
from pathlib import Path
from recomb_confirm_v2_diagnostics import read,sha,write
from recomb_benign_catalytic_identity_20261017 import retained_identity
from recomb_confirm_v2_general_evaluate import effect

BASE=Path('reports/recomb_confirm_v2_Enzyme80_catalytic_identity')
FZ=Path('configs/recomb_confirm_v2_Enzyme80_catalytic_identity_freeze.json')
OUTPUT=Path('reports/recomb_confirm_v2_application_outputs')
PANEL=Path('reports/recomb_confirm_v2_Enzyme80_constructs/Enzyme80_panel.json')


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('Enzyme primary identity analysis already frozen')
    done=read(root/OUTPUT/'completion.json')
    if done['logical_outputs']!=1460:raise ValueError('Application six-method outputs incomplete')
    paths=['analysis/recomb_confirm_v2_enzyme_identity.py','analysis/recomb_benign_catalytic_identity_20261017.py',
        'analysis/recomb_confirm_v2_general_evaluate.py','analysis/recomb_confirm_v2_diagnostics.py',
        'slurm/recomb_confirm_v2_enzyme_identity.sbatch',str(PANEL),str(OUTPUT/'completion.json'),str(OUTPUT/'all_logical_outputs.json')]
    native={p['native_assembly_reference']:p['native_assembly_reference_sha256'] for p in read(root/PANEL)}
    write(root/FZ,dict(status='ENZYME80_PREDECLARED_CATALYTIC_AA_IDENTITY_ENDPOINT_FROZEN_BEFORE_GEOMETRIC_OUTCOMES',
        files={p:sha(root/p) for p in paths},native_reference_sha256=native,
        primary_endpoint='all annotated catalytic positions uniquely retained with the exact WT amino acid',
        Raygun='deleted/substituted fails; ambiguous identity retains lower/upper bounds, not a false point score',
        annotation_tiers='79 M-CSA complete networks plus1 reviewed UniProt Active site, separately reported',
        methods='same six fixed generators and same target roster; generation budget differs by native protocol',
        paired_statistics='component bootstrap10000 seed20261016,95% interval; no seed pseudo-replication',
        no_model_calls=True,new_structure_predictions=0))
    (root/BASE).mkdir(parents=True,exist_ok=True);print(sha(root/FZ),flush=True)


def run(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm only')
    f=read(root/FZ)
    for p,h in {**f['files'],**f['native_reference_sha256']}.items():
        path=Path(p) if Path(p).is_absolute() else root/p
        if sha(path)!=h:raise ValueError('Frozen enzyme endpoint input changed')
    out=root/BASE
    if (out/'completion.json').exists():raise FileExistsError('Enzyme sequence-identity endpoint already complete')
    panel={p['protein_id']:p for p in read(root/PANEL)};annotations={}
    if len(panel)!=80:raise ValueError('Fixed enzyme80 roster changed')
    for pid,p in panel.items():
        native=read(p['native_assembly_reference']);sites=[]
        for s in native['catalytic']:
            i=int(s['wt_index']);aa=s['WT_residue']
            if p['sequence'][i]!=aa:raise ValueError('Catalytic original AA annotation mismatch')
            sites.append(dict(status='PASS',wt_index=i,wt_residue=aa))
        if not sites or ('catalytic_residue_count' in p and len(sites)!=p['catalytic_residue_count']):raise ValueError('Missing or changed catalytic annotation')
        annotations[pid]=sites
    rows=[]
    for r in read(root/OUTPUT/'all_logical_outputs.json'):
        if r['cohort']!='Enzyme80':continue
        pid=r['protein_id'];p=panel[pid]
        if r['WT_sha256']!=p['WT_sha256']:raise ValueError('Output WT differs from annotated enzyme')
        rows.append(dict(protein_id=pid,method=r['method'],budget=r['budget'],evaluation_role=r['evaluation_role'],
            annotation_tier=p['annotation_tier'],EC=p['ec_numbers'],domain_training_exposure=p['training_domain_like_exposure'],
            **retained_identity(annotations[pid],r['edited_sequence'],r['correspondence'])))
    if len(rows)!=920:raise ValueError('Enzyme140conditions x6 plus80WT roster differs')
    grouped=defaultdict(list)
    for r in rows:grouped[r['budget'],r['method']].append(r)
    summaries=[]
    for (beta,method),group in sorted(grouped.items()):
        n=len(group)
        summaries.append(dict(method=method,budget=beta,N=n,components=len({panel[r['protein_id']]['evaluation_sequence_component'] for r in group}),
            annotation_tier_counts=dict(__import__('collections').Counter(r['annotation_tier'] for r in group)),
            all_sites_retained_cases_lower=sum(r['all_verified_identities_retained_lower'] for r in group),
            all_sites_retained_cases_upper=sum(r['all_verified_identities_retained_upper'] for r in group),
            all_sites_retained_fraction_lower=sum(r['all_verified_identities_retained_lower'] for r in group)/n,
            all_sites_retained_fraction_upper=sum(r['all_verified_identities_retained_upper'] for r in group)/n,
            macro_site_retention_lower=sum(r['site_identity_retention_lower'] for r in group)/n,
            macro_site_retention_upper=sum(r['site_identity_retention_upper'] for r in group)/n,
            certain_deletion_cases=sum(bool(r['certain_deleted_sites']) for r in group),
            certain_substitution_cases=sum(bool(r['certain_substituted_sites']) for r in group),
            ambiguous_identity_cases=sum(bool(r['unresolved_mapping_sites']) for r in group)))
    main={(r['protein_id'],r['method']):r for r in rows if r['budget']==.2}
    methods=sorted({r['method'] for r in rows if r['budget']==.2})
    components={pid:p['evaluation_sequence_component'] for pid,p in panel.items()};comparisons=[]
    for method in methods:
        if method=='delepref_unlimited':continue
        lower=[];upper=[]
        for pid in sorted(panel):
            a=main[pid,method];b=main[pid,'delepref_unlimited']
            lower.append((pid,float(a['all_verified_identities_retained_lower'])-float(b['all_verified_identities_retained_upper'])))
            upper.append((pid,float(a['all_verified_identities_retained_upper'])-float(b['all_verified_identities_retained_lower'])))
        lo=effect(lower,components);hi=effect(upper,components)
        comparisons.append(dict(method=method,versus='delepref_unlimited',budget=.2,primary_endpoint='complete_catalytic_AA_identity',
            paired_difference_lower=lo,paired_difference_upper=hi,
            CI95_conservative_envelope=[lo['CI95'][0],hi['CI95'][1]],
            partial_identity_bounds_not_measured_activity=True))
    for name,value in [('per_output_identity.json',rows),('summary.json',summaries),('primary_comparisons.json',comparisons)]:write(out/name,value)
    write(out/'completion.json',dict(status='ALL80_ENZYME_SIX_METHOD_CATALYTIC_AA_IDENTITIES_AUDITED_NOT_ACTIVITY_VALIDATION',
        WT=80,logical_outputs=len(rows),primary_nonWT=480,annotation_sites=sum(len(s) for s in annotations.values()),
        new_model_calls=0,new_structure_predictions=0,no_generator_reselection=True,
        outputs={n:sha(out/n) for n in ['per_output_identity.json','summary.json','primary_comparisons.json']},job_id=os.environ['SLURM_JOB_ID']))
    print(read(out/'completion.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();prepare(a.root) if a.prepare else run(a.root)
