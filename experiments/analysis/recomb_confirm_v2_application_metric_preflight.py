"""Ordinary native-reference fixtures for application metrics; no model predictions."""
import argparse
from collections import Counter
import hashlib
import os
from pathlib import Path
import numpy as np
from recomb_confirm_v2_diagnostics import read,sha,write
from recomb_confirm_v2_application_metrics import (native_role_maps,choose_role_map,interface_contacts,
    backbone_components,enzyme_catalytic_geometry)

BASE=Path('reports/recomb_confirm_v2_application_metric_preflight')
FZ=Path('configs/recomb_confirm_v2_application_metric_preflight_freeze.json')


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('Application metric preflight already frozen')
    from boltz.data import const
    from Bio.SeqUtils import seq3
    required={a:list(const.ref_atoms[seq3(a).upper()]) for a in 'ACDEFGHIKLMNPQRSTVWY'}
    write(root/BASE/'required_protein_heavy_atoms.json',required)
    names=['analysis/recomb_confirm_v2_application_metric_preflight.py','analysis/recomb_confirm_v2_application_metrics.py',
        'analysis/recomb_benign_binding_native_20261017.py','analysis/recomb_confirm_v2_diagnostics.py',
        'analysis/tests/test_recomb_confirm_v2_application_metrics.py','slurm/recomb_confirm_v2_application_metric_preflight.sbatch',
        str(BASE/'required_protein_heavy_atoms.json')]
    panel=[]
    for cohort in ['Binder60','Enzyme80']:
        p=Path('reports/recomb_confirm_v2_'+cohort+'_constructs/'+cohort+'_panel.json');names.append(str(p))
        panel.extend(dict(r,cohort=cohort) for r in read(root/p))
    native={r['native_assembly_reference']:r['native_assembly_reference_sha256'] for r in panel}
    protocol=dict(status='APPLICATION_METRIC_INSTANTIATION_BEFORE_ANY_APPLICATION_VARIANT_FOLD',
        Binder_primary='observed primary H to fixed-antigen native heavy-atom residue-pair contact recovery, strict distance<5A',
        antibody_light_chain_interface='separate secondary field, not included in H-antigen denominator',
        role_maps='identity plus globally closing deposited biological assembly operator actions preserving source labels/roles/sequences; choose once by global fixed-CA RMSD, tie1e-6 identity-first',
        no_free_interface_maximization=True,original_observed_native_denominator=True,
        unknowns='ambiguous fixed correspondence, missing residue or incomplete predicted canonical heavy-atom set keep conservative contact/F1/Jaccard bounds',
        backbone_components='native10A interface backbone N/CA/C/O RMSD, plus antigen-fit H backbone RMSD, retained atom counts and full observed atom coverage; undefined if all design interface removed',
        standard_DockQ_score=False,custom_components_not_mislabeled_DockQ=True,
        DockQ_primary_source='https://github.com/wallnerlab/DockQ',
        Enzyme_primary='separately audited exact catalytic AA identity; not inferred from geometry',
        Enzyme_key_secondary='position-correspondence catalytic-neighbor native CA contacts, existing definition retained',
        Enzyme_additional='identity-aware catalytic-neighbor contacts and identity-preserved catalytic-pair CA MAE with coverage',
        all_gene_copies_edited=True,primary_native_design_chain_predefined=True,
        native_reference_is_fixture_only_not_predicted_structure=True,new_model_calls=0,new_structure_predictions=0)
    write(root/BASE/'metric_protocol.json',protocol);names.append(str(BASE/'metric_protocol.json'))
    write(root/FZ,dict(status='APPLICATION_METRIC_SOFTWARE_AND_NATIVE_FIXTURES_FROZEN',files={n:sha(root/n) for n in names},
        native_reference_sha256=native,no_variant_structural_results_used=True,new_structure_predictions=0))
    print(sha(root/FZ),flush=True)


def run(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm only')
    frozen=read(root/FZ)
    for p,h in {**frozen['files'],**frozen['native_reference_sha256']}.items():
        path=Path(p) if Path(p).is_absolute() else root/p
        if sha(path)!=h:raise ValueError('Frozen metric fixture/input changed')
    out=root/BASE
    if (out/'completion.json').exists():raise FileExistsError('Metric native fixture checks already completed')
    required=read(out/'required_protein_heavy_atoms.json');maps=[];checks=[]
    for cohort in ['Binder60','Enzyme80']:
        panel=read(root/('reports/recomb_confirm_v2_'+cohort+'_constructs/'+cohort+'_panel.json'))
        for row in panel:
            native=read(row['native_assembly_reference']);allowed=native_role_maps(native)
            ids=[p['predictor_chain_id'] for p in native['chains']]
            if len(ids)!=len(set(ids)):raise ValueError('Native assembly instance labels are not unique')
            primary=next(p for p in native['chains'] if p['predictor_chain_id']==native['primary_target_chain'])
            if primary['sequence']!=row['sequence']:raise ValueError('Primary WT sequence differs')
            n=len(row['sequence']);correspondence=dict(wt_to_edit=list(range(n)),position_states=['stable']*n)
            fixture={p['predictor_chain_id']:dict(sequence=p['sequence'],coordinates=p['native_CA'],heavy_atoms=p['heavy_atoms']) for p in native['chains']}
            mapping,diagnostic=choose_role_map(native,fixture,correspondence,allowed)
            if mapping!=allowed[0]:raise ValueError('Native identity fixture lost its identity chain mapping')
            check=dict(protein_id=row['protein_id'],cohort=cohort,protein_instances=len(ids),allowed_native_role_maps=len(allowed),
                raw_source_chain_labels=len({p['chain_id'] for p in native['chains']}),native_identity_role_map_selected=True)
            if cohort=='Binder60':
                result=interface_contacts(native,fixture,correspondence,mapping,required)
                if result['contact_recovery_lower']!=1. or result['interface_F1_upper']!=1.:raise ValueError('Native identity interface contact fixture differs')
                bb=backbone_components(native,fixture,correspondence,mapping)
                for field in ['native_interface_backbone_RMSD_A','antigen_fit_H_backbone_RMSD_A']:
                    if bb[field] is not None and abs(bb[field])>1e-6:raise ValueError('Native identity backbone geometry differs')
                check.update(native_antigen_contacts=result['native_contacts'],observed_native_atom_coverage=result['native_design_atom_coverage'])
            else:
                result=enzyme_catalytic_geometry(native,fixture,correspondence,mapping)
                if result['identity_retained_catalytic_sites']!=len(native['catalytic']):raise ValueError('Native catalytic AA identity differs')
                value=result['identity_aware_catalytic_neighbor_contact_recovery_lower']
                if value is not None and value!=1.:raise ValueError('Native catalytic neighborhood fixture differs')
                value=result['catalytic_pair_CA_distance_MAE_A']
                if value is not None and abs(value)>1e-8:raise ValueError('Native catalytic CA-pair fixture differs')
                check.update(catalytic_sites=len(native['catalytic']),annotation_tier=row['annotation_tier'])
            maps.append(dict(protein_id=row['protein_id'],native_assembly_sha256=row['native_assembly_reference_sha256'],allowed_maps=allowed))
            checks.append(check);print('NATIVE_METRIC_FIXTURE_PASS '+str(len(checks))+'/140 '+cohort,flush=True)
    write(out/'native_fixture_audit.json',checks);write(out/'allowed_role_maps.json',maps)
    done=dict(status='ALL140_ORDINARY_NATIVE_FIXTURES_PASS_APPLICATION_METRICS_NOT_MODEL_RESULTS',
        cases=len(checks),Binder60=60,Enzyme80=80,unique_instance_ids_verified=True,
        native_operator_role_map_counts=dict(Counter(r['allowed_native_role_maps'] for r in checks)),
        new_model_calls=0,new_structure_predictions=0,no_prediction_quality_measured_by_fixtures=True,
        outputs={n:sha(out/n) for n in ['native_fixture_audit.json','allowed_role_maps.json','metric_protocol.json','required_protein_heavy_atoms.json']},job_id=os.environ['SLURM_JOB_ID'])
    write(out/'completion.json',done);print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();prepare(a.root) if a.prepare else run(a.root)
