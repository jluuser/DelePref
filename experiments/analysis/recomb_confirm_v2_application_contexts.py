"""Compile shared, nonspatial application chemistry; never infer activity from parser success."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
import gzip
import hashlib
import os
from pathlib import Path
import numpy as np
from recomb_confirm_v2_diagnostics import read,sha,write
from recomb_confirm_v2_enzyme_context import rows,values

BASE=Path('reports/recomb_confirm_v2_application_contexts')
FZ=Path('configs/recomb_confirm_v2_application_contexts_freeze.json')
COFACTOR_CODES={'FAD','FMN','NAD','NAP','NAI','NDP','COA','ACO','TPP','PLP','PMP','HEM','HEC','SAM','SAH','GSH','BTN','SF4','FES','F3S'}
BIO_METALS={'MG','MN','CA','ZN','FE','FE2','FE3','CO','CU','CU1','CU2','NI','K','NA'}
ANNOTATED={'CHEBI:58210':{'FMN'},'CHEBI:57692':{'FAD'},'CHEBI:49883':{'SF4'},
    'CHEBI:18420':{'MG'},'CHEBI:29035':{'MN'},'CHEBI:29108':{'CA'},'CHEBI:58937':{'TPP'},
    'CHEBI:597326':{'PLP'},'CHEBI:48828':{'CO'},'CHEBI:29033':{'FE2'},'CHEBI:190135':{'FES'},
    'CHEBI:24875':{'FE','FE2','FE3'},'CHEBI:57540':{'NAD'},'CHEBI:57925':{'GSH'},'CHEBI:29105':{'ZN'}}


def signature(matrix,vector):return tuple(np.round(np.asarray(matrix).ravel(),10))+tuple(np.round(vector,10))


def protein_index_map(part,polymers,canonical):
    full=''.join(polymers[part['entity_id']]['pdbx_seq_one_letter_code_can'].split());seq=part['sequence']
    if part['role']=='design_enzyme':
        mapping=canonical['canonical_to_entity']
        return {j+1:i for i,j in enumerate(mapping) if j is not None and 0<=j<len(full) and seq[i]==full[j]}
    if 'label_seq_start' in part:
        start=int(part['label_seq_start'])-1
        if full[start:start+len(seq)]!=seq:raise ValueError('Native contiguous protein mapping changed')
    else:
        starts=[i for i in range(len(full)-len(seq)+1) if full[i:i+len(seq)]==seq]
        if len(starts)!=1:return {}
        start=starts[0]
    return {start+i+1:i for i in range(len(seq))}


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('Application chemistry rules already frozen')
    inputs=[];names=['analysis/recomb_confirm_v2_application_contexts.py','analysis/recomb_confirm_v2_diagnostics.py',
        'analysis/recomb_confirm_v2_enzyme_context.py','analysis/tests/test_recomb_confirm_v2_application_contexts.py',
        'docs/superpowers/plans/2026-10-18-recomb-v2-application-chemical-input-contract.md',
        'slurm/recomb_confirm_v2_application_contexts.sbatch',
        'reports/recomb_confirm_v2_chemical_context/per_case_chemical_inventory.json']
    for cohort in ['Binder60','Enzyme80']:
        p=root/('reports/recomb_confirm_v2_'+cohort+'_constructs/'+cohort+'_panel.json');names.append(str(p.relative_to(root)))
        for r in read(p):inputs.append(dict(r,cohort=cohort))
    write(root/BASE/'inputs.json',inputs);names.append(str(BASE/'inputs.json'))
    write(root/FZ,dict(status='APPLICATION_SHARED_NONSPATIAL_CHEMISTRY_RULES_FROZEN_BEFORE_APPLICATION_VARIANT_STRUCTURES',
        files={p:sha(root/p) for p in names},cofactor_codes=sorted(COFACTOR_CODES),metal_codes=sorted(BIO_METALS),
        annotation_CCD_mapping={k:sorted(v) for k,v in ANNOTATED.items()},
        no_native_coordinates_templates_pocket_or_contact_constraints=True,
        glycan='deposited unit graph only when linked to the positive protein assembly; preserve source coverage',
        known_native_parser_limit='standard CCD/bond representation may retain leaving atoms; record explicitly, no strict valence or function guarantee',
        nonidentity_covalent_symmetry='unmapped, flag rather than invent a bond',
        structure_prediction_allowed=False,new_model_calls=0))
    print(sha(root/FZ),flush=True)


def build_one(item):
    row,inventory=item;ref=read(row['native_assembly_reference']);path=Path(ref['native_file'])
    if sha(path)!=ref['native_file_sha256'] or sha(row['native_assembly_reference'])!=row['native_assembly_reference_sha256']:
        raise ValueError('Frozen native context identity changed')
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as handle:d=MMCIF2Dict(handle)
    import gemmi
    structure=gemmi.read_structure(str(path));assembly=next(a for a in structure.assemblies if a.name==str(ref['assembly_id']))
    polymers={r['entity_id']:r for r in rows(d,'_entity_poly.')};asym={r['id']:r['entity_id'] for r in rows(d,'_struct_asym.')}
    chemical={c['CCD']:c for c in inventory['chemical_components']};conns=inventory['native_chemical_connections']
    protein_labels={p['native_label_chain'] for p in ref['chains']};positive_entities={p['entity_id'] for p in ref['chains']}
    if {asym[l] for l in protein_labels}-positive_entities:raise ValueError('Protein role/entity differs')
    canonical=read(ref['canonical_native_reference']) if row['cohort']=='Enzyme80' else None
    prot={};proteins=[];flags=[]
    for part in ref['chains']:
        sig=signature(part['assembly_matrix'],part['assembly_vector']);mapping=protein_index_map(part,polymers,canonical)
        prot[part['native_label_chain'],sig]=(part,mapping)
        proteins.append({k:part[k] for k in ['predictor_chain_id','role','sequence','sequence_sha256','target_gene_copy']})
    # Nonprotein identities are gathered without exporting target coordinates.
    atoms={k.split('.',1)[1]:values(d,k) for k in d if k.startswith('_atom_site.')};observed=defaultdict(set)
    for i in range(len(atoms['id'])):
        if atoms.get('pdbx_PDB_model_num',['1']*len(atoms['id']))[i]!='1':continue
        if atoms['type_symbol'][i].upper() in ('H','D') or float(atoms.get('occupancy',['1']*len(atoms['id']))[i])<=0:continue
        observed[atoms['label_asym_id'][i]].add((atoms['auth_asym_id'][i],atoms['auth_seq_id'][i],atoms['label_comp_id'][i]))
    branch_byentity=defaultdict(list)
    for unit in inventory['branch_units']:branch_byentity[unit['entity_id']].append(unit)
    nonpoly={r['entity_id']:r for r in inventory['native_nonpolymer_entities']}
    available={l for l,eid in asym.items() if observed[l] and (eid in nonpoly or eid in branch_byentity)}
    sugar=set()
    for l in available:
        eid=asym[l]
        if eid in branch_byentity or 'saccharide' in str((chemical.get(nonpoly.get(eid,{}).get('comp_id'),{}).get('native_chemical_definition') or {}).get('type','')).lower():sugar.add(l)
    attached=set();changed=True
    while changed:
        changed=False
        for c in conns:
            if c['conn_type_id']!='covale':continue
            a,b=c['ptnr1_label_asym_id'],c['ptnr2_label_asym_id']
            for left,right in [(a,b),(b,a)]:
                if left in sugar and (right in protein_labels or right in attached) and left not in attached:
                    attached.add(left);changed=True
    annotated=set()
    for a in inventory['canonical_cofactor_annotations']:
        for c in a.get('cofactors',[]):annotated.update(ANNOTATED.get(c.get('cofactorCrossReference',{}).get('id'),set()))
    selected={};omitted=[]
    for label in sorted(available):
        eid=asym[label]
        if label in attached:
            if eid in branch_byentity:
                units=sorted(branch_byentity[eid],key=lambda u:int(u['num']))
                if len({u['num'] for u in units})!=len(units):flags.append(dict(reason='ambiguous_declared_glycan_unit',label=label));continue
                units=[dict(number=u['num'],CCD=u['comp_id']) for u in units]
            else:units=[dict(number='1',CCD=nonpoly[eid]['comp_id'])]
            role='covalent_glycan'
        else:
            if eid not in nonpoly:omitted.append(dict(label=label,reason='branched_context_without_verified_protein_attachment'));continue
            code=nonpoly[eid]['comp_id'];name=str((chemical.get(code,{}).get('native_chemical_definition') or {}).get('name',''))
            metal=any(c['conn_type_id']=='metalc' and ((c['ptnr1_label_asym_id']==label and c['ptnr2_label_asym_id'] in protein_labels) or(c['ptnr2_label_asym_id']==label and c['ptnr1_label_asym_id'] in protein_labels)) for c in conns)
            adduct=code in {'PLV','PDD'} and 'PLP' in annotated and 'PYRIDOX' in name.upper()
            if code in annotated:role='native_observed_annotated_cofactor'
            elif code in COFACTOR_CODES:role='native_observed_biochemical_cofactor_or_cosubstrate'
            elif code in BIO_METALS and metal:role='author_recorded_protein_coordinated_physiological_metal'
            elif adduct:role='native_adduct_of_annotated_cofactor'
            else:omitted.append(dict(label=label,CCD=code,reason='water_buffer_solvent_substrate_inhibitor_or_unclassified_not_assumed_required'));continue
            obs=sorted(observed[label]);units=[dict(number=str(i+1),CCD=code,auth_chain=a,auth_number=n) for i,(a,n,c) in enumerate(obs) if c==code]
        if not units:continue
        selected[label]=dict(native_label_chain=label,entity_id=eid,role=role,units=units)
    # Assembly metadata determines copies. Only explicit chemical-to-protein links can supply an omitted chemical label.
    operations={};opmeta={}
    for g in assembly.generators:
        labels=set(g.subchains)
        if not labels:labels={p['native_label_chain'] for p in ref['chains'] if p['source_native_chain'] in g.chains}
        for op in g.operators:
            sig=signature(op.transform.mat.tolist(),op.transform.vec.tolist());opmeta[sig]=op.name
            for label in labels&selected.keys():operations.setdefault(label,set()).add(sig)
    for label in selected:
        if operations.get(label):continue
        linked={c['ptnr2_label_asym_id'] if c['ptnr1_label_asym_id']==label else c['ptnr1_label_asym_id']
            for c in conns if c['conn_type_id'] in ('covale','metalc') and label in (c['ptnr1_label_asym_id'],c['ptnr2_label_asym_id'])}
        sigs={sig for pl,sig in prot if pl in linked}
        if sigs:operations[label]=sigs;flags.append(dict(reason='chemical_copy_derived_from_explicit_protein_connection',label=label))
    ligands=[];ligmap={}
    for label in sorted(selected):
        for sig in sorted(operations.get(label,set())):
            base=selected[label];cid='L%04d'%(len(ligands)+1)
            ligand=dict(base,id=cid,operation_name=opmeta.get(sig),units=copy.deepcopy(base['units']))
            ligand['ccd']=[u['CCD'] for u in ligand['units']]
            if any(not chemical.get(c,{}).get('cached_Boltz_molecule_available') for c in ligand['ccd']):raise ValueError('Required cached CCD missing')
            ligands.append(ligand);ligmap[label,sig]=ligand
    def endpoint(c,side,sig):
        prefix='ptnr'+str(side)+'_';label=c[prefix+'label_asym_id'];atom=c[prefix+'label_atom_id']
        if (label,sig) in prot:
            p,mp=prot[label,sig];value=c[prefix+'label_seq_id']
            if not value.isdigit() or int(value) not in mp:return None
            i=mp[int(value)]
            return dict(kind='protein',chain=p['predictor_chain_id'],residue=i+1,atom=atom,
                WT_AA=p['sequence'][i],target_gene_copy=p['target_gene_copy'])
        if (label,sig) not in ligmap:return None
        lig=ligmap[label,sig];eid=lig['entity_id'];index=None
        if eid in branch_byentity:
            matches=[]
            for s in inventory['branch_scheme']:
                if s['asym_id']!=label or s['mon_id']!=c[prefix+'label_comp_id']:continue
                if ((s.get('pdb_asym_id')==c[prefix+'auth_asym_id'] and s.get('pdb_seq_num')==c[prefix+'auth_seq_id']) or
                    (s.get('auth_asym_id')==c[prefix+'auth_asym_id'] and s.get('auth_seq_num')==c[prefix+'auth_seq_id'])):matches.append(s['num'])
            choices={i for i,u in enumerate(lig['units'],1) if u['number'] in set(matches)}
            if len(choices)==1:index=next(iter(choices))
        else:
            choices=[i for i,u in enumerate(lig['units'],1) if u.get('auth_chain',c[prefix+'auth_asym_id'])==c[prefix+'auth_asym_id'] and u.get('auth_number',c[prefix+'auth_seq_id'])==c[prefix+'auth_seq_id'] and u['CCD']==c[prefix+'label_comp_id']]
            if len(choices)==1:index=choices[0]
        if index is None:return None
        return dict(kind='ligand',chain=lig['id'],residue=index,atom=atom)
    bonds=[];seen=set()
    def add_bond(a,b,origin):
        key=tuple(sorted((tuple(a[k] for k in ('chain','residue','atom')),tuple(b[k] for k in ('chain','residue','atom')))))
        if key not in seen:seen.add(key);bonds.append(dict(atom1=a,atom2=b,source=origin))
    for c in conns:
        if c['conn_type_id']!='covale':continue  # A coordination annotation is not a forced covalent bond.
        if c.get('ptnr1_symmetry','1_555') not in ('1_555','.') or c.get('ptnr2_symmetry','1_555') not in ('1_555','.'):
            flags.append(dict(reason='nonidentity_covalent_symmetry_not_invented',connection=c['id']));continue
        candidate_sigs=set(sig for _,sig in ligmap)|set(sig for _,sig in prot)
        for sig in candidate_sigs:
            a,b=endpoint(c,1,sig),endpoint(c,2,sig)
            if a is not None and b is not None:add_bond(a,b,'native_struct_conn:'+c['id'])
            elif all((c['ptnr'+str(side)+'_label_asym_id'],sig) in prot or (c['ptnr'+str(side)+'_label_asym_id'],sig) in ligmap for side in (1,2)):
                flags.append(dict(reason='covalent_endpoint_mapping_unresolved',connection=c['id']))
    # The branch table also covers deposited units without observed atoms.
    for (label,sig),lig in ligmap.items():
        if lig['role']!='covalent_glycan':continue
        nums={u['number']:i for i,u in enumerate(lig['units'],1)}
        for edge in inventory['branch_linkages']:
            if edge['entity_id']!=lig['entity_id']:continue
            a,b=edge['entity_branch_list_num_1'],edge['entity_branch_list_num_2']
            if a in nums and b in nums:
                add_bond(dict(kind='ligand',chain=lig['id'],residue=nums[a],atom=edge['atom_id_1']),
                    dict(kind='ligand',chain=lig['id'],residue=nums[b],atom=edge['atom_id_2']),'deposited_branch_link:'+edge['link_id'])
    present={u['CCD'] for l in ligands for u in l['units']}
    for ligand in ligands:
        if ligand['role']=='covalent_glycan' and not any((b['atom1']['chain']==ligand['id'] and b['atom2']['kind']=='protein') or (b['atom2']['chain']==ligand['id'] and b['atom1']['kind']=='protein') for b in bonds):
            flags.append(dict(reason='glycan_attachment_mapping_not_established',chain=ligand['id']))
    return dict(protein_id=row['protein_id'],cohort=row['cohort'],WT_sha256=row['WT_sha256'],proteins=proteins,ligands=ligands,covalent_bonds=bonds,
        omitted_chemical_context=omitted,context_flags=flags,annotated_CCD_present=sorted(annotated&present),
        annotated_CCD_not_observed=sorted(annotated-present),native_protein_copy_count=len(proteins),
        native_assembly_reference=row['native_assembly_reference'],native_assembly_reference_sha256=row['native_assembly_reference_sha256'],
        only_composition_and_covalent_adjacency_for_model=True,no_native_XYZ_templates_pocket_or_contact_restraints=True,
        native_CCD_leaving_atom_validation_pending=True,strict_valence_or_function_not_established=True,
        structure_prediction_allowed=False)


def apply_variant(context,edited,correspondence):
    result=copy.deepcopy(context);flags=[];bonds=[];anchors=defaultdict(list)
    target={p['predictor_chain_id']:p for p in context['proteins'] if p['target_gene_copy']}
    for p in result['proteins']:
        if p['target_gene_copy']:p.update(sequence=edited,sequence_sha256=hashlib.sha256(edited.encode()).hexdigest())
    def remap(a):
        if a['kind']!='protein' or a['chain'] not in target:return dict(a)
        i=a['residue']-1;state=correspondence['position_states'][i];j=correspondence['wt_to_edit'][i]
        if state!='stable' or j is None or edited[j]!=a['WT_AA']:
            flags.append(dict(reason='covalent_anchor_deleted_substituted_or_ambiguous',chain=a['chain'],WT_position=i+1,state=state));return None
        return dict(a,residue=j+1)
    glycan_ids={l['id'] for l in result['ligands'] if l['role']=='covalent_glycan'}
    for bond in context['covalent_bonds']:
        a,b=bond['atom1'],bond['atom2'];ma,mb=remap(a),remap(b)
        if a['chain'] in glycan_ids and b['kind']=='protein':anchors[a['chain']].append(mb is not None)
        if b['chain'] in glycan_ids and a['kind']=='protein':anchors[b['chain']].append(ma is not None)
        if ma is not None and mb is not None:bonds.append(dict(bond,atom1=ma,atom2=mb))
    removed={g for g,states in anchors.items() if states and not any(states)}
    result['ligands']=[l for l in result['ligands'] if l['id'] not in removed]
    result['covalent_bonds']=[b for b in bonds if b['atom1']['chain'] not in removed and b['atom2']['chain'] not in removed]
    result['variant_context_flags']=flags;result['glycan_chains_removed_with_lost_anchor']=sorted(removed)
    return result


def run(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm only')
    f=read(root/FZ)
    for p,h in f['files'].items():
        if sha(root/p)!=h:raise ValueError('Frozen application context source changed')
    out=root/BASE
    if (out/'execution_claim.json').exists():raise FileExistsError('Context build claimed; inspect existing job')
    write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID']))
    inventory={r['protein_id']:r for r in read(root/'reports/recomb_confirm_v2_chemical_context/per_case_chemical_inventory.json')}
    with ProcessPoolExecutor(max_workers=8) as executor:
        result=list(executor.map(build_one,[(r,inventory[r['protein_id']]) for r in read(out/'inputs.json')]))
    write(out/'WT_contexts.json',result)
    done=dict(status='APPLICATION_NONSPATIAL_WT_CHEMICAL_CONTEXT_TEMPLATES_BUILT_NATIVE_PARSER_PENDING',
        cases=len(result),ligand_instances=sum(len(r['ligands']) for r in result),covalent_bonds=sum(len(r['covalent_bonds']) for r in result),
        context_flags=dict(Counter(x['reason'] for r in result for x in r['context_flags'])),
        with_ligands=sum(bool(r['ligands']) for r in result),with_covalent_glycan=sum(any(l['role']=='covalent_glycan' for l in r['ligands']) for r in result),
        output_sha256=sha(out/'WT_contexts.json'),job_id=os.environ['SLURM_JOB_ID'],new_model_calls=0,new_structure_predictions=0,fold_allowed=False)
    write(out/'completion.json',done);print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();prepare(a.root) if a.prepare else run(a.root)
