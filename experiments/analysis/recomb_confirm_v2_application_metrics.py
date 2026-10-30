"""Fixed biological roles and observed-native application metrics with missingness bounds."""
import numpy as np
from recomb_benign_binding_native_20261017 import contact_pairs


def coordinates(points):
    out=np.full((len(points),3),np.nan)
    for i,p in enumerate(points):
        if p is not None:out[i]=p
    return out


def rigid_fit(reference,prediction):
    ref=np.asarray(reference,float);pred=np.asarray(prediction,float)
    if ref.shape!=pred.shape or ref.ndim!=2 or ref.shape[1]!=3 or len(ref)<3 or not np.isfinite(ref).all() or not np.isfinite(pred).all():
        raise ValueError('A rigid fit requires at least three finite corresponding points')
    x=pred-pred.mean(0);y=ref-ref.mean(0);u,_,vt=np.linalg.svd(x.T@y)
    correction=np.eye(3);correction[-1,-1]=np.linalg.det(u@vt)
    rotation=u@correction@vt;translation=ref.mean(0)-pred.mean(0)@rotation
    rmsd=float(np.sqrt(np.mean(np.sum((pred@rotation+translation-ref)**2,axis=1))))
    return rotation,translation,rmsd


def native_role_maps(native,tolerance=1e-5):
    """Only assembly-operator actions closing over every original protein instance."""
    chains=native['chains'];identity={c['predictor_chain_id']:c['predictor_chain_id'] for c in chains}
    maps=[identity];seen={tuple(sorted(identity.items()))}
    transforms=[(np.eye(3),np.zeros(3))]+[(np.asarray(c['assembly_matrix'],float),np.asarray(c['assembly_vector'],float)) for c in chains]
    for matrix,vector in transforms:
        mapping={}
        for c in chains:
            composed=matrix@np.asarray(c['assembly_matrix']);shift=matrix@np.asarray(c['assembly_vector'])+vector
            choices=[q for q in chains if q['native_label_chain']==c['native_label_chain'] and q['role']==c['role'] and q['sequence_sha256']==c['sequence_sha256']
                and np.allclose(q['assembly_matrix'],composed,atol=tolerance,rtol=0) and np.allclose(q['assembly_vector'],shift,atol=tolerance,rtol=0)]
            if len(choices)!=1:break
            mapping[c['predictor_chain_id']]=choices[0]['predictor_chain_id']
        if len(mapping)!=len(chains) or len(set(mapping.values()))!=len(chains):continue
        key=tuple(sorted(mapping.items()))
        if key not in seen:seen.add(key);maps.append(mapping)
    return maps


def stable_index(chain,i,correspondence):
    if not chain['target_gene_copy']:return i
    if correspondence['position_states'][i]!='stable':return None
    return correspondence['wt_to_edit'][i]


def choose_role_map(native,predicted,correspondence,allowed_maps):
    scores=[]
    for mapping in allowed_maps:
        ref=[];pred=[]
        for c in native['chains']:
            output=predicted.get(mapping[c['predictor_chain_id']],{})
            points=coordinates(output.get('coordinates',[]))
            for i,xyz in enumerate(c['native_CA']):
                j=stable_index(c,i,correspondence)
                if xyz is not None and j is not None and 0<=j<len(points) and np.isfinite(points[j]).all():ref.append(xyz);pred.append(points[j])
        score=rigid_fit(ref,pred)[2] if len(ref)>=3 else None
        scores.append(dict(mapping=mapping,global_fixed_CA_RMSD_A=score,known_CA_pairs=len(ref)))
    defined=[r for r in scores if r['global_fixed_CA_RMSD_A'] is not None]
    if not defined:return allowed_maps[0],dict(status='INSUFFICIENT_GLOBAL_CA_FOR_EQUIVALENCE_SELECTION',candidates=scores)
    best=min(r['global_fixed_CA_RMSD_A'] for r in defined)
    chosen=next(r for r in scores if r['global_fixed_CA_RMSD_A'] is not None and r['global_fixed_CA_RMSD_A']<=best+1e-6)
    return chosen['mapping'],dict(status='NATIVE_OPERATOR_EQUIVALENCE_SELECTED_BY_GLOBAL_CA_ONLY',candidates=scores,
        selected=chosen['mapping'],no_interface_score_or_method_label_used=True)


def atom_table(atoms):
    result={}
    for a in atoms:
        if a.get('element','').upper() in ('H','D') or not np.isfinite(a['xyz']).all():continue
        key=int(a['residue_index']),a['name']
        if key in result:raise ValueError('Atom alternates must be resolved before metric evaluation')
        result[key]=a['xyz']
    return result


def interface_contacts(native,predicted,correspondence,chain_map,required_atoms,partner_role='fixed_antigen'):
    primary=next(c for c in native['chains'] if c['predictor_chain_id']==native['primary_target_chain'])
    if len(primary['sequence'])!=len(correspondence['position_states']):raise ValueError('Interface design WT length differs')
    partners=[c for c in native['chains'] if c['role']==partner_role]
    source='observed_H_antigen_contacts' if partner_role=='fixed_antigen' else 'observed_H_light_contacts'
    truth={(int(r['design_index']),r['partner_chain'],int(r['partner_index'])) for r in native[source]}
    d_native={int(a['residue_index']) for a in primary['heavy_atoms']}
    possible={i for i in d_native if correspondence['position_states'][i]!='deleted'}
    inverse={j:i for i,j in enumerate(correspondence['wt_to_edit']) if correspondence['position_states'][i]=='stable' and j is not None}
    design=predicted.get(chain_map[primary['predictor_chain_id']],{})
    d_table=atom_table(design.get('heavy_atoms',[]));d_seq=design.get('sequence','')
    mapped_atoms=[dict(a,residue_index=inverse[int(a['residue_index'])]) for a in design.get('heavy_atoms',[]) if int(a['residue_index']) in inverse and inverse[int(a['residue_index'])] in d_native and a.get('element','').upper() not in ('H','D') and np.isfinite(a['xyz']).all()]
    complete=set()
    for j,i in inverse.items():
        if i not in d_native or not 0<=j<len(d_seq):continue
        expected=set(required_atoms.get(d_seq[j],[]))
        present={name for pos,name in d_table if pos==j}
        if expected and expected<=present:complete.add(i)
    positive=set();uncertain_total=0;complete_partner={};native_partner={}
    for partner in partners:
        cid=partner['predictor_chain_id'];p=predicted.get(chain_map[cid],{})
        known={int(a['residue_index']) for a in partner['heavy_atoms']};native_partner[cid]=known
        table=atom_table(p.get('heavy_atoms',[]));seq=p.get('sequence','');full=set()
        present_by_position={}
        for pos,name in table:present_by_position.setdefault(pos,set()).add(name)
        for i in known:
            expected=set(required_atoms.get(seq[i],[])) if i<len(seq) else set()
            if expected and expected<=present_by_position.get(i,set()):full.add(i)
        complete_partner[cid]=full
        atoms=[a for a in p.get('heavy_atoms',[]) if int(a['residue_index']) in known and a.get('element','').upper() not in ('H','D') and np.isfinite(a['xyz']).all()]
        observed={(i,cid,j) for i,j in contact_pairs(mapped_atoms,atoms) if i in possible and j in known}
        positive.update(observed)
        observed_uncertain=sum(i not in complete or j not in full for i,_,j in observed)
        uncertain_total += len(possible)*len(known)-len(complete)*len(full)-observed_uncertain
    unknown_native=sum((i,c,j) not in positive and i in possible and (i not in complete or j not in complete_partner[c]) for i,c,j in truth)
    hit=len(truth&positive);unknown_other=uncertain_total-unknown_native
    if unknown_other<0:raise ValueError('Native uncertainty exceeds all unresolved contacts')
    n=len(truth);p=len(positive);union=n+p-hit
    result=dict(native_contacts=n,known_predicted_contacts=p,recovered_known_contacts=hit,
        unknown_native_contacts=unknown_native,unknown_non_native_pairs=unknown_other,
        contact_recovery_lower=hit/n if n else None,contact_recovery_upper=(hit+unknown_native)/n if n else None,
        interface_F1_lower=2*hit/(n+p+unknown_other) if n+p+unknown_other else None,
        interface_F1_upper=2*(hit+unknown_native)/(n+p+unknown_native) if n+p+unknown_native else None,
        interface_jaccard_lower=hit/(union+unknown_other) if union+unknown_other else None,
        interface_jaccard_upper=(hit+unknown_native)/union if union else None,
        observed_native_design_residues=len(d_native),complete_mapped_prediction_design_residues=len(complete),
        native_design_atom_coverage=len(d_native)/len(primary['sequence']),
        certain_deleted_native_contact_count=sum(correspondence['position_states'][i]=='deleted' for i,_,_ in truth),
        partner_role=partner_role,primary_design_instance=native['primary_target_chain'],
        minimum_heavy_atom_contact_cutoff_A=5.,partial_predicted_atoms_and_mapping_unknowns_bounded=True,
        native_unresolved_contact_universe='observed native residue pairs only; unobserved native regions separately reported')
    return result


def backbone_components(native,predicted,correspondence,chain_map):
    """Native 10A interface backbone and antigen-fit H backbone RMSD, with coverage."""
    primary=next(c for c in native['chains'] if c['predictor_chain_id']==native['primary_target_chain'])
    partners=[c for c in native['chains'] if c['role']=='fixed_antigen']
    memberships={primary['predictor_chain_id']:set()}
    for c in partners:
        pairs=contact_pairs(primary['heavy_atoms'],c['heavy_atoms'],cutoff=10.)
        memberships[primary['predictor_chain_id']].update(i for i,j in pairs)
        memberships[c['predictor_chain_id']]={j for i,j in pairs}
    inter_ref=[];inter_pred=[];receptor_ref=[];receptor_pred=[];H_ref=[];H_pred=[]
    native_interface_atoms=0;native_H_atoms=0;known_interface_H_atoms=0;known_interface_partner_atoms=0
    for c in [primary]+partners:
        output=predicted.get(chain_map[c['predictor_chain_id']],{})
        atoms=atom_table(output.get('heavy_atoms',[]));refatoms=atom_table(c['heavy_atoms'])
        for (i,name),xyz in refatoms.items():
            if name not in ('N','CA','C','O'):continue
            is_interface=i in memberships[c['predictor_chain_id']]
            native_interface_atoms+=int(is_interface)
            is_H=c['predictor_chain_id']==primary['predictor_chain_id'];native_H_atoms+=int(is_H)
            j=stable_index(c,i,correspondence)
            if j is None or (j,name) not in atoms:continue
            pred=atoms[j,name]
            if is_interface:
                inter_ref.append(xyz);inter_pred.append(pred)
                known_interface_H_atoms+=int(is_H);known_interface_partner_atoms+=int(not is_H)
            if is_H:H_ref.append(xyz);H_pred.append(pred)
            else:receptor_ref.append(xyz);receptor_pred.append(pred)
    irmsd=rigid_fit(inter_ref,inter_pred)[2] if len(inter_ref)>=3 and known_interface_H_atoms and known_interface_partner_atoms else None
    lrmsd=None
    if len(receptor_ref)>=3 and H_ref:
        r,t,_=rigid_fit(receptor_ref,receptor_pred)
        lrmsd=float(np.sqrt(np.mean(np.sum((np.asarray(H_pred)@r+t-np.asarray(H_ref))**2,axis=1))))
    return dict(native_interface_backbone_RMSD_A=irmsd,known_interface_backbone_atoms=len(inter_ref),
        observed_native_interface_backbone_atoms=native_interface_atoms,
        interface_backbone_atom_coverage=len(inter_ref)/native_interface_atoms if native_interface_atoms else None,
        known_interface_H_backbone_atoms=known_interface_H_atoms,known_interface_antigen_backbone_atoms=known_interface_partner_atoms,
        antigen_fit_H_backbone_RMSD_A=lrmsd,known_H_backbone_atoms=len(H_ref),observed_native_H_backbone_atoms=native_H_atoms,
        H_backbone_atom_coverage=len(H_ref)/native_H_atoms if native_H_atoms else None,
        antigen_fit_backbone_atoms=len(receptor_ref),native_interface_threshold_A=10.,
        conditional_RMSDs_do_not_replace_deleted_or_unknown_atoms_with_zero=True,
        standard_DockQ_score=False,fixed_biological_roles_not_longest_chain_reassignment=True)


def enzyme_catalytic_geometry(native,predicted,correspondence,chain_map):
    primary=next(c for c in native['chains'] if c['predictor_chain_id']==native['primary_target_chain'])
    ref=coordinates(primary['native_CA']);output=predicted.get(chain_map[primary['predictor_chain_id']],{})
    pred=coordinates(output.get('coordinates',[]));sites={int(s['wt_index']) for s in native['catalytic']}
    pairs=[];errors=[];pair_universe=len(sites)*(len(sites)-1)//2
    known_identity=set()
    for i in sites:
        j=stable_index(primary,i,correspondence)
        if j is not None and j<len(output.get('sequence','')) and output['sequence'][j]==primary['sequence'][i]:known_identity.add(i)
    for i in sorted(sites):
        for j in sorted(sites):
            if i>=j or i not in known_identity or j not in known_identity:continue
            a=stable_index(primary,i,correspondence);b=stable_index(primary,j,correspondence)
            if max(a,b)<len(pred) and np.isfinite(ref[[i,j]]).all() and np.isfinite(pred[[a,b]]).all():
                pairs.append((i,j));errors.append(abs(float(np.linalg.norm(ref[i]-ref[j])-np.linalg.norm(pred[a]-pred[b]))))
    contacts=[]
    for i in sites:
        if not np.isfinite(ref[i]).all():continue
        for j in range(len(ref)):
            if abs(i-j)>=3 and np.isfinite(ref[j]).all() and np.linalg.norm(ref[i]-ref[j])<8.:contacts.append(tuple(sorted((i,j))))
    contacts=set(contacts);hits=unknown=0;identity_disrupted=0
    for i,j in contacts:
        if any(correspondence['position_states'][p]=='deleted' for p in [i,j]):identity_disrupted+=1;continue
        a=stable_index(primary,i,correspondence);b=stable_index(primary,j,correspondence)
        if any(p in sites and correspondence['position_states'][p]=='stable' and p not in known_identity for p in [i,j]):identity_disrupted+=1;continue
        if a is None or b is None or max(a,b)>=len(pred) or not np.isfinite(pred[[a,b]]).all():unknown+=1;continue
        hits+=int(np.linalg.norm(pred[a]-pred[b])<8.)
    return dict(annotated_catalytic_sites=len(sites),identity_retained_catalytic_sites=len(known_identity),
        identity_aware_catalytic_neighbor_contacts=len(contacts),
        identity_aware_catalytic_neighbor_contact_recovery_lower=hits/len(contacts) if contacts else None,
        identity_aware_catalytic_neighbor_contact_recovery_upper=(hits+unknown)/len(contacts) if contacts else None,
        deleted_or_substituted_native_neighbor_contacts=identity_disrupted,
        unknown_native_neighbor_contacts=unknown,catalytic_pair_CA_distance_MAE_A=float(np.mean(errors)) if errors else None,
        observed_annotated_site_pair_universe=pair_universe,known_identity_preserved_site_pairs=len(pairs),
        catalytic_pair_coverage=len(pairs)/pair_universe if pair_universe else None,
        all_deleted_sites_produce_undefined_geometry_not_zero=True,not_activity_measurement=True)
