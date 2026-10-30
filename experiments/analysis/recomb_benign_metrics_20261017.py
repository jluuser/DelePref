"""Metrics on fixed logical correspondences, with explicit mapping uncertainty."""
import numpy as np
from delepref_v8_structure_metrics import relationship_metrics, fixed_full_wt_tm
from recomb_benign_binding_native_20261017 import contact_pairs


def geometry(reference, prediction, correspondence, binary=None):
    reference = np.asarray(reference,dtype=float); prediction = np.asarray(prediction,dtype=float)
    if reference.ndim!=2 or reference.shape[1]!=3 or prediction.ndim!=2 or prediction.shape[1]!=3:
        raise ValueError('invalid coordinate shape')
    mp = correspondence['wt_to_edit']; states = correspondence['position_states']
    if len(mp)!=len(reference) or len(states)!=len(reference):
        raise ValueError('WT mapping length mismatch')
    stable = [i for i,s in enumerate(states) if s=='stable']
    js = [mp[i] for i in stable]
    if any(j is None or not 0<=j<len(prediction) for j in js) or js!=sorted(set(js)):
        raise ValueError('invalid stable one-to-one correspondence')
    possible = np.array([i for i,s in enumerate(states) if s!='deleted'],dtype=int)
    aligned = np.full((len(possible),3),np.nan)
    for k,i in enumerate(possible):
        if states[i]=='stable': aligned[k]=prediction[mp[i]]
    result = relationship_metrics(reference,aligned,possible)
    stable_result = relationship_metrics(reference,prediction[js],np.asarray(stable,dtype=int))
    ambiguous = sum(s=='ambiguous' for s in states)
    result.update(stable_correspondences=len(stable),ambiguous_WT_positions=ambiguous,
        certain_deleted_positions=sum(s=='deleted' for s in states),edited_length=len(prediction),
        stable_correspondence_coverage=len(stable)/len(reference),
        stable_subset_ca_lddt_lower=stable_result['retained_ca_lddt_lower'],
        stable_subset_ca_lddt_upper=stable_result['retained_ca_lddt_upper'],
        static_relationship_loss_lower=result['static_relationship_loss'],
        static_relationship_loss_upper=stable_result['static_relationship_loss'])
    if ambiguous:
        # Membership and per-residue neighbour denominators are uncertain.
        # The stable-only calculation is diagnostic, not the complete retained mean.
        result['retained_ca_lddt_lower']=0.;result['retained_ca_lddt_upper']=1.
        result['retained_ca_lddt_status']='MAPPING_UNCERTAIN_COARSE_BOUND_STABLE_SUBSET_DIAGNOSTIC_SEPARATE'
    else:
        result['retained_ca_lddt_status']='FIXED_RETAINED_CORRESPONDENCE'
    if binary is not None:
        tm = fixed_full_wt_tm(reference,aligned,possible,binary)
        unknown_max = max(0,min(len(possible),len(prediction))-tm['known_correspondences'])
        tm['upper']=min(tm['upper'],tm['lower']+unknown_max/len(reference))
        result['full_WT_TM']=tm
    return result


def binding_contacts(native, prediction_chains, correspondence, chain_map):
    """Observed-native residue contact recovery/Jaccard; unknowns are bounded."""
    mapping = {m['native_chain_id']:m['prediction_chain_id'] for m in chain_map}
    design = native['chains'][0]; wt_length = len(design['sequence'])
    states = correspondence['position_states']; mp = correspondence['wt_to_edit']
    if len(states)!=wt_length or len(mp)!=wt_length:
        raise ValueError('interface design mapping length mismatch')
    predicted_design = prediction_chains.get(mapping[design['native_chain_id']],{}).get('heavy_atoms',[])
    inverse = {j:i for i,j in enumerate(mp) if states[i]=='stable' and j is not None}
    atoms_design = [dict(a,residue_index=inverse[a['residue_index']]) for a in predicted_design if a['residue_index'] in inverse]
    design_known = {a['residue_index'] for a in atoms_design}
    design_native_known = {a['residue_index'] for a in design['heavy_atoms']}
    design_possible = {i for i in design_native_known if states[i]!='deleted'}
    true = {(r['design_index'],r['partner_chain'],r['partner_index']) for r in native['observed_native_heavy_atom_contacts']}
    positive = set(); uncertain = set()
    for partner in native['chains'][1:]:
        chain = partner['native_chain_id']
        atoms = prediction_chains.get(mapping[chain],{}).get('heavy_atoms',[])
        native_known = {a['residue_index'] for a in partner['heavy_atoms']}
        partner_known = {a['residue_index'] for a in atoms}
        positive.update((a,chain,b) for a,b in contact_pairs(atoms_design,atoms) if a in design_native_known and b in native_known)
        uncertain.update((a,chain,b) for a in design_possible for b in native_known if a not in design_known or b not in partner_known)
    hit = len(true & positive); unknown_native = len(true & uncertain); unknown_other = len(uncertain-true)
    union = len(true | positive)
    return dict(native_contacts=len(true),known_predicted_contacts=len(positive),recovered_known_contacts=hit,
        unknown_native_contacts=unknown_native,unknown_non_native_pairs=unknown_other,
        contact_recovery_lower=hit/len(true) if true else None,
        contact_recovery_upper=(hit+unknown_native)/len(true) if true else None,
        interface_jaccard_lower=hit/(union+unknown_other) if union+unknown_other else None,
        interface_jaccard_upper=(hit+unknown_native)/union if union else None,
        native_unknown_regions_excluded_from_observed_native_contact_universe=True,
        native_design_atom_coverage=len(design_native_known)/wt_length,
        observed_native_universe_definition='minimum heavy-atom distance<5A, native resolved design-partner residue pairs',
        stable_design_correspondence_coverage=len(inverse)/wt_length)


def catalytic_geometry(native,prediction,correspondence):
    prediction = np.asarray(prediction,dtype=float); reference = np.full((len(native['native_CA']),3),np.nan)
    for i,xyz in enumerate(native['native_CA']):
        if xyz is not None:reference[i]=xyz
    states=correspondence['position_states'];mp=correspondence['wt_to_edit']
    sites = sorted({int(r['wt_index']) for r in native['catalytic'] if r['status']=='PASS'})
    pairs=[];errors=[];deleted=[];unknown=[]
    for i in sites:
        if states[i]=='deleted':deleted.append(i)
        elif states[i]!='stable' or not np.isfinite(prediction[mp[i]]).all():unknown.append(i)
    for n,i in enumerate(sites):
        for j in sites[n+1:]:
            if states[i]=='stable' and states[j]=='stable' and np.isfinite(reference[[i,j]]).all() and np.isfinite(prediction[[mp[i],mp[j]]]).all():
                errors.append(abs(float(np.linalg.norm(reference[i]-reference[j])-np.linalg.norm(prediction[mp[i]]-prediction[mp[j]]))))
                pairs.append([i,j])
    neighbors=set()
    for i in sites:
        if np.isfinite(reference[i]).all():
            for j in range(len(reference)):
                if abs(i-j)>=3 and np.isfinite(reference[j]).all() and np.linalg.norm(reference[i]-reference[j])<8.:
                    neighbors.add(tuple(sorted((i,j))))
    hits=uncertain=0
    for i,j in neighbors:
        if states[i]=='deleted' or states[j]=='deleted':continue
        if states[i]!='stable' or states[j]!='stable' or not np.isfinite(prediction[[mp[i],mp[j]]]).all():uncertain+=1
        else:hits+=int(np.linalg.norm(prediction[mp[i]]-prediction[mp[j]])<8.)
    return dict(annotated_catalytic_sites=len(sites),certain_deleted_catalytic_sites=deleted,unknown_catalytic_sites=unknown,
        all_catalytic_sites_retained_lower=(not deleted and not unknown) if sites else None,
        all_catalytic_sites_retained_upper=(not deleted) if sites else None,
        catalytic_pair_CA_distance_MAE_A=float(np.mean(errors)) if errors else None,
        catalytic_neighbor_contacts=len(neighbors),catalytic_neighbor_contact_recovery_lower=hits/len(neighbors) if neighbors else None,
        catalytic_neighbor_contact_recovery_upper=(hits+uncertain)/len(neighbors) if neighbors else None,
        catalytic_pair_count=len(pairs),not_activity_or_all_atom_function_validation=True)
