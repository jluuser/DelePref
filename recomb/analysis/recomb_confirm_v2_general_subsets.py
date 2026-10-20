"""Freeze balanced nested General subsets using input hashes, before variants."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
from recomb_confirm_v2_diagnostics import read, sha, write


def main(root):
    out = root/'reports/recomb_confirm_v2_General100_subsets'
    if (out/'completion.json').exists(): raise FileExistsError('General subsets already frozen')
    panel_path = root/'reports/recomb_confirm_v2_general_native_v2/General100_panel.json'
    panel = read(panel_path)
    if len(panel)!=100 or Counter(r['stratum'] for r in panel)!=dict(pdb_chain=50,cath_domain=50):
        raise ValueError('Balanced native-qualified General100 required')
    if len({r['evaluation_sequence_component'] for r in panel})!=100:
        raise ValueError('Input sequence components are not unique')
    groups = {s:sorted([r for r in panel if r['stratum']==s],key=lambda r:hashlib.sha256(
        ('recomb-confirm-v2-subset|'+r['WT_sha256']+'|'+r['candidate_id']).encode()).hexdigest())
        for s in ('pdb_chain','cath_domain')}
    subsets = {}
    for name,n in [('budget40',20),('second_predictor30',15),('cost_and_SCISOR_K8_20',10),('full_pool12',6)]:
        subsets[name] = [dict(protein_id=r['protein_id'],WT_sha256=r['WT_sha256'],stratum=r['stratum'],
            component=r['evaluation_sequence_component']) for s in groups for r in groups[s][:n]]
    protocol = dict(status='INPUT_ONLY_GENERAL100_NESTED_SUBSETS_FROZEN',
        main_budget=.2,other_budget_levels=[.1,.3,.4],generation_seed=13,CPU_stability_seed=29,
        second_predictor_seeds=[20261017,20261018,20261019],prediction_repeats_not_independent_N=True,
        subset_rule='one fixed input-hash order within each of two source strata; balanced prefixes',
        mechanism40='conditional input exchangeability after evidence review, not model/structure effect; report effective N',
        primary_statistics='WT paired component bootstrap 10000; seed 20261018; all directions and unknowns retained',
        model_outputs_used=False,retrieval_support_used=False,variant_structures_used=False,
        full_confirmation_prediction_release=False)
    write(out/'subsets.json',subsets);write(out/'protocol.json',protocol)
    paths=[panel_path,Path(__file__),out/'subsets.json',out/'protocol.json']
    result=dict(status='GENERAL100_NESTED_SUBSETS_READY',N={k:len(v) for k,v in subsets.items()},
        files={str(p.relative_to(root)):sha(p) for p in paths},new_model_calls=0,new_structure_calls=0)
    write(out/'completion.json',result);print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);main(p.parse_args().root)
