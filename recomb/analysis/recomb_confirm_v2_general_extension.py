"""Reserve independent enzyme families, then extend ordinary whole-chain inputs."""
import argparse
import hashlib
from pathlib import Path
from recomb_confirm_v2_diagnostics import read,write,sha


def main(root):
    out=root/'reports/recomb_confirm_v2_general_input_extension'
    if (out/'completion.json').exists():raise FileExistsError('General input extension already prepared')
    source=root/'reports/recomb_confirm_v2_native_audit_release/eligible_WT_native_audit_inputs.json'
    rows=read(source);general=[r for r in rows if r['role'] in ('general_domain','general_chain')]
    used={r['evaluation_sequence_component'] for r in general}
    enzymes=sorted([r for r in rows if r['role']=='enzyme'],key=lambda r:hashlib.sha256(
        ('recomb-confirm-v2-enzyme-reserve|'+r['WT_sha256']).encode()).hexdigest())
    reserve=[]
    for r in enzymes:
        comp=r['evaluation_sequence_component']
        if comp in used:continue
        used.add(comp);reserve.append(r)
        if len(reserve)==80:break
    if len(reserve)!=80:raise RuntimeError('Insufficient independent input-only Enzyme80 reserve')
    reserved_components={r['evaluation_sequence_component'] for r in reserve}
    extra=[dict(r,role='general_chain',origin_role='enzyme',
                reason='whole_chain_native_coverage_buffer_extension_not_variant_result_selection')
           for r in enzymes if r['evaluation_sequence_component'] not in reserved_components]
    write(out/'provisional_Enzyme80_reserve.json',reserve)
    write(out/'General_native_candidate_pool.json',general+extra)
    report=dict(status='GENERAL_INPUT_EXTENSION_WITHOUT_LOWERING_NATIVE_COVERAGE',
        original_general_candidates=len(general),ordinary_MCSA_topup_candidates=len(extra),
        provisional_independent_Enzyme80_reserved=80,reserved_families_not_used_for_topup=True,
        minimum_native_CA_coverage_unchanged=.9,source_sha256=sha(source),
        input_only=True,no_variant_or_predicted_structure_outcomes_used=True,new_model_calls=0)
    write(out/'completion.json',report);print(report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
