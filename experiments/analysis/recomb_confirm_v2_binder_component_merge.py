"""Merge antigen/sequence-defined clone groups before any new Binder generation."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
from recomb_confirm_v2_diagnostics import read,sha,write
from delepref_v8_panel import Components,alignment

BASE=Path('reports/recomb_confirm_v2_binder_components')
FZ=Path('configs/recomb_confirm_v2_binder_component_merge_freeze.json')


def gene_family(gene,kind):
    if not gene or not gene.get('name'):return None
    name=gene['name'].split('*',1)[0]
    match=re.match(r'IGHV\d+',name) if kind=='V' else None
    return match[0] if match else name


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('Component merge already frozen')
    paths=['analysis/recomb_confirm_v2_binder_component_merge.py','analysis/recomb_confirm_v2_diagnostics.py',
        'analysis/delepref_v8_panel.py','slurm/recomb_confirm_v2_binder_component_merge.sbatch',
        str(BASE/'case_inputs.json'),str(BASE/'query_inputs.json'),str(BASE/'input_completion.json'),
        'reports/recomb_confirm_v2_binder_numbering/numbering_audit.json','reports/recomb_confirm_v2_binder_numbering/completion.json',
        'reports/recomb_confirm_v2_binder_exposure/per_query_exposure.json','reports/recomb_confirm_v2_binder_exposure/completion.json',
        'reports/recomb_confirm_v2_binder_exposure/antigen_sequence_edges.json','reports/recomb_confirm_v2_binder_exposure/self_hits.tsv']
    write(root/FZ,dict(status='INPUT_ONLY_ANTIGEN_AND_CLONE_PROXY_MERGE_FROZEN_BEFORE_VARIANT_OUTPUTS',
        files={p:sha(root/p) for p in paths},
        clone_proxy='same V family and allele-stripped J gene, same CDR3 length, >=90% ungapped CDR3 AA identity; connected components',
        biological_lineage_not_proven=True,HMM_species_not_source_taxon=True,
        antigen_family='same source UniProt accession or confirmed native antigen sequence edge at identity_shorter_full>=30%, both coverage80%, full lengths>=50',
        legacy_antigen_proxy='legacy Binder20 has numbered H; non-numbered-IG participants form conservative candidate antigen references',
        legacy_antigen_homology='identity_shorter_full>=30%, at least one coverage80%, min full length50; exact IDs also flag',
        eligibility='no exact trained design, prior exact development/General100 design, development clone proxy or old Binder antigen component; typed scaffold and listed protein context closure',
        selection='input hash within VHH/Fab source strata; first one per combined component, then max2 per component; unique heavy SHA and PDB',
        target_cases=60,minimum_components=40,model_output_or_retrieval_support_used=False,
        no_model_release_until_full_assembly_constructs=True))
    print(sha(root/FZ),flush=True)


def run(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    f=read(root/FZ)
    for p,h in f['files'].items():
        if sha(root/p)!=h:raise ValueError('Component audit input/source changed')
    out=root/BASE
    if (out/'component_completion.json').exists():raise FileExistsError('Component audit complete; do not repeat')
    cases=read(out/'case_inputs.json');queries={q['query_id']:q for q in read(out/'query_inputs.json')}
    numbered={q['query_id']:q for q in read(root/'reports/recomb_confirm_v2_binder_numbering/numbering_audit.json')}
    exposures=read(root/'reports/recomb_confirm_v2_binder_exposure/per_query_exposure.json')
    case_by={c['native_case_key']:c for c in cases};ids=sorted(case_by)
    ag=Components(ids);combined=Components(ids)
    ag_uses=defaultdict(set);accession_uses=defaultdict(set);heavy_query={};heavy_domains={}
    for c in cases:
        key=c['native_case_key']
        for p in c['participants']:
            if p['role']=='fixed_antigen':ag_uses[p['query_id']].add(key)
            elif p['role']=='design_antibody_H':heavy_query[key]=p['query_id']
        for accession in c['antigen_accessions']:accession_uses[accession].add(key)
    def unite(groups,components):
        for group in groups:
            a=sorted(group)
            for b in a[1:]:components.union(a[0],b)
    unite(ag_uses.values(),ag);unite(accession_uses.values(),ag)
    for h in read(root/'reports/recomb_confirm_v2_binder_exposure/antigen_sequence_edges.json'):
        unite([ag_uses[h['query']]|ag_uses[h['target']]],ag)
    antigen_groups=defaultdict(set)
    for key in ids:antigen_groups[ag.find(key)].add(key)
    unite(antigen_groups.values(),combined)
    nodes={};by_proxy=defaultdict(list);development_nodes=set()
    for qid,row in numbered.items():
        for index,d in enumerate(row['domains']):
            if d['chain_type']!='H':continue
            node=qid+':H'+str(index);v=gene_family(d['V_gene'],'V');j=gene_family(d['J_gene'],'J')
            nodes[node]=dict(query_id=qid,domain_index=index,CDR3=d['CDR3'],V_family=v,J_gene=j)
            if v and j:by_proxy[v,j,len(d['CDR3'])].append(node)
            if any(u['kind']=='development' for u in queries[qid]['uses']):development_nodes.add(node)
    clones=Components(sorted(nodes));clone_edges=0
    for group in by_proxy.values():
        for i,a in enumerate(sorted(group)):
            for b in sorted(group)[i+1:]:
                x,y=nodes[a]['CDR3'],nodes[b]['CDR3']
                if sum(p==q for p,q in zip(x,y))/len(x)>=.9:clones.union(a,b);clone_edges+=1
    development_clone={clones.find(n) for n in development_nodes}
    clone_cases=defaultdict(set);case_clone={}
    for key,qid in heavy_query.items():
        domains=[n for n,d in nodes.items() if d['query_id']==qid]
        heavy_domains[key]=domains;case_clone[key]={clones.find(n) for n in domains}
        for component in case_clone[key]:clone_cases[component].add(key)
    unite(clone_cases.values(),combined)
    legacy=defaultdict(set)
    for qid,q in queries.items():
        for u in q['uses']:
            if u['kind']=='development' and u.get('dataset')=='Binder20':legacy[u['protein_id']].add(qid)
    legacy_ag=set();legacy_cases=[]
    for pid,participants in legacy.items():
        h=[q for q in participants if any(d['chain_type']=='H' for d in numbered.get(q,{}).get('domains',[]))]
        antigen=[q for q in participants if not numbered.get(q,{}).get('domains')]
        if h:legacy_ag.update(antigen)
        legacy_cases.append(dict(protein_id=pid,H_queries=h,conservative_antigen_queries=antigen if h else [],
            true_binder_roles_not_inferred_if_H_absent=not bool(h)))
    old_ag_cases={key for q in legacy_ag for key in ag_uses.get(q,[])}
    with (root/'reports/recomb_confirm_v2_binder_exposure/self_hits.tsv').open() as handle:
        for line in handle:
            fields=line.rstrip('\n').split('\t')
            if fields[0] not in ag_uses or fields[1] not in legacy_ag:continue
            h=alignment(fields,queries)
            if min(h['query_length'],h['target_length'])>=50 and h['identity_shorter_full']>=.3 and max(h['query_coverage'],h['target_coverage'])>=.8:
                old_ag_cases.update(ag_uses[h['query']])
    old_ag_components={ag.find(key) for key in old_ag_cases}
    audit=[]
    for key in ids:
        c=case_by[key];qid=heavy_query[key];q=queries[qid];reasons=[];domains=heavy_domains[key]
        if exposures[qid]['exact_training_sequence_ids']:reasons.append('exact_complete_student_training_design')
        if any(u['kind'] in ('development','General100') for u in q['uses']):reasons.append('exact_prior_development_or_General100_design')
        if case_clone[key]&development_clone:reasons.append('development_CDR3_VJ_proxy_component')
        if ag.find(key) in old_ag_components:reasons.append('development_binder_antigen_component')
        if not domains:reasons.append('H_variable_domain_or_exact_CDR3_mapping_unresolved')
        elif any(not nodes[n]['V_family'] or not nodes[n]['J_gene'] for n in domains):reasons.append('VJ_proxy_assignment_unresolved')
        if not c['context_closure_known']:reasons.append('listed_protein_context_not_closed')
        if c['scaffold'] not in ('explicitly_annotated_VHH_or_nanobody','curated_antibody_heavy_chain_with_fixed_light'):
            reasons.append('main_scaffold_annotation_not_explicit_VHH_or_fixed_light_Fab')
        row=dict(c,antigen_component=ag.find(key),combined_component=combined.find(key),
            sequence_defined_clone_components=sorted(case_clone[key]),H_domain_count=len(domains),
            heavy_training_exposure=exposures[qid],eligibility_reasons=reasons,
            eligibility='INPUT_ELIGIBLE_FOR_FROZEN_ROSTER' if not reasons else 'INPUT_AUDIT_HOLD_NO_MODEL_RELEASE')
        audit.append(row)
    eligible=[r for r in audit if not r['eligibility_reasons']]
    types=['explicitly_annotated_VHH_or_nanobody','curated_antibody_heavy_chain_with_fixed_light']
    ordered={t:sorted([r for r in eligible if r['scaffold']==t],key=lambda r:hashlib.sha256(
        ('recomb-confirm-v2-Binder60|'+r['WT_sha256']+'|'+r['native_case_key']).encode()).hexdigest()) for t in types}
    selected=[];used_keys=set();used_hash=set();used_pdb=set();component_count=Counter()
    for cap in [1,2]:
        cursor={t:0 for t in types}
        while len(selected)<60:
            changed=False
            for t in types:
                while cursor[t]<len(ordered[t]):
                    row=ordered[t][cursor[t]];cursor[t]+=1;key=row['native_case_key']
                    if key in used_keys or row['WT_sha256'] in used_hash or row['PDB'] in used_pdb or component_count[row['combined_component']]>=cap:continue
                    selected.append(row);used_keys.add(key);used_hash.add(row['WT_sha256']);used_pdb.add(row['PDB']);component_count[row['combined_component']]+=1;changed=True;break
                if len(selected)==60:break
            if not changed:break
        if len(selected)==60:break
    agN=len({r['antigen_component'] for r in selected});compN=len(component_count)
    ready=len(selected)==60 and min(agN,compN)>=40
    for r in selected:
        r.update(protein_id='Binder60v2|'+r['native_case_key'][:16],WT_encoding_allowed=False,
            deletion_generation_allowed=False,structure_prediction_allowed=False,full_assembly_construct_release_pending=True)
    write(out/'per_case_components.json',audit);write(out/'legacy_binder_role_proxy.json',legacy_cases)
    write(out/'clone_proxy_nodes.json',nodes);write(out/'Binder60_selected_case_ids.json',[r['native_case_key'] for r in selected])
    write(out/'Binder60_roster_pending_constructs.json',selected)
    report=dict(status='BINDER60_INPUT_CASES_SELECTED_FULL_ASSEMBLY_CONSTRUCTS_PENDING' if ready else 'BINDER60_INPUT_SHORTFALL_NO_MODEL_RELEASE',
        audited_cases=len(cases),eligible_cases=len(eligible),eligibility_reasons=dict(Counter(x for r in audit for x in r['eligibility_reasons'])),
        antigen_components=len(antigen_groups),combined_components=len({r['combined_component'] for r in audit}),
        sequence_defined_clone_components=len({clones.find(n) for n in nodes}),clone_proxy_edges=clone_edges,
        legacy_numbered_H_binder_cases=sum(bool(r['H_queries']) for r in legacy_cases),
        selected_cases=len(selected),selected_antigen_components=agN,selected_combined_components=compN,
        selected_unique_H=len(used_hash),selected_unique_PDB=len(used_pdb),scaffold=dict(Counter(r['scaffold'] for r in selected)),
        no_retrieval_or_variant_outcomes_used=True,biological_clonal_lineage_not_proven=True,
        model_release_allowed=False,new_model_calls=0,new_structure_predictions=0,job_id=os.environ['SLURM_JOB_ID'],
        selected_roster_sha256=sha(out/'Binder60_roster_pending_constructs.json'))
    write(out/'component_completion.json',report);print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();prepare(a.root) if a.prepare else run(a.root)
