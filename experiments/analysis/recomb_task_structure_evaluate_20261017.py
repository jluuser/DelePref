"""Four-budget paired analysis, preserving old metrics and all unknown bounds."""
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
from recomb_task_budget_20261017 import ROOT, read, write
from delepref_v8_runtime import sha
from delepref_v8_structure_metrics import relationship_metrics, fixed_full_wt_tm
sys.path.insert(0,str(ROOT/'code'))
from budget_ledger import BudgetLedger
from paired_statistics import paired_bootstrap
BASE='reports/recomb_task_structure_20261017'
OUT='reports/recomb_task_structures_20261017'


def main():
    freeze=read('configs/recomb_task_structure_freeze_20261017.json')
    for r in freeze['files']:assert sha(r['path'])==r['sha256'],r['path']
    protocol=read(BASE+'/statistical_protocol.json');subset=set(protocol['paired_roster'])
    panel={x['protein_id']:x for x in read('reports/delepref_v8_native_panel/main_panel.json')}
    roster={p:panel[p]['panel_component'] for p in subset}
    old=read('reports/delepref_v8_structure_evaluation/per_output_metrics.json')
    results=[]
    for row in old:
        if row['protein_id'] in subset and row['predictor']=='esmfold' and row['method'] in protocol['methods']:
            if row['role']=='primary_20pct':results.append(dict(row,budget=.2,artifact_reuse='old_frozen_metrics'))
            elif row['role']=='secondary_10pct':results.append(dict(row,budget=.1,artifact_reuse='old_frozen_metrics'))
    new_attempts=BudgetLedger(ROOT/BASE/'statistical_protocol.json',ROOT/OUT/'budget_ledger.jsonl').snapshot()['attempts']
    states={x['task_id']:x for x in new_attempts}
    states.update(read(BASE+'/reused_old_requests.json'))
    physical=read(BASE+'/physical_requests.json')
    assert all(x['task_id'] in states and states[x['task_id']]['state'] in ('succeeded','failed','cache_hit') for x in physical)
    natives={}
    for p in subset:
        wt=panel[p];assert sha(wt['wt_native_json'])==wt['wt_native_json_sha256']
        coords=np.full((len(wt['sequence']),3),np.nan)
        for k,v in json.loads(Path(wt['wt_native_json']).read_text())['coordinates'].items():coords[int(k)]=v
        natives[p]=coords
    for row in read(BASE+'/logical_requests.json'):
        state=states[row['physical_id']]
        if state['state'] in ('succeeded','cache_hit'):
            assert sha(state['output_path'])==state['output_sha']
            with np.load(state['output_path'],allow_pickle=False) as a:
                assert a['sequence_sha256'].item()==row['edited_sequence_sha256'];coords=a['coordinates'].copy()
        else:coords=np.full((len(row['kept']),3),np.nan)
        metrics=relationship_metrics(natives[row['protein_id']],coords,row['kept'])
        tm=fixed_full_wt_tm(natives[row['protein_id']],coords,row['kept'],ROOT/'provenance/usalign_20261017/USalign')
        results.append(dict(row,metrics=metrics,full_WT_fixed_TM=tm,prediction_status=state['state']))
    assert len(results)==240
    assert len({(x['protein_id'],x['budget'],x['method']) for x in results})==240
    summaries=[];paired=[]
    for budget in protocol['budgets']:
        for method in protocol['methods']:
            rows=[x for x in results if x['budget']==budget and x['method']==method];assert len(rows)==20
            summary=dict(budget=budget,method=method,WT=20,prediction_status=dict(Counter(r['prediction_status'] for r in rows)))
            for metric in ['T_lower','T_upper','retained_ca_lddt_lower','retained_ca_lddt_upper',
                           'contact_recovery_lower','contact_recovery_upper','static_relationship_loss','static_contact_loss']:
                values=[x['metrics'][metric] for x in rows if x['metrics'][metric] is not None]
                summary[metric]=float(np.mean(values)) if values else None;summary[metric+'_defined_WT']=len(values)
            summary['full_WT_fixed_TM_lower']=float(np.mean([x['full_WT_fixed_TM']['lower'] for x in rows]))
            summaries.append(summary)
        for metric in ['T','retained_ca_lddt','contact_recovery','full_WT_fixed_TM']:
            observations=[]
            for x in [r for r in results if r['budget']==budget]:
                m=x['full_WT_fixed_TM'] if metric=='full_WT_fixed_TM' else x['metrics']
                lo='lower' if metric=='full_WT_fixed_TM' else metric+'_lower';hi='upper' if metric=='full_WT_fixed_TM' else metric+'_upper'
                observations.append(dict(wt_id=x['protein_id'],group_id=roster[x['protein_id']],method=x['method'],lower=m[lo],upper=m[hi]))
            for baseline in ['delepref_unlimited','terminal']:
                paired.append(dict(budget=budget,metric=metric,comparison='rag_unlimited minus '+baseline,
                    result=paired_bootstrap(observations,'rag_unlimited',baseline,roster,replicates=10000,seed=20261017)))
    write(BASE+'/per_output_metrics.json',results);write(BASE+'/budget_summaries.json',summaries);write(BASE+'/paired_statistics.json',paired)
    write(BASE+'/evaluation_completion.json',dict(status='ALL_FOUR_BUDGET_SUPPLEMENT_OUTPUTS_EVALUATED',logical_outputs=240,WT=20,
        new_physical_requests=len(physical),new_actual_calls=len(new_attempts),
        all_positive_negative_and_unknown_outputs_retained=True,measured_function_validated=False))
    print(json.dumps(read(BASE+'/evaluation_completion.json')),flush=True)


if __name__=='__main__':main()
