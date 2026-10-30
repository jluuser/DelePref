"""Standalone budget figures from frozen paired outputs and known annotations."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/recomb_task_aligned_final_20261017'


def main():
    read=lambda n:json.loads((OUT/n).read_text())
    results=read('budget_per_output_metrics.json');generation=read('plot_generation_inputs.json')
    features={r['protein_id']:r for r in read('mapped_features.json')}
    methods=['delepref_unlimited','rag_unlimited','terminal'];labels=['DelePref (unlimited)','DelePref + retrieval','Terminal control']
    colors=['#31688e','#e8871e','#666666'];budgets=[.1,.2,.3,.4]
    roster=sorted({r['protein_id'] for r in results});assert len(roster)==20
    site_sets={p:sorted({x for f in features[p]['features'] if f['type'] in ['Active site','Binding site','Disulfide bond']
                       for x in f['WT_positions_0based']}) for p in roster}
    known=[p for p in roster if site_sets[p]]
    fig,axes=plt.subplots(2,3,figsize=(12,7.8))
    fig.subplots_adjust(left=.06,right=.985,bottom=.075,top=.82,hspace=.36,wspace=.22)
    names=[('T','Full-native relationship retention T'),('retained_ca_lddt','Retained CA lDDT'),
           ('full_WT_fixed_TM','Fixed-correspondence full-WT TM'),('contact_recovery','Native 8A contact retention')]
    sample=np.random.default_rng(20261017).integers(0,20,(10000,20))
    for ax,(metric,title) in zip(axes.flat,names):
        for method,label,color in zip(methods,labels,colors):
            means=[];low=[];high=[]
            for budget in budgets:
                by_wt={r['protein_id']:r for r in results if r['budget']==budget and r['method']==method}
                lo=[];hi=[]
                for p in roster:
                    m=by_wt[p]['full_WT_fixed_TM'] if metric=='full_WT_fixed_TM' else by_wt[p]['metrics']
                    lo.append(m['lower' if metric=='full_WT_fixed_TM' else metric+'_lower'])
                    hi.append(m['upper' if metric=='full_WT_fixed_TM' else metric+'_upper'])
                lo=np.array(lo);hi=np.array(hi);means.append(lo.mean());low.append(np.quantile(lo[sample].mean(1),.025));high.append(np.quantile(hi[sample].mean(1),.975))
            ax.plot(np.array(budgets)*100,means,'o-',color=color,label=label,lw=1.7,ms=4)
            ax.fill_between(np.array(budgets)*100,low,high,color=color,alpha=.10,lw=0)
        ax.set_title(title,fontsize=10);ax.set_ylim(0,1)
    ax=axes[1,1]
    for method,label,color in zip(methods,labels,colors):
        means=[]
        for budget in budgets:
            rs={r['protein_id']:r for r in generation if r['method']==method and r['budget']==budget}
            values=[]
            for p in known:
                deleted=set()
                for a,b in json.loads(rs[p]['deleted_intervals']):deleted.update(range(a,b+1))
                values.append(1-len(deleted&set(site_sets[p]))/len(site_sets[p]))
            means.append(float(np.mean(values)) if values else np.nan)
        ax.plot(np.array(budgets)*100,means,'o-',color=color,lw=1.7,ms=4)
    ax.set_title('Annotated site retention (known WT N='+str(len(known))+')',fontsize=10);ax.set_ylim(0,1.02)
    ax=axes[1,2]
    for method,label,color in zip(methods,labels,colors):
        vals=[[r['span_count'] for r in generation if r['method']==method and r['budget']==b] for b in budgets]
        ax.plot(np.array(budgets)*100,[np.median(v) for v in vals],'o-',color=color,lw=1.7,ms=4)
        ax.fill_between(np.array(budgets)*100,[np.quantile(v,.25) for v in vals],[np.quantile(v,.75) for v in vals],color=color,alpha=.10)
    ax.set_title('Deletion span count: median and IQR',fontsize=10)
    for ax in axes.flat:
        ax.set_xticks([10,20,30,40]);ax.set_xlabel('Target deletion budget (%)',fontsize=9);ax.grid(alpha=.18)
        ax.axvspan(25,42,color='#aaaaaa',alpha=.07);ax.tick_params(labelsize=8)
    handles,legend_labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(.5,.915),ncol=3,frameon=False,fontsize=10)
    fig.suptitle('Frozen 20-WT supplementary budget panel\nStructure proxies and annotation retention; 30-40% budgets extrapolate beyond training',fontsize=12,y=.985)
    fig.savefig(OUT/'budget_task_dimensions.png',dpi=180);fig.savefig(OUT/'budget_task_dimensions.pdf')
    report=dict(status='BUDGET_FIGURE_EXPORTED',WT=20,annotated_site_WT=len(known),
        band='Structural panels: component/WT bootstrap10000 sampling plus coordinate-missing intervals; last panel IQR',
        annotation='Known mapped sites only; unknown annotations excluded with explicit N, no measured activity claim',
        input_sha256={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in
                      ['budget_per_output_metrics.json','plot_generation_inputs.json','mapped_features.json']})
    (OUT/'figure_manifest.json').write_text(json.dumps(report,indent=2)+'\n',newline='\n')
    print(json.dumps(report))


if __name__=='__main__':main()
