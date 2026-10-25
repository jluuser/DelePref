"""Recreate paper tables/figures from frozen measurements; no new model calls."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def read_gz(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)


def save_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})


def flatten_application(rows,cohort):
    result=[]
    for row in rows:
        if row['cohort']!=cohort:continue
        item={k:row[k] for k in ['cohort','method','N_all','budget']}
        for endpoint,e in row['endpoints'].items():
            for bound in ['mean_lower','mean_upper','N_defined']:item[endpoint+'__'+bound]=e[bound]
        result.append(item)
    return result


def plot_supplements(tables,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime,timezone
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':9,
      'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'delepref-reproduction-v1'})
    bundle=tables['structural_case_and_mechanism'];eps=['structure_mapped_CA_lddt','structure_mapped_native_contact_recovery']
    methods=['rag_unlimited','delepref_unlimited','gap_frequency','terminal','scisor_L_native','raygun_native']
    labels=['DelePref','DelePref-Base','Gap-frequency','Terminal','SCISOR-L K1','Raygun']
    colors=['#087F78','#45556B','#729947','#B45309','#AE5845','#7556A7'];markers=['o','D','^','s','P','x']
    def save(fig,name):
        fig.savefig(out/(name+'.pdf'),metadata={'CreationDate':datetime(2026,10,25,tzinfo=timezone.utc)})
        fig.savefig(out/(name+'.svg'),metadata={'Date':'2026-10-25'});fig.savefig(out/(name+'.png'),dpi=300);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(174/25.4,72/25.4));fig.subplots_adjust(left=.10,right=.98,bottom=.20,top=.77,wspace=.35)
    for ax,ep,title in zip(axes,eps,['Local Cα geometry','Native-contact recovery']):
        for m,label,col,marker in zip(methods,labels,colors,markers):
            rows=sorted([r for r in bundle['budget'] if r['method']==m],key=lambda r:r['budget'])
            assert len(rows)==4 and all(r['N']==40 for r in rows)
            ax.plot([r['budget']*100 for r in rows],[r[ep] for r in rows],label=label,color=col,marker=marker,lw=1.3,ms=3)
        ax.set(xlabel='Deletion budget (%)',ylabel=title,xticks=[10,20,30,40]);ax.grid(axis='y',ls='--',lw=.4,color='#E3E7EC')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,fontsize=8)
    save(fig,'supplement_general40_budget')
    fig,ax=plt.subplots(figsize=(100/25.4,76/25.4));fig.subplots_adjust(left=.37,right=.97,bottom=.20,top=.91)
    for y,e in zip([4,3,1,0],bundle['effects']):
        ax.errorbar(e['mean'],y,xerr=[[e['mean']-e['lower']],[e['upper']-e['mean']]],fmt='o' if not e['posthoc'] else 's',
                    color='#087F78',mfc='white' if e['posthoc'] else '#087F78',capsize=2,ms=4)
    ax.axvline(0,color='#AAB5C2',ls='--',lw=.8)
    ax.set(yticks=[4,3,1,0],yticklabels=['Uniform / local','Uniform / contacts','Morphology* / local','Morphology* / contacts'],
           xlabel='Policy - control',xlim=(-.012,.104),ylim=(-.6,4.8))
    fig.text(.04,.95,'20 WT, archived joint 97.5% intervals',fontsize=8.5)
    fig.text(.04,.04,'*Post-hoc morphology control; 15/20 outputs identical.',fontsize=8)
    save(fig,'supplement_matched_seed_effects')
    fig,ax=plt.subplots(figsize=(100/25.4,80/25.4));fig.subplots_adjust(left=.20,right=.97,bottom=.23,top=.94)
    for row in bundle['changed']:
        x,y=row['common_local_delta'],row['common_contact_delta'];col='#087F78' if x>0 and y>0 else '#C65B58' if x<0 and y<0 else '#8D9BAF'
        ax.scatter(x,y,s=18,c=col)
    ax.axvline(0,color='#AAB5C2',lw=.7,ls='--');ax.axhline(0,color='#AAB5C2',lw=.7,ls='--')
    ax.set(xlabel='Δ local Cα',ylabel='Δ native contacts');ax.text(.02,.94,'23 changed / 77 unchanged',transform=ax.transAxes,fontsize=8)
    save(fig,'supplement_general100_paired_effects')


def reproduce(data_dir,out,*,sec6=False):
    data_dir=Path(data_dir);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    path=data_dir/'paper_tables.json.gz';tables=read_gz(path)
    save_csv(out/'table_general100_all_predictors.csv',tables['general100'])
    save_csv(out/'table_general100_main.csv',[r for r in tables['general100'] if r['predictor']=='esmfold'])
    save_csv(out/'table_enzyme80.csv',flatten_application(tables['application_means'],'Enzyme80'))
    save_csv(out/'table_binder60.csv',flatten_application(tables['application_means'],'Binder60'))
    save_csv(out/'table_completion_uniform.csv',tables['e9_uniform'])
    shape=[]
    for r in tables['e9_morphology']:
        shape.append(dict(endpoint=r['endpoint'],N=r['lower_effect']['N'],mean_difference=r['lower_effect']['mean_difference'],
                          joint_CI97_5=r['joint_CI97_5_envelope'],post_outcome_mechanism_supplement=r['post_outcome_mechanism_supplement']))
    save_csv(out/'table_completion_morphology.csv',shape)
    save_csv(out/'table_runtime_descriptive.csv',tables['runtime'])
    plot_supplements(tables,out)
    if sec6:
        source=data_dir/'sec6_case'
        if not (source/'data/case_data.json').exists():raise FileNotFoundError('Download/extract the sec6_case asset first')
        destination=out/'sec6_case'
        if destination.exists():raise FileExistsError('Use a fresh output directory for the immutable Sec6 case copy')
        shutil.copytree(source,destination)
        from .sec6_figure import compose
        compose(destination)
        shutil.copyfile(Path(__file__).resolve().parent/'assets/Sec6_caption.md',destination/'Figure2_caption_en.txt')
    report=dict(status='FROZEN_RESULTS_REPRODUCED',source_data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
       source_artifact_hashes=tables['source_sha256'],new_model_calls=0,new_structure_predictions=0,new_bootstrap=0,
       scope='Recreates tables and visualizations from archived measurements and intervals; does not claim a new full folding or training run.')
    (out/'reproduction_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
