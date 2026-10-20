"""Export complete-primary figures and a candid report; no new model computation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from recomb_scope_screen_20261017 import read,sha,write


def build(root,output_name='presentation'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    base=root/'reports/recomb_benign_primary_readout_20261017'
    done=read(base/'completion.json')
    for n,h in done['outputs'].items():
        if sha(base/n)!=h:raise ValueError('complete primary results changed')
    out=base/output_name
    if out.exists():raise FileExistsError('primary presentation already exists')
    out.mkdir()
    rows=read(base/'RAG_coverage_and_differences.json');summaries=read(base/'paired_comparisons.json')
    absolute=read(base/'absolute_metrics.json');counts=Counter(r['participation'] for r in rows)
    categories=['direct_guidance_selected','partial_guidance_selected','retrieval_proposed_policy_selected','no_usable_proposal_fallback']
    labels=['Direct natural proposal','Homology + Policy completion','Proposed, Policy selected','No usable proposal: fallback']
    colors=['#009E73','#56B4E9','#E69F00','#999999']
    color=dict(zip(categories,colors));name=dict(zip(categories,labels))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    def save(fig,stem):
        fig.savefig(out/(stem+'.png'),dpi=200,bbox_inches='tight');fig.savefig(out/(stem+'.svg'),bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,2.5));left=0
    for category in categories:
        n=counts[category];ax.barh(0,n,left=left,color=color[category],label=name[category])
        if n:ax.text(left+n/2,0,str(n),ha='center',va='center',fontsize=11)
        left+=n
    ax.set_xlim(0,44);ax.set_yticks([]);ax.set_xlabel('Number of constructs (fixed N=44)')
    ax.set_title('RAG participation at 20% deletion budget',pad=28)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.4),ncol=2,frameon=False)
    ax.text(.5,1.03,'Changed deletion plans: 10/44; changed candidate pools: 19/44',transform=ax.transAxes,ha='center',fontsize=9)
    save(fig,'RAG_participation')
    endpoints=['retained_ca_lddt','contact_recovery','full_WT_TM']
    titles=['Retained Cα-lDDT','Native-contact recovery','Full-WT fixed-correspondence TM']
    fig,axes=plt.subplots(1,3,figsize=(13,3.6),sharex=True)
    for ax,endpoint,title in zip(axes,endpoints,titles):
        for i,row in enumerate(rows,1):
            v=row['differences'][endpoint];lo,hi=100*v['lower'],100*v['upper'];middle=(lo+hi)/2
            ax.plot([i,i],[0,middle],color=color[row['participation']],alpha=.45,lw=.8)
            ax.errorbar(i,middle,yerr=[[max(0,middle-lo)],[max(0,hi-middle)]],fmt='o',ms=3,
                color=color[row['participation']],elinewidth=1,capsize=1)
        ax.axhline(0,color='black',lw=.8,ls='--');ax.set_title(title);ax.set_xlabel('Construct (fixed ID order)');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('RAG − original DelePref (score ×100)')
    fig.suptitle('All 44 prespecified constructs; positive and negative effects retained',fontsize=11)
    fig.text(.5,-.04,'TM bars show missing-coordinate bounds; interval midpoints are display positions, not a new point estimator.',ha='center',fontsize=9)
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],marker='o',color=c,lw=0,label=label) for c,label in zip(colors,labels)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,-.22),ncol=2,frameon=False,fontsize=9)
    fig.tight_layout();save(fig,'RAG_per_construct_differences')
    fig,axes=plt.subplots(1,3,figsize=(13,3.6))
    for ax,endpoint,title in zip(axes,endpoints,titles):
        for category in categories:
            selected=[r for r in rows if r['participation']==category]
            x=[100*r['reliable_gap_coverage'] for r in selected]
            lower=[100*r['differences'][endpoint]['lower'] for r in selected];upper=[100*r['differences'][endpoint]['upper'] for r in selected]
            middle=(np.asarray(lower)+np.asarray(upper))/2
            if selected:ax.errorbar(x,middle,yerr=[np.maximum(0,middle-np.asarray(lower)),np.maximum(0,np.asarray(upper)-middle)],fmt='o',ms=4,alpha=.7,color=color[category],label=name[category]+' (n='+str(len(selected))+')')
        ax.axhline(0,color='black',ls='--',lw=.8);ax.set_title(title);ax.set_xlabel('Reliable gap-position coverage (% WT)');ax.grid(alpha=.15)
    axes[0].set_ylabel('RAG − original DelePref (score ×100)')
    handles,legend_labels=axes[0].get_legend_handles_labels();fig.legend(handles,legend_labels,loc='lower center',bbox_to_anchor=(.5,-.13),ncol=2,frameon=False,fontsize=9)
    fig.suptitle('Coverage is fixed from evidence; coincident points overlap',fontsize=11)
    fig.tight_layout();save(fig,'RAG_coverage_vs_effect')
    influence=[];group=[]
    for endpoint in endpoints:
        lower=np.array([r['differences'][endpoint]['lower'] for r in rows]);upper=np.array([r['differences'][endpoint]['upper'] for r in rows])
        mid=(lower+upper)/2;index=int(np.argmax(np.abs(mid)))
        influence.append(dict(endpoint=endpoint,largest_absolute_effect_construct=rows[index]['protein_id'],
            largest_case_lower=float(lower[index]),largest_case_upper=float(upper[index]),
            fraction_of_net_midpoint_sum=float(mid[index]/mid.sum()) if mid.sum() else None,
            posthoc_influence_diagnostic_not_a_reselected_primary_analysis=True))
        for category in categories:
            selected=[r for r in rows if r['participation']==category]
            group.append(dict(endpoint=endpoint,participation=category,N=len(selected),
                mean_difference_lower=float(np.mean([r['differences'][endpoint]['lower'] for r in selected])) if selected else None,
                mean_difference_upper=float(np.mean([r['differences'][endpoint]['upper'] for r in selected])) if selected else None))
    write(out/'influence_diagnostic.json',influence);write(out/'participation_group_effects.json',group)
    table={(r['method'],r['endpoint']):r for r in absolute}
    names={'WT':'WT预测校准','delepref_unlimited':'原DelePref（K8，同Scorer）','rag_unlimited':'DelePref-RAG（K8，同Scorer）',
        'scisor_native':'SCISOR U90-S（原生K1）','raygun_native':'Raygun（原生500→50→1）','terminal':'Terminal（K8，同Scorer）'}
    def number(r):
        lo,hi=r['macro_lower'],r['macro_upper']
        if lo is None or hi is None:return '未定义'
        return f'{lo:.3f}' if abs(lo-hi)<1e-8 else f'[{lo:.3f}, {hi:.3f}]'
    text=['# 通用44构建、20%删除预算的完整主面板读数','',
        '这是固定开发面板的主预测器结构读数，全部44构建和五种方法均保留，另有WT校准；应用、第二预测器和其余预算/机制结构汇总尚未完成。沿用冻结权重、输入、同源来源、候选、选择器和统计口径，没有据结果改模型或重选样本。',
        '', '| 方法 | Retained Cα-lDDT | 原生接触保持 | Full-WT TM |','|---|---:|---:|---:|']
    for method in ['WT','delepref_unlimited','rag_unlimited','scisor_native','raygun_native','terminal']:
        text.append('| '+names[method]+' | '+' | '.join(number(table[(method,e)]) for e in endpoints)+' |')
    text += ['', '表内区间是坐标/对应未知界，不是跨蛋白置信区间。不同方法的原生搜索预算不同，不能当作等计算量比较。Raygun的对应歧义使完整保留lDDT无法给出有信息的界，稳定对应子集几何只作诊断；不能把[0,1]解释为质量0，也不能从当前口径宣称完整可比的Raygun胜负。',
        '', '## RAG相对原DelePref的配对变化','',
        '| 指标 | 全44例平均差值 | 95%组件bootstrap及未知界包络 | 改善/退化/数值不变/未知 |',
        '|---|---:|---:|---:|']
    for r in summaries:
        if r['method']!='rag_unlimited':continue
        lo,hi=r['construct_macro_difference_lower'],r['construct_macro_difference_upper'];ci=r['uncertainty_and_bootstrap_interval95']
        point=f'{lo:+.4f}' if abs(lo-hi)<1e-8 else f'[{lo:+.4f}, {hi:+.4f}]'
        text.append(f"| {titles[endpoints.index(r['endpoint'])]} | {point} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {r['confirmed_improved']}/{r['confirmed_degraded']}/{r['numerically_unchanged']}/{r['unresolved']} |")
    text += ['', '两项局部指标的全队列平均差值为正，当前配对区间也高于零；TM方向偏正但保守区间仍跨零。10个改变决策的病例中，局部lDDT为7升3降，接触为8升2降。其余34个删除决策完全相同。TM对边际未知界保守相减，未利用相同物理输出/映射的相关性，因此“区间未知数”不能代替“改变决策数”。这些区间是当前面板的描述性不确定性，不是独立新确认或功能证明。',
        '', '收益分布不均：唯一直接天然提案贡献约62%的lDDT净增量、52%的接触净增量；补全提案同时有改善和退化。影响度分析是事后诊断，不删除该病例、不替换全44主分析，也不能概括为所有蛋白都获益。',
        '', 'Terminal在这个20%通用面板的三项结构指标均较高，所以当前结果不支持RAG在通用结构保持上总体最优。其他预算和界面/催化区域需要单列完整结果；不能用尚未完成的应用结果替当前表解释优势。SCISOR-K8共同Scorer及机制对照另有固定20例，不能拿其未汇总结果补主表。',
        '', '所有结构值均为预测代理。原Scorer仍在超过3段的方案上使用，属于训练范围外外推。随机化同源事件有效扰动仅2/20、min8有效0/10，不能据此声称联合机制或稀缺鲁棒性成立。',
        '', '图和逐例数据保留完整44例：RAG参与堆叠图、三指标逐例差值图、预定证据覆盖散点图。散点中的覆盖不等于同源数量，也不预设单调关系。论文未修改。',
        '', '源数据：`../completion.json`及其逐文件SHA；freeze SHA `'+done['freeze_sha256']+'`。']
    (out/'primary_result_zh.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    report=dict(status='ALL_PRIMARY44_FIGURES_AND_BALANCED_READOUT_EXPORTED',N=44,source_completion_sha256=sha(base/'completion.json'),
        source_script_sha256=sha(Path(__file__)),new_model_forwards=0,new_structure_predictions=0,
        files={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    write(out/'completion.json',report);print(json.dumps(dict(status=report['status'],files=len(report['files']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
    p.add_argument('--output-name',default='presentation');a=p.parse_args();build(a.root,a.output_name)
