"""Paper-style tables, complete-round figures and an evidence-limited decision."""
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

from recomb_scope_screen_20261017 import read,sha,write

METHODS=['delepref_unlimited','rag_unlimited','terminal','scisor_native','raygun_native']
LABELS={'WT':'WT calibration','delepref_unlimited':'DelePref (K8)','rag_unlimited':'DelePref-RAG (K8)',
        'terminal':'Terminal (K8)','scisor_native':'SCISOR U90-S (native K1)','raygun_native':'Raygun (native500/50/top1)',
        'rag_position_gap_frequency':'Position gap frequency','rag_marginal_gap':'Marginal event bundles',
        'rag_cooccurrence_null':'Matched event randomization','rag_direct_only':'Direct-only retrieval',
        'rag_no_scorer':'Same RAG pool, Policy selector','scisor_K8_same_scorer':'SCISOR K8 + original Scorer'}
ENDPOINTS=['retained_ca_lddt','contact_recovery','full_WT_TM']
TITLES=['Retained CA-lDDT','Native-contact recovery','Full-WT fixed-correspondence TM']


def bounds(row,endpoint):
    if endpoint=='full_WT_TM':return row['geometry']['full_WT_TM']['lower'],row['geometry']['full_WT_TM']['upper']
    block='interface' if endpoint.startswith('interface_') else 'catalytic' if endpoint.startswith('catalytic_') else 'geometry'
    key='contact_recovery' if endpoint=='interface_contact_recovery' else endpoint
    return row[block].get(key+'_lower'),row[block].get(key+'_upper')


def average(rows,endpoint):
    intervals=[bounds(r,endpoint) for r in rows]
    if not rows or any(x is None or y is None for x,y in intervals):return None,None
    return float(np.mean([x for x,_ in intervals])),float(np.mean([y for _,y in intervals]))


def fmt(lo,hi,precision=4):
    if lo is None or hi is None:return '未定义'
    return f'{lo:.{precision}f}' if abs(lo-hi)<1e-8 else f'[{lo:.{precision}f}, {hi:.{precision}f}]'


def md_table(header,rows):
    return ['| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows]


def build(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=root/'reports/recomb_benign_final_report_20261017'
    if out.exists():raise FileExistsError('final report exists; preserve it')
    full=root/'reports/recomb_benign_structure_evaluation_20261017';done=read(full/'completion.json')
    for n,h in done['outputs'].items():
        if sha(full/n)!=h:raise ValueError('final metrics changed')
    qa=root/'reports/recomb_benign_final_quality_20261017';read(qa/'completion.json')
    identity_dir=root/'reports/recomb_benign_catalytic_identity_20261017';read(identity_dir/'completion.json')
    rows=read(full/'per_output_metrics.json');paired=read(full/'paired_comparisons.json')
    general=root/'reports/recomb_benign_general_readout_20261017';curve=read(general/'absolute_metrics.json')
    contrasts_dir=root/'reports/recomb_benign_mechanism_contrasts_20261017';contrasts=read(contrasts_dir/'paired_contrasts.json')
    identity={r['method']:r for r in read(identity_dir/'summary.json')}
    cost=read(root/'reports/recomb_benign_efficiency_20261017/computation_accounting.json')
    if len(rows)!=867 or done['failures_or_missing']!=0 or cost['structure']['actual_new_calls']!=419:raise ValueError('unexpected completed roster/calls')
    out.mkdir();(out/'figures').mkdir();tables={}
    def group(cohort,profile,role=None):
        return [r for r in rows if r['cohort']==cohort and r['profile_name']==profile and (role is None or r['evaluation_role']==role)]
    primary=[r for r in group('general','esmfold') if r['budget']==.2 and r['evaluation_role']=='comparison' or r['method']=='WT']
    table=[]
    for method in ['WT']+METHODS:
        selected=[r for r in primary if r['method']==method]
        if len(selected)!=44:raise ValueError('primary table roster mismatch')
        table.append(dict(method=LABELS[method],N=44,**{e:fmt(*average(selected,e)) for e in ENDPOINTS}))
    tables['01_general44_primary']=table
    table=[]
    for beta in [.1,.2,.3,.4]:
        for method in METHODS:
            selected=[r for r in curve if r['experiment']=='general_budget_curve' and r['method']==method and r['budget']==beta]
            table.append(dict(budget=f'{beta:.0%}',method=LABELS[method],N=20,
                **{e:fmt(next(r['macro_lower'] for r in selected if r['endpoint']==e),next(r['macro_upper'] for r in selected if r['endpoint']==e)) for e in ENDPOINTS}))
    tables['02_budget20_all_methods']=table
    table=[]
    for right in ['rag_position_gap_frequency','rag_marginal_gap','rag_cooccurrence_null','rag_direct_only','rag_no_scorer']:
        selected=[r for r in contrasts if r['left_method']=='rag_unlimited' and r['right_method']==right]
        record=dict(contrast='Full RAG minus '+LABELS[right],N=20)
        for endpoint in ENDPOINTS:
            r=next(r for r in selected if r['endpoint']==endpoint)
            record[endpoint]=fmt(r['construct_macro_difference_lower'],r['construct_macro_difference_upper'])
            record[endpoint+'_CI95']=fmt(*r['uncertainty_and_bootstrap_interval95'])
        table.append(record)
    tables['03_mechanism20_direct_pairs']=table
    for cohort,profile,key,expected in [('binding_interface','boltz_binding','04_binding12',12),('enzyme','boltz_enzyme','05_enzyme11',11),('general','boltz_general','06_second_predictor8',8)]:
        table=[]
        for method in ['WT','delepref_unlimited','rag_unlimited','scisor_native','raygun_native']+(['terminal'] if cohort=='general' else []):
            selected=[r for r in group(cohort,profile) if r['method']==method]
            if len(selected)!=expected:raise ValueError('application/second table roster mismatch')
            record=dict(method=LABELS[method],N=expected,**{e:fmt(*average(selected,e)) for e in ENDPOINTS})
            if cohort=='binding_interface':
                for e in ['interface_contact_recovery','interface_jaccard']:record[e]=fmt(*average(selected,e))
            if cohort=='enzyme':
                record['catalytic_neighbor_contact_recovery']=fmt(*average(selected,'catalytic_neighbor_contact_recovery'))
                ident=identity[method];record['all_catalytic_identities_case_fraction']=fmt(ident['all_sites_identity_retained_case_fraction_lower'],ident['all_sites_identity_retained_case_fraction_upper'])
                n=sum(r['catalytic']['catalytic_pair_count'] for r in selected)
                total=sum(r['catalytic']['annotated_catalytic_sites']*(r['catalytic']['annotated_catalytic_sites']-1)//2 for r in selected)
                record['catalytic_CA_pair_coverage']=f'{n}/{total}'
                record['conditional_CA_pair_distance_MAE_A']=f"{sum((r['catalytic']['catalytic_pair_CA_distance_MAE_A'] or 0)*r['catalytic']['catalytic_pair_count'] for r in selected)/n:.3f}" if n else '未定义'
            table.append(record)
        tables[key]=table
    for name,table in tables.items():
        with (out/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    write(out/'all_tables.json',tables)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    colors=dict(zip(METHODS,['#0072B2','#009E73','#E69F00','#CC79A7','#777777']))
    def save(fig,name):
        for suffix in ['png','svg']:fig.savefig(out/'figures'/(name+'.'+suffix),dpi=200,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.5))
    for ax,endpoint,title in zip(axes,ENDPOINTS,TITLES):
        for method in METHODS:
            data=sorted([r for r in curve if r['experiment']=='general_budget_curve' and r['method']==method and r['endpoint']==endpoint],key=lambda r:r['budget'])
            x=[100*r['budget'] for r in data];lo=[r['macro_lower'] for r in data];hi=[r['macro_upper'] for r in data]
            ax.plot(x,lo,'o-',color=colors[method],label=LABELS[method],lw=1.5,ms=3)
            if any(abs(a-b)>1e-8 for a,b in zip(lo,hi)):
                ax.plot(x,hi,':',color=colors[method],lw=1);ax.fill_between(x,lo,hi,color=colors[method],alpha=.08)
        ax.set_title(title);ax.set_ylim(0,1.02);ax.set_xticks([10,20,30,40]);ax.set_xlabel('Deletion budget (%)');ax.grid(alpha=.15)
    axes[0].set_ylabel('Score (fixed same20 constructs)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.17),ncol=3,frameon=False,fontsize=8)
    fig.suptitle('Budget curves: identical frozen20 at every budget; shaded areas are mapping/coordinate bounds',fontsize=10)
    fig.tight_layout();save(fig,'budget_curves20')
    fig,axes=plt.subplots(1,2,figsize=(11,3.6),sharey=True)
    controls=['rag_position_gap_frequency','rag_marginal_gap','rag_cooccurrence_null','rag_direct_only','rag_no_scorer']
    for ax,endpoint,title in zip(axes,ENDPOINTS[:2],TITLES[:2]):
        for i,right in enumerate(controls):
            r=next(r for r in contrasts if r['left_method']=='rag_unlimited' and r['right_method']==right and r['endpoint']==endpoint)
            point=100*(r['construct_macro_difference_lower']+r['construct_macro_difference_upper'])/2;lo,hi=[100*v for v in r['uncertainty_and_bootstrap_interval95']]
            ax.errorbar(point,i,xerr=[[max(0,point-lo)],[max(0,hi-point)]],fmt='o',color='#0072B2',capsize=3)
        ax.set_yticks(range(5),[LABELS[m] for m in controls]);ax.invert_yaxis();ax.axvline(0,color='black',ls='--',lw=.8);ax.set_title(title);ax.set_xlabel('Full RAG − control (score ×100)');ax.grid(axis='x',alpha=.15)
    fig.suptitle('Same fixed20 component pairs;95% component bootstrap. Randomization effective only2/20.',fontsize=10)
    fig.tight_layout();save(fig,'component_contrasts20')
    calibration=read(qa/'WT_calibration.json')
    ordered=[next(r for r in calibration if r['profile_name']==p) for p in ['esmfold','boltz_general','boltz_binding','boltz_enzyme']]
    fig,ax=plt.subplots(figsize=(8,3.6));x=np.arange(4)
    ax.bar(x-.16,[r['WT_retained_CA_lddt'] for r in ordered],.32,label='WT CA-lDDT',color='#0072B2')
    ax.bar(x+.16,[r['WT_native_contact_recovery'] for r in ordered],.32,label='WT native-contact recovery',color='#56B4E9')
    ax.axhline(1,color='black',ls='--',lw=.8);ax.set_ylim(0,1.07);ax.set_xticks(x,['ESMFold\nGeneral44','Boltz2\nGeneral8','Boltz2\nInterfaces12','Boltz2\nEnzymes11'])
    ax.set_ylabel('WT prediction vs native');ax.set_title('Frozen WT calibration: absolute Boltz enzyme/general fidelity is limited');ax.legend(frameon=False)
    fig.tight_layout();save(fig,'WT_calibration')
    body=['# RECOMB普通蛋白质缩小：本轮完整实验报告','',
        '本轮固定流程已执行完成：原4000步Policy/Scorer＋同源检索引导/补全/回退，SCISOR、Raygun和简单参考全部按既定名单运行。没有新增训练。419个新增结构全部成功、无失败或重试；232个精确兼容缓存复用，合计651个预测身份、867条逻辑评价。论文未修改。',
        '', '结论是有限且分场景的：RAG在通用面板带来局部结构/接触增量，第二预测器方向一致；结合界面接触有正向平均变化但区间含零，酶催化位点身份未改善。简单gap-frequency与完整RAG接近，联合事件优势没有得到独立支持。不能据此宣称普遍功能保持、所有蛋白改善或通用结构总体最优。',
        '', '## 1. 数据与方法','',
        '通用实际N=44（29PDB链、15CATH域）；固定预算/机制N=20（10+10），证据稀缺N=10，第二预测器N=8（4+4）。应用为12个结合界面病例、11个酶；不能继续标作General100/Binder20/Enzyme20的实际结果。原100中25排除、31待核验，待核验不等同有风险。全部输入和检索证据均经过正向来源核验，新增预测没有涉及范围外靶标或partner。',
        '', '所有当前方法不限删除段数。原Policy的3潜在槽和Scorer段数/3尺度保持原样，超过3段和高预算属于训练范围外使用。主方法K8使用同一个原Scorer；SCISOR U90-S原生K1、Raygun原生500→50→1另标计算协议，20例SCISOR-K8共同Scorer控制用于补充，不能冒称等搜索预算。',
        '', '## 2. 通用44：20%预算主表','']
    body+=md_table(['方法','N','Retained Cα-lDDT','原生接触保持','Full-WT TM'],[[r['method'],r['N']]+[r[e] for e in ENDPOINTS] for r in tables['01_general44_primary']])
    body+=['', '区间单元格是未知坐标/对应的界，不是跨蛋白置信区间。Raygun有大量最优编辑比对歧义，完整lDDT界为[0,1]，不能当作0或据此强行排名；TM未知界条件于已知CA拟合，不是未知结构重优化界。Terminal在此通用面板三指标均较高，结论不能写成RAG总体最优。',
        '', 'RAG与原模型相比：局部Cα-lDDT平均+0.02116，95%组件bootstrap[0.00209,0.05242]；接触保持+0.02008，[0.00444,0.04472]。10个改变决策的病例中分别7升3降、8升2降，另外34例决策相同。Full-WT TM差值界[0.01246,0.02016]，保守区间[-0.00263,0.04788]跨零。唯一直接天然提案贡献约62%的局部净增量、52%的接触净增量，结果集中而非普遍；所有负例和零差病例保留。',
        '', '## 3. 同一20例预算与机制','',
        '全部五方法、四预算见02_budget20_all_methods.csv；budget_curves20图使用完全相同的20例。RAG的局部/接触增量在20%和30%较明确，10%和40%仍不确定；Terminal在这20例的10%–40%三指标也均较高，因此不能预设高预算会反转该比较。',
        '', '下面直接比较完整RAG减对照，正值支持完整RAG；每项N=20，三个指标及全部区间在03_mechanism20_direct_pairs.csv。']
    body+=md_table(['对照','Δ局部lDDT','95%区间','Δ接触','95%区间'],[[r['contrast'],r['retained_ca_lddt'],r['retained_ca_lddt_CI95'],r['contact_recovery'],r['contact_recovery_CI95']] for r in tables['03_mechanism20_direct_pairs']])
    body+=['', '完整RAG相对简单位置gap没有可辨优势；相对定义的边际事件组合有小正差，但随机化有效输入仅2/20、输出变化仅1例，不能证明联合机制。补全和Scorer的接触增量为正，局部lDDT区间仍跨零。证据归零10例精确回退；min8并未减少任何一例的证据，不能当作稀缺鲁棒性成功。相同物理预测且同WT对应的对照差值精确为0，其余传播原未知界；主表保守差分未更改。',
        '', '## 4. 第二预测器8例','']
    body+=md_table(['方法','N','局部Cα-lDDT','原生接触','Full-WT TM'],[[r['method'],r['N']]+[r[e] for e in ENDPOINTS] for r in tables['06_second_predictor8']])
    body+=['', 'RAG相对原模型局部+0.01274，区间[0.00390,0.02519]；接触+0.02725，[0.00602,0.05559]；TM仍不确定。名单在预测前固定，未因主结果挑选。不过此空MSA快速Boltz配置的WT局部校准仅约0.534，正向配对差只能作为有限一致性证据。',
        '', '## 5. 结合界面12例','']
    body+=md_table(['方法','N','重原子原生接触恢复','界面Jaccard','局部Cα-lDDT'],[[r['method'],r['N'],r['interface_contact_recovery'],r['interface_jaccard'],r['retained_ca_lddt']] for r in tables['04_binding12']])
    body+=['', 'RAG比原模型界面接触平均+0.09342，组件区间[0,0.16815]；Jaccard+0.04831，[0,0.08696]。均为3升2降7不变，12病例对应8个输入组件。绝对接触恢复约0.260，WT预测校准也只有约0.421，不能称结合功能保持或把26%叫实验成功率。5个修正连续构建的旧复合物未复用，所有方法使用同一新构建和固定partner。',
        '', '## 6. 酶11例','']
    body+=md_table(['方法','N','完整催化身份保留比例','催化邻域接触','局部Cα-lDDT','可用催化CA配对/注释总配对'],[[r['method'],r['N'],r['all_catalytic_identities_case_fraction'],r['catalytic_neighbor_contact_recovery'],r['retained_ca_lddt'],r['catalytic_CA_pair_coverage']] for r in tables['05_enzyme11']])
    body+=['', '完整催化身份保留：原DelePref8/11，RAG7/11，SCISOR6/11；Raygun因歧义为[0,6/11]，另有4例存在已确定的催化残基替换。68个已核验位点全部纳入。冻结的all_catalytic_sites_retained字段实际描述位置/坐标保留；单独的催化身份审计检查真实编辑后残基，不能把替换当身份保留。原指标及预测文件没有修改。',
        '', '酶催化邻域接触差约+0.00035，区间[-0.01636,0.01741]，未见明确提升。当前Boltz酶WT局部lDDT约0.398、接触约0.469，绝对几何校准较弱。全部67个原生自比较为1、137个新Boltz CIF的独立CA解析一致，未发现坐标解析错误；低校准是当前预测证据的限制，不通过重跑配置追正结果。条件CA几何MAE必须与可用配对数和催化身份保留一起读，不把删掉位点后的缺失误作0误差。',
        '', '## 7. 成本、核验与可复现性','',
        f"新增预测419/419成功、0重试；历史758次保持，本项目历次实际调用1177。当前八分片GPU预留合计{cost['structure']['Slurm_reserved_GPU_hours']:.3f} A100卡时，始终最多2卡。651独立预测身份中的232个缓存逐一验证原成功账本和SHA，867逻辑记录全部保留。",'',
        '同60个额外预算条件的原生中位时间：SCISOR K1约0.890秒，Raygun500/50/top1约54.498秒，两者搜索量不同，不能改写成等计算量速度比。原/RAG混合CPU批次没有单独计时，未捏造RAG加速倍数；44例编码缓存来自既有100WT批次，未虚构独立44例冷启动测量。',
        '', '指标代码、两预测器资产和profile、候选、选择、原生映射及统计均在新预测前冻结。264条早期主读数及704条通用读数与完整867汇总数值一致。所有SHA核验、独立解析、逐病例指标、全部对照和计费账本均保留。没有完整候选池结构，因此没有oracle/regret结果，也没有把RAG覆盖率、Scorer分数或模型置信度当作功能验证。',
        '', '这是已观察过部分历史结果的固定开发面板，不冒称完全独立的新确认。区间为预定组件bootstrap的描述性结果及未知界传播，不做未声明的多重显著性保证。',
        '', '## 8. 本轮开发决定','',
        '保留已实现的检索引导生成、原Policy补全、原Scorer选择及精确回退，作为已跑通且有有限结构收益的系统；当前证据不足以支撑联合事件优势、普遍功能保持或通用结构最优。应用Terminal未在本轮固定名单内，不能据本轮数据证明其应用功能劣势。若继续修改研究，应先独立规划功能保护约束、预测器WT校准及可比较的Raygun评价，明确与本轮开发数据的关系；本轮不再改配置、增样本或追加预测追分。',
        '', '本轮全部预定实验和必要核验已完成；交付后暂停现有自动跟进。论文仍暂停，没有发布代码或投稿。']
    (out/'README_zh.md').write_text('\n'.join(body)+'\n',encoding='utf-8')
    inputs=[full/'completion.json',qa/'completion.json',general/'completion.json',contrasts_dir/'completion.json',identity_dir/'completion.json',root/'reports/recomb_benign_efficiency_20261017/completion.json']
    decision=dict(status='FIXED_EXPERIMENT_ROUND_COMPLETE_WITH_LIMITED_MIXED_EVIDENCE',
        pipeline_complete=True,paper_modified=False,new_training=False,new_structure_predictions_after419=False,
        supported=['limited local-geometry/contact gains vs original DelePref','directional local consistency on frozen second predictor subset','traceable proposals and exact fallback'],
        not_established=['joint-event advantage over position-gap reference','general structural superiority','biological function preservation','enzyme catalytic improvement','broad independent confirmation'],
        limitations=['effect concentration','Terminal stronger on fixed general budgets','Raygun mapping uncertainty','weak empty-MSA Boltz WT calibration','sparse effective mechanism interventions'],
        phase_should_pause_after_delivery=True,input_completion_sha256={str(p.relative_to(root)):sha(p) for p in inputs})
    write(out/'development_decision.json',decision)
    report=dict(status='FULL_REPORT_TABLES_AND_FIGURES_COMPLETE',logical_outputs=867,actual_new_structure_calls=419,physical_identities=651,
        all_predeclared_positive_negative_and_unknown_results_retained=True,source_script_sha256=sha(Path(__file__)),
        files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    write(out/'completion.json',report);print(json.dumps(dict(status=report['status'],files=len(report['files']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
    build(p.parse_args().root)
