"""Readable Chinese summary of all fixed supplementary conditions; no new inference."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent.parent
base=root/'reports/recomb_confirm_v2_General100_ESM_supplement'
def read(name):return json.loads((base/name).read_text(encoding='utf-8'))
done=read('completion.json')
for name,h in done['outputs'].items():
    if hashlib.sha256((base/name).read_bytes()).hexdigest()!=h:raise ValueError('Supplement output SHA differs: '+name)
labels={'structure_mapped_CA_lddt':'局部Cα几何','structure_mapped_native_contact_recovery':'原生接触恢复','full_WT_TM':'Full-WT TM'}
endpoints=list(labels)
def number(value):return '未定义' if value is None else f'{value:+.5f}'
def cell(r):return number(r.get('mean_difference'))+' ['+', '.join(number(x) for x in r.get('CI95',[]))+']'
lines=['# General100 预算与机制实验结果','',
    'General100的全部1864条ESMFold逻辑记录、1192个实际结构均已完成，无失败。以下是预定次要与描述性分析；主20%全100例结论保持不变。第二预测器仍在运行，酶80与Binder60尚无完整结构结论。','',
    '12例完整池显示：RAG在2例上提供了更高的池内最优局部几何和接触，但最终选中结果为0升、1降、11不变。更好的候选未必被原Scorer选中，是当前瓶颈的线索；这不是修改本轮模型或选择器的依据，也不能从12例外推全体。','',
    '下列差值均按同一蛋白配对；数值为均值差及普通95%组件bootstrap区间，10000次、seed20261016。次要分析没有进行跨预算、控制和指标的多重校正，不能挑其中一个阳性格子替代主结果。','',
    '## 固定40例预算曲线','',
    '四个预算使用同一40例、同一六方法；下表列RAG减原DelePref。该40例在20%下的结果不是全100例主表，不能混用分母。','',
    '| 删除预算 | 局部Cα几何差 [95%CI] | 原生接触差 [95%CI] | Full-WT TM差 [95%CI] |',
    '|---|---|---|---|']
budget=read('budget_comparisons.json')
for beta in [.1,.2,.3,.4]:
    selected={r['endpoint']:r for r in budget if r['method']=='rag_unlimited' and r['budget']==beta}
    lines.append('| '+str(round(beta*100))+'% | '+' | '.join(cell(selected[e]) for e in endpoints)+' |')
lines += ['',
    '10%、20%、30%下的RAG差值区间均跨零。40%下局部几何的普通95%区间为正，原生接触仍跨零；这是次要观察，不能称为跨预算一致增益。全部六方法的逐预算均值在 budget_table.json，未省略不利对照。','',
    '## 固定40例机制控制','',
    '下表是控制减完整RAG，负值表示控制较差。形态匹配在39/40例有有效输入干预，联合关系在14/40例有效；有效输入不等于最后选择变化。','',
    '| 控制 | 全队列N | 相对RAG最终序列改变数 | 局部Cα几何差 [95%CI] | 原生接触差 [95%CI] | Full-WT TM差 [95%CI] |',
    '|---|---:|---:|---|---|---|']
rows=read('per_output_metrics.json')
by={(r['protein_id'],r['method'],r['budget']):r for r in rows if r['evaluation_role']!='full_pool'}
mech=read('mechanism_comparisons.json')
names={'rag_morphology_randomized':'形态匹配随机位置','rag_joint_null':'联合关系扰动','rag_direct_only':'无Policy补全','rag_no_scorer':'去Scorer'}
changed={}
for method,title in names.items():
    group=[r for r in rows if r['method']==method]
    changed[method]=sum(r['edited_sequence_sha256']!=by[r['protein_id'],'rag_unlimited',.2]['edited_sequence_sha256'] for r in group)
    selected={r['endpoint']:r for r in mech if r['method']==method and r['cohort']=='all_predefined40'}
    lines.append('| '+title+' | 40 | '+str(changed[method])+' | '+' | '.join(cell(selected[e]) for e in endpoints)+' |')
lines += ['',
    '联合扰动最终仅改变2/40例，14例有效输入子集的三项差值区间也均跨零，目前没有联合事件优势的支持。无补全的接触下降在普通95%区间下可见，但它是多项次要比较之一；不能据此宣称所有机制已成立。有效39/14例的独立统计均保存在 mechanism_comparisons.json。','',
    '## 固定20例证据稀缺','',
    '下表为减少证据后的方案减全证据RAG；50%、25%、0%是输入哈希规则下的嵌套证据子集。','',
    '| 证据保留比例 | 相对全证据序列改变数 | 局部Cα几何差 [95%CI] | 原生接触差 [95%CI] | Full-WT TM差 [95%CI] |',
    '|---|---:|---|---|---|']
scarcity=read('evidence_scarcity_comparisons.json')
for frac in [50,25,0]:
    method='rag_evidence_'+str(frac);group=[r for r in rows if r['method']==method]
    selected={r['endpoint']:r for r in scarcity if r['method']==method}
    n=sum(r['edited_sequence_sha256']!=by[r['protein_id'],'rag_unlimited',.2]['edited_sequence_sha256'] for r in group)
    changed[method]=n
    lines.append('| '+str(frac)+'% | '+str(n)+' /20 | '+' | '.join(cell(selected[e]) for e in endpoints)+' |')
lines += ['',
    '零证据20例均已核验精确退回原DelePref。回退正确属于行为验证，不能单独作为结构鲁棒性证据，也不预设随着证据减少质量单调下降。','',
    '## 固定12例完整候选池','',
    '每例同时预测原模型K8和RAG K8的完整并集，保留重复方案的实际身份。池内最好值是诊断上界，未用来选择最终输出；每个指标分别计算，没有临时综合分。','',
    '| RAG池减原池 | 局部Cα几何差 [95%CI] | 原生接触差 [95%CI] | Full-WT TM差 [95%CI] |',
    '|---|---|---|---|']
pool=read('full_pool_comparisons.json')
for field,title in [('best','池内最好值'),('selected','固定Scorer选中值'),('regret','最好值减选中值的差距')]:
    selected={r['endpoint']:r for r in pool if r['field']==field}
    lines.append('| '+title+' | '+' | '.join(cell(selected[e]) for e in endpoints)+' |')
lines += ['',
    '局部几何和接触的池内最好值均为2升、0降、10不变；最终选中值均为0升、1降、11不变。局部几何/接触的选优差距在3例增大。池内最好值与差距的95%下界可到0，样本仅12例，应作为瓶颈线索。','',
    '把原赢家保留在K8的配额控制在全部12例上均未改变三项结构指标，因此本子集没有显示“恢复被挤掉的原赢家”能解决当前选优问题。该控制没有替换主4+4方法。','',
    '## 后续实验状态','',
    'General第二预测器Boltz-2按固定30例和三个seed继续运行；全部seed在WT内部先汇总。应用Raygun补完后，既有CPU依赖会核验六方法、构建完整链/化学/MSA输入并逐项解析。应用折叠仍需最终指标、执行器和统计整包冻结。论文未修改。','',
    '全部记录、失败分母与来源见本目录JSON文件；共同结构几何与原始编辑身份两层都已保留。所有结果均为计算结构代理，不是活性、稳定性或结合实测。','']
dest=base/'supplement_result_zh.md';dest.write_text('\n'.join(lines),encoding='utf-8')
report=dict(status='READABLE_ALL_SUPPLEMENT_RESULTS_FROM_VERIFIED_OUTPUTS',
    completion_sha256=hashlib.sha256((base/'completion.json').read_bytes()).hexdigest(),
    report_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),control_sequence_changes_vs_RAG=changed,new_model_calls=0,new_predictions=0)
(base/'presentation_completion.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(report)
