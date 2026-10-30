"""Add complete fairness/source diagnostics, preserving the generated report."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import zipfile

from recomb_scope_screen_20261017 import read,sha,write


def build(root):
    out=root/'reports/recomb_benign_final_report_20261017';done=read(out/'completion.json')
    if done.get('revision',1)!=1:raise ValueError('report already supplemented')
    for p,h in done['files'].items():
        if sha(out/p)!=h:raise ValueError('generated report changed')
    base=root/'reports/recomb_benign_final_diagnostics_v2_20261017';d=read(base/'completion.json')
    for p,h in d['outputs'].items():
        if sha(base/p)!=h:raise ValueError('diagnostic changed')
    history=out/'history';history.mkdir()
    with zipfile.ZipFile(history/'base_generated_report.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in done['files']:z.write(out/p,p)
        z.write(out/'completion.json','completion.json')
        z.write(root/'analysis/recomb_benign_final_report_20261017.py','source/recomb_benign_final_report_20261017.py')
    controls=read(base/'SCISOR_K8_same_scorer20.json')
    with (out/'07_SCISOR_K8_control20.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(controls[0]));w.writeheader();w.writerows(controls)
    shutil.copyfile(base/'training_homology44.csv',out/'08_training_homology44.csv')
    report=(out/'README_zh.md').read_text(encoding='utf-8')
    algorithm='实际在线流程：WT只编码一次，正式训练Policy产生删除倾向与多样化原提案；同源比对中的可信内部缺失形成完整天然提案或must-delete/保留锚点，缺少的预算由同一Policy精确补全。最多4个原提案和4个检索提案去重混合并补足K8，原完整方案Scorer选择最终输出；没有可用证据时原K8路径精确回退。结构预测只评价已选终输出，不参与候选后验选择。这里不是冷启动随机方案加teacher U，也没有重训残差评分器。\n\n下表“DelePref”指原权重在当前无限段数适配器下的基线；这是用户更改任务约束后的公平同适配器比较，不冒称原论文原生≤3段系统的数值复现。'
    report=report.replace('所有当前方法不限删除段数。',algorithm+'\n\n所有当前方法不限删除段数。',1)
    report+='\n## 9. 来源、训练暴露与匹配K8补充\n\n'
    report+=f"44构建长度{d['length_min']}–{d['length_max']}aa；来源为29PDB/15CATH，真核21、细菌13、古菌10。既有输入注释有{d['unique_exact_functional_maps']}例唯一精确功能映射；未用检索成功或新结构表现补样。\n\n"
    report+='| 方法（同20例20%） | 局部Cα-lDDT | 原生接触 | Full-WT TM界 |\n|---|---:|---:|---:|\n'
    for r in controls:
        report+=f"| {r['method']} | {r['CA_lddt']:.4f} | {r['native_contact_recovery']:.4f} | [{r['full_WT_TM_lower']:.4f}, {r['full_WT_TM_upper']:.4f}] |\n"
    report+='\n这是原生K1与K8共同Scorer的补充控制，不以较有利版本替代主表。当前Scorer未让SCISOR提案达到原Policy水平；这不是所有SCISOR模型/采样规模的最优性能证明。完整逐值表见07_SCISOR_K8_control20.csv。\n\n'
    common=d['common_retained_relationship_effect'];ci=common['uncertainty_and_bootstrap_interval95']
    report+=f"在RAG/原模型共同保留的完全相同原生关系上，四阈值距离保持诊断的平均差值为{common['construct_macro_difference_lower']:+.5f}，区间[{ci[0]:+.5f},{ci[1]:+.5f}]，8升2降34不变。它支持部分收益不只是改变保留集合的表面效应，但仍是结构代理诊断。逐例共同关系及静态删除损失均保存在final_diagnostics_v2目录。\n\n"
    spans=d['span_greater3_by_method']
    report+=f"原DelePref {spans['delepref_unlimited']}/44、RAG {spans['rag_unlimited']}/44、SCISOR {spans['scisor_native']}/44主输出超过原3段训练范围，Terminal为0/44。必须披露原模型大量外推；唯一直接天然提案的大增量同时伴随段数形态变化，不能把所有收益都解释成新的联合学习机制。\n\n"
    report+='训练序列审计复用了269915WT参考和预先固定MMseqs检索：排除identity_shorter_full≥0.4且双向覆盖均≥0.8的命中，44例均无该配置下阈值命中。**这不是严格家族隔离证明**：其中7例最近返回命中对查询覆盖高、对较长训练序列覆盖低，仍存在域级同源暴露。搜索是近似最近返回结果，不是全对全或远同源穷尽证明；逐例identity/双向coverage/命中数在08_training_homology44.csv，不删掉这些病例来改善结果。自然同源接触也不能直接等同实验标签泄漏。\n\n'
    report+='补充CPU诊断首版1118015因服务器缺少既有输入覆盖元数据副本，在任何模型调用前失败；v2只同步已存在本地元数据的正向44例过滤版本后完成，旧源码/失败日志保留。没有新检索、训练、生成或结构重跑，也没有更改已冻结指标。\n'
    (out/'README_zh.md').write_text(report,encoding='utf-8')
    done.update(revision=2,generated_source_sha256=done['source_script_sha256'],supplement_source_sha256=sha(Path(__file__)),
        diagnostics_completion_sha256=sha(base/'completion.json'),original_generated_report_preserved=True,
        files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='completion.json'})
    write(out/'completion.json',done);print(json.dumps(dict(status=done['status'],revision=2,files=len(done['files']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
    build(p.parse_args().root)
