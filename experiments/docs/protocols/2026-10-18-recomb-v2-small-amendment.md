# RECOMB v2 小修订：生成贡献、候选配额与双层评价

> For agentic workers: execute inline under the user's existing authorization. Existing jobs continue. No new thread, delegation, training, or edits to bound scientific sources.

**Goal:** 保持原v2计算和结果，补齐RAG生成贡献、等K候选对照与编辑身份视角，优先完成应用主实验。

**Architecture:** 原确认包、候选、模型、结构和统计只读保留；新增分析写入独立目录。General100结果已经可见，本修订的统计与新增K12实验均明确标为结果可见后的补充，不能倒写成原预设确认终点。新计算只补缺失身份。

**Tech Stack:** 现有Python/NumPy、原4000步Policy/Scorer、真实WT缓存、Slurm gpu_gua/gpu_gub、原ESMFold/Boltz配置。

日期：2026-10-18。用户最后明确“原有的那些跑的接着跑，后续有些东西小改一下”。本文覆盖后续分析与新增补充，不撤销原v2冻结实验；GPU无本任务固定卡数上限。论文继续暂停。

## 1. 执行顺序与不变项

1. 已运行及已排依赖继续，包括General完整评价1119265、应用WT数组1119685/1119705与WT评价1119708，不取消、不重提、不改绑定源。
2. 全140应用WT读数后，优先六方法20%主实验及主表；保留全部WT、失败和未知，不依据WT成绩换名单或参数。
3. CPU只读路径、配额审计及双层表可以并行进行，不阻塞主预测。现成预算、机制、稀缺、第二预测器、完整池与成本结果直接复用。
4. 新K12的GPU补充排在应用主实验之后；先复用现成候选，完成全100 CPU补充，再只对预先存在的General budget40子集补缺失K12赢家结构。原应用预算/重复继续按既定矩阵完成。
5. 原D1/E1–E8编号保持，新增用R1/R2/R3，不给旧job/表/结果改编号。Terminal保留；不新增靶标、种子搜索、训练、替代评分器或任意综合效果分。

General冻结SHA为7c42d417fea99f5aee5b90b2094c5a15b43532e78de2ce743555e72f55f6a04d；应用hardware-v3冻结SHA为518367be541ca621e2a690ca47a74ef51181e08ef7e5eac664fbba00d8d62814。已生成的同WT硬件类、完整链/化学/MSA、权重、seed不变。

## 2. R1：全100生成路径与已有候选配额审计

输入使用General100主20%、generation seed13，不混预算或CPU seed29：

- `reports/recomb_confirm_v2_General100_plans/{completion,generation_diagnostics,candidate_scores,all_selections}.json`
- `retrieval/recomb_confirm_v2_General100/profiles.json`
- `reports/recomb_confirm_v2_General100_ESM_supplement/per_output_metrics.json`
- 已完成 `reports/recomb_confirm_v2_RAG_measurement_audit/incumbent_and_decision_audit_20261018.json`

输出目录：`reports/recomb_confirm_v2_RAG_generation_path/`，包括逐WT CSV、完整JSON、来源SHA和中文解释；无新增模型或结构调用。

| 指标 | 固定分子／分母与边界 |
|---|---|
| Reviewed-homolog coverage | 至少一个正向审核代表的WT / 全100；与可靠内部缺失事件分开 |
| Reliable-event coverage | 至少一个满足现有锚点规则的内部事件WT / 全100；复用原natural_events规则，不调整门槛 |
| Proposal Rate | diagnostics.retrieved_unique>0的WT / 全100；提案已满足完整删除预算，包括Policy补全，不能都称完整天然删除 |
| Selection Rate | 原日志实际胜出候选generation_kind以retrieval开头的WT / 全100；同时报占有提案WT比例；直接、补全、refined分别列 |
| DCR_mask、DCR_sequence | 各自与原DelePref最终mask/序列不同的WT / 全100；两个率不合并 |
| Structural Gain | 每个端点分别报全100配对差、改善/退化/相同/未知；另列改变方案子集，不能用条件子集替代全体效果 |

按mask去重可能使提案来源标签归于先加入的Policy；“日志选中检索来源”不等于该mask只能由检索产生。额外列pool内检索mask是否与原Policy重合；不补写不存在的来源证据。

路径展示保留旁支：没有事件、有事件但无法生成提案、提案未选中、选中但mask/序列未变、选中或Policy改选后的质量。不能把Selection Rate、DCR、改善率串成没有交叉项的单条漏斗，不能丢掉未知病例。数值容差1e-8只用于符号分类，不是生物学成功阈值。

全100原赢家移出及放回反事实审计已经完成，直接引用，不重新生成候选或扩大原完整池结构。尚无完整候选结构者不得报告oracle/regret。

## 3. R2：等K12与保留原Policy8的候选控制

本项是新增探索性机制补充，不加入原六方法确认主表。CPU覆盖原General100；结构验证只使用原冻结 `General100_subsets/subsets.json` 的budget40名单（按原输入哈希确定，不按效果重选）。

### 3.1 固定候选规则

- 基准K8、RAG K8及原Scorer胜出者全部从旧日志读取，不重复生成或折叠。
- **Policy K12：** 必须保留原Policy K8的8个mask与原顺序。真实WT缓存、原Policy logits、condition_seed、perturbation_scale均沿用原配置；从该条件已记录的original_generation.attempts继续，attempt编号a的噪声seed仍为condition_seed+a×1000003，仅生成另外4个唯一mask。总尝试上限12×原max_attempts_multiplier（当前为192）。不以结构效果重试或换seed。
- **RAG K12：** 先保留原Policy8，再按原raw_candidate_rank加入现有RAG K8中带retrieval来源的提案（最多4个）；mask重复只算一次，再按顺序用Policy第9–12个补到12。在原前4个Policy中已被去重的检索mask已属于原Policy8，不需要重检索。
- 无可用检索提案时精确回退Policy K12。不能生成12个唯一mask的条件保存不足和原因，不换样、不改尝试上限、不隐藏失败。
- K按不同mask计，同时报告不同序列数；重复氨基酸可让不同mask产生同序列。已有每候选Scorer分数复用，只对新增mask调用原Scorer；排序严格沿用分数降序、删除区间字典序平局规则。Scorer原三槽、段数/3尺度不变，>3段继续披露外推。
- 检索、WT编码及原候选日志复用；新Policy/Scorer CPU调用、尝试数与耗时单独记账，不能称新增全部模型调用为0。

### 3.2 比较及正确解释

| 比较 | 可识别的效应 |
|---|---|
| 原K8 vs 原RAG K8 | 原冻结系统整体效果，已有结构复用 |
| Policy K12 vs RAG K12 | 同K、共享原Policy8时，新增检索提案相对新增Policy候选的差异 |
| RAG K8 vs RAG K12 | 扩容并保留全部原候选的组合效应；不能单独称为“配额挤出”的因果效应 |
| 原赢家保留K8 | 复用原E7固定12完整池控制；全100仅日志反事实，不把未折叠候选当已测质量 |

仅预测固定40例的两个K12最终赢家，上限80条逻辑新增请求；先与全部已有同序列/同profile/同权重/同MSA身份去重，实际调用以缺失catalog为准。无需预测全部K12候选。Pool Success@K、Selected Success@K不混用；本修订不自动启动K=1/2/4/8/16整条曲线。

嵌套候选池的最高Scorer分数不下降只是集合性质，不能作为结构质量或同源信息收益的证据；结论仍依靠实际赢家的配对结构结果。

后续实现采用独立 `analysis/recomb_confirm_v2_K12_supplement.py`、`configs/recomb_confirm_v2_K12_supplement_freeze.json` 与 `reports/recomb_confirm_v2_K12_supplement/`，不修改父生成器。放行前核验：旧8候选前缀完全一致、12唯一mask和精确预算、原8不被移除、去重来源及回退正确、无新PLM、旧分数与平局规则保持。先冻结代码/输入/候选/名单/统计及新赢家，再安排缺失结构。

## 4. 形态与三段外推

原E4形态匹配由可选说明提升为核心消融展示，直接复用已完成40例，其中39例有效匹配；匹配失败保留。它使用非同源位置、段长/末端类型匹配和同Policy补全，不冒称从条件Policy分布进行形态匹配。原D1、碎片化、段数/段长/接头数对照继续保留，不恢复主任务三段限制。

现有联合扰动有效14/40、最终改2/40，只能按实际有效干预解释；不能因缺乏增益另选病例、联合规则或评分器。

## 5. R3：双层评价与报告视角

- **原确认分析保留：** 运行/排队的原evaluator继续输出共同结构对应两主要端点及原95%/97.5%区间，原结论不覆盖。
- **后续删除任务重点表：** 对有明确原始编辑对应的方法，突出固定原位置的Native-contact recovery、Retained Cα局部几何、覆盖及静态关系损失；使用已有 `fixed_identity_geometry`，不重折叠。标注“查看General结果后的评价视角修订/补充”，不能称新盲测确认。
- **统一形状表：** Full-WT TM、结构对应Cα几何与接触分开给六方法，用于整体形状可比性，不能替代催化或界面原身份。
- **应用端点不变：** 酶完整催化AA身份为主要、邻域几何/接触为次要；Binder固定角色重原子<5Å原生界面接触为主要，F1和适用几何分量次要。自定义分量不称标准DockQ总分。Raygun歧义与缺失用既定界，不挑最佳对应，也不静默置0。
- 新双层表使用独立 `reports/recomb_confirm_v2_edit_identity_presentation/`，直接读取已有逐输出指标，组件bootstrap10000、seed20261016；缺失/未知保留。原结果与新补充并列，不能选有利的一层作为唯一结论。

## 6. 本次工作清单与持续状态

- [x] 阅读原v2、硬件修订、当前账本及源代码；确认现有作业继续、不重新启动4796条计划量。
- [x] 冻结本补充的指标、分母、K12规则、固定40名单来源、优先级和结果可见后分析标签。
- [x] 对原日志完成全100 R1路径表、逐WT数据及来源SHA；沿用已完成原赢家CPU审计。
- [x] 对已有结构指标输出R3固定编辑身份重点表，保留原共同结构对应表和结论。
- [ ] 应用主结构读数后按R2契约实现/冻结全100 CPU K12补充及固定40缺失赢家预测；不抢占应用主实验优先级。
- [ ] 原预算、三seed、第二预测器、应用完整报告和必要核验继续完成；不因本修订废弃已完成结果。

论文、公开代码、发布/投稿均保持暂停。新K12缺失调用另立账，不把4796当停止线，也不把它当必须重新运行的清单。
