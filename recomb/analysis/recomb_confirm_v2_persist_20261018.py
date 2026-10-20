"""Persist verified progress; do not change frozen scientific inputs."""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def write(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

snapshot = read('reports/recomb_confirm_v2_execution/live_snapshot_20261018_0555.json')
path = Path('reports/recomb_confirm_v2_execution/execution.json')
execution = read(path)
now = datetime.now(timezone(timedelta(hours=8))).isoformat()
status = 'GENERAL100_PRIMARY_COMPLETE_EXTENDED_PREDICTIONS_RUNNING_APPLICATION_PREPARATION'
execution.update(status=status, last_updated=now)
for name, job, state in [
    ('General100_ESMFold_main',1119261,'COMPLETED'),
    ('General100_ESMFold_extended',1119263,'RUNNING_ARRAY_THROTTLE1'),
    ('General100_primary_readout',1119269,'COMPLETED'),
    ('application_chemistry_compiler_v1',1119329,'FAILED_MISSING_IMPORT_NO_MODEL_CALL'),
    ('application_chemistry_compiler_v2',1119334,'COMPLETED'),
    ('application_context_WT_MSA',1119344,'COMPLETED'),
    ('application_context_schema_v1',1119346,'AUDIT_ERROR_WRONG_STRUCTUREV2_BOND_ATTRIBUTE_NO_MODEL_CALL'),
    ('application_context_schema_v2',1119369,'RUNNING_CPU')]:
    execution['jobs'][name] = dict(job_id=job, status=state)
execution['notification_flags'].update(
    General100_primary_complete_and_mixed_results_notified=True,
    application_chemistry_compiler_missing_import_and_repair_notified=True,
    application_WT_chemical_context_compile_complete_notified=True,
    application_native_Boltz2_bond_attribute_fix_notified=True)
primary = dict(status='COMPLETE_ALL100_WT_700_LOGICAL_545_PHYSICAL',job_id=1119269,
    logical_outputs=700, physical_predictions=545, missing_or_failed=0,
    RAG_minus_DelePref=dict(common_CA_local_geometry_mean=.0082217762,
        local_CI95=[-.0011084512,.0201935013], local_joint_CI975=[-.0022366088,.0221848598],
        native_contact_mean=.0127512721,contact_CI95=[.00084185045,.0284298916],
        contact_joint_CI975=[-.0001276895,.0309865472],
        FullWT_TM_mean=.0113093349,TM_CI95=[.0005413305,.0256694375],
        local_improved_degraded_unchanged=[13,10,77],contact_improved_degraded_unchanged=[11,12,77]),
    joint_confirmation_supported=False,not_a_functional_validation=True,
    no_model_or_selector_changed_after_readout=True,
    readable_report='reports/recomb_confirm_v2_General100_primary_presentation/primary_result_zh.md',
    table_sha256='202841a15d24bd724db6e8949783a44adf2ec167c1c5d6a1e254297b764fec8d',
    comparisons_sha256='ffe8f3b5e23c6d1631caa6cf7f48ca898a94c213c31e6311e2d09ed40b3f7dbf')
execution['General100_primary_result'] = primary
execution['General100_prediction'].update(status='PRIMARY_COMPLETE_EXTENDED_RUNNING',actual_snapshot=snapshot)
execution['structure_counts_snapshot'] = snapshot
execution['application_baseline_snapshot']['Raygun_completed_conditions'] = snapshot['app_Raygun_conditions']
execution['application_chemical_input_preparation'] = dict(
    compiler_v2_job=1119334,WT_cases=140,ligand_instances=240,covalent_connections=136,
    WT_context_sha256='09dfe6ad55cbf57f6b55b6d2d87ca7c4940177fdb067c612ff164713e583b7c0',
    real_WT_MSA_queries=231,reused_uniform_WT_queries=150,new_uniform_alignments=81,
    native_v1_job=1119346,native_v1_error='StructureV2 has no attribute connections',
    native_v1_all140_technical_errors_not_biological_unsupported=True,
    native_v2_job=1119369,native_v2_freeze_sha256='22b150e4ccd77c86ad219ce27b0458dbc8e4f7a9f0222ff6c9a3404e0b3b95bc',
    native_v2_bond_check='exact endpoint multiset of unified bonds excluding intra-residue CCD bonds',
    regression_tests_passed=3,MSAs_and_chemistry_inputs_unchanged=True,
    new_model_calls=0,new_structure_predictions=0,application_fold_allowed=False,
    native_CCD_leaving_atoms_are_an_explicit_representation_limit=True)
write(path,execution)
state = read('reports/current_recomb_state.json')
state.update(status=status,last_updated=now,active_GPU_jobs=[1119142,1119263],active_GPU_count=2,
    General100_primary_complete=True,General100_primary_result=primary,
    new_stage_structure_calls=snapshot['v2_actual_started_total'],
    cumulative_structure_calls=snapshot['cumulative_actual_started_total'],
    snapshot=dict(started=snapshot['v2_actual_started_total'],succeeded=160+snapshot['successful'],
        failed=snapshot['failed'],inflight=snapshot['inflight']),
    application_context_schema_v2_job=1119369,application_context_schema_v2_status='RUNNING_CPU',
    application_structure_prediction_released=False)
write('reports/current_recomb_state.json',state)
progress = Path('reports/recomb_confirm_v2_execution/progress_zh.md')
heading = '# 最新进展：General100主实验完成，预定双指标一致增益尚未获支持'
text = f'''{heading}

- 1119261主数组545个实际结构全部成功；CPU1119269读取全部100WT/700逻辑输出。RAG相对原模型局部Cα几何均值+0.00822，95%CI[-0.00111,+0.02019]；原生接触+0.01275，95%CI[+0.00084,+0.02843]。预定联合校正97.5%区间分别[-0.00224,+0.02218]和[-0.00013,+0.03099]，均跨零；不支持双主要终点一致增益的确认结论。TM是并列次要终点，不能拿它替换主要终点。
- 局部几何13升/10降/77不变，接触11升/12降/77不变；全队列和不利结果保留。RAG与简单gap均值接近；本组20%通用测试Terminal三项均值更高，不能据此推断应用功能优劣。中文报告在reports/recomb_confirm_v2_General100_primary_presentation/primary_result_zh.md。新主结果已通知。
- 额外预算/机制ESMFold1119263继续运行，固定第二预测器1119264及完整统计1119265保持原依赖，不因主结果改模型、候选或子集。GPU仍General一张+应用Raygun一张，总2张。应用Raygun本次快照{snapshot['app_Raygun_conditions']}/220，SCISOR-L220已完成。
- cu9快照{snapshot['timestamp']}：General实际启动{snapshot['actual_started']}、成功{snapshot['successful']}、进行{snapshot['inflight']}、失败{snapshot['failed']}；v2累计{snapshot['v2_actual_started_total']}、含旧历史1177累计{snapshot['cumulative_actual_started_total']}。该主机时钟约快599秒，保留来源，不按跨主机墙钟混排。
- 应用化学编译v1 CPU1119329因漏MMCIF2Dict导入失败，无模型调用；v2仅导入和版本路径修复，1119334成功。140WT构建得到240配体实例、136共价连接；原输入与化学规则不变。真实WT MSA1119344完成231唯一序列，150复用、81统一重比对，无新供体搜索。
- 原生接口CPU1119346的140例均在同一软件断言失败：StructureV2使用统一bonds表，没有connections属性；不能把这些当生物输入不支持。保留旧源/冻结/日志，v2改为逐共价端点多重集核验，3项回归在本机和cu9通过；CPU1119369运行，复用全部231已完成MSA。尚未放行应用折叠。标准CCD离去原子等表示局限明确披露，不把解析通过当化学/功能验证。
- 论文保持暂停，无新增训练，原4000步Policy/Scorer、RAG4+4与不限段数不变。

---

'''
old = progress.read_text(encoding='utf-8')
if not old.startswith(heading):
    progress.write_text(text+old,encoding='utf-8')
print(status)
