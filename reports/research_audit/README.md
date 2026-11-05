# UMM Fairness：联合研究审计

审计日期：2026-11-03（北京时间）。覆盖 **52个顶层实验目录、docs原有36份文件**，按真实研究目的重建为 **8个研究问题**。目录数量不是独立实验数量；复用图片、BASE、模型检查点和噪声的诊断不重复计作独立证据。

当前最可靠的判断：**任务回答改善、人口公平改善、生成分布变化和跨端公平迁移必须分开。** VILA-U匹配实验表明，监督方向与肖像/工作提示共同决定G变化；U职业收益未形成稳定的人口公平源收益，暂不能声称发现公平被某条通路阻断。固定图片换读出和旧解析问题进一步说明，测量变化可以制造部分表面收益。

## 按问题阅读

| RQ | 问题 | 当前证据 |
|---|---|---|
| [RQ1](research_questions/RQ1_measurement.md) | 公平分数受测量方式影响多大？ | 固定图片换读出可改变/反转差距；有直接诊断 |
| [RQ2](research_questions/RQ2_source_qualification.md) | 数据与源任务有无公平学习机会？ | 可学视觉任务已建立，但职业真值/源差距仍不足 |
| [RQ3](research_questions/RQ3_u_fairness_learning.md) | 公平监督有无普通训练之外的U收益？ | 所测路线未见稳定增量，部分退化 |
| [RQ4](research_questions/RQ4_sft_transfer.md) | I→T/T→I监督、预算和场景如何影响两端？ | 匹配三seed结果完整；肖像/工作差异明显 |
| [RQ5](research_questions/RQ5_behavior_and_probability.md) | 少说属性是否信息丢失或更公平？ | 模板化与部分字段可读共存；真实面板亦已完成，标签/类别覆盖仍有限 |
| [RQ6](research_questions/RQ6_generation_and_codec.md) | CFG/视觉处理可否解释或改善偏差？ | 低CFG无可靠联合收益；codec群体信息损失未证实 |
| [RQ7](research_questions/RQ7_bidirectional_transfer.md) | 原生迁移是否双向有效？ | 有行为影响，尚无可靠双端公平协同 |
| [RQ8](research_questions/RQ8_pathway_causality.md) | 哪些共享路径对行为有因果作用？ | 部分功能干预成立，公平起源/完整链未建立 |

逻辑关系：RQ1测量资格与RQ2数据资格约束RQ3源公平；RQ4/RQ5解释监督学到了什么；RQ6区分生成设置作用；RQ7测对端；RQ8只能在相应有效行为上解释本次更新，不追溯唯一历史来源。

## 全部交付入口

- [研究脉络](research_overview.md)、[演变时间线](research_timeline.md)：为何不断产生版本。
- [实验清单](experiment_inventory.md)、[文档清单](docs_inventory.md)：逐项用途、执行状态、版本关系与保留决策。
- [结果总表](result_summary.md)、[设计—实验—证据矩阵](evidence_matrix.md)：区分Proposed/Executed/Observed/Interpreted。
- [冲突及核查](inconsistencies.md)、[整体研究判断](research_assessment.md)。
- [实验整理记录](cleanup_manifest.md)、[文档整理记录](docs_cleanup_manifest.md)。
- [验证范围及复算](validation.md)、[逐条复算JSON](evidence/recomputed_checks.json)、[运行记录CSV](evidence/run_records.csv)、[日志扫描CSV](evidence/log_audit.csv)、[状态记录CSV](evidence/status_records.csv)。
- [逐文件资产清单](evidence/filesystem_inventory.jsonl.gz)、[文本及哈希](evidence/file_inventory.json)、[原报告表格与行号](evidence/experiment_tables.md)、[文档原文定位](evidence/reading_index.md)。后两项是证据检索副本，原文件仍是权威记录。

## 如何使用证据

结果优先级：实际输入/配置及逐条输出 → 机器汇总 → 阶段报告 → 研究计划。完成标记只表示执行成功；机器一致只表示代理一致。旧Janus轨迹及部分权重已在历史清理删除，报告数表可保留研究史，但缺少底层证据时不作为可复算发现。面板开始/完成状态分别记录快照，不因整理停止或重启。

全部分析是已有材料的审计，没有启动训练、图像生成或新模型评审。322项主实验点估计/覆盖/导出核验及44项审计期间完成的真实面板计数核验通过，但未重做全部置信区间、全量图像人审或全部权重哈希。完整边界见validation。
