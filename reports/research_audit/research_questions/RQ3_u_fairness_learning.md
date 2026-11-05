# RQ3：公平监督是否比匹配的普通理解训练带来额外源端收益？

## 1. 研究问题（Research Question）

GroupDRO、反事实配对、证据说明和规范分布等监督，能否在任务正确性保持时减少人口相关误差？先区分普通任务学习与公平特有收益，再讨论U→G。

## 2. 实验设计（Experimental Setup）

本线包含CelebA属性的shared-prefix/校准隔离及两模型ERM-CDRO确认实验，Winogender纯文本共指，SWiG角色训练，旧A–E语义试验，以及occupation v1–v3文本证据/规范先验。它们测量对象不同，绝不合并平均。CelebA确认集512身份，每label×Male cell128人；最大条件失败差距为固定真实标签内两Male标签组错误率差的最大值。文本职业v3数据为作者合成文本，无输入图像。

## 3. 实际结果（Observed Results）

| CelebA模型/状态 | 总准确率% | 最大条件失败差距pp | 最差cell% |
|---|---:|---:|---:|
| BAGEL BASE | 91.02 | 1.56 | 89.06 |
| BAGEL ERM，seed29/30/31 | 91.60 / 91.80 / 91.80 | 1.56 / 3.12 / 0.78 | 90.62 / 88.28 / 90.62 |
| BAGEL CDRO，同seed | 91.21 / 91.21 / 91.21 | 3.12 / 3.91 / 4.69 | 89.06 / 87.50 / 86.72 |
| SenseNova BASE | 91.80 | 0.78 | 90.62 |
| SenseNova ERM，同seed | 92.38 / 92.58 / 92.77 | 4.69 / 3.12 / 3.91 | 88.28 / 91.41 / 90.62 |
| SenseNova CDRO，同seed | 92.77 / 92.77 / 92.58 | 3.91 / 3.12 / 4.69 | 89.84 / 91.41 / 89.84 |

seed29/30/31指20261029/30/31，不是推理噪声。来源：[源/目标完整JSON](../../../experiments/fairness_transfer_v2/summary.json)；本次表保持全部seed，不挑BAGEL ERM seed31的最低差距宣称稳定改善。G明确发色请求Qwen所有状态100%，接近天花板，不适合证明额外迁移。

| 其他源任务 | 普通/基座 | 公平/干预 | 观察 |
|---|---|---|---|
| 校准隔离CelebA | PLAIN BA91.41%，差距4.69pp | FAIR/ROUTED BA92.19%，差距6.25pp | 最弱task×group均84.38%，强组多对1题 |
| Winogender44留出职业，352题 | BASE217；ordinary两seed225/207 | prior-local207/212；random-local215/226 | 跨seed无稳定定位优势 |
| SWiG 60图 | BASE弱组80.00% | U-CE79.58%，U-CDRO79.17% | 源公平未改善 |
| SenseNova文本职业v3 test，256题 | BASE208；ordinary249 | parallel248，sequential249，pair250 | 普通训练解释大量正确率收益 |
| 同版heldout，256题 | BASE192；ordinary249 | parallel/sequential/pair均251 | 源任务泛化不等于规范公平 |

来源：[校准源分析](../../../experiments/calibration_split_fair_transfer_v1/source_analysis.json)、[Winogender及replication44](../../../experiments/source_fairness_winogender_v1/RESULTS.md)、[角色](../../../experiments/role_transfer_pilot_v1/source_analysis.json)、[职业阶段表](../../../experiments/occupation_transfer_v3/INTERIM_RESULTS.md)。职业表是首seed阶段可比状态，非最终全臂结果。

occupation v3实际有首seed九个完整训练/评价分支及route_q部分输出，训练通常128步；BAGEL未通过资格。[实际中止标记](../../../experiments/occupation_transfer_v3/main/sensenova/seed_20261031/status.json)为interrupted_for_official_reference_check，pipeline held；旧报告“正在运行”已过时。v1/v2不是额外独立seed，部分权重已在历史清理删除。


### 职业v3：按实际文件补齐全部已完成状态

以下为SenseNova训练seed20261031，采用各臂最终test/heldout及test_prior汇总；不是只沿用早期四臂阶段报告。事实问答分母每划分256，属性读取另256不混入。自然回答TV为人口配对的完整原生语义分布总变差，含Unknown/other/失败；64重复配对只有一个共同无职业证据事实簇，不作事实总体泛化区间。

| 状态 | test职业正确/256 | heldout职业正确/256 | test_prior原生语义TV | 自然分布语义覆盖 | 已生成/192 | Qwen J / RD | Phi J / RD |
|---|---:|---:|---:|---:|---:|---|---|
| BASE | 208 | 192 | 0.1211 | 72.41% | 192 | 0.0000 / 1.0000 | 0.0625 / 0.9138 |
| ordinary | 249 | 249 | 0.1191 | 61.13% | 192 | 0.0000 / 1.0000 | 0.0521 / 0.9242 |
| evidence_parallel | 248 | 251 | 0.1396 | 59.28% | 192 | 0.0000 / 1.0000 | 0.0521 / 0.9174 |
| evidence_sequential | 249 | 251 | 0.1309 | 62.40% | 192 | 0.0000 / 1.0000 | 0.0833 / 0.8854 |
| evidence_pair | 250 | 251 | 0.1299 | 63.04% | 192 | 0.0000 / 1.0000 | 0.0417 / 0.9341 |
| semantic_control | 247 | 251 | 0.1191 | 55.62% | 192 | 0.0000 / 1.0000 | 0.0729 / 0.8958 |
| semantic_forward | 250 | 251 | 0.1094 | 54.35% | 192 | 未评分 | 未评分 |
| semantic_reverse | 248 | 251 | 0.0977 | 56.79% | 192 | 未评分 | 未评分 |
| semantic_bidirectional | 250 | 249 | 0.1016 | 55.76% | 192 | 未评分 | 未评分 |
| route_kv | 209 | 203 | 0.1250 | 67.43% | 192 | 未评分 | 未评分 |
| route_q | 未完成 | 未完成 | 未完成 | 未完成 | 无ledger | 未评分 | 未评分 |

来源：[逐状态直接提取JSON](../evidence/occupation_v3_extracted.json)，底层同目录各状态test/heldout/test_prior/summary.json和review/qwen或phi/状态/summary.json。J按全部计划分母计算平衡合格产出，RD只在各职业合格二类子集定义；未评分不填零，零覆盖职业RD未定义，不能默认为均衡。

九个训练臂有完整源/生成文件，route_q仅部分评估且无G ledger；只有BASE与最先五训练臂各192图完成两评审。没有观察到已评状态的跨评审稳定增量。自然TV很小可能包含Unknown集中、有限采样和重复事实；候选质量与覆盖必须保留，不能只从更小TV推断职业公平重学习。

## 4. 判断与分析（Judgment & Analysis）

**观察事实：** 普通训练能提高任务表现；所测额外公平目标没有跨seed稳定超过匹配普通对照。部分硬差距变差，较弱组没有获益。文本职业增益主要包括弱/无证据答案策略，不能称视觉公平。

**合理解释：** 小基础差距、监督不匹配目标构念、输出格式和候选集支持质量，会限制或混淆源公平收益。校准隔离试验明确减少私有标量吸收公平梯度的机会，但仍无双端收益，故“只要换梯度路由就成功”的简单解释缺乏支持。

**尚未验证：** 所有公平监督都无效，或公平学习本质上比任务学习更难。不同数据、目标、参数范围不支持这样的总体结论。Winogender原小分区成功被大职业复核削弱，应保留两者，不能选择有利子集。

## 5. 结论（Conclusion）

本项目尚未建立一个可重复、相对强普通对照有稳定增量且保持效用的U公平源模型。这是迁移研究的首要缺口，不是所有训练都未学习。当前证据足以停止把已有CDRO、校准隔离和7–13层定位路线作为成功方法继续包装。

## 6. 下一步（Next Steps）

只优先验证一个清楚的源公平终点：图像证据充分时对人口编辑保持正确，事实变化时仍响应；使用真正匹配的普通监督对照和独立来源。若已冻结设计在此终点仍无增量，应停止扩展其G矩阵，保留负面结果。不得在测试后重选解析器、职业或检查点。
