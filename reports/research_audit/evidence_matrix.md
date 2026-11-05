# 假设—设计—执行—证据矩阵

Proposed=计划/假设；Executed=实际调用/训练；Observed=结果文件事实；Interpreted=解释与适用边界。四者分栏，未执行不填零，不从失败数据构建推断方法无效。

| RQ/假设 | 设计文档（Proposed） | 对应实验（Executed） | 实际证据（Observed） | 当前判断（Interpreted） | 证据来源 |
|---|---|---|---|---|---|
| 1 自动公平分数反映图片变化 | benchmark、figure_contract | role benchmark / measurement / pilot | 固定BASE图换U读者差距37.5→25pp，总体22.92→16.67% | 直接评分混杂，不能称G去偏 | [证据](../../experiments/role_transfer_pilot_v1/RESULTS.md) |
| 1 旧B训练学到更公平语义 | training_redesign历史 | A–E三seed；事后首词敏感性 | SenseNova BASE122/128、ordinary123、fair相同 | 旧10.94pp差距含解析影响 | [证据](../../experiments/fairness_transfer_study/diagnostics/B_parser_all_seeds_20261031/README.md) |
| 2 共享前缀不能学视觉 | method_candidate、早期protocol | shared_prefix v1 / visual_dependence v2 | 普通SP-CE最终121/128、BA94.53%；无图50% | 简单不可学假设被当前开发结果反驳；原因未单独归因 | [证据](../../experiments/visual_dependence_v2/RESULTS.md) |
| 2 新职业图文对可直接训练 | sft_source/data/execution_design | BAGEL源图v1–v3资格/生成 | v1训练只有6对5职业，v2/v3中止 | 数据/配置失败，不是SFT公平失败 | [证据](../../experiments/understanding_generation_sft_v1/experiment_v1/LIVE_RESULTS.md) |
| 3 CDRO稳定超越普通U训练 | KTP-follow、training_redesign | fairness_transfer_v2，2模型3seed | 差距及最弱cell未稳定改善 | 本设置无稳定额外源收益 | [证据](../../experiments/fairness_transfer_v2/summary.json) |
| 3 证据/配对改善职业公平 | research_plan、职业v3协议 | SenseNova首seed部分完成 | ordinary已249/256；额外监督小差异 | 一般答题进步，不足以称规范公平重学习 | [证据](../../experiments/occupation_transfer_v3/INTERIM_RESULTS.md) |
| 4 U学公平但G不接收 | sft_plan / paper_audit | paper_sft复现+followup匹配 | 职业提升，编辑公平变化区间含零 | 前提未成立，不能定位G阻断 | [证据](../../experiments/u_g_transfer_followup_v1/matched_evaluation_v1/paired_results.json) |
| 4 U/G差异只来自轮数 | followup_controls、matched PROTOCOL | 6训练、3seed、共同曝光/文本 | G肖像RD下降而工作RD上升，方向差异仍在 | 轮数不解释全部；token/目标难度仍不同 | [证据](../../experiments/u_g_transfer_followup_v1/matched_evaluation_v1/generation_results.json) |
| 5 少说人口词就是公平 | sft_plan与诊断设计 | description/attribute/object_action | 44/44固定FEMALE与帽饰31/36共存；一般细节减少 | 不成立；模板/输出策略混杂 | [证据](../../experiments/attribute_probe_v1/reviewed_results.json) |
| 5 输出恒定说明图像分数无信息 | probability diagnostic design | 旧22图概率复核 | 小参考上AUC1.0，采样/排序题式敏感 | 恒答不等于无输入敏感性；不能证内部知识完整 | [证据](../../experiments/answer_probability_diagnostic_v1/results.json) |
| 5 真实照片题式与格式效应 | probability design / analysis_contract | 48图、两状态各2352条件均完成 | 呈现词严格一致501→577/1152，辅助语义1092→838；字母题改善而顺序敏感 | 格式变化可反转解释；非身份真值，帽饰阳性仅1图 | [真实结果](../../experiments/answer_probability_real_panel_v1/results.json) |
| 5 信息范围与多样性可修复模板化 | i2t_supervision_factorial_design | 无四臂训练 | 无结果 | Proposed，未验证 | [证据](../../docs/i2t_supervision_factorial_design_20261102.md) |
| 6 降CFG改善公平且保持效用 | umm_bias protocol | 720图/双评审 | 合格/明确联合完成减少，J增量区间未确立 | 当前不支持作为去偏方法 | [证据](../../experiments/umm_bias_bagel_sensenova_v1/REPORT.md) |
| 6 codec造成群体信息损失 | umm_bias_research_plan | 96真实开发图恢复 | 整体失真组差区间含零；ROI未标 | 证据不足，不排除局部作用 | [证据](../../experiments/visual_distortion_audit_v1/summary.json) |
| 7 单侧监督可双端协同 | difflens/KTP-follow | Winogender / G-source / prior / semantic | 局部G变化不稳定，text-U无可靠公平增量 | 尚未建立；不能推一般不可能 | [证据](../../experiments/g_source_balanced_distillation_v1/RESULTS.md) |
| 8 可达层就是共享偏差起源 | method/receiver、hook audit | 前缀/层干预、norm probe、撤销 | 部分路径和更新对行为有功能作用 | 可修复位置不等于唯一历史来源 | [证据](../../experiments/u_g_transfer_followup_v1/adapter_ablation_v1/REPORT.md) |
| 8 交互/SAE/自产计划已成有效方法 | research_plan、receiver、interaction | CPU/候选提取；未完成全链 | 42前缀候选，非公平终点 | Proposed/技术准备，非已验证收益 | [证据](../../experiments/interaction_transfer_v1/summary.json) |

每份docs的完整对应及当前有效性见docs_inventory；每个顶层实验的状态见experiment_inventory。版本之间无独立样本时，证据强度不因目录增加而提高。
