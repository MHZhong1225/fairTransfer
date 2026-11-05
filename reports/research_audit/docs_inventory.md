# docs 联合清单

逐份覆盖 docs 下所有文件；状态是本次审计判断。Superseded只在新索引标记，不修改冻结历史正文；原文件原位保留。

| 文档 | 实际用途 | 当前状态/有效性 | RQ | 对应实验 | 处理 |
|---|---|---|---|---|---|
| [docs/answer_probability_diagnostic_design_20261102.md](../../docs/answer_probability_diagnostic_design_20261102.md) | 输出概率诊断设计 | 历史设计，旧图及真实面板均已完成；设计稿保留历史状态 | 5 | answer_probability_diagnostic_v1; answer_probability_real_panel_v1 | 原位保留，已索引 |
| [docs/bias_map_protocol.md](../../docs/bias_map_protocol.md) | 正式双端偏差图协议 | 正式未执行；开发实验另记 | 1,6 | bias_map_v1 | 原位保留，已索引 |
| [docs/bpo_2502_03429v1_reading_20261031.md](../../docs/bpo_2502_03429v1_reading_20261031.md) | 论文v1证据及数学核读 | 有效的本地文献核查，非实验数值 | 4,8 | bias_origin_v1; paper_sft_replication_v2 | 原位保留，已索引 |
| [docs/difflens_transferability_review.md](../../docs/difflens_transferability_review.md) | 稀疏表示与路径方法借鉴 | 候选，未运行完整DiffLens公平方法 | 8 | pathway_transfer_v2; receiver_transfer_proposal_v3 | 原位保留，已索引 |
| [docs/experiment_log.md](../../docs/experiment_log.md) | 阶段流水与失败解释 | 历史记录；顶部日期及运行状态不代表当前 | 全部 | 多实验，见实验清单 | 原位保留，已索引 |
| [docs/fairness_transfer_benchmark_proposal.md](../../docs/fairness_transfer_benchmark_proposal.md) | 共同语义图文dataset与评估pipeline | 现行实施设计；多数新增方案尚未实现 | 1,2,3,4 | occupation_transfer_v3是旧文本实验，非本设计完成 | 原位保留，已索引 |
| [docs/fairness_transfer_training_redesign_20261029.md](../../docs/fairness_transfer_training_redesign_20261029.md) | 监督候选及来源演变 | 历史候选，当前范围由research_plan与冻结协议决定 | 3,7,8 | fairness_transfer_study; occupation_transfer_v3 | 原位保留，已索引 |
| [docs/figure_contract.md](../../docs/figure_contract.md) | 探索性自动筛查图展示合同 | 有效；非U/G关联证据 | 1,6 | bias_map_v1 | 原位保留，已索引 |
| [docs/i2t_supervision_factorial_design_20261102.md](../../docs/i2t_supervision_factorial_design_20261102.md) | 信息范围×任务多样性2×2 | Proposed，未执行训练 | 5 | 无四臂结果 | 原位保留，已索引 |
| [docs/knowledge_transfer_probing_follow_20261029.md](../../docs/knowledge_transfer_probing_follow_20261029.md) | KTP知识访问与公平迁移的区别 | 文献/实现核读 | 7 | fairness_transfer_v2; g_source_balanced_distillation_v1 | 原位保留，已索引 |
| [docs/method_candidate.md](../../docs/method_candidate.md) | 通路引导单侧公平训练提案 | Superseded的早期方法范围；保留思想历史 | 8 | pathway_transfer_v2; interaction_transfer_v1 | 原位保留，已索引 |
| [docs/paper_sft_followup_controls_20261031.md](../../docs/paper_sft_followup_controls_20261031.md) | 同预算与效用补充入口 | 后续已由matched训练执行；不是独立实验 | 4 | u_g_transfer_followup_v1 | 原位保留，已索引 |
| [docs/receiver_transfer_proposal_v3.md](../../docs/receiver_transfer_proposal_v3.md) | 接收端路径候选 | Proposed/CPU核心检查；未完成真实公平验证 | 8 | receiver_transfer_proposal_v3 | 原位保留，已索引 |
| [docs/research_plan.md](../../docs/research_plan.md) | 当前研究定义、主线与候选入口 | 现行概念总纲；状态须由新结果校正 | 全部 | 全部；不当作执行台账 | 原位保留，已索引 |
| [docs/script_cleanup_completed_20261029.json](../../docs/script_cleanup_completed_20261029.json) | 清理/去重/下载与恢复操作明细 | 历史审计凭证；不是新的实验完成证明 | 维护 | 清理涉及实验，逐路径见原JSON/JSONL | 原位保留，已索引 |
| [docs/script_cleanup_inventory_20261029.json](../../docs/script_cleanup_inventory_20261029.json) | 清理/去重/下载与恢复操作明细 | 历史审计凭证；不是新的实验完成证明 | 维护 | 清理涉及实验，逐路径见原JSON/JSONL | 原位保留，已索引 |
| [docs/storage_adapter_duplicates_20261031.json](../../docs/storage_adapter_duplicates_20261031.json) | 清理/去重/下载与恢复操作明细 | 历史审计凭证；不是新的实验完成证明 | 维护 | 清理涉及实验，逐路径见原JSON/JSONL | 原位保留，已索引 |
| [docs/storage_cleanup_20261031.md](../../docs/storage_cleanup_20261031.md) | 历史存储删除与硬链接记录 | 必须保留；约41.6GiB是历史报告量非本次释放 | 维护 | Janus/旧权重/计数及正式数据 | 原位保留，已索引 |
| [docs/storage_cleanup_20261031_adapter_result.json](../../docs/storage_cleanup_20261031_adapter_result.json) | 清理/去重/下载与恢复操作明细 | 历史审计凭证；不是新的实验完成证明 | 维护 | 清理涉及实验，逐路径见原JSON/JSONL | 原位保留，已索引 |
| [docs/storage_cleanup_20261031_dedup_summary.json](../../docs/storage_cleanup_20261031_dedup_summary.json) | 清理/去重/下载与恢复操作明细 | 历史审计凭证；不是新的实验完成证明 | 维护 | 清理涉及实验，逐路径见原JSON/JSONL | 原位保留，已索引 |
| [docs/storage_cleanup_20261031_deduplicated.jsonl](../../docs/storage_cleanup_20261031_deduplicated.jsonl) | 清理/去重/下载与恢复操作明细 | 历史审计凭证；不是新的实验完成证明 | 维护 | 清理涉及实验，逐路径见原JSON/JSONL | 原位保留，已索引 |
| [docs/storage_cleanup_20261031_delete_plan.json](../../docs/storage_cleanup_20261031_delete_plan.json) | 清理/去重/下载与恢复操作明细 | 历史审计凭证；不是新的实验完成证明 | 维护 | 清理涉及实验，逐路径见原JSON/JSONL | 原位保留，已索引 |
| [docs/storage_cleanup_20261031_deleted.json](../../docs/storage_cleanup_20261031_deleted.json) | 清理/去重/下载与恢复操作明细 | 历史审计凭证；不是新的实验完成证明 | 维护 | 清理涉及实验，逐路径见原JSON/JSONL | 原位保留，已索引 |
| [docs/storage_cleanup_20261031_generation_audit.json](../../docs/storage_cleanup_20261031_generation_audit.json) | 清理/去重/下载与恢复操作明细 | 历史审计凭证；不是新的实验完成证明 | 维护 | 清理涉及实验，逐路径见原JSON/JSONL | 原位保留，已索引 |
| [docs/storage_cleanup_20261031_integrity.json](../../docs/storage_cleanup_20261031_integrity.json) | 清理/去重/下载与恢复操作明细 | 历史审计凭证；不是新的实验完成证明 | 维护 | 清理涉及实验，逐路径见原JSON/JSONL | 原位保留，已索引 |
| [docs/storage_cleanup_20261031_plan.json](../../docs/storage_cleanup_20261031_plan.json) | 清理/去重/下载与恢复操作明细 | 历史审计凭证；不是新的实验完成证明 | 维护 | 清理涉及实验，逐路径见原JSON/JSONL | 原位保留，已索引 |
| [docs/u_dataset_options.md](../../docs/u_dataset_options.md) | 来源、许可、职业线索适用性 | 历史候选；不是来源已全部通过 | 2 | u_visogender_pilot; u_public_pilot_expanded | 原位保留，已索引 |
| [docs/umm_bias_research_plan_20261031.md](../../docs/umm_bias_research_plan_20261031.md) | 属性可读、不当关联、最终图像因果链 | 现行机制问题；完整因果链未实现 | 6,8 | umm_bias_bagel_sensenova_v1; visual_distortion_audit_v1 | 原位保留，已索引 |
| [docs/umm_u_to_g_hook_audit_20261031.md](../../docs/umm_u_to_g_hook_audit_20261031.md) | 模块投影、缓存、动态token结构核查 | 有效结构证据；非公平终点 | 8 | mechanism_probe_v1; occupation_transfer_v3 | 原位保留，已索引 |
| [docs/understanding_generation_sft_data_audit_20261031.md](../../docs/understanding_generation_sft_data_audit_20261031.md) | 旧36图及180图基线数据可用性 | 历史P0依据；后续新数据未自动继承资格 | 2,4 | understanding_generation_sft_v1 | 原位保留，已索引 |
| [docs/understanding_generation_sft_execution_design_20261031.md](../../docs/understanding_generation_sft_execution_design_20261031.md) | BAGEL十二臂执行顺序和预算 | Superseded执行建议；未完成十二臂 | 2,4 | understanding_generation_sft_v1/v2/v3 | 原位保留，已索引 |
| [docs/understanding_generation_sft_existing_g_evidence_20261031.md](../../docs/understanding_generation_sft_existing_g_evidence_20261031.md) | 旧G蒸馏的原图、噪声和评审复算 | 有效历史证据，修正旧README未完成说法 | 7 | g_source_balanced_distillation_v1 | 原位保留，已索引 |
| [docs/understanding_generation_sft_paper_audit_20261031.md](../../docs/understanding_generation_sft_paper_audit_20261031.md) | 论文/公开代码/训练任务差异 | 有效设计审查；后续训练以新合同为准 | 4 | paper_sft_replication_v1/v2 | 原位保留，已索引 |
| [docs/understanding_generation_sft_plan_20261031.md](../../docs/understanding_generation_sft_plan_20261031.md) | 同图文双监督方向的重心调整 | 思想仍有效；原BAGEL执行已暂停 | 4 | understanding_generation_sft_v1/v2/v3; paper_sft_replication_v2 | 原位保留，已索引 |
| [docs/understanding_generation_sft_source_design_20261031.md](../../docs/understanding_generation_sft_source_design_20261031.md) | 十二职业源材料与评估建议 | 历史设计，不等于数据通过或已训练 | 2,4 | understanding_generation_sft_v1/v2/v3 | 原位保留，已索引 |
| [docs/understanding_generation_sft_training_audit_20261031.md](../../docs/understanding_generation_sft_training_audit_20261031.md) | caption/flow目标、梯度与更新位置资格 | 技术已执行部分有效；不能当完整SFT结果 | 2,4,8 | understanding_generation_sft_v1/qualification/gradient_probe | 原位保留，已索引 |

## 阅读范围与恢复边界

所有Markdown正文与JSON均读取；维护JSONL逐条解析与统计，见 run_records.csv。大JSON中的操作路径、状态和删除/硬链接记录保留，未重复执行旧清理。文献笔记的论文数值不是本地复现实验结果。

仓库根没有 .git，git status 无法给出版本恢复保障；上游子项目的Git不代表本项目有历史恢复能力。故不移动有引用关系的原文，不永久删除唯一实验产物。AGENTS.md→CLAUDE.md软链接有效；规则文件25行/2212字节，无需扩写。
