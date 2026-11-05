# docs 与研究入口整理记录

保留docs原36份研究/维护文档（25 Markdown、10 JSON、1 JSONL），新增docs/README导航。原设计正文、数表及清理日志不删除；已过时文档的Superseded状态在docs_inventory标记，不覆写其历史承诺。

| 原路径 | 动作 | 当前路径 | 备份/恢复 | 原因 |
|---|---|---|---|---|
| README.md | repair_link_with_backup | [README.md](../../README.md) | [_archive/research_audit_20261103/before_link_repair/README.md](../../_archive/research_audit_20261103/before_link_repair/README.md) | 仅修复已确定的canonical benchmark文件名；不改变实验数字或计划 |
| docs/experiment_log.md | repair_link_with_backup | [docs/experiment_log.md](../../docs/experiment_log.md) | [_archive/research_audit_20261103/before_link_repair/docs/experiment_log.md](../../_archive/research_audit_20261103/before_link_repair/docs/experiment_log.md) | 仅修复已确定的canonical benchmark文件名；不改变实验数字或计划 |
| docs/fairness_transfer_training_redesign_20261029.md | repair_link_with_backup | [docs/fairness_transfer_training_redesign_20261029.md](../../docs/fairness_transfer_training_redesign_20261029.md) | [_archive/research_audit_20261103/before_link_repair/docs/fairness_transfer_training_redesign_20261029.md](../../_archive/research_audit_20261103/before_link_repair/docs/fairness_transfer_training_redesign_20261029.md) | 仅修复已确定的canonical benchmark文件名；不改变实验数字或计划 |
| docs/method_candidate.md | repair_link_with_backup | [docs/method_candidate.md](../../docs/method_candidate.md) | [_archive/research_audit_20261103/before_link_repair/docs/method_candidate.md](../../_archive/research_audit_20261103/before_link_repair/docs/method_candidate.md) | 仅修复已确定的canonical benchmark文件名；不改变实验数字或计划 |
| docs/research_plan.md | repair_link_with_backup | [docs/research_plan.md](../../docs/research_plan.md) | [_archive/research_audit_20261103/before_link_repair/docs/research_plan.md](../../_archive/research_audit_20261103/before_link_repair/docs/research_plan.md) | 仅修复已确定的canonical benchmark文件名；不改变实验数字或计划 |
| 新建 | create_research_index | [docs/README.md](../../docs/README.md) | 新文件，无原文覆盖 | 按RQ归并导航；不移动依赖中的证据 |
| README.md | add_root_navigation | [README.md](../../README.md) | [_archive/research_audit_20261103/before_root_navigation/README.md](../../_archive/research_audit_20261103/before_root_navigation/README.md) | 提供统一结果入口；保留原研究说明 |

## 修复范围

仅将不存在的fairness_transfer_benchmark_proposal_20261028.md引用改到实存的fairness_transfer_benchmark_proposal.md；内容包含现行和历史设计，是可确定的同一入口。修改README、experiment_log、training_redesign、method_candidate、research_plan共5份文件；README再加统一报告入口，原文件及链接修复后版本均有备份。没有将旧实验解释自动升级成新结果。

旧提案不物理迁移的原因是仍有引用、运行依赖与历史研究价值；索引明确其现行/历史/候选/维护状态。清理JSON/JSONL全部保留，已逐条解析，不重新执行。

## 仍缺失的引用

[缺失引用逐条清单](evidence/unresolved_document_links.json)按源文件、行号和目标记录，已识别`:行号`和尖括号路径，避免把有效代码定位误计为缺失。部分旧研究报告、路由审计、文献综述及已清理脚本没有同内容替代，不能猜测修复。报告的新证据引用与这些历史缺失分开核验；不得把新索引当作恢复了旧文件。

## 不变的规则与恢复约束

AGENTS.md仍为CLAUDE.md软链接，25行/2212字节。没有追加实验流水，也没有修改上游模型/权重或全局记忆。根目录无.git，不能依赖git checkout恢复；所有本次原文编辑均有项目内备份。恢复README需注意两次编辑：before_link_repair是最初原文，before_root_navigation是修完链接但尚未加导航的版本。

机器明细：[cleanup_actions.json](evidence/cleanup_actions.json)。报告目录中的审计派生产物可重新生成，但safe_cleanup为一次性操作脚本，不应对已整理工作区盲目重跑。
