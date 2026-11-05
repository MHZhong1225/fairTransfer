# 统一多模态模型的公平性迁移

研究目标：在统一多模态模型中，先验证理解训练和生成训练是否改善各自任务的公平性，再研究这种改善能否跨任务迁移，以及哪些计算影响迁移。这里 benchmark 包含自建 dataset 与评估；两端的本任务能力和跨任务变化分别测量。当前优先比较相同图文数据的 I→T 与 T→I SFT。

## 从这里开始

- [全部实验与文档的联合研究审计](reports/research_audit/README.md)：52个实验目录、8个研究问题，含实际结果、证据矩阵、冲突和整理记录。

- [最新已完成：U 学到了什么，以及 G 为何没有同步改善](experiments/u_g_transfer_followup_v1/研究结果综合报告.md)：六次匹配训练、三个种子、36,960 张正式生成与六个参数撤销版本全部完成。区分职业回答、人口公平、提示场景与生成质量；原始结果及完整性核验均可追溯。
- [当前重点：按论文训练并比较理解与生成公平性](experiments/paper_sft_replication_v2/README.md)：核对论文 v1 与公开代码，优先使用作者数据和 VILA-U。此前 [BAGEL 第三版](experiments/understanding_generation_sft_v3/README.md)是改编设置，已停止源图生成并暂停训练队列；尚未完成多步 SFT，不能据此判断公平性。
- [当前完整思想总纲](docs/research_plan.md)：整合有效讨论、研究定义、全部主要候选与验证标准，替代早期相互重叠的研究路线。
- [整体 pipeline 实施设计](docs/fairness_transfer_benchmark_proposal.md)：共同场景数据、监督、推理接口、对照及阶段产物；以顶部 2026-10-31 设计为当前版本。
- [文献与监督候选核读](docs/fairness_transfer_training_redesign_20261029.md)：设计来源与历史候选，采用何种分支以总纲及具体实验协议为准。

核心任务是职业与人口属性关联。规范的 50:50 目标、现实统计保真、证据支持的个人判断和明确属性遵从分别定义；公平指标与职业正确性、有效输出和生成质量一起报告。

## 现有实现与运行

第一批具体模型为 BAGEL 与 SenseNova-U1.5。已有原生 U/G 后端、U-only 训练、文本职业数据与评价、原始结果保存和部分缓存/路径干预能力。新总纲中的完整图文数据与自产计划→G pipeline 尚未全部实现，不能将现有文本队列视为完整新方案。

| 入口 | 内容 |
|---|---|
| [v3 冻结协议](experiments/occupation_transfer_v3/PROTOCOL.md) | 当前既有文本 U→原生 G 实验的实际数据、分支及评价合同 |
| [实时状态](experiments/occupation_transfer_v3/LIVE_STATUS.md) | 运行阶段与最近输出；只是一份监控快照 |
| [职业训练阶段结果](experiments/occupation_transfer_v3/INTERIM_RESULTS.md) | 普通训练与额外监督的实际对比，分别说明答题效果、公平指标和生成端待评内容 |
| [BAGEL 评估问题](experiments/occupation_transfer_v3/BAGEL_QUALIFICATION_REVIEW.md) | 为什么小规模训练完成后仍未进入正式训练，以及现有评分的限制 |
| [运行说明](experiments/occupation_transfer_v3/RUNNING.md) | 队列、恢复与监控入口 |
| [当前主流程代码](scripts/occupation_transfer/) | 数据、训练、评价、生成观察器、汇总与监控 |
| [旧实验结果](experiments/fairness_transfer_study/RESULTS.md) | 已有探索的结果与解释边界，不替代新职业主线 |

运行状态以实际进度和原始产物为准，进程存在、技术资格通过或训练 loss 下降均不代表公平收益。运行中的数据、评分器和训练合同保持冻结；新方案使用单独版本。环境入口为 /home/ubuntu/anaconda3/envs/ucp/bin/python，具体命令见运行说明。

## 研究材料

| 材料 | 用途 |
|---|---|
| [Knowledge-Transfer-Probing 核验](docs/knowledge_transfer_probing_follow_20261029.md) | 学习来源、跨任务访问与 alignment 边界 |
| [IRIS 协议核验](docs/iris_protocol_audit_20261027.md) | 双端公平、现实保真与评价口径 |
| [去偏与迁移文献](docs/fairness_transfer_literature_20261028_v2.md) | 相关方法与竞争性对照 |
| [BPO 精读与偏差来源研究](experiments/bias_origin_v1/REPORT.md) | 六条分层假设、旧轨迹复算与新评审提示干预；明确尚未识别的来源 |
| [完整 UMM 偏差研究方案](docs/umm_bias_research_plan_20261031.md) | 属性表征、不当关联、最终生成的因果作用；重点验证 U 中的偏差相关计算是否通过实际通路影响 G |
| [U→G 结构核查](docs/umm_u_to_g_hook_audit_20261031.md) | BAGEL / SenseNova 的专家、前缀缓存、候选干预位置和实现缺口；只读检查，尚非机制实验结果 |
| [BAGEL / SenseNova 实际结果](experiments/umm_bias_bagel_sensenova_v1/RESULTS.md) | 720张图与两套评分已完成；降低引导未给出可靠公平收益，内容完成变差且评审分歧较大；[运行与记录入口](experiments/umm_bias_bagel_sensenova_v1/RUNNING.md) |
| [DiffLens 与机制分析](docs/difflens_transferability_review.md) | 候选表示、干预、恢复及架构边界 |
| [实验文件清理记录](docs/storage_cleanup_20261031.md) | 旧试跑删除范围、重复文件合并、释放空间和保留内容；历史代码清理另见[原记录](experiments/fairness_transfer_v2/CODE_CLEANUP.md) |

日期命名的旧方案和保留下来的实验记录供复核使用；已删除的旧试跑见清理记录。其中的“下一步”或旧命令不自动成为当前计划。当前研究思想以总纲为入口，实际执行以各运行目录的冻结协议为准。

## 主要参考文献

以下文献是本项目当前研究设计、对照实验或机制分析中实际核读和引用的主要来源。它们提供问题定义、数据/监督先例或分析工具，不意味着其结论已经在本项目模型上复现。

1. [On Fairness of Unified Multimodal Large Language Model for Image Generation](https://arxiv.org/abs/2502.03429)（BPO；本项目的论文式 U/G SFT 对照与精读记录见 [`experiments/paper_sft_replication_v2/`](experiments/paper_sft_replication_v2/) 和 [`docs/bpo_2502_03429v1_reading_20261031.md`](docs/bpo_2502_03429v1_reading_20261031.md)）。
2. [Can Visual Generation Expand What Unified Multimodal Models Know?](https://arxiv.org/abs/2610.21459)（Knowledge-Transfer-Probing；用于区分监督来源、学会内容和推理时可访问内容）。
3. [Transferability Between Understanding and Generation in Unified Multimodal Models](https://arxiv.org/abs/2607.04423)（用于单侧训练、对端评价和架构对照；一般能力迁移不等于公平迁移）。
4. [IRIS](https://arxiv.org/abs/2603.00590)（用于区分理想公平、现实统计保真、双端评价和条件遵从）。
5. [BBQ: A Hand-Built Bias Benchmark for Question Answering](https://aclanthology.org/2022.findings-acl.165/)（用于证据不足/充分和 unknown 处理的评价思想）。
6. [FairVLP](https://arxiv.org/abs/2207.01056)（用于视觉—语言表示的群体配对与公平性对照）。
7. [Addressing Both Statistical and Causal Gender Fairness](https://aclanthology.org/2024.findings-naacl.38/)（用于分别报告统计公平与因果/配对一致性）。
8. [Debiasing Vision-Language Models via Biased Prompts](https://arxiv.org/abs/2302.00070)（用于配对语义校准与视觉—语言去偏的对照）。
9. [Attention, May I Have Your Decision? Localizing Generative Choices in Diffusion Models](https://arxiv.org/abs/2604.06052)（ICM；用于生成决策定位和路径干预的分析参考）。
10. [Dissecting and Mitigating Diffusion Bias via Mechanistic Interpretability](https://arxiv.org/abs/2503.20483)（DiffLens；用于表示分解、归因、受控编辑和恢复验证）。
11. [NEO-unify: Building Native Multimodal Unified Models End to End](https://arxiv.org/abs/2605.12500)（SenseNova-U1；对应本项目使用的统一多模态模型实现背景）。

更完整的文献核读、限制和与本项目假设的对应关系见[公平性迁移文献与监督候选核读](docs/fairness_transfer_training_redesign_20261029.md)及[研究审计阅读索引](reports/research_audit/evidence/reading_index.md)。
