# 实验总清单

审计日期：2026-11-03（北京时间）。完整枚举 **52 个顶层实验目录**；目录是资产容器，不等于独立科学实验。RQ关联允许多选。全部原路径保留。

状态中“完成”描述执行，可信度由限制列及RQ报告确定；不代表公平改善成功。

| 原始目录 | 真实目的 | 执行/证据状态 | RQ | 证据入口 | 重复/版本关系与限制 | 归档 |
|---|---|---|---|---|---|---|
| `answer_probability_diagnostic_v1` | 精确候选概率、题式、顺序和采样路径诊断 | 完成，诊断证据有限 | 5 | [experiments/answer_probability_diagnostic_v1/REPORT.md](../../experiments/answer_probability_diagnostic_v1/REPORT.md) | 736条件完成；旧22图复用，人口参考只有8图；AUC不等于准确率 | 原位保留 |
| `answer_probability_real_panel_v1` | VISOGENDER实际256视图上的概率诊断复核 | 已完成输出，诊断参考仍有限 | 5 | [experiments/answer_probability_real_panel_v1/question.json](../../experiments/answer_probability_real_panel_v1/question.json) | 审计中自然完成：2352/2352条件/模型；48真实图；帽饰YES仅1图 | 原位保留 |
| `attribute_probe_v1` | 单属性词回答与反序选择题 | 完成，诊断证据有限 | 5 | [experiments/attribute_probe_v1/REPORT.md](../../experiments/attribute_probe_v1/REPORT.md) | 44/44 FEMALE与帽饰保留共存；8/18/14图属性参考不同 | 原位保留 |
| `bias_map_v1` | 职业U/G基线与同场景生成开发 | 开发已执行；正式协议未执行 | 1,6 | [experiments/bias_map_v1/design.json](../../experiments/bias_map_v1/design.json) | design.json的未执行状态仅指正式bias map；开发图被多个后续实验复用 | 原位保留 |
| `bias_origin_v1` | 偏差来源六层假设、旧轨迹及评审上下文干预 | 部分完成；部分历史证据已删除 | 6,8 | [experiments/bias_origin_v1/REPORT.md](../../experiments/bias_origin_v1/REPORT.md) | 不支持唯一LM起源；Janus轨迹只能作为历史摘要 | 原位保留 |
| `bpo_token_affinity_v1` | 同图像历史下next-code条件接近性和BAGEL速度场 | 历史完成；Janus原始产物已删除 | 8 | [experiments/bpo_token_affinity_v1/RESULTS.md](../../experiments/bpo_token_affinity_v1/RESULTS.md) | 完整序列JSD饱和、读出依赖；BAGEL速度场不是概率JSD | 原位保留 |
| `calibration_split_fair_transfer_v1` | 隔离私有标量和共享更新的公平梯度路由 | 完成，未建立双端收益 | 3,7 | [experiments/calibration_split_fair_transfer_v1/RESULTS.md](../../experiments/calibration_split_fair_transfer_v1/RESULTS.md) | 五臂各64步；较弱组不变；软分数降低不代表硬差距改善 | 原位保留 |
| `conditional_transfer_v1` | 真实照片活动识别的源公平资格 | 资格完成；未开展公平训练 | 2 | [experiments/conditional_transfer_v1/RESULTS.md](../../experiments/conditional_transfer_v1/RESULTS.md) | Janus/Lumina未通过稳定源差距门槛；新照片未重现驾驶子组差异 | 原位保留 |
| `cross_source_update_geometry_v1` | 两类监督最终有效更新几何 | 完成，事后描述 | 8 | [experiments/cross_source_update_geometry_v1/README.md](../../experiments/cross_source_update_geometry_v1/README.md) | 四种seed配对全部保留；近零余弦不证明机制独立 | 原位保留 |
| `description_coverage_v1` | 固定22图开放描述与追问覆盖 | 完成，诊断证据有限 | 5 | [experiments/description_coverage_v1/REPORT.md](../../experiments/description_coverage_v1/REPORT.md) | 职业模板化同时减少一般细节；少说人口词不是公平 | 原位保留 |
| `fairness_transfer_study` | 两模型三seed A–E监督与路径探索 | 训练/生成及双评审报告已完成；解释受评分混杂限制 | 1,3,8 | [experiments/fairness_transfer_study/RESULTS.md](../../experiments/fairness_transfer_study/RESULTS.md) | B的旧语义收益受解析器混杂；部分Janus评审已清理 | 原位保留 |
| `fairness_transfer_v2` | CelebA发色条件误差公平：ERM/CDRO | 完成，未建立额外公平迁移 | 3 | [experiments/fairness_transfer_v2/RESULTS.md](../../experiments/fairness_transfer_v2/RESULTS.md) | 512身份确认集；三seed；G明确发色请求接近天花板 | 原位保留 |
| `g_source_balanced_distillation_v1` | G-only均衡teacher训练及G→text-U迁移 | 完成，结果有限 | 7 | [experiments/g_source_balanced_distillation_v1/RESULTS.md](../../experiments/g_source_balanced_distillation_v1/RESULTS.md) | 4臂各96步；chef仅seed28有2/16女性；U无可靠公平收益 | 原位保留 |
| `interaction_transfer_v1` | 职业×人口交互跨度候选提取 | 准备及CPU检查完成；训练未执行 | 8 | [experiments/interaction_transfer_v1/summary.json](../../experiments/interaction_transfer_v1/summary.json) | 42前缀，除层0外候选秩8；不是已验证偏差跨度 | 原位保留 |
| `iris_public_release_audit_20261027` | IRIS公开资产与指标可复现性 | 审计完成；非模型迁移实验 | 1 | [experiments/iris_public_release_audit_20261027/audit.json](../../experiments/iris_public_release_audit_20261027/audit.json) | 不把论文报告数值当成本项目生成结果 | 原位保留 |
| `mechanism_probe_v1` | 前缀替换、teacher-forced、7–13层条件干预 | 多项诊断完成；非公平学习 | 8 | [experiments/mechanism_probe_v1/u_wording_visogender_v1/protocol.md](../../experiments/mechanism_probe_v1/u_wording_visogender_v1/protocol.md) | 新职业复验96图；强属性条件路径可控不定位历史来源 | 原位保留 |
| `object_action_probe_v1` | 物体/动作单字段、双字段、检查点描述曲线 | 完成，诊断证据有限 | 5 | [experiments/object_action_probe_v1/REPORT.md](../../experiments/object_action_probe_v1/REPORT.md) | 4000条请求并非4000图；312步前无检查点；CODE诱导另列 | 原位保留 |
| `occupation_transfer` | 职业文本数据v1–v6与模型技术资格资产 | 数据及技术资产；非独立主实验 | 3 | [experiments/occupation_transfer/data_v5/data.json](../../experiments/occupation_transfer/data_v5/data.json) | 多版数据不能合并；供v1/v2/v3引用，保持路径 | 原位保留 |
| `occupation_transfer_v1` | 首版文本职业训练预检 | 失败/被后续版本替代 | 3 | [experiments/occupation_transfer_v1/PREFLIGHT_DIAGNOSIS.json](../../experiments/occupation_transfer_v1/PREFLIGHT_DIAGNOSIS.json) | 旧权重已清理，保留回答及失败；不能以旧active快照认定在运行 | 原位保留 |
| `occupation_transfer_v2` | 文本职业训练与自然先验评价修订 | 失败/被v3替代 | 3 | [experiments/occupation_transfer_v2/PREFLIGHT_DIAGNOSIS.json](../../experiments/occupation_transfer_v2/PREFLIGHT_DIAGNOSIS.json) | 旧适配器已清理；不构成独立成功复验 | 原位保留 |
| `occupation_transfer_v3` | 文本证据监督、规范分布与参数路由 | 部分完成，官方配置核验暂停 | 3,8 | [experiments/occupation_transfer_v3/INTERIM_RESULTS.md](../../experiments/occupation_transfer_v3/INTERIM_RESULTS.md) | SenseNova首seed有结果，route_q不完整；BAGEL资格失败，非两模型三seed完成 | 原位保留 |
| `official_generation_check_20261031` | 官方参考配置下基座生成资格 | 单样例/技术检查完成 | 2 | [experiments/official_generation_check_20261031/README.md](../../experiments/official_generation_check_20261031/README.md) | 检查生成配置与可运行性，不能代表总体质量及公平通过 | 原位保留 |
| `paper_sft_replication_v1` | BPO公开数据与VILA-U SFT下载复现 | 下载失败，被v2替代 | 4 | [experiments/paper_sft_replication_v1/README.md](../../experiments/paper_sft_replication_v1/README.md) | 训练未启动；保留manifest与网络失败 | 原位保留 |
| `paper_sft_replication_v2` | 公开代码I→T/T→I SFT及双端评价 | 完成；结论受单训练seed及预算混杂限制 | 4,5 | [experiments/paper_sft_replication_v2/研究结果综合报告.md](../../experiments/paper_sft_replication_v2/研究结果综合报告.md) | 四训练分支；G纸面50×160，另11×160工作提示；DEBUG100失败排除 | 原位保留 |
| `pathway_transfer_v2` | 路径方法CPU激活缓存审计 | 仅CPU检查；无公平结果 | 8 | [experiments/pathway_transfer_v2/cpu_audit/result.json](../../experiments/pathway_transfer_v2/cpu_audit/result.json) | 只有cpu_audit等资产，不等于真实训练/生成完成 | 原位保留 |
| `prior_transfer_v1` | Janus文本50:50先验监督与G-only对照 | 完成，源门槛未通过 | 3,7 | [experiments/prior_transfer_v1/RESULTS.md](../../experiments/prior_transfer_v1/RESULTS.md) | 四臂两seed；格式质量提高但条件先验偏离更大 | 原位保留 |
| `receiver_transfer_proposal_v3` | 接收端方法及核心检查 | 提案/CPU检查；无实测公平结果 | 8 | [experiments/receiver_transfer_proposal_v3/core_checks.log](../../experiments/receiver_transfer_proposal_v3/core_checks.log) | core_checks.log不能当训练日志 | 原位保留 |
| `role_measurement_v4` | 224张旧角色图同格式读出及措辞诊断 | 完成，测量诊断 | 1 | [experiments/role_measurement_v4/analysis.json](../../experiments/role_measurement_v4/analysis.json) | 复用role benchmark图片，不是新增生成独立证据 | 原位保留 |
| `role_spatial_qualification_v2` | 60张SWiG空间/镜像定位资格 | 完成，测量诊断 | 1,2 | [experiments/role_spatial_qualification_v2/analysis.json](../../experiments/role_spatial_qualification_v2/analysis.json) | 与swig_grounding共用60张照片；480回答/模型不等于480图 | 原位保留 |
| `role_transfer_benchmark_v1` | 角色分配、条件绑定和默认呈现开发测量 | 生成与诊断完成；人工真值未完成 | 1 | [experiments/role_transfer_benchmark_v1/RESULTS.md](../../experiments/role_transfer_benchmark_v1/RESULTS.md) | 224图，两读出导致Janus组差方向反转 | 原位保留 |
| `role_transfer_pilot_v1` | 真实SWiG U-only训练与生成×读出交叉重放 | 完成；未建立双端公平收益 | 1,3 | [experiments/role_transfer_pilot_v1/RESULTS.md](../../experiments/role_transfer_pilot_v1/RESULTS.md) | 40训练图/60校验图，120主图；读出变化可独立改变分数 | 原位保留 |
| `semantic_transfer_v1` | 已学会图文属性匹配的U适配器→G | 完成，未观察一致额外迁移 | 7 | [experiments/semantic_transfer_v1/RESULTS.md](../../experiments/semantic_transfer_v1/RESULTS.md) | 260主图；两noise流复用，三读出覆盖不同 | 原位保留 |
| `shared_prefix_fairness_v1` | CelebA共享文本前缀的U公平训练 | 完成；源端捷径/公平未建立 | 2,3 | [experiments/shared_prefix_fairness_v1/RESULTS.md](../../experiments/shared_prefix_fairness_v1/RESULTS.md) | 四臂48步；全答Yes等问题；v2重建设计，不覆盖v1 | 原位保留 |
| `source_fairness_winogender_v1` | text-U共指反事实公平与定位/随机层对照 | 完成；跨职业不稳定 | 3,7,8 | [experiments/source_fairness_winogender_v1/RESULTS.md](../../experiments/source_fairness_winogender_v1/RESULTS.md) | 额外44职业352题；原四职业不足以支持定位优势；G中性无匹配类别变化 | 原位保留 |
| `source_task_redesign_v1` | 活动语义/字母混淆、图像转移与人工审查准备 | 诊断完成；人工审核未完成 | 1,2 | [experiments/source_task_redesign_v1/README.md](../../experiments/source_task_redesign_v1/README.md) | 1248支持度是重复测量；空白140项包不是人工标签 | 原位保留 |
| `swig_grounding_v1` | SWiG原角色/框参考下60照片定位资格 | 完成，开发资格 | 1,2 | [experiments/swig_grounding_v1/analysis.json](../../experiments/swig_grounding_v1/analysis.json) | 与role pilot/spatial复用照片；作者参考不是新增人工金标准 | 原位保留 |
| `swig_role_metadata_audit_v1` | SWiG train/dev人物角色元数据审计 | 审计完成；后续另有照片调用 | 2 | [experiments/swig_role_metadata_audit_v1/README.md](../../experiments/swig_role_metadata_audit_v1/README.md) | README主要是上游说明；1173候选不自动是职业真值 | 原位保留 |
| `training_feasibility_v1` | 随机模型路由、真实单侧更新和范数匹配探针 | 技术检查完成；公平结论不足 | 2,8 | [experiments/training_feasibility_v1/README.md](../../experiments/training_feasibility_v1/README.md) | 子实验真实训练与根README随机张量范围必须区分；缩放59.8倍是事后干预 | 原位保留 |
| `u_g_transfer_followup_v1` | 匹配训练、跨图/编辑U诊断、双提示G及更新撤销 | 预定队列完成；公平解释仍有限 | 4,5,8 | [experiments/u_g_transfer_followup_v1/研究结果综合报告.md](../../experiments/u_g_transfer_followup_v1/研究结果综合报告.md) | 6训练、3seed、15参考/检查点、36960新增正式图；322项审计点估计核验包含本线 | 原位保留 |
| `u_genderbias_vl_development` | GenderBias-VL编辑图职业线索开发 | 数据/开发诊断，不是确认实验 | 2 | [experiments/u_genderbias_vl_development/manifest.json](../../experiments/u_genderbias_vl_development/manifest.json) | 6配对；编辑会同时改变非人口细节 | 原位保留 |
| `u_genderbias_vl_stratified_development` | 分层选样与编辑职业线索复核 | 部分图像/诊断资产 | 2 | [experiments/u_genderbias_vl_stratified_development/ai_cue_review.json](../../experiments/u_genderbias_vl_stratified_development/ai_cue_review.json) | 检查实际缺失edit图；不能按selection计划补齐已完成配对 | 原位保留 |
| `u_public_pilot` | SocialCounterfactuals初始职业识别 | 探索完成；样本与expanded重叠 | 2 | [experiments/u_public_pilot/manifest.json](../../experiments/u_public_pilot/manifest.json) | 12次早期回答不能与expanded32相加为独立图片 | 原位保留 |
| `u_public_pilot_expanded` | SocialCounterfactuals职业线索与盲审扩展 | 开发完成；保留样本未调用 | 2 | [experiments/u_public_pilot_expanded/README.md](../../experiments/u_public_pilot_expanded/README.md) | 医生/护士共享视觉线索，不能把提示职业直接当准确率真值 | 原位保留 |
| `u_stable_bias_development` | 稳定偏差候选与图像可见性检查 | 开发资产，结论不足 | 2 | [experiments/u_stable_bias_development/manifest.json](../../experiments/u_stable_bias_development/manifest.json) | 24图及AI审查；不预设可靠因果配对 | 原位保留 |
| `u_visogender_pilot` | 真实职业照片开发和来源审计 | 探索完成 | 2 | [experiments/u_visogender_pilot/README.md](../../experiments/u_visogender_pilot/README.md) | 20张；仅评估许可，不能默认进入训练 | 原位保留 |
| `umm_bias_bagel_sensenova_v1` | CFG4/1与BAGEL重构生成过程诊断 | 完成，未建立有效公平收益 | 6 | [experiments/umm_bias_bagel_sensenova_v1/RESULTS.md](../../experiments/umm_bias_bagel_sensenova_v1/RESULTS.md) | 720图，双评审726次/人；部分均衡伴随合格和明确遵从下降 | 原位保留 |
| `umm_u_training_v1` | 旧计数训练与模型下载资产 | 历史产物已删除；当前路线退役 | 2 | [experiments/umm_u_training_v1/README.md](../../experiments/umm_u_training_v1/README.md) | 只保留下载核验；不能复算旧计数结果 | 原位保留 |
| `understanding_generation_sft_v1` | BAGEL三路径资格、180图基线及192源图构建 | 技术完成；多步训练在资格处失败 | 2,4 | [experiments/understanding_generation_sft_v1/RESULTS.md](../../experiments/understanding_generation_sft_v1/RESULTS.md) | 10合格对，其中训练6对5职业，未达12职业门槛 | 原位保留 |
| `understanding_generation_sft_v2` | 修复构图的BAGEL图文SFT数据 | 暂停；63/384源图，未多步训练 | 2,4 | [experiments/understanding_generation_sft_v2/README.md](../../experiments/understanding_generation_sft_v2/README.md) | 因官方配置要求被v3替代；不是U-SFT无效 | 原位保留 |
| `understanding_generation_sft_v3` | 官方配置下BAGEL三监督/位置SFT | 暂停；16/384源图，未多步训练 | 2,4 | [experiments/understanding_generation_sft_v3/README.md](../../experiments/understanding_generation_sft_v3/README.md) | 后转向VILA-U公开代码复现；原冻结目录保留 | 原位保留 |
| `visual_dependence_v2` | 共享前缀视觉依赖的固定描述配对修复 | 完成，源任务通过；未证明公平 | 2,3 | [experiments/visual_dependence_v2/RESULTS.md](../../experiments/visual_dependence_v2/RESULTS.md) | SP-CE与SP-PAIR到200步BA同为94.53%；配对损失无额外硬答案收益 | 原位保留 |
| `visual_distortion_audit_v1` | 真实照片VAE往返与SenseNova噪声恢复 | 96图开发计算完成；属性/ROI标注未完成 | 6 | [experiments/visual_distortion_audit_v1/REPORT.md](../../experiments/visual_distortion_audit_v1/REPORT.md) | 288/1152输出，整体失真非群体信息损失证明；正式960未执行 | 原位保留 |

## 子实验、配置、原始输出与图表

顶层表不将子版本压成一次运行。以下机器清单提供逐文件追踪：

- [逐文件元数据及已读取文本哈希](evidence/file_inventory.json)：源文件、字节数、修改时间、SHA256（符合扫描规则的文本）。
- [运行记录逐文件统计](evidence/run_records.csv)：JSONL总行数、状态、重复ID、解析失败及稳定性。
- [实验报告原表摘录](evidence/experiment_tables.md)：按源路径及行号保留原表，不跨协议合并。
- [完整文件系统资产清单](evidence/filesystem_inventory.jsonl.gz)：含图像、权重、环境及源码，记录软链接；只记录元数据，不逐个加载模型或图片。
- [审计复算](evidence/recomputed_checks.json)：从保存的分类器输出与原始U答案复核点估计；不重复运行模型。

主要子线：training_feasibility 的 u_technical_smoke / u_technical_replication / g_technical_pilot / u_image_resolution / update_norm_matched；mechanism_probe 的前缀、teacher-forced及layer_replication；Winogender 的replication44、mechanistic_masks及g_transfer；BAGEL SFT的qualification/gradient_probe/baseline/experiment_v1；VILA-U 的runs_v2/runs_v3、paper/own G、U与token诊断；followup的paired_visual、work_image、noimage、response、text、matched_training/evaluation及adapter_ablation均保留独立源路径。

图表是派生证据：SVG/PDF/PNG不能替代原始计数。旧图可能沿用旧评分器或已删Janus数据；引用前须查RQ和清理记录。没有对所有图像做人类视觉审查，也没有将元数据扫描宣称为逐图标注。

## 审计期间自然完成的输出

`answer_probability_real_panel_v1`两状态最终均2352/2352条件，status均completed；48图结果、六题序及43图辅助敏感性已纳入RQ5。其输出在初始扫描中增长，最终补充哈希见[evidence/late_completed_panel_inventory.json](evidence/late_completed_panel_inventory.json)，44项原始计数复核见[evidence/probability_real_checks.json](evidence/probability_real_checks.json)。这不是本次审计启动的新实验。
