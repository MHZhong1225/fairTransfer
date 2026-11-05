# 跨实验关键结果总表

只在同行内比较匹配设置。不同数据集、模型、seed和协议不能横向数值排名。百分比差距以pp表示百分点；RD、JS、BA的定义见各RQ。322项复算限于VILA-U主结果及部分撤销点估计/覆盖，其余明确指向原分析文件，未宣称全部独立重算。

| 研究问题 | 实验/方法 | 对照设置 | 核心结果 | 判断 | 证据来源 |
|---|---|---|---|---|---|
| RQ1 测量 | role交叉重放 | 固定BASE图，仅换读者 | 差距37.50→25.00pp；整体22.92→16.67%；弱组不变 | 评分变化不是G改善 | [证据](../../experiments/role_transfer_pilot_v1/RESULTS.md) |
| RQ1 解析 | A–E B三seed | 冻结原分数/统一首词敏感性分列 | SenseNova BASE122/128，ordinary/fair123/128 | 旧额外公平解释不成立 | [证据](../../experiments/fairness_transfer_study/RESULTS.md) |
| RQ2 视觉可学 | SP-CE vs SP-PAIR | 同曝光/200步 | BA均94.53%，控制请求答案相同；无图均50% | 普通CE足够当前资格；无额外硬答案收益 | [证据](../../experiments/visual_dependence_v2/RESULTS.md) |
| RQ2 数据失败 | BAGEL SFT源图v1 | 192候选、冻结资格 | 合格10对，训练6对5职业 | 未执行训练，不算SFT负迁移 | [证据](../../experiments/understanding_generation_sft_v1/experiment_v1/LIVE_RESULTS.md) |
| RQ3 公平增量 | BAGEL CDRO三seed | BASE/ERM同预算，512身份 | 最大条件差距BASE1.56pp；CDRO3.12/3.91/4.69 | 未稳定改善，含退化 | [证据](../../experiments/fairness_transfer_v2/summary.json) |
| RQ3 文本证据 | SenseNova职业v3首seed | BASE/ordinary/额外监督 | test正确208/249/248–250，分母256 | 任务提高，公平增量不足 | [证据](../../experiments/occupation_transfer_v3/INTERIM_RESULTS.md) |
| RQ4 作者代码SFT | VILA-U性别U/G | 同基座但3/10轮、文案不同 | 论文性别RD .8675→.8467(U)/.3008(G) | 有变化，预算混杂 | [证据](../../experiments/paper_sft_replication_v2/paper_g_evaluation_v3/results_gender.json) |
| RQ4 匹配SFT | VILA-U U/G，3seed，epoch3 | 相同图文/顺序/936步 | 肖像RD BASE.9148，U .6866–.8807，G .4347–.5142 | 预算不是全部解释 | [证据](../../experiments/u_g_transfer_followup_v1/matched_evaluation_v1/generation_results.json) |
| RQ4 场景反例 | 同模型工作提示 | 同11职业/每状态704图 | RD BASE.2098，G .5565/.5568/.6038 | 生成收益不跨场景保持 | 同上 |
| RQ4 源公平 | U最终三seed | 人口编辑56对 | 准确率提高；组差/答案改变率变化区间均含零 | 源公平未确立 | [证据](../../experiments/u_g_transfer_followup_v1/matched_evaluation_v1/paired_results.json) |
| RQ5 读取 | 属性微调后 | 固定22图，参考子集 | 性别词44/44 FEMALE；帽饰31/36参考一致 | 模板化不等于所有信息不可读 | [证据](../../experiments/attribute_probe_v1/reviewed_results.json) |
| RQ5 概率 | 旧图流程复核 | 8人口/18帽饰参考图 | 平均分数AUC1.0，微调人口词采样21/64 | 小样本排序不等于真实识别/采样正确 | [证据](../../experiments/answer_probability_diagnostic_v1/results.json) |
| RQ5 真实照片复核 | 48图、六排列 | 各题式1152次采样 | 呈现词严格501→577，辅助语义1092→838；帽饰YES仅1图 | 格式改善与语义退化可并存；非人审真值 | [真实汇总](../../experiments/answer_probability_real_panel_v1/results.json) |
| RQ6 生成过程 | CFG4→1 | 两模型、双评审、720图 | BAGEL/Qwen合格26→4/48，明确女性21→8/48 | 低CFG无可靠联合收益 | [证据](../../experiments/umm_bias_bagel_sensenova_v1/REPORT.md) |
| RQ6 视觉处理 | 96图codec/恢复 | 职业内两来源组各48 | BAGEL SSIM .9856/.9862；组差区间含零 | 未证明群体信息损失 | [证据](../../experiments/visual_distortion_audit_v1/summary.json) |
| RQ7 反向 | G-source U/G参数 | 三职业、两训练seed | chef女性0→2/16仅部分seed；U289→287/286或不变/480 | 无可靠双端公平协同 | [证据](../../experiments/g_source_balanced_distillation_v1/RESULTS.md) |
| RQ8 撤销 | VILA-U首seed U后半更新撤销 | full对照、随机半组 | U98.44→89.78%，G变化区间均含零 | 两端贡献可分离，非严格零G效应 | [证据](../../experiments/u_g_transfer_followup_v1/adapter_ablation_v1/REPORT.md) |
| RQ8 层组 | G前/后更新撤销 | 同seed/最终G | 肖像RD .5142→.8523/.3718 | 更新有相反贡献，位置/幅度混杂 | 同上 |

完整15检查点和9状态撤销表见RQ4/RQ8；旧失败、小幅/不改善与未知没有被丢弃。更多历史表格见[evidence/experiment_tables.md](evidence/experiment_tables.md)，其中旧Janus缺失原始数据的数字仅属研究史。
