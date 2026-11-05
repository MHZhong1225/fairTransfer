# RQ7：单侧训练是否产生可靠的原生双向公平迁移？

## 1. 研究问题（Research Question）

U-only训练能否改善G，G-only训练能否改善U？必须按监督来源和参数位置分别回答：G loss更新U参数依然是G监督；显式属性遵从、默认人口比例与源条件误差公平是不同终点。

## 2. 实验设计（Experimental Setup）

Winogender三种U源设置×两seed（合计六臂）的文本共指与G职业分布；G-source均衡teacher四臂（仅G loss，更新U或G v_proj，全28层rank1，114688参数，各96步）反向测480条文本U；Janus prior四臂两seed各48步；semantic_transfer复用已训练final200 U适配器；calibration_split五臂各64步。VILA-U两方向行为见RQ4，不与BAGEL/Janus合并。

G-source仅36旧图、24训练/12开发，三职业，均衡来自请求不是人工身份。G中性JS为与50:50规范分布的Jensen–Shannon散度（bit），不是二类RD；二类全偏时约0.311278。显式joint32请求是固定条件/两条noise流复用，不能当32独立试验。

## 3. 实际结果（Observed Results）

| BAGEL G-source分支 | chef女性/16（Qwen；Phi） | 其余两职业女性/16 | 中性macro JS（Qwen；Phi） | 反向U正确/480 |
|---|---|---|---|---:|
| BASE | 0；0 | 均0 | .311278；.311278 | 289 |
| G loss/U参数 seed27 | 0；0 | 均0 | .311278；.311278 | 287 |
| G loss/U参数 seed28 | 2；2 | 均0 | .248937；.248937 | 286 |
| G loss/G参数 seed27 | 0；1 | 均0 | .311278；.270353 | 289 |
| G loss/G参数 seed28 | 2；2 | 均0 | .248937；.248937 | 289 |

来源：[完整源/目标和能力预警](../../../experiments/g_source_balanced_distillation_v1/RESULTS.md)、[全部主生成/评审完整性](../../../experiments/g_source_balanced_distillation_v1/main1096_judge_integrity_audit_v1.json)。实际source-G315图；main1096是跨A/B源、阶段路径及能力预警的联合评审审计，每名评审1096项（1036生成项＋60能力项），并非该G-source实验新增1096图，不能与315相加。G专家限定更新在所测text-U路径不可达，U逐项不变是结构对照。

| 其他方向 | 基座/普通 | 干预结果 | 结论限制 |
|---|---|---|---|
| Winogender text-U→G | 四职业BASE均单一二类呈现 | 普通/定位/随机两seed均无匹配二类翻转 | 未决和职业成功有变化；未观察到翻转不等于严格零效应 |
| Janus U先验→G | 源偏离20.64% | ordinary45.00/46.64%，generic28.92/27.38%，prior24.86/27.73% | 所有U臂未过源门槛 |
| Janus G-only→text-U | 源偏离20.64% | 19.92/19.05% | 两seed、小面板，不足以证明公平反向迁移 |
| semantic U→G显式joint | BASE Qwen30/32，Janus31/32 | SP-PAIR30/32、32/32；SP-CE30/32、31/32 | 两流复用，无跨评审一致额外优势 |
| 校准路由U→G显式joint | PRIVATE-FAIR28/32、32/32 | ROUTED29/32、32/32 | 弱组/默认分布不支持方法成立 |

来源：[U→G原计数](../../../experiments/source_fairness_winogender_v1/g_transfer_v1/final_analysis_v1/main_qwen_analysis.json)、[Janus先验损失分解](../../../experiments/prior_transfer_v1/source_loss_diagnostic.json)、[属性语义迁移](../../../experiments/semantic_transfer_v1/RESULTS.md)、[校准源×目标](../../../experiments/calibration_split_fair_transfer_v1/final_analysis.json)。

Janus prior full CE由1.2577到0.8643/0.9068，同时条件先验偏离更大；候选质量项−log m从0.3967到0.0079/0.0048。损失改善主要包含更愿输出指定标签，而非主先验指标改善。四臂两seed384次更新/306图已完成，负面结果不能删除。

## 4. 判断与分析（Judgment & Analysis）

**观察事实：** G-only可以通过U位置影响G，局部职业/seed有少量人口变化；对应text-U没有可靠公平收益。若干U源任务学会一般语义但G额外收益不一致；部分U公平前提本身未通过。

**合理解释：** 不同条件方向的目标未被同一监督直接约束，参数共享不保证同义使用。text-U对视觉U的代表性有限，当前负结果也可能反映端点不匹配或测量精度不足。

**尚未验证：** 一般性的单向/双向不可能定律，或某架构先天迁移更好。没有共同可靠源收益、同端点、等容量/成本和人审对照，不支持这种排名。微小显式joint增加还需考虑原基座天花板和评审不确定。

## 5. 结论（Conclusion）

目前没有可靠的双端公平协同方法，也没有证明单侧训练完全不能影响对端。最明确的是“参数可达”“输出变化”“有效公平能力迁移”三者不同；本地反向G→U证据不足，不能用个别seed的2张变化宣布成功。

## 6. 下一步（Next Steps）

先选择一个已验证源收益及与目标同语义的方向，再测试迁移。无须继续扩大不通过源门槛的Janus先验训练或仅两条noise流的显式属性矩阵。若使用自产计划/筛选，另设相同接口BASE、规则50:50及调用预算对照，不归入原生参数迁移。
