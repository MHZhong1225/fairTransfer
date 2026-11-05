# RQ2：数据与源任务能否提供可检验的公平学习机会？

## 1. 研究问题（Research Question）

图像中的职业证据、人口呈现及源任务视觉依赖是否成立？若数据无可靠职业真值、基础差距很小或源任务只学格式，后续迁移失败无法定位到G。

## 2. 实验设计（Experimental Setup）

早期VisoGender/SocialCounterfactuals、GenderBias-VL与稳定偏差开发图检查职业证据；之后使用CelebA属性匹配验证shared-prefix视觉依赖，用SWiG真实角色和PISC活动检查源差距。BAGEL图文SFT又经历三版源图构建：首版构图/资格失败，第二版修构图，第三版恢复官方配置后转向VILA-U公开数据。技术资格包括随机模型路由、真实单步caption/flow、保存重载与更新范数；这些不承担公平收益结论。

BA（balanced accuracy）是标签两类召回率的等权平均。CelebA开发集的128判断、50个联通来源组与普通准确率和图片数分别记录。原版v1、v2同时改变数据组织、EOS与步数，不是单因素消融。

## 3. 实际结果（Observed Results）

| 设置 | 实际完成 | 结果 | 状态判断 |
|---|---|---|---|
| BAGEL shared-prefix v1 | 四臂各48步 | M0/A/B/C/D平均BA50.78%，未建立源公平 | 实现可运行，任务仍有捷径 |
| visual_dependence v2 SP-CE | 48/100/200步 | BA52.34/91.41/94.53% | 普通源学习已能通过视觉资格 |
| 同版SP-PAIR | 同步数/曝光 | BA53.91/90.62/94.53% | 最终无额外硬答案收益 |
| 同版IF-CE | 同步数/曝光 | BA89.84/94.53/94.53% | 早期较快，最终无确定优势 |
| Janus真实活动开发 | 80图，重复顺序/控制 | 总体93.44%；作者man91.25%、woman95.63% | 差距−4.38 pp，区间[−12.50,3.75]，未过门槛 |
| Lumina活动开发 | 同80图及后续诊断 | 总体组差8.75 pp | 未达10 pp门槛，驾驶子组新图未重现 |
| BAGEL SFT源图v1 | 192候选生成+双评审 | 合格10对；训练6对/5职业 | 未达12职业覆盖，训练前失败 |
| 源图v2 / v3 | 63/384；16/384 | 未执行多步SFT | 暂停，不能记作训练无效 |

来源：[v2补充分析](../../../experiments/visual_dependence_v2/run_seed20261027/supplemental_analysis.json)；[Janus原计数](../../../experiments/conditional_transfer_v1/baseline_janus_analysis.json)；[Lumina追踪](../../../experiments/conditional_transfer_v1/RESULTS.md)；[源图资格失败](../../../experiments/understanding_generation_sft_v1/experiment_v1/LIVE_RESULTS.md)；[v2中止](../../../experiments/understanding_generation_sft_v2/README.md)；[v3中止](../../../experiments/understanding_generation_sft_v3/README.md)。

v2最终SP-CE答对121/128；无图与空白图均BA50%，正常相对下降44.53 pp。SP-PAIR与SP-CE最终640个控制请求答案相同，不能把经验差为零的bootstrap区间当总体等效证明。

早期SocialCounterfactuals的32张开发图中，医生/护士16图的AI盲审认为14张只有共享医疗线索，另2张依赖护士帽。初始12次推理与扩展集重叠，不能相加。[线索审查](../../../experiments/u_public_pilot_expanded/blinded_cue_review/reviewer_ai.json)。VisoGender20张开发仅限评价；GenderBias-VL编辑同时改变服饰/工具的可能性没有被排除。分层目录存在仅base、缺edit的资产，选样计划不等于已完成配对。

首版固定8条训练判断的优化分解中，语义答案CE为1.234→1.161，EOS CE为8.297→1.393，约99%的该小样总损失下降来自EOS。这是训练样本的优化诊断，不能外推公平收益；它解释了为何后续版本需分开语义决策和结束/格式监督，而非只检查总loss。

## 4. 判断与分析（Judgment & Analysis）

**观察事实：** 后续普通CE修复了共享前缀的图像判别；首版源图构图与合格率不足；多个真实照片任务没有通过事前差距资格。官方生成单图与随机/真实梯度资格已做，但不等于完整数据和任务质量合格。

**合理解释：** 早期失败至少部分来自构造和输出偏置，不能全部归因于架构不支持共享前缀。基础准确率高、差距低时，用小样本测额外公平收益缺乏空间和精度。

**尚未验证：** 可靠、独立、职业证据明确且存在可改进人口捷径的源任务。来源标签、图片请求和机器筛查均存在效度边界。新版成功同时改变数项条件，不能归因于单独移除EOS。

## 5. 结论（Conclusion）

已有证据证明部分源任务可学且依赖图像，未证明它们已经形成稳定的公平学习机会。数据资格失败和差距门槛未通过都是有价值的负面结果，不能写成公平迁移不存在。

## 6. 下一步（Next Steps）

优先核验已有真实/合成图在模型实际输入尺寸下的职业证据、属性可见性与编辑副作用，再独立冻结源任务；不要按当前模型错误挑最终测试。停止重跑已近天花板或职业真值不明的小面板。只在真实监督/冻结路径有未解决错误时补技术检查。
