# 本轮实际运行

用户已要求继续实验，只使用 BAGEL 与 SenseNova-U1.5。范围见 [PROTOCOL.md](PROTOCOL.md)。本轮全部新增 GPU 工作使用 GPU0；原有 GPU1 的职业迁移训练继续运行。

本轮已于 2026-10-31 16:36（北京时间）完成生成、两名评审、最终审计、分析、绘图及自动报告。先读[主要结果与中文解释](RESULTS.md)，完整表格见 [REPORT.md](REPORT.md)。主要结论是：本轮没有足够证据支持把引导从4直接降到1作为去偏办法；部分均衡读数上升，但合格图片与明确条件下的联合完成数下降，机器评审分歧也很大。内部 U→G 机制尚未得到验证。

## 已核验

- BAGEL 于 2026-10-31 15:20（北京时间）通过真实 GPU 资格：相同初始噪声重放的 PNG、最终 latent、19 步轨迹完全一致。
- BAGEL 已完成 288 张原始图与 144 张后验均值重构图，全部成功保存。
- SenseNova 已通过原生、捕获及实际噪声回放的一致性资格，完成 288 张原始图，全部成功保存。
- 共 720 张图的文件、计划对应关系及实际噪声检查通过。两名机器评审各完成 726 次调用，回答全部符合格式规则；统计、图表和独立复算均已完成。格式合法不代表图像判断正确，技术通过不代表公平改善。

## 实时记录

- [BAGEL 生成进度](bagel/generation/status.json)
- [BAGEL 资格](bagel/qualification/qualification.json)
- [BAGEL 冻结合同](bagel/freeze.json)
- [BAGEL 重构计划](bagel/codec/plan.jsonl)
- [SenseNova 冻结合同](sensenova/generation/freeze.json)
- [顺序运行状态](generation_queue.json)
- [生成后评审、分析和报告流程](pipeline_status.json)
- [Qwen 正式评审进度](review/qwen/status.json)
- [Phi 正式评审进度](review/phi/status.json)
- [评分结束后的最终审计](AUDIT_post_review.json)
- [独立统计与图表复核](verification/INDEPENDENT_AUDIT_20261031.json)

队列已完成全部阶段，真实记录齐全且相应检查通过后生成了 `REPORT.md`。本轮没有生成或评审执行失败，没有自动换提示、换种子或重试样本。

代码已通过 42 项 CPU 检查。这个数字是执行与分析实现的检查数，不是研究样本数，也不是偏差实验的结果。

状态应从这些输出和进程核对，不能把本文静态进度当作实时计数。过程保留原始图像、实际噪声、哈希、失败和日志；不会自动换种子或删除失败。

## 本轮回答什么

比较降低生成引导后，人物分布、职业内容、条件遵从和图像质量怎样共同变化。BAGEL 另检验再次经过 VAE 编解码能否保留这些可见信息。SenseNova 当前直接在像素空间生成，不运行 VAE 重构。两种模型的结果按各自原生结构解释。
