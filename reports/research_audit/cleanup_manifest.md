# 实验与工作区安全整理记录

本次遵循先审计、再建立研究索引、最后可逆整理。永久删除0；移动/重命名原实验目录0；修改原结果、日志、权重、冻结协议0；启动/停止/重启实验0。没有将历史41.6GiB释放量算作本次清理收益。

| 操作 | 原始路径 | 处理后路径 | 依据与恢复方式 |
|---|---|---|---|
| archive_regenerable_cache | .pytest_cache | [_archive/research_audit_20261103/pytest_cache](../../_archive/research_audit_20261103/pytest_cache) | 只含可再生测试缓存；未归档训练/评审/数据；缓存可直接移回原路径，索引可移除，不影响实验。 |
| create_research_index | 新建 | [experiments/RESEARCH_INDEX.md](../../experiments/RESEARCH_INDEX.md) | 按RQ归并导航；不移动依赖中的证据；缓存可直接移回原路径，索引可移除，不影响实验。 |
| create_research_index | 新建 | [docs/README.md](../../docs/README.md) | 按RQ归并导航；不移动依赖中的证据；缓存可直接移回原路径，索引可移除，不影响实验。 |

## 为什么保留52个实验原目录

原目录被冻结合同、报告、训练/评审manifest及代码中的绝对路径引用；根项目没有Git恢复保障。已被后续版本取代的实验仍承担失败机制、版本演化和负结果证据；已完成图像/权重还有历史硬链接去重，原位覆写会影响多个路径。因此本次在experiments创建按RQ聚合的RESEARCH_INDEX，而不为表面整齐移动依赖中的历史目录。

旧权重和Janus轨迹的永久删除发生在10-31，详见[历史记录](../../docs/storage_cleanup_20261031.md)及相邻JSON。本次未再次执行任何删除/去重，不把已删除工件标成可恢复归档。

不确定用途或只有CPU/候选资产的pathway、receiver、interaction等保留并标明状态。失败SFT源图、下载失败、DEBUG100、旧评分和原技术错误全部保留。

## 可审计明细

[操作前后SHA256、备份位置及缓存逐文件清单](evidence/cleanup_actions.json)。缓存归档只涉及根目录.pytest_cache，不涉及活动进程缓存。所有主要证据路径保持有效；新增报告链接由validate_audit核验。文档链接修复及原文备份单列[文档整理记录](docs_cleanup_manifest.md)。

## 审计派生材料压缩与归档

初始全资产JSONL清单约51MiB，已无损压缩为[evidence/filesystem_inventory.jsonl.gz](evidence/filesystem_inventory.jsonl.gz)；约447MiB的中间文本语料已压缩归档为[原始读取快照](../../_archive/research_audit_20261103/source_text_index.json.gz)。解压后SHA256与压缩前完全一致，原实验/原文档未删除。四个一次性报告生成/清理脚本移至[_archive中的脚本目录](../../_archive/research_audit_20261103/report_build_scripts/)，保留验证脚本供独立复核。这些均是本次新建的派生产物整理，不是历史实验迁移或永久删除。具体旧/新路径和哈希已追加cleanup_actions.json。
