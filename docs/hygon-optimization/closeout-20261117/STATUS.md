# Hygon unique2 收尾状态

整理日期：2026-11-17（Asia/Shanghai）。

正确性修复候选：空输入、disabled inverse/count 输出的设备/形状/定义张量处理，历史310工况两路严格forward验证通过；第5个fresh性能进程超时，性能baseline无效，未完成全部语义与性能验收。

个人 fork 分支：`closeout/hygon-unique2-wip-20261117`。原始源码提交：`6d9ac5543fb2a543ec00a7770c8c583f1a9f1f29`。本分支用于保存、复现和继续验证，不声明已完成DAS90验收。

## 后续动作

先在当前集成底座复核schema、empty/optional输出、alias、完整forward和out语义，再完成新鲜性能baseline与支持域测试。

## 底层运行逻辑与原理分析

Hygon _unique2先处理empty输出；非空仍由原sort/unique链执行，disabled可选输出补同设备int64空张量。该补丁包含原有torch.sort路径，非新的排序kernel；责任在FlagGems输出契约，性能与后向/流语义尚未完整验收。

本次整理前的原始工作树保持独立；补丁、文件SHA256、历史结果和收尾台账在本地 analysis/hygon-closeout-20261117，实验来源在11月8–14日各原始目录。
