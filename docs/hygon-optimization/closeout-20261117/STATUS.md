# Hygon mul 收尾状态

整理日期：2026-11-17（Asia/Shanghai）。

待继续优化：11月14日标准 event-time 复测16工况，0达标、14稳定低于90%、2 A/A不稳定；正确性16/16通过。

个人 fork 分支：`closeout/hygon-mul-wip-20261117`。原始源码提交：`5ce293e6fcb4930ce8c295d17ef097247d0d2ca4`。本分支用于保存、复现和继续验证，不声明已完成DAS90验收。

## 后续动作

先解决A/A稳定性，再针对稳定未达标工况调优；主指标为冻结版本do_bench event time，历史kernel-sum列保持各自口径。

## 底层运行逻辑与原理分析

相同dense BF16布局沿物理顺序处理，tail使用mul_dense_bf16_tail_kernel；独立DAS走boxing/native。已有producer/source/codegen证据支持实际命中，不能由机制变化推导event-time过线。

本次整理前的原始工作树保持独立；补丁、文件SHA256、历史结果和收尾台账在本地 analysis/hygon-closeout-20261117，实验来源在11月8–14日各原始目录。
