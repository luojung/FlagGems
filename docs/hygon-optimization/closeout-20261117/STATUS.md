# Hygon sum 收尾状态

整理日期：2026-11-17（Asia/Shanghai）。

待继续优化：11月14日标准 event-time 复测37工况，0达标、5稳定低于90%、32 A/A不稳定；正确性37/37通过。

个人 fork 分支：`closeout/hygon-sum-wip-20261117`。原始源码提交：`c913afc2cd7180b91be546a61842bb42ca1cd855`。本分支用于保存、复现和继续验证，不声明已完成DAS90验收。

## 后续动作

先解决A/A稳定性，再针对稳定未达标工况调优；主指标为冻结版本do_bench event time，历史kernel-sum列保持各自口径。

## 底层运行逻辑与原理分析

元数据重建输入进入PrivateUse1/flagos设备；实际FG路径落到Hygon sum.py。新增域为连续FP32/FP64末维3、1024<=rows<8192，改变BLOCK_M/num_warps，不改变算术；Triton/FlagTree生成HIP kernel。DAS独立boxing/native；11月14日event-time稳定性不足，不能用此前纯kernel点值作为新口径达标结论。

本次整理前的原始工作树保持独立；补丁、文件SHA256、历史结果和收尾台账在本地 analysis/hygon-closeout-20261117，实验来源在11月8–14日各原始目录。
