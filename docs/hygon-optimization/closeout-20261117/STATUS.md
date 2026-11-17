# Hygon silu 收尾状态

整理日期：2026-11-17（Asia/Shanghai）。

保留探索：rcp候选171工况finite/special验证通过，ISA/PMC证明除法链缩短；代表大工况仅改善约1.33%，小工况观测回退约3.53%；独立确认未完成，无发布性能结论。

个人 fork 分支：`closeout/hygon-silu-wip-20261117`。原始源码提交：`6d9ac5543fb2a543ec00a7770c8c583f1a9f1f29`。本分支用于保存、复现和继续验证，不声明已完成DAS90验收。

## 后续动作

用新鲜A/A与AB/BA样本确认低幅收益和特殊值语义；收益不稳定时保留为探索，不晋升。

## 底层运行逻辑与原理分析

Hygon silu_forward将完整fdiv链改为v_rcp_f32乘x，backward未修改；既有ISA/PMC支持VALU和VGPR减少，launch/访存及测量稳定性可能限制收益，尚无独立性能晋升证据。

本次整理前的原始工作树保持独立；补丁、文件SHA256、历史结果和收尾台账在本地 analysis/hygon-closeout-20261117，实验来源在11月8–14日各原始目录。
