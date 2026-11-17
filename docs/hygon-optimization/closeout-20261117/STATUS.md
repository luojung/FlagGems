# Hygon rsqrt-guarded 收尾状态

整理日期：2026-11-17（Asia/Shanghai）。

保留继续研究：128工况 finite/special 两路验证通过；历史完整设备矩阵79过线、45未过线、4空路径，不构成整体性能验收。通用文件中的修改尚需限定Hygon作用域，不能据此推到其他后端。

个人 fork 分支：`closeout/hygon-rsqrt-guarded-wip-20261117`。原始源码提交：`6d9ac5543fb2a543ec00a7770c8c583f1a9f1f29`。本分支用于保存、复现和继续验证，不声明已完成DAS90验收。

## 后续动作

改为Hygon限定实现并对照已有#8404的覆盖差异；补完整支持域与稳定配对性能，确认45项未达标的实际瓶颈。

## 底层运行逻辑与原理分析

通用rsqrt_func用整数位识别FP32次正规数，mantissa重建正常数后调用rsqrt，再缩放恢复并处理负次正规NaN；Triton/FlagTree/HIP。通用文件影响范围大于Hygon，跨后端没有验证。

本次整理前的原始工作树保持独立；补丁、文件SHA256、历史结果和收尾台账在本地 analysis/hygon-closeout-20261117，实验来源在11月8–14日各原始目录。
