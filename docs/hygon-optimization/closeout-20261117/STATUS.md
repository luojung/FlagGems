# Hygon integration 收尾状态

整理日期：2026-11-17（Asia/Shanghai）。

定向正确性已验证、完整性能待验收：从远端19d8acfb建立；移植官方#9045独立linear实现，仅在现有MM配置未命中时启用projection，K1 affine接入现有Hygon addmm入口；保留#7回归测试，不注册第二套重叠精度实现。尚未合入hygon-optimization。

个人 fork 分支：`integration/hygon-closeout-20261117`。原始源码提交：`19d8acfb98210fe410161ec320b6c9782050af0c`。本分支用于保存、复现和继续验证，不声明已完成DAS90验收。

## 后续动作

定向验证#7长向量精度/stride/out/alpha/beta保护、K1与general-MM回归，再做完整109工况及必要性能/稳定性验证。#6以此替代提案处理，#7先核实测试与适用域；旧PR不自动合并或关闭。

## 底层运行逻辑与原理分析

公开linear经Composite/Autograd及view/t/matmul分解为mm/addmm/bmm，经PrivateUse1/Torch-FL到实际Hygon提供函数；现有MM配置优先，未命中才尝试有域保护的projection。长向量使用既有_precise_vector/finish和FP64 partial，无torch.compile/Inductor跨算子融合。此整合不能继承官方单PR的完整性能资格。

本次整理前的原始工作树保持独立；补丁、文件SHA256、历史结果和收尾台账在本地 analysis/hygon-closeout-20261117，实验来源在11月8–14日各原始目录。

## 本次新增硬件证据

作业1146777：基线36/36、整合提案48/48，COMPLETED / 0:0。作业1146916：2个FP32和5个BF16域两臂均正确，设备stride与host契约一致，实际专化JIT调用见证齐全，COMPLETED / 0:0。BF16 split-K域输出不要求与旧版bitwise一致，双方均通过默认BF16容差；其他6域本次输出hash相同。

首次七域尝试1146841虽数值通过，但实际为通用fallback，原因在设备输入重建布局，未被算作专化覆盖。保留其未资格化回执；修正版先传连续backing再在设备重建转置view。这里是Python/JIT launch见证和同步硬件结果，不是GPU producer-linked profiler trace。

未采FG/DAS性能样本、未完成完整109工况、反向或非默认流验收。相关源码格式检查及4项元数据检查通过；全仓库pre-commit未通过，7个原底座MM文件有继承的E226格式错误，本次没有改动这些文件。
