# #6 linear旧PR的整合处理

日期2026-11-17。旧PR head954ae05f616473f0f06af213840664b22c4315d1不再原样覆盖当前MM/ADDMM。以远端hygon-optimization的19d8acfb建立独立整合提案，复制官方#9045的隔离linear_projection.py，再适配旧集成底座的入口。

- 当前MM的select_config优先级保留；只有cfg=None才调用try_linear_projection，未命中继续原_mm_default。
- K1 affine接入现有addmm_kernel90.py的公开addmm入口；其余精度和addmm_out保留现有路径。
- 移植已有限域guard，wide-vocabulary在分配前回退；通用FP64保持原路线。
- 不把旧#6中的整份mm.py或通用addmm.py覆盖当前实现。

定向硬件48项通过；另有7个实际专化域的输入保护、独立FP64参考、默认容差与JIT launch见证。此证据没有完成109工况性能或FG/DAS稳定配对，所以本分支仍是整合候选。旧#6保持OPEN；在完整整合资格完成并合入替代提交后，按“由新整合实现替代”关闭旧PR。

## 底层运行逻辑与原理分析

CPU构造输入→传连续backing到flagos→设备transpose view保持权重stride(1,K)→公开FlagGems mm提供函数→已有配置或try_linear_projection→FP32 static/BF16 static及split-K JIT→同步→CPU FP64参考对比。K1由try_linear_affine执行单个手写乘加kernel；未启用torch.compile/Dynamo/Inductor跨算子融合。split-K使用FP32临时partial及两个kernel，本次不以单kernel时间代替完整调用成本。

本轮直接调用FlagGems提供函数；ATen schema/PrivateUse1 keys为环境/输入记录，不冒充已重新验证整个Torch-FL公共linear分解链。完整模型/dispatcher/fallback覆盖仍需后续109工况与端到端检查。
