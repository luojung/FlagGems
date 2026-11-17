# #7 长向量addmm修复的去重处理

日期：2026-11-17。目标底座19d8acfb98210fe410161ec320b6c9782050af0c；旧PR #7 head b3244d9d39c931fd2321f2661707361c390959df。

当前底座的addmm_kernel90.py已含精度kernel。只将PRECISE形参替换为实际调用的True并展开dtype条件后，_precise_vector AST与#7逐节点相同；_precise_vector_finish无需改写就相同。对#7有效FP32长向量域，RK=4096、BO/parts/FP64 partial/warp/fusion选项一致。

本提交保留现有生产实现，只补入#7缺失的36项数值回归，包括R186、stride/storage backing、out、alpha/beta、beta-zero NaN bias、split边界与cancellation。核查证明的是kernel核心和有限正确性覆盖；wrapper在alpha=0、eligibility和异常输入验证方面仍有差异，不宣称全部alias/stream/error语义等价。

处理建议：审查并合入该测试补充后，旧#7可按“生产精度实现已由#1覆盖，缺失测试由本提交补齐”处理为被替代。不要再将addmm_precision.py整体覆盖当前导出。旧PR在本次整理中保留OPEN，未执行关闭或合并操作。

定向硬件回执另随本次验收记录保存；该测试提交不带任何新性能达标承诺。

2026-11-17硬件回执：作业1146777在Hygon BW上完成，现有生产源码配合本测试文件36/36通过，进程及Slurm退出0；未新增生产精度实现。详见targeted-validation-20261117.json。
