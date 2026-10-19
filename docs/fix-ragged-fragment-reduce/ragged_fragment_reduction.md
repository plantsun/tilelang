# Ragged Fragment Reduction：复现与布局分析

## 问题概述

Fragment 的归约宽度与线程数不匹配时，`T.reduce_sum` 可能生成不兼容的输入和输出布局，导致编译失败。
以 `threads=128` 为例，`N=64/128/256` 编译成功且数值正确，`N=96/100/120/127` 在归约 lowering 失败。

Issue #2399 对应归约布局问题。Issue #2398 对应 Fragment 上的逐元素循环布局问题，两者均可复现。
归约问题的证据支持 ownership 与 replication 不兼容的解释，具体修复方案仍需正确性验证。

## 实验配置

| 配置项 | 设置 |
| --- | --- |
| GPU | NVIDIA A100 |
| 软件组件 | Python、PyTorch、TileLang、CUDA |
| 数据类型 | float32 |
| 线程数 | 128 |
| 归约输入形状 | `(1, N)` |
| 归约输出形状 | `(1,)` |
| 归约维度 | 1 |

## Python 依赖

Python 运行依赖由 `requirements.txt` 管理，源码构建与开发工具由 `requirements-dev.txt` 管理。
归约复现与回归使用 `requirements-ragged-reduction.txt`，其中引用前两份清单，安装命令为：

```bash
python -m pip install -r requirements-ragged-reduction.txt
```

依赖包含张量计算、编译接口、数值计算、源码构建、测试和静态检查工具。
CUDA Toolkit 与支持 CUDA 代码编译的 C++ 编译器由系统工具链提供。

## 数值验证方法

归约实验保持输入形状、数据类型、线程数和归约维度不变，分别执行编译和 GPU 数值比较。

- 归约输入包含 arange、全 1 和固定随机种子的正态分布数据，参考结果为 `torch.sum(dim=1)`。
- 逐元素实验包含 arange 和固定随机输入，参考结果为 `X * 2`。
- 输出在执行前填充 NaN，GPU 执行结束后同步，再进行比较。
- 归约比较使用 `rtol=0, atol=1e-4`；逐元素比较使用 `rtol=0, atol=0`。
- 编译缓存关闭，避免复用其他配置的编译产物。

只有编译成功且 GPU 输出通过比较才记为 `PASS`。
`COMPILE_FAIL`、`WRONG_RESULT`、`RUNTIME_FAIL` 分别表示编译失败、数值错误和运行失败。

## 归约测试结果

| N | threads | 编译 | GPU 数值 | arange 参考和 | 结果 |
| --- | --- | --- | --- | --- | --- |
| 64 | 128 | 成功 | 三类输入通过 | 2016 | PASS |
| 96 | 128 | 失败 | 未执行 | 4560 | COMPILE_FAIL |
| 100 | 128 | 失败 | 未执行 | 4950 | COMPILE_FAIL |
| 120 | 128 | 失败 | 未执行 | 7140 | COMPILE_FAIL |
| 127 | 128 | 失败 | 未执行 | 8001 | COMPILE_FAIL |
| 128 | 128 | 成功 | 三类输入通过 | 8128 | PASS |
| 256 | 128 | 成功 | 三类输入通过 | 32640 | PASS |

四种失败宽度触发相同的归约布局一致性检查，归约结果布局的 replication 分别为 96、100、120、127。
`N=96` 的三次独立执行均触发同一错误。编译成功的三种宽度均未出现数值错误。

## 失败阶段

`N=96` 能完成 `LayoutInference` 和 `ReducerPlanAndMaterialize`，在 `LowerTileOp` 内的归约 lowering 失败。
出错操作为 `ReduceLowerer<cuda::Reduce>::Lower`，错误信息为 `Inconsistent layouts between src and dst in ReduceOp`。

目标标量布局的 replication 为 128，输入推导出的归约结果布局的 replication 为 96。
两者均绑定线程范围 `[0, 128)`，但绑定范围相同不代表实际参与线程相同。
一致性条件 `dst replication <= reduced replication` 不成立，编译在 GPU kernel 执行前终止。

## Fragment 布局

布局探针读取 `LayoutInference` 生成的 buffer annotation。`j` 表示归约维度坐标，`rep` 表示复制坐标。

| N | 输入 thread 映射 | 每线程 slot | 输入 replication | 实际 thread extent | 归约结果 replication | 目标 replication |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | `rep * 64 + j` | `0` | 2 | 128 | 128 | 128 |
| 96 | `j` | `0` | 1 | 96 | 96 | 128 |
| 100 | `j` | `0` | 1 | 100 | 100 | 128 |
| 120 | `j` | `0` | 1 | 120 | 120 | 128 |
| 127 | `j` | `0` | 1 | 127 | 127 | 128 |
| 128 | `j` | `0` | 1 | 128 | 128 | 128 |
| 256 | `j // 2` | `j % 2` | 1 | 128 | 128 | 128 |

输入和目标布局来自实际推断结果。失败组的归约结果 replication 由编译异常确认，其余组根据输入映射与 `ComputeReducerLayout` 的归约及 condense 规则确定。

`N=96` 只有线程 0–95 持有逻辑输入，目标标量却要求 128 份复制。
`N=64` 的输入复制两份，生成的 CUDA 使用独立的 64 线程 AllReduce 分组，输出没有重复累加。
`N=128/256` 使用 128 线程 AllReduce，`N=256` 每线程持有两个连续值。

归约后的布局由输入布局计算得到，不能将它等同于目标标量布局，也不能仅凭 CTA 线程数推断参与线程数。

## 对照实验

对照实验固定 `N=96, threads=128`，保留全部编译器校验，只改变指定因素。

| 对照 | 改变的因素 | 结果 | 证据含义 |
| --- | --- | --- | --- |
| 移除输入 copy | 用 Parallel 直接生成 arange | 同一 replication 不兼容错误 | 输入 `T.copy` 不是唯一触发因素 |
| 全线程复制输入 | 每个线程持有完整的 96 个输入值 | 编译、执行和 arange 比较通过，结果为 4560 | 相同逻辑宽度存在兼容布局，默认 ownership 选择影响结果 |
| 限制目标复制数 | 目标 replication 指定为 96 | 通过原一致性检查，随后因 AllReduce 宽度不满足约束而失败 | 仅缩小目标 replication 不足以修复问题 |

XOR-butterfly AllReduce 要求逻辑宽度 `threads / scale` 为正的 2 次幂，96 不满足该约束。
全线程复制输入只验证了 arange 数值，不能视为完整的正确性方案或性能优化。

## 原因分析

默认布局推断选择了无法由归约 lowering 实现的 ownership 与 replication 组合。
短 ragged reduction 还涉及无输入线程的单位元、参与线程和输出结果传播，单独放宽检查不能保证正确性。

源级证据包括：

1. 默认输入采用紧凑布局，非规则短宽度只覆盖部分 CTA 线程，输出标量仍被推断为 128 份复制。
2. `ComputeReducerLayout` 将归约维度转换为 replication，再执行 condense。
   `N=64` 的两份输入复制形成 128 份结果；`N=96` 只能形成 96 份结果。
3. `ReduceOpNode::InferLayout` 调用 `ProveFragmentContains` 检查已有目标与归约结果布局的兼容性。
   该检查对 `IsCompletedReplicated()` 存在直接成功条件，而此条件只判断线程表达式是否为 `_rep`。
   归约结果 replication 为 96 的布局也能满足该条件，因而不兼容组合可以通过布局推断。
4. `ReduceLowerer` 要求目标 replication 不超过归约结果 replication，因此在 `128 > 96` 时拒绝该组合。
5. 将目标 replication 改为 96 仍会遇到 AllReduce 的 2 次幂宽度约束，说明目标复制数不是唯一需要解决的问题。

`num_thread / thread_extent` 参与输入布局构造，但直接改为向上取整会改变元素复制与线程 ownership。
现有证据不能证明这种替换能够保持正确结果。

需要验证的修复方向包括：兼容的布局选择、带单位元初始化的 padding、参与线程规划，以及输出消费者所需的结果传播。
收紧 containment 检查后的候选选择与通信代价也需要单独评估。

## 逐元素循环测试结果

Issue #2398 使用 Fragment 上的 `T.Parallel` 执行逐元素乘二。

| N | threads | 结果 | 数值验证或失败阶段 |
| --- | --- | --- | --- |
| 128 | 128 | PASS | arange、随机输入逐元素正确 |
| 129 | 128 | COMPILE_FAIL | LayoutInference，无可用布局 |
| 192 | 128 | COMPILE_FAIL | LayoutInference，无可用布局 |
| 229 | 128 | COMPILE_FAIL | LayoutInference，无可用布局 |
| 256 | 128 | PASS | arange、随机输入逐元素正确 |
| 257 | 128 | COMPILE_FAIL | LayoutInference，无可用布局 |

仅执行布局推断时，129、229、257 同样失败，最终触发候选搜索的 `has_best` 检查。
`DetectInjectiveMapping` 包含静态单射及其他证明后备，候选首次被拒绝的内部原因尚未确定。
“无可用布局”只能确认搜索失败，不能直接证明整数除法或某一条单射检查是根因。

## 修复约束与验证范围

候选修复需要同时满足以下条件：

- 无效 lane 使用 sum 的单位元 0，copy 不越界，每个有效元素恰好贡献一次。
- 参与 warp、shuffle mask 与 barrier 范围一致，输出消费者能够读取正确的标量结果。
- 保留布局一致性和单射校验，不能通过删除断言消除编译错误。
- 全线程复制引入的重复加载和寄存器开销需要评估。
- 回归覆盖七种归约宽度、随机及抵消输入、NaN 输出哨兵、多种线程配置和输出复制。
- 若扩展到 max/min，分别使用正确单位元并增加数值验证。

验证覆盖 13 组形状、三个诊断对照和归约失败的重复执行。
两个既有测试分别检查部分线程归约的数值正确性和非 2 次幂宽度的拒绝行为，结果均为通过。
Python 静态检查及文档格式检查通过；完整回归套件和编译器修复的正确性仍待验证。
