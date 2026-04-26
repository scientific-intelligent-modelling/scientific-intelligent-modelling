# full uDSR 严格复现计划

## 当前结论

当前仓库里的 `udsr` 是 **uDSR-trunk**，不是 full uDSR。

已经完成的组件：

- DSR 风险寻优主干：基于 DSO `DeepSymbolicOptimizer`
- GP：启用 `gp_meld`
- LM/LINEAR：启用 DSO 中的 `poly` token
- benchmark 集成：统一 runner、normalizer、artifact、分钟级快照、timeout recovery

尚未完成的 full uDSR 论文组件：

- AIF：AI Feynman 式递归问题简化、子问题求解、子解重组
- LSPT：Set Transformer encoder + controller 的预训练初始化

因此在 AIF 和 LSPT 未接入并通过 smoke 前，不能把当前算法标称为 full uDSR。

## 论文口径

NeurIPS 2022 uDSR 论文把 full uDSR 定义为五个组件的统一框架：

- AIF：递归问题简化
- DSR：神经引导搜索
- LSPT：大规模预训练
- GP：遗传编程内循环
- LM/LINEAR：线性模型 token

论文实验设置还包括：

- SRBench 252 个问题
- 每个问题 10 seeds
- 每个子问题最多 2,000,000 expression evaluations
- 每次 run 最长 24h walltime

本项目 benchmark 可以使用更短预算，但论文复现状态必须明确标注预算差异。

## 严格复现验收门

只有同时满足以下条件，才能把工具名从 `uDSR-trunk` 升级为 `full-uDSR`：

- `AIF` 递归简化链路能输出子问题树和重组 metadata
- 子问题叶节点不是用 AI Feynman 自带 BF/PF 结束，而是调用当前 DSO/GP/LINEAR/LSPT trunk 求解
- root problem 必须作为候选保留，保证 AIF 子问题不会劣化原问题结果
- `LSPT` 具备 Set Transformer encoder-controller 架构
- 能加载论文口径的预训练权重，或有可复现实验脚本训练同口径权重
- 结果 artifact 中记录每个组件开关：`aif=true, dsr=true, lspt=rl, gp=true, linear=true`
- 至少通过真实数据集 5 分钟 smoke，输出最终公式、canonical artifact、有限 ID/OOD 指标和子问题日志

## 实施顺序

1. `E0`：完成当前 `uDSR-trunk` 远端 smoke 和 timeout recovery 修复。
2. `E1`：接入 AIF 2.0 代码为可选依赖，先只暴露问题简化树，不直接采用 BF/PF 最终解。
3. `E2`：实现 AIF 子问题 runner，把每个 leaf subproblem 转成标准数据集并调用 `UDSRRegressor`。
4. `E3`：实现子解重组 artifact，保留 root trunk 解和 AIF 重组解的可比指标。
5. `E4`：补 LSPT 架构和权重加载接口；如果没有公开权重，则建立预训练数据生成与训练脚本。
6. `E5`：新增 `udsr_full` manifest、参数快照、check 脚本和真实数据 smoke。

## 当前阻塞点

- 官方 DSO 仓库公开文档支持 DSO、GP-meld、LINEAR/poly，但当前 vendored 代码里没有 AIF 子问题 API。
- AI Feynman 2.0 有独立 PyPI/GitHub 实现，但默认入口是完整 BF/PF 求解器，不是 uDSR 论文所需的“只做递归简化、再交给 trunk 求解”接口。
- LSPT 需要预训练 Set Transformer encoder-controller；当前 vendored DSO 代码没有这套架构和论文权重。

## 下一步最小闭环

先完成 `E1 + E2`，也就是 AIF 递归简化接入和子问题 trunk 求解。LSPT 单独作为下一阶段，因为它需要架构和权重两部分同时对齐。

## 参考来源

- NeurIPS 2022 paper: https://proceedings.neurips.cc/paper_files/paper/2022/file/dbca58f35bddc6e4003b2dd80e42f838-Paper-Conference.pdf
- OpenReview entry: https://openreview.net/forum?id=2FNnBhwJsHK
- DSO official repository: https://github.com/dso-org/deep-symbolic-optimization
- AI Feynman repository: https://github.com/SJ001/AI-Feynman
- AI Feynman PyPI: https://pypi.org/project/aifeynman/2.0.2/
