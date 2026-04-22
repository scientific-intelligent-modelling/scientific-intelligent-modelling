# iaaccn 环境盘点（2026-04-23）

## 范围

这次盘点覆盖：

- CPU 机器：`iaaccn22~29`
- GPU 机器：`iaaccn48~55`

检查的 benchmark 环境：

- `sim_base`
- `sim_llm`
- `sim_dso`
- `sim_tpsr`
- `sim_e2esr`
- `sim_qLattice`
- `sim_iMCTS`

对应算法族：

- `sim_base`
  - `gplearn / pysr / pyoperon`
- `sim_llm`
  - `llmsr / drsr`
- `sim_dso`
  - `dso`
- `sim_tpsr`
  - `tpsr`
- `sim_e2esr`
  - `e2esr`
- `sim_qLattice`
  - `QLattice`
- `sim_iMCTS`
  - `iMCTS`

详细机器表见：

- [iaaccn_env_audit_20260423.csv](./iaaccn_env_audit_20260423.csv)

## 总体结论

### 1. CPU 机 `iaaccn22~29`

当前 CPU 池并不是全环境统一：

- `iaaccn22`
  - 环境最全，`7` 个 benchmark 环境都在
- `iaaccn23~28`
  - 共同具备：
    - `sim_base`
    - `sim_llm`
    - `sim_dso`
  - 共同缺失：
    - `sim_tpsr`
    - `sim_e2esr`
    - `sim_qLattice`
    - `sim_iMCTS`
- `iaaccn29`
  - 只有：
    - `sim_base`
    - `sim_llm`
  - 额外还缺：
    - `sim_dso`

### 2. GPU 机 `iaaccn48~55`

当前 `48~55` 基本可以认为：

- **没有装任何 benchmark 环境**

其中：

- `iaaccn48`
  - 登录时会输出 `.bashrc` 文本噪音
  - 但实际 `~/anaconda3/envs` 是空的
- `iaaccn49~55`
  - 同样没有这批 benchmark 环境

## 按算法看当前可用机器

### 可以直接跑

- `gplearn / pysr / pyoperon`
  - `iaaccn22~29`
- `llmsr / drsr`
  - `iaaccn22~29`
- `dso`
  - `iaaccn22~28`

### 目前只能在单机或极少数机器上跑

- `tpsr`
  - 目前只有 `iaaccn22`
- `e2esr`
  - 目前只有 `iaaccn22`
- `QLattice`
  - 目前只有 `iaaccn22`
- `iMCTS`
  - 目前只有 `iaaccn22`

## 对 E1 的直接影响

如果按当前环境不补装直接跑：

- `W1`
  - `gplearn + llmsr`
  - 可以直接跑
- `W2`
  - `pyoperon + drsr`
  - 可以直接跑
- `W3`
  - `pysr`
  - 可以直接跑
- `W4`
  - `dso`
  - 可以在 `iaaccn22~28` 跑
- `W5`
  - `tpsr`
  - 只能在 `iaaccn22` 单机跑

## 最小建议

### 如果只想推进 E1

最先需要补的环境是：

1. `sim_tpsr`
   - 至少补到 `iaaccn23~25`

这样 `tpsr` 才能和 `gplearn / pysr / pyoperon` 一样按 `22~25` 四机对称分发。

### 如果想把 10 个算法全部纳入最终 leaderboard

下一批应该补的环境是：

1. `sim_e2esr`
2. `sim_qLattice`
3. `sim_iMCTS`

并优先补到：

- `iaaccn23~25`

## 一句话总结

当前 `iaaccn` 机器群的 benchmark 环境可以分成三层：

1. **完整机**
   - `iaaccn22`
2. **E1 主力机**
   - `iaaccn23~28`
3. **只适合 base/llm 的机子**
   - `iaaccn29`
4. **GPU 但当前未装 benchmark 环境**
   - `iaaccn48~55`
