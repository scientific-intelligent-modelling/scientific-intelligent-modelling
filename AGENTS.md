# 项目补充约定

## PySR / Julia 缓存

- 本机已准备好的 `juliapkg` 缓存目录：
  - `/home/family/pyjuliapkg_pysr`
```bash
PYTHON_JULIAPKG_PROJECT=/home/family/pyjuliapkg_pysr PYTHONPATH=. python /tmp/sim_runner_smoke/run_one.py pysr ...
```

## 远程服务器相关
远程服务器有 iaaccn22~29,这里是纯CPU机子
另外，还有iaaccn48~55，这里每台机子都有6~8卡的3090环境，如果没有必要，我会

## 远程执行与同步注意事项

### 1. 远程执行 Python / Shell 复杂逻辑时，优先发脚本，不要内联长命令

- 对远程机器执行多行 Python、包含引号、f-string、here-doc、列表/字典字面量时：
  - **优先做法**：先把脚本写到本地临时文件，再用 `scp` 发到远端 `/tmp/*.py`，最后执行：

```bash
scp /tmp/do_remote_check.py iaaccn22:/tmp/do_remote_check.py
ssh iaaccn22 'python /tmp/do_remote_check.py'
```

- 不要优先使用这类高风险形式：

```bash
ssh iaaccn22 'python - <<\"PY\" ... PY'
ssh iaaccn22 'python -c \"...很长的 Python 代码...\"'
```

- 原因：
  - 容易出现引号转义错误
  - 容易出现 here-doc 边界错误
  - 容易在嵌套 `ssh -> ssh -> python -c` 时被 shell 提前解释
  - 一旦出错，远端日志很难读

### 2. 多跳远程时，本地先上 `iaaccn22`，再由 `iaaccn22` 内网分发

- 当本地到 `iaaccn23~29` 的链路不稳定时：
  - **优先做法**：
    1. 本地先同步到 `iaaccn22`
    2. 再由 `iaaccn22` 通过内网 `10.10.100.23~29` 分发

- 典型模式：

```bash
rsync ... iaaccn22:/home/zhangziwen/...
ssh iaaccn22 'rsync ... 10.10.100.23:/home/zhangziwen/...'
```

- 不要在本地同时直接 fan-out 到 `iaaccn22~29`，链路不稳时很容易出现：
  - 部分机器成功
  - 部分机器卡死
  - 部分机器超时
  - 难以判断到底是同步失败还是代理链失败

### 3. 对远程连通性探测要加超时

- 所有远程探测、抽样校验、批量执行，默认加：
  - `timeout`
  - `ssh -o ConnectTimeout=...`
  - `ssh -o BatchMode=yes`

- 推荐形式：

```bash
timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 iaaccn22 'hostname'
```

- 原因：
  - 避免卡住本地会话
  - 避免大量僵尸 `ssh -W` 代理进程残留

### 4. 批量同步数据文件时，不要误以为 Git 会带过去

- `sim-datasets-data/` 下的数据文件通常**不在主仓库 Git 跟踪内**。
- 因此：
  - 本地修复了 CSV
  - **不等于**远端仓库自动有这些修复

- 对数据文件修复必须显式同步：
  - 用 `rsync/scp`
  - 同步到远端真实数据目录

### 5. 远端真实数据目录与仓库目录不是同一个位置

- 远端仓库根目录：
  - `/home/zhangziwen/workplace/scientific-intelligent-modelling`

- 远端真实数据目录：
  - `/home/zhangziwen/sim-datasets-data`

- 不要默认远端数据就在：
  - `repo/sim-datasets-data/...`

- 真实运行、白名单核验、launcher 读取时，优先按：
  - `/home/zhangziwen/sim-datasets-data/...`
 处理。

### 6. 对 `datasets_to_run.csv` 做远端校验时，要做路径映射

- `datasets_to_run.csv` 里的 `dataset_dir` 可能是：
  - 相对路径，例如 `sim-datasets-data/...`
  - 本地路径风格

- 在远端做全量校验时，必须先映射到：
  - `/home/zhangziwen/sim-datasets-data/...`

- 否则会出现假阴性：
  - 文件其实存在
  - 但脚本误报“数据集目录不存在”

### 7. 远端 launcher 必须在正确环境里启动

- `launch_pysr_probe.py` 必须从 `sim_base` 启动
- `launch_llmsr_probe.py` 必须从 `sim_llm` 启动

- 原因：
  - launcher 内部子任务会继承 `sys.executable`
  - 如果启动环境错了，后续所有 worker 都会错环境

### 8. 遇到 Git LFS pointer，先判断是数据没落下来，不要直接误判为列名错误

- 若 CSV 首行像这样：

```text
version https://git-lfs.github.com/spec/v1
```

- 这说明文件还是 LFS pointer，不是真实 CSV。
- 这时不能直接下结论说：
  - `metadata.target.name` 和 CSV 列名不一致

- 正确顺序：
  1. 先确认是不是 LFS pointer
  2. 如果是，先补真实数据
  3. 再做目标列/表头一致性判断

### 9. 当前这批双探针数据修复的真实规则

- 不要误记为“改了 metadata”。
- 实际修复规则是：
  1. 批量修了 `343` 个数据集的 `ood_test.csv` 表头
     - 把最后一列 `target`
     - 改成 `metadata.yaml` 里的真实 `target.name`
  2. 通过 `ModelScope + git-lfs` 补齐了最后 `3` 个 `srbench1.0/feynman` 数据集的真实 CSV

- 后续如需在新机器复刻修复，按这两个规则操作，不要直接批量改 metadata。
