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

### 10. 用 `tmux` 起远程任务时，优先直接执行脚本文件，不要把长命令内联给 `tmux`

- **优先做法**：

```bash
tmux new-session -d -s smoke_xxx /bin/bash /tmp/run_xxx.sh
```

- 不要优先使用：

```bash
tmux new-session -d -s smoke_xxx "bash -lc '很长的一串命令 ...'"
```

- 原因：
  - 多层 shell 时，命令字符串非常容易被提前吃掉
  - 表面上 `tmux` 会话已经创建，但 pane 里只剩一个空 `bash`
  - 这类错误很隐蔽，往往没有日志、没有报告文件，容易误判为算法失败

### 11. 通过 `/tmp/*.py` 调仓库代码时，要显式设置 `PYTHONPATH=.`

- 如果远程临时脚本在：
  - `/tmp/run_probe_smoke_one.py`
- 但脚本内部要 `import scientific_intelligent_modelling...`
- 必须在仓库根目录下执行，并显式带：

```bash
cd /home/zhangziwen/workplace/scientific-intelligent-modelling
PYTHONPATH=. conda run -n sim_base python /tmp/run_probe_smoke_one.py ...
```

- 否则容易出现：

```text
ModuleNotFoundError: No module named 'scientific_intelligent_modelling...'
```

- 不要假设：
  - `cd repo` 之后 Python 会自动把当前仓库加入模块搜索路径

### 12. 临时 smoke / one-shot 任务不要依赖 launcher 内部的路径兜底，优先把 `dataset_dir` 写成远端绝对路径

- launcher 内部已经做过相对路径到远端真实数据目录的映射。
- 但临时 one-shot 脚本、独立 smoke 脚本、调试脚本未必复用了那层逻辑。

- 因此在远端 smoke CSV 里，优先直接写：

```text
/home/zhangziwen/sim-datasets-data/...
```

- 不要依赖：

```text
sim-datasets-data/...
```

- 原因：
  - 临时脚本常常比正式 launcher 少一层路径映射
  - 很容易出现 `FileNotFoundError`
  - 这种错误会伪装成“算法没跑起来”

### 13. 从 `iaaccn22` 再去访问 `23~29` 时，优先用内网 IP，不要继续用主机别名

- 在本地可以用：
  - `iaaccn22`
  - `iaaccn23`
  - ...

- 但在 `iaaccn22` 内部再次访问其它机器时，优先用：
  - `10.10.100.23`
  - `10.10.100.24`
  - ...

- 不要优先继续用：
  - `iaaccn23`
  - `iaaccn24`
  - ...

- 原因：
  - 远端机器自己的 `~/.ssh/config` 可能带跳板、代理或私钥路径
  - 别名解析后可能再次走外层 hub
  - 容易出现：
    - 缺失私钥
    - 权限拒绝
    - banner 超时
    - 明明内网可达却连不上

### 14. 远端批量重试脚本不要直接 `set -e` 串到底

- 对 8 台机器批量起任务、同步脚本、同步切片时：
  - 单台机器临时连不上是常见情况

- 因此批量重试脚本里：
  - 单机失败要记录并继续
  - 不要让第一台失败直接中断整轮分发

- 推荐形式：

```bash
ssh host '...' || { echo PREP_FAIL host; continue; }
rsync ... || { echo SYNC_FAIL host; continue; }
tmux new-session ... && echo STARTED host || echo START_FAIL host
```

- 适用场景：
  - 网络抖动
  - 某几台机器偶发超时
  - 需要 22 上本地控制器持续重试把剩余机器逐步拉起来

### 15. `task_status.jsonl` 里的 `FileNotFoundError` 不一定是数据路径错了，要先看单任务日志

- 在 `pysr` / `subprocess_runner` 这条链路里，如果底层子进程异常退出、没有成功写出 `.result` 文件：
  - 上层很可能只看到：

```text
FileNotFoundError(2, 'No such file or directory')
```

- 这时不要立刻下结论说：
  - `dataset_dir` 路径错了
  - 远端缺少数据集目录

- 正确顺序：
  1. 先看 `__launcher__/logs/*.log`
  2. 再判断是：
     - 真正的路径缺失
     - 还是底层运行时崩了但没写 `.result`

- 这次真实案例里：
  - `task_status` 表面是 `FileNotFoundError`
  - 但单任务日志里真正根因是：

```text
ERROR: ArgumentError: Package Pkg not found in current path.
```

### 16. 如果远端 `pyjuliapkg_pysr` 连 `import Pkg` 都失败，不要只修 `Project.toml`，直接整目录替换

- 诊断命令：

```bash
/home/zhangziwen/pyjuliapkg_pysr/pyjuliapkg/install/bin/julia --startup-file=no -e 'import Pkg; println(\"pkg_ok\")'
```

- 如果这条都失败，说明问题已经不是：
  - `PySR` 参数
  - `datasets_to_run.csv`
  - `Project.toml` 某一项依赖

- 而是：
  - 远端 Julia 安装/stdlib/共享缓存本身坏了

- 最稳修法：
  1. 停掉正在跑的 `pysr` 任务
  2. 备份旧的 `~/pyjuliapkg_pysr`
  3. 用一份已验证可用的完整缓存整目录覆盖
  4. 再逐台执行一次：

```bash
PYTHON_JULIAPKG_PROJECT=/home/zhangziwen/pyjuliapkg_pysr \
PYTHONPATH=. conda run -n sim_base python -c 'from pysr import PySRRegressor; print("pysr_import_ok")'
```

  5. 预热成功后再重启全量 probe

- 不要优先做：
  - 只改 `Project.toml`
  - 只删 `lock.pid`
  - 只重试全量任务

- 因为这类坏缓存会让整片任务系统性报错。

### 17. 如果远端仓库内存在 `repo/sim-datasets-data` 残缺镜像，launcher 的 `dataset_dir` 解析必须优先命中 `~/sim-datasets-data`

- 当 `dataset_dir` 形如：

```text
sim-datasets-data/...
```

- 远端通常同时存在两份候选路径：
  1. `repo/sim-datasets-data/...`
  2. `~/sim-datasets-data/...`

- 如果仓库内那份只是之前同步留下的**局部镜像/残缺目录**，而 launcher 又优先选了它，
  就会出现：
  - 目录本身存在
  - 但 `metadata.yaml` / 某些 split 文件不存在
  - 上层表现成大量 `FileNotFoundError`

- 正确规则：
  - 只要 `dataset_dir` 以 `sim-datasets-data/` 开头，
  - **优先尝试 `Path.home() / dataset_dir`**
  - 再尝试 `Path.cwd() / dataset_dir`

- 不要默认：
  - `cwd/path` 在远端永远比 `home/path` 更可信

- 这次真实案例里：
  - `task_status` 大面积出现 `FileNotFoundError`
  - 表面像是数据集路径全错
  - 实际是 launcher 命中了仓库内不完整的 `repo/sim-datasets-data/...`
