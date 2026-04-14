# 项目补充约定

## PySR / Julia 缓存

- 本机已准备好的 `juliapkg` 缓存目录：
  - `/home/family/pyjuliapkg_pysr`
- 其中可执行 Julia 路径：
  - `/home/family/pyjuliapkg_pysr/pyjuliapkg/install/bin/julia`
- 后续运行 `pysr`、`juliacall`、`juliapkg` 相关任务时，优先复用该缓存，避免再次在线下载 Julia。
- 推荐在命令前显式设置：

```bash
PYTHON_JULIAPKG_PROJECT=/home/family/pyjuliapkg_pysr
```

- 如需做单次 runner 验收，可使用：

```bash
PYTHON_JULIAPKG_PROJECT=/home/family/pyjuliapkg_pysr PYTHONPATH=. python /tmp/sim_runner_smoke/run_one.py pysr ...
```

- 若发现 `juliapkg` 再次尝试下载 Julia，优先检查：
  1. 是否遗漏设置 `PYTHON_JULIAPKG_PROJECT`
  2. 目标环境是否改用了新的 Python/conda 环境
  3. 缓存目录中的 `Project.toml` / `Manifest.toml` 是否被改动
