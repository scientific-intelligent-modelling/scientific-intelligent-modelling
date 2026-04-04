from __future__ import annotations

from pathlib import Path
import argparse
import json
import subprocess
import time


DEFAULT_HOSTS = [
    "iaaccn22",
    "iaaccn23",
    "iaaccn24",
    "iaaccn25",
    "iaaccn26",
    "iaaccn27",
    "iaaccn28",
    "iaaccn29",
]


# 目标是找出 PySR 在远程机器上“速度起来”的并行区间，
# 因此主要扫描 populations / procs 的组合，其他参数保持不变。
DEFAULT_MATRIX = [
    {"label": "cfg01_p1_pop1", "niterations": 300, "population_size": 64, "populations": 1, "procs": 1, "maxsize": 30},
    {"label": "cfg02_p4_pop4", "niterations": 300, "population_size": 64, "populations": 4, "procs": 4, "maxsize": 30},
    {"label": "cfg03_p8_pop8", "niterations": 300, "population_size": 64, "populations": 8, "procs": 8, "maxsize": 30},
    {"label": "cfg04_p16_pop16", "niterations": 300, "population_size": 64, "populations": 16, "procs": 16, "maxsize": 30},
    {"label": "cfg05_p32_pop32", "niterations": 300, "population_size": 64, "populations": 32, "procs": 32, "maxsize": 30},
    {"label": "cfg06_p8_pop16", "niterations": 300, "population_size": 64, "populations": 8, "procs": 16, "maxsize": 30},
    {"label": "cfg07_p16_pop32", "niterations": 300, "population_size": 64, "populations": 16, "procs": 32, "maxsize": 30},
    {"label": "cfg08_p32_pop8", "niterations": 300, "population_size": 64, "populations": 32, "procs": 8, "maxsize": 30},
]


def build_remote_script(repo_root: str, batch: str, cfg: dict) -> str:
    # 用远程脚本文件承接参数，避免 tmux/ssh 多层引号导致 JSON 失真。
    return f"""#!/usr/bin/env bash
set -euo pipefail
REPO={repo_root}
BATCH={batch}
RUN_NAME={cfg['label']}
cd "$REPO"
mkdir -p "$BATCH/logs" /tmp/dcodex_runs
cat > /tmp/dcodex_runs/run_${{RUN_NAME}}.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd {repo_root}
conda run --no-capture-output -n sim_base python /tmp/dcodex_runs/run_stressstrain_pysr_benchmark.py \\
  --output-root {batch} \\
  --label {cfg['label']} \\
  --niterations {cfg['niterations']} \\
  --population-size {cfg['population_size']} \\
  --populations {cfg['populations']} \\
  --procs {cfg['procs']} \\
  --maxsize {cfg['maxsize']} \\
  --random-state 1314 \\
  --progress \\
  --verbosity 1 \\
  2>&1 | tee {batch}/logs/{cfg['label']}.log
SH
chmod +x /tmp/dcodex_runs/run_${{RUN_NAME}}.sh
tmux has-session -t "${{RUN_NAME}}" 2>/dev/null && tmux kill-session -t "${{RUN_NAME}}" || true
tmux new-session -d -s "${{RUN_NAME}}" "bash /tmp/dcodex_runs/run_${{RUN_NAME}}.sh"
echo STARTED:${cfg['label']}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-name",
        default=f"bench_results/stressstrain_pysr_speed_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument(
        "--repo-root",
        default="/home/zhangziwen/projects/scientific-intelligent-modelling",
    )
    parser.add_argument(
        "--hosts",
        nargs="*",
        default=DEFAULT_HOSTS,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if len(args.hosts) < len(DEFAULT_MATRIX):
        raise ValueError("可用机器数不足以覆盖默认 8 组测速配置")

    runner = Path("check/run_stressstrain_pysr_benchmark.py").resolve()
    if not runner.exists():
        raise FileNotFoundError(f"缺少 runner 脚本: {runner}")

    assignments = list(zip(args.hosts[: len(DEFAULT_MATRIX)], DEFAULT_MATRIX))
    print(json.dumps({"batch": args.batch_name, "assignments": assignments}, ensure_ascii=False, indent=2))

    if args.dry_run:
        return

    for host, cfg in assignments:
        subprocess.run(["ssh", host, "mkdir -p /tmp/dcodex_runs"], check=True)
        subprocess.run(["scp", str(runner), f"{host}:/tmp/dcodex_runs/run_stressstrain_pysr_benchmark.py"], check=True)
        local_script = Path(f"/tmp/{host}_{cfg['label']}.sh")
        local_script.write_text(build_remote_script(args.repo_root, args.batch_name, cfg), encoding="utf-8")
        subprocess.run(["scp", str(local_script), f"{host}:/tmp/{host}_{cfg['label']}.sh"], check=True)
        proc = subprocess.run(["ssh", host, f"bash /tmp/{host}_{cfg['label']}.sh"], capture_output=True, text=True)
        print(f"===== {host} =====")
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr)
            raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
