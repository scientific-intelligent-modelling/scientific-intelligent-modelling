# stressstrain 200-generation environment and best formulas

This file records the environment observed on `iaaccn22` and the best formulas archived in this bundle.

## Environment Snapshot

Collected from `iaaccn22` at `2026-04-25 00:23 CST`.

| Item | Value |
| --- | --- |
| SSH alias | `iaaccn22` |
| Hostname | `cpu-4` |
| OS/kernel | `Linux cpu-4 4.15.0-213-generic x86_64` |
| CPU | `AMD EPYC 7702 64-Core Processor` |
| CPU layout | `2 sockets, 64 cores/socket, 2 threads/core, 256 logical CPUs` |
| Memory | `251G total` |
| GPU | `nvidia-smi: unavailable` |
| Disk checked | `/dev/sdb2`, `880G total`, `605G available` |
| Remote code root checked | `/home/zhangziwen/workplace/scientific-intelligent-modelling` |
| Remote repo note | The checked remote directory is a package-root copy without parent `.git` metadata. Wrapper file hashes are recorded below instead. |

## Dependency Versions

### PySR Environment

PySR uses conda env `sim_base`.

| Component | Version |
| --- | --- |
| Python | `3.10.20` |
| Julia | `1.12.6` |
| PySR | `1.5.9` |
| SymbolicRegression.jl | `1.11.3` |
| PythonCall.jl | `0.9.26` |
| juliacall | `0.9.26` |
| juliapkg | `0.1.23` |
| numpy | `2.2.6` |
| pandas | `2.3.3` |
| scipy | `1.15.3` |
| scikit-learn | `1.7.2` |
| sympy | `1.14.0` |
| PyYAML | `6.0.3` |

### LLM-SR / DRSR Environment

LLM-SR and DRSR use conda env `sim_llm`.

| Component | Version |
| --- | --- |
| Python | `3.10.18` |
| numpy | `1.26.4` |
| pandas | `2.2.1` |
| scipy | `1.12.0` |
| sympy | `1.13.1` |
| requests | `2.31.0` |
| PyYAML | `6.0.3` |

LLM calls are made through the repository `requests` clients, not through an installed `openai` package.

### Remote Wrapper Hashes

| File on `iaaccn22` | SHA256 |
| --- | --- |
| `algorithms/drsr_wrapper/wrapper.py` | `823ecadf5b35b3509ea1f31cc6cb3b6d8d0a8ec2af577e3dae0b222eaa27ad72` |
| `algorithms/llmsr_wrapper/wrapper.py` | `dd240d6bed347c22d6604fecfde044b2f5dc23f1c9bec6da345fe0119b6b4b52` |
| `algorithms/pysr_wrapper/wrapper.py` | `aa80b817b77a1c601d54405603e0e998985fb2e55b1f4a18c904cbcf383b4059` |

## Shared Data And LLM Configuration

| Item | Value |
| --- | --- |
| Dataset | `stressstrain` |
| Features | `strain`, `temp` |
| Target | `stress` |
| Train rows | `2161` |
| ID test rows | `1442` |
| OOD test rows | `738` |
| LLM host | `api.bltcy.ai` |
| LLM model | `blt/gpt-3.5-turbo` |
| LLM max tokens | `1024` |
| LLM temperature | `0.6` |
| LLM top_p | `0.3` |
| Credential handling | `api_key` is intentionally redacted in `shared/llm.config.redacted.json`. |

## Key Parameter Settings

| Tool | Environment | Main settings |
| --- | --- | --- |
| `drsr` | `sim_llm` | `niterations=200`, `samples_per_iteration=4`, `max_params=10`, `max_samples=800`, `samples_per_prompt=4`, `evaluate_timeout_seconds=20`, configured `timeout_in_seconds=3600`, `persist_all_samples=false` |
| `llmsr` | `sim_llm` | `niterations=200`, `samples_per_iteration=4`, `max_params=12`, configured `timeout_in_seconds=3600`, `inject_prompt_semantics=false`, `persist_all_samples=false` |
| `pysr` | `sim_base` | `niterations=200`, `population_size=64`, `populations=8`, `ncycles_per_iteration=500`, `maxsize=30`, `maxdepth=10`, `parsimony=0.001`, `precision=32`, `deterministic=true`, `parallelism=serial`, `procs=1`, `model_selection=best` |

PySR operator set:

```text
binary_operators = ["+", "-", "*", "/"]
unary_operators = ["square", "cube", "exp", "log", "sin", "cos"]
constraints = {"/": [-1, 9], "square": 9, "cube": 9, "exp": 7, "log": 7, "sin": 9, "cos": 9}
nested_constraints = {
  "exp": {"exp": 0, "log": 1},
  "log": {"exp": 0, "log": 0},
  "square": {"square": 1, "cube": 1, "exp": 0, "log": 0},
  "cube": {"square": 1, "cube": 1, "exp": 0, "log": 0}
}
```

Note: DRSR observed runtime exceeded the configured `timeout_in_seconds=3600` in both archived runs. The archived metrics use the completed final results.

## Best Formula By Tool

The following "best" row is selected by lowest OOD RMSE among the two archived reruns for each tool.

| Tool | Selected batch | ID RMSE | OOD RMSE | Formula |
| --- | --- | ---: | ---: | --- |
| `drsr` | `rerun200_llm_drsr_20260406_111429` | `0.0481463838` | `0.0401321623` | `p0 + p1*x0 + p2*x0^2 + p3*exp(-k*x0) + p5*sigmoid(k*(x0-xc)) + p6*log(1+abs(x1)+eps) + p7*x1^2 + p8*abs(x0)^0.5*abs(x1)^1.5 + p9*x0*x1` |
| `llmsr` | `rerun200_llm_drsr_20260406_111429` | `0.0555734593` | `0.0499374750` | `p0 + p1*exp(-p2*T)*eps + p3*eps^2/(1+p4*eps^2+1e-8) + p5*T + p6*T^2 + p7*T^3 + p8*eps*T + p9*eps^2*T + p10*tanh(100*eps) + 0.5*p10*eps^3 + p11*log1p(eps)_positive + 0.3*p11*eps^3*T^2 + 0.05*p5*eps^2*T^2` |
| `pysr` | `rerun200_llm_drsr_20260406_111429` | `0.0539810568` | `0.0541847773` | `x0*(-0.23151687 + cos(x1/(-0.7882476))/(x0 + 0.009546288))` |

Variables: `x0 = strain`, `x1 = temp`, `eps = strain`, `T = temp`, `stress = y`.

## Best Formula Parameters

### DRSR selected formula

Source: `runs/rerun200_llm_drsr_20260406_111429/drsr/artifacts/top_samples/top01_samples_643.json`

```text
params = [
  0.8130894612086434,
  0.5473169189779569,
  -1.2081764140259132,
  -0.8011479307919599,
  -79.35028597195576,
  0.07308422026840106,
  0.05591541463791215,
  -0.035771866057800555,
  -2.40467395141181,
  2.1112950870898604
]
k = abs(p4) = 79.35028597195576
xc = abs(p3) = 0.8011479307919599
train_score = -0.0022622921115126524
```

### LLM-SR selected formula

Source: `runs/rerun200_llm_drsr_20260406_111429/llmsr/artifacts/top_samples/top01_samples_574.json`

```text
params = [
  -0.045523565162647994,
  12.312966669441272,
  0.1779700087986147,
  -7.739377963060558,
  0.1899916452038879,
  1.0278299943309386,
  -2.220209894895133,
  1.0816285070826086,
  -1.7662894303919265,
  7.025929107386874,
  0.6005532654901811,
  -10.138843999203509
]
train_mse = 0.00310469518850044
train_nmse = 0.040319416738688156
```

### PySR selected formula

Source: `runs/rerun200_llm_drsr_20260406_111429/pysr/result.json`

```text
formula = x0*(-0.23151687 + cos(x1/(-0.7882476))/(x0 + 0.009546288))
sympy = x0*(-0.23151687 + cos(1.2686369105342*x1)/(x0 + 0.009546288))
inline_constants = [-0.23151687, -0.7882476, 0.009546288]
parameter_values = null
ast_node_count = 14
tree_depth = 6
```

## All Six Run Formula Parameters

| Batch | Tool | ID RMSE | OOD RMSE | Top-sample parameters / constants |
| --- | --- | ---: | ---: | --- |
| `rerun200_llm_drsr_20260405_150543` | `drsr` | `0.0433494665` | `0.0401904881` | `[-0.12781552052053868, 0.03390443501660197, 0.038657536337436155, -0.3644948068223763, 0.3302841941438914, -0.6065881564777371, -0.42810172464533836, 80.16739170091819, -1.0752324475355275, 0.7869404800976196]`; the executable function allocates 14 slots, so `p10..p13` default to `0.0`. |
| `rerun200_llm_drsr_20260405_150543` | `llmsr` | `0.0558788980` | `0.1229632315` | `[106.00453859081094, 1.2123945545849213, -149.4789396533355, -1.4334092598449237, -1.6348739460336192, 42.657088971714956, 2.219798704282065, -0.7002314941785567, 122.7235999261276, -7.3804049780379835, 0.5962066609279294, 66.50538789676153]` |
| `rerun200_llm_drsr_20260405_150543` | `pysr` | `0.0585231917` | `0.0681401524` | `formula = x0*(-0.19282249 + cos(x1)**2/(x0 + 0.008700118)); inline_constants = [-0.19282249, 0.008700118]` |
| `rerun200_llm_drsr_20260406_111429` | `drsr` | `0.0481463838` | `0.0401321623` | `[0.8130894612086434, 0.5473169189779569, -1.2081764140259132, -0.8011479307919599, -79.35028597195576, 0.07308422026840106, 0.05591541463791215, -0.035771866057800555, -2.40467395141181, 2.1112950870898604]` |
| `rerun200_llm_drsr_20260406_111429` | `llmsr` | `0.0555734593` | `0.0499374750` | `[-0.045523565162647994, 12.312966669441272, 0.1779700087986147, -7.739377963060558, 0.1899916452038879, 1.0278299943309386, -2.220209894895133, 1.0816285070826086, -1.7662894303919265, 7.025929107386874, 0.6005532654901811, -10.138843999203509]` |
| `rerun200_llm_drsr_20260406_111429` | `pysr` | `0.0539810568` | `0.0541847773` | `formula = x0*(-0.23151687 + cos(x1/(-0.7882476))/(x0 + 0.009546288)); inline_constants = [-0.23151687, -0.7882476, 0.009546288]` |

Exact executable formula sources are kept in each `runs/<batch>/<tool>/result.json`. For LLM-SR and DRSR, the corresponding optimized parameter vectors are kept in `runs/<batch>/<tool>/artifacts/top_samples/top01_*.json`.
