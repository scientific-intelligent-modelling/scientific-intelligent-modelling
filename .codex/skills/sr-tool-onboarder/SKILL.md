---
name: sr-tool-onboarder
description: 接入新的符号回归仓库到当前 scientific-intelligent-modelling 框架时使用。适用于生成 algorithms/<tool>_wrapper 脚手架、注册 toolbox/env 配置、生成 check/check_<tool>.py 验收脚本，并用 manifest + 校验脚本统一接入流程。
---

# SR Tool Onboarder

用于把新的符号回归仓库接入当前项目的统一流程。

## 何时使用

- 用户要新增一个符号回归算法到当前框架。
- 用户要把外部仓库整理成 `scientific_intelligent_modelling/algorithms/<tool>_wrapper` 下的子工具。
- 用户希望减少手工修改 `toolbox_config.json`、`envs_config.json`、`check/check_<tool>.py` 的重复劳动。

## 快速流程

1. 先读接入契约：
   - `references/integration_contract.md`
2. 按仓库类型选择包装模式：
   - Python API / 可编辑安装：读 `references/wrapper_patterns.md`
   - CLI-only：也读 `references/wrapper_patterns.md`
3. 新建或修改 manifest：
   - `tools/sr_onboarder/manifests/<tool>.json`
   - 可直接复制 `tools/sr_onboarder/manifests/example_external_sr.json`
4. 运行脚手架生成器：
   - `python3 tools/sr_onboarder/scripts/create_sr_tool.py --manifest tools/sr_onboarder/manifests/<tool>.json`
5. 补齐生成的 `wrapper.py` 中剩余 TODO。
6. 运行校验：
   - 结构校验：`python3 tools/sr_onboarder/scripts/validate_sr_tool.py --manifest tools/sr_onboarder/manifests/<tool>.json`
   - 运行时校验：`python3 tools/sr_onboarder/scripts/validate_sr_tool.py --manifest tools/sr_onboarder/manifests/<tool>.json --runtime-check`

## 强约束

- 新工具优先使用小写 `tool_name`，避免继续扩散大小写混用的工具 ID。
- 每个新工具优先独立 `env`，除非你能明确证明可安全复用现有环境。
- 包装器必须满足当前 `BaseWrapper` 契约，并能被 `subprocess_runner.py` 动态导入。
- 离线 smoke check 必须先通过，再做在线或远程批量实验。
- API key 只能运行时注入，不能写进仓库。

## 需要重点看的文件

- 包装器基类：`scientific_intelligent_modelling/algorithms/base_wrapper.py`
- 子进程动态加载：`scientific_intelligent_modelling/srkit/subprocess_runner.py`
- 工具注册：`scientific_intelligent_modelling/config/toolbox_config.json`
- 环境注册：`scientific_intelligent_modelling/config/envs_config.json`
- 现有验收脚本目录：`check/`

## 生成器做什么

- 创建 `scientific_intelligent_modelling/algorithms/<tool>_wrapper/`
- 生成 `wrapper.py` 与 `__init__.py`
- 生成 `check/check_<tool>.py`
- 更新 `toolbox_config.json`
- 更新 `envs_config.json`

## 生成器不做什么

- 不会自动猜中外部仓库的真实训练入口。
- 不会自动解决不兼容依赖。
- 不会自动为 CLI-only 工具补完完整的命令行适配。
- 默认不会覆盖已存在但内容不同的文件，除非显式传 `--overwrite`

## 参考资料

- 接入契约：`references/integration_contract.md`
- 包装模式：`references/wrapper_patterns.md`
- 验收规则：`references/acceptance_rules.md`
