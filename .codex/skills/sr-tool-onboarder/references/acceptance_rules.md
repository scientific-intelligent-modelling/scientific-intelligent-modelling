# 验收规则

## 最低验收标准

每个新工具接入后，至少要通过下面 3 类检查。

1. 结构检查
   - manifest 可解析
   - `wrapper.py` 存在
   - `check/check_<tool>.py` 存在
   - `toolbox_config.json` 与 `envs_config.json` 中有对应注册

2. 导入检查
   - `scientific_intelligent_modelling.algorithms.<tool>_wrapper.wrapper` 可导入
   - manifest 声明的包装器类真实存在

3. 离线 smoke check
   - 能实例化 `SymbolicRegressor("<tool>")`
   - 能跑 `fit`
   - 能拿到非空最优方程
   - 若支持预测，能拿到正确形状的 `predict` 结果

## 推荐扩展检查

- `python3 -m py_compile` 检查新脚本语法
- 若使用新环境，验证 `post_install_commands` 合理
- 若依赖外部仓库，确认 vendor 路径真实存在
- 若工具需要在线服务，在线 check 必须和离线 check 分开

## 不通过时优先排查

1. `tool_name` 与包装器目录不一致
2. `toolbox_config.json` 中的 regressor 类名不一致
3. `envs_config.json` 的 env 名与 toolbox 注册不一致
4. 输入形状没有适配
5. 底层模型无法序列化，导致 fit 后 predict 失效
6. 最优方程提取逻辑写在了 check 脚本里，而不是包装器里
