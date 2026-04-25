# E1 Remaining Nonvalid Explanation 2026-04-26

## 判定口径

一个 run 被计为有效输出，需要同时满足：数据集身份匹配、存在可解析表达式、表达式在 valid/id_test/ood_test 上都能产生有限预测，并且能得到有限的 ID/OOD 指标。

当前剩余无效项不是数据错配或远端同步问题；`wrong_dataset_collision = 0`。它们是算法在 3600s E1 预算下没有留下完整可评估结果。

## 剩余无效项

- `pysr` `g0007` `CRK22`: `partial_output`, valid_nmse=NaN, id_nmse=NaN, ood_nmse=NaN
- `pyoperon` `g0012` `feynman-i.26.2`: `not_timeout`, valid_nmse=1.8232454008701329, id_nmse=0.9420362393383256, ood_nmse=NaN
- `tpsr` `g0022` `feynman-ii.11.20`: `partial_output`, valid_nmse=NaN, id_nmse=NaN, ood_nmse=NaN
- `tpsr` `g0025` `feynman-bonus.4`: `no_valid_output`, valid_nmse=NaN, id_nmse=NaN, ood_nmse=NaN
- `tpsr` `g0039` `feynman-ii.11.3`: `no_valid_output`, valid_nmse=NaN, id_nmse=NaN, ood_nmse=NaN
- `pyoperon` `g0078` `III.21.20_2_0`: `not_timeout`, valid_nmse=0.2755598773630295, id_nmse=0.2703471637964043, ood_nmse=NaN
- `dso` `g0130` `Keijzer-10`: `partial_output`, valid_nmse=0.0002071856507272, id_nmse=NaN, ood_nmse=0.0001544311185422
- `pysr` `g0130` `Keijzer-10`: `partial_output`, valid_nmse=1.3137600289333627e-32, id_nmse=NaN, ood_nmse=1.04924468663337e-32
- `dso` `g0141` `Keijzer-15`: `partial_output`, valid_nmse=0.0478951076538949, id_nmse=NaN, ood_nmse=NaN
- `pysr` `g0143` `MatSci8`: `partial_output`, valid_nmse=2.1726618911947393e-09, id_nmse=NaN, ood_nmse=1.88619729472192e-08
- `tpsr` `g0174` `feynman-ii.35.18`: `partial_output`, valid_nmse=NaN, id_nmse=NaN, ood_nmse=NaN
- `tpsr` `g0175` `feynman-bonus.2`: `partial_output`, valid_nmse=NaN, id_nmse=NaN, ood_nmse=NaN
- `pyoperon` `g0184` `III.15.14_1_0`: `not_timeout`, valid_nmse=0.8902041412518361, id_nmse=0.8440711574259224, ood_nmse=NaN
- `pyoperon` `g0193` `II.21.32_3_0`: `not_timeout`, valid_nmse=0.482493165826908, id_nmse=0.5105030673248439, ood_nmse=NaN
- `pyoperon` `g0194` `II.21.32_2_0`: `not_timeout`, valid_nmse=0.4163492185306029, id_nmse=0.4474141716559776, ood_nmse=NaN

## 代表性例子

### `pyoperon` / `g0193` / `II.21.32_3_0`

- 原因：`log(tanh(...))` 在 OOD 上可能遇到非正输入，导致 OOD 预测出现 NaN；ID/valid 可算，但 OOD 指标不可算。
- 表达式：`((-3.164) + ((-9.403) * sin(log(tanh((1.084 * X1))))))`
- 指标：valid_nmse=0.482493165826908, id_nmse=0.5105030673248439, ood_nmse=NaN

### `pyoperon` / `g0012` / `feynman-i.26.2`

- 原因：表达式包含 `log(sin(...))`，当测试点上 `sin(...) <= 0` 时数学域非法，OOD 指标为 NaN。
- 表达式：`((-8.525) + (15.244 * (log(sin((0.933 * X4))) * ((((-0.101) * X4) + ((-1.435) * X2)) / (sqrt(1 + 0.066 ^ 2))))))`
- 指标：valid_nmse=1.8232454008701329, id_nmse=0.9420362393383256, ood_nmse=NaN

### `pysr` / `g0130` / `Keijzer-10`

- 原因：表达式 `exp(x1 * log(x0))` 在 `x0 <= 0` 的 split 上不可评估，因此只得到部分 split 指标。
- 表达式：`exp(x1 * log(x0))`
- 指标：valid_nmse=1.3137600289333627e-32, id_nmse=NaN, ood_nmse=1.04924468663337e-32

### `dso` / `g0141` / `Keijzer-15`

- 原因：表达式包含 `1/(2*x2)`、`sin(...)` 分母和 `log(...)`，同时存在除零/非正 log 输入风险，导致 ID/OOD 指标缺失。
- 表达式：`x2*log(x2/sin(x2 + 1/(2*x2)))`
- 指标：valid_nmse=0.0478951076538949, id_nmse=NaN, ood_nmse=NaN

### `tpsr` / `g0025` / `feynman-bonus.4`

- 原因：跑满预算后没有留下 canonical artifact 或可解析表达式，属于 no_valid_output。
- 表达式：`NaN`
- 指标：valid_nmse=NaN, id_nmse=NaN, ood_nmse=NaN

### `tpsr` / `g0175` / `feynman-bonus.2`

- 原因：表达式含多层高次幂，预测容易溢出为非有限值，valid/id/ood 指标都不可算。
- 表达式：`(0.002 + ((0.001 + (0.006 * ((0.001 + (0.9 * ((70.0 + (0.001 * x_3)))**3)))**3)) * (0.001 + (-0.9 * (((0.001 + (0.009 * x_1)) + ((0.001 + (0.009 * x_0)) + ((0.001 + (0.009 * x_5)) + (0.009 + (0.9 * ((0.009 + (90.0 * 0)))**2))))))**3))))`
- 指标：valid_nmse=NaN, id_nmse=NaN, ood_nmse=NaN

