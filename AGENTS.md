# 符号回归 benchmark 文档整理存档

这份存档把你上传的 PDF 文档整理成了两类 Markdown 文件，方便后续长期保存、检索和复用。

## 解读规则

`source_markdown/` 和 `thematic_notes/` 的主要作用，是保留你当时阶段性的想法、问题意识、研究叙事和设计草稿。

这些内容可以帮助后续代理或协作者理解：

- 你之前是怎么想的
- 你当时为什么会往某个方向推进
- 你的 benchmark 叙事和方法论是如何逐步形成的

但需要明确：

- 这些目录 **不是** 当前代码实现、当前运行协议、当前结果口径的唯一真值来源
- 其中部分内容可能与当前仓库的代码实践、工具接口、实验流程已经发生漂移
- 文档中的愿景、草稿、阶段性判断，不应被机械地视为“当前必须满足的实现约束”

当归档文档与当前实现不一致时，默认按以下优先级理解：

1. 当前代码实现与可运行路径
2. 明确声明“以当前实现为准”的工具文档
3. `source_markdown/` 与 `thematic_notes/` 中的历史思路归档

后续在阅读、分析、回答或修改项目时，必须显式区分：

- **历史想法 / 阶段性叙事**
- **当前代码现实 / 当前可执行协议**

## 目录说明

### 1. `source_markdown/`

按原始 PDF 一一对应整理的 Markdown 版本。  
特点是：

- 保留原文的主要结构与信息顺序
- 对 PDF 中的断行、分页和格式噪声做了轻度清洗
- 在每份文档前加入了“核心内容”摘要，便于快速回看

### 2. `thematic_notes/`

按主题重新整理的综合笔记。  
适合后续继续扩写成：

- 周报 / 月报
- 论文实验设计说明
- benchmark 项目介绍
- 开题、汇报或答辩材料

## 文件清单

### 原始文档整理版

- `source_markdown/01_Symbolic_Regression_Benchmark.md`
- `source_markdown/02_指标.md`
- `source_markdown/03_自问自答.md`
- `source_markdown/04_11月第一周.md`
- `source_markdown/05_11月第三周.md`
- `source_markdown/06_11月第四周.md`
- `source_markdown/07_IDEA.md`
- `source_markdown/08_如何进行实验.md`
- `source_markdown/09_TODO.md`
- `source_markdown/10_实验setting.md`

### 主题整合版

- `thematic_notes/01_项目定位与愿景.md`
- `thematic_notes/02_工程架构与实验基础设施.md`
- `thematic_notes/03_评测漏斗与核心数据集设计.md`
- `thematic_notes/04_实验设置_成本与指标.md`
- `thematic_notes/05_待解决问题与论文叙事.md`
- `thematic_notes/06_这几个月的思考总结.md`

## 使用建议

如果你后面要继续写论文、做汇报或继续补实验，推荐优先从 `thematic_notes/` 里继续改。  
如果你想回溯原始思路和具体语句，去看 `source_markdown/` 更合适。

## 说明

- 这批 Markdown 是基于 PDF 文本提取后重新整理的版本。
- 少量由于 PDF 自动换行产生的断句，已经按语义做了轻度合并与清洗。
- 原始 PDF 仍然建议保留，作为最终存档来源。

# AGENTS相关
这是一个工具相关的项目，所有的错误应该尽可能的提前暴露，尽量少使用try来兜底。
