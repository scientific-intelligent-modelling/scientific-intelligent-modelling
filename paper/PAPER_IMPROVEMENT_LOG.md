# Paper Improvement Log

## Score Progression

| Round | Score | Verdict | Key Changes |
|-------|-------|---------|-------------|
| Round 0 (baseline) | 6.0/10 | Almost | 题目已成型，但 `Auto-Review` 叙事弱于标题，方法中的 review 机制不够具体 |
| Round 1 | 7.0/10 | Almost | 强化 review-before-evolve 主线，补充 review 动作边界，并在实验与结论中明确当前验证范围 |

## Round 1 Review

### Summary

当前稿件已经具备清晰的系统论文骨架，但题目中的 `Auto-Review and Evolution` 与正文主线仍有轻微脱节。更具体地说，论文已经很好地说明了统一 substrate、schema 和 skills，但对 `review` 这一控制机制的定义、边界和作用还不够突出，容易让读者把系统误解为“包装器集合 + 审计”，而不是“review-gated evolution framework”。

### Strengths

1. 故事主线已经稳定，系统贡献边界较清楚。
2. 数据 schema、工具 registry、Figure 1 和 pilot slice 能支撑真实实现感。
3. 当前篇幅控制较好，主文没有被背景和 related work 撑爆。

### Weaknesses

1. `CRITICAL`: 标题强调 `Auto-Review`，但摘要、引言和方法没有把 review 机制定义成一等公民。
2. `MAJOR`: 方法中的 `Review and Archive` 段落过轻，没有说明 reviewer 读什么、能做什么、不能做什么。
3. `MAJOR`: 实验部分没有明确告诉读者“这一版验证的是 auto-review/evolution 所需基础设施，而不是已经完成闭环强化结果”。
4. `MINOR`: 结论没有把 `review-before-evolve` 作为下一阶段最关键的验证点。

## Round 1 Fixes Implemented

1. 在摘要中加入 `run artifacts are first reviewed` 的表述，把 review 明确成 evolution 前置控制器。
2. 在引言中补出 `review before evolve` 设计原则，并把最后一条贡献改成 `review-first audited evolution loop`。
3. 在方法章节开头与 loop 小节中补充 review gating 逻辑，明确 reviewer 的动作集合与红线边界。
4. 在实验与限制中明确当前版本验证的是支持 auto-review 的基础设施，而非完整 autonomous improvement results。
5. 在结论中把下一阶段重点收束到固定预算下验证完整的 review-before-evolve loop。

## Notes

- 本轮按用户要求只做了 **1 轮** 自动进化，没有继续执行第 2 轮。
- `paper/main_round0_original.pdf` 已保留为本轮修改前的基线快照。
