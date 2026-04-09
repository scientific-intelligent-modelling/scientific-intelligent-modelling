# Paper Improvement Log

## Score Progression

| Round | Score | Verdict | Key Changes |
|-------|-------|---------|-------------|
| Round 0 (baseline) | 6.0/10 | Almost | 题目已成型，但 `Auto-Review` 叙事弱于标题，方法中的 review 机制不够具体 |
| Round 1 | 7.0/10 | Almost | 强化 review-before-evolve 主线，补充 review 动作边界，并在实验与结论中明确当前验证范围 |
| Round 2 | 7.8/10 | Yes | 压实贡献表述，建立 claim-evidence 对应关系，并提升实验段落的定量说服力 |

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

- 本日志现已包含 **2 轮** 自动进化结果。
- `paper/main_round0_original.pdf` 已保留为本轮修改前的基线快照。
- `paper/main_round1.pdf` 已保留为第 1 轮后的快照。

## Round 2 Review

### Summary

经过第 1 轮修正后，论文已经在系统定位上更加稳定，但仍有两个会影响说服力的问题。第一，贡献列表仍偏“功能列举”，缺少更强的可验证口径。第二，实验虽然真实，但还没有明确告诉读者每组实验到底支持哪条 claim，导致读者需要自己做归纳，削弱了主文的说服效率。

### Strengths

1. 系统定位、标题与方法主线已经一致。
2. 真实数据、真实工具表和真实 pilot runs 让论文有明显落地感。
3. 页数仍可控，适合继续强化论证而不是继续压缩。

### Weaknesses

1. `CRITICAL`: 贡献表述还不够“可证伪/可验证”，更像功能清单而不是论文 claim。
2. `MAJOR`: 实验缺少 claim-evidence mapping，导致说服链条不够直接。
3. `MAJOR`: 摘要缺少一处最强定量结果，不利于快速建立可信度。
4. `MINOR`: 结论可以更明确地点出“这些实验已经支持了本文三条核心 claim”。

## Round 2 Fixes Implemented

1. 将引言中的 contribution bullets 改写为更可验证的 claim 口径，加入 `$10$ tools / $8$ environments` 与 formula verification 的明确描述。
2. 在引言中加入一条总括句，明确当前稿件由三类证据支撑。
3. 在实验开头显式定义 `C1/C2/C3` 三条 claim，并在后续段落中逐一回扣。
4. 在 formula validation 小节中明确说明其直接支持 `C2`。
5. 在 pilot slice 中加入更直接的定量对比，突出 `oscillator2` 上显著的 OOD 差异与 `stressstrain` 上更接近的行为分布。
6. 在摘要中加入最强定量事实之一，即公式校验可达 `$10^{-15}$` 量级 NMSE。
7. 在结论中明确三项实证能力与核心 claims 的对应关系。
