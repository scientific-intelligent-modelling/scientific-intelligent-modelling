# Problem Definition / Contributions Figure

## Goal

This figure should communicate, in one glance, why SR-Workbench is needed and what it contributes beyond a collection of wrappers.

## Recommended Layout

Use a left-to-right 3-panel figure.

### Panel A: Fragmented SR Research Workflow

Show the current pain points before SR-Workbench.

Visual elements:
- Multiple heterogeneous method repositories:
  - PySR
  - DRSR
  - LLMSR
  - DSO
  - TPSR
- Multiple heterogeneous datasets:
  - SRBench
  - LLM-SRBench
  - SRSD
- Broken arrows or mismatched interfaces between them
- Labels such as:
  - incompatible environments
  - inconsistent file structures
  - different split conventions
  - inconsistent expression outputs
  - weak OOD protocol

Message:
```
Existing symbolic regression research is fragmented across repositories,
runtime stacks, data schemas, and evaluation conventions.
```

### Panel B: SR-Workbench Core Architecture

Show SR-Workbench as a layered middle substrate.

Recommended stack:
1. Unified Method Substrate
   - wrappers
   - subprocess execution
   - environment isolation
2. Canonical Dataset Substrate
   - train / valid / id_test / ood_test
   - metadata
   - optional formula
3. Skill Layer
   - tool onboarding
   - data homogenization
   - validation
   - protocol compilation
4. Lightweight Audited Evolution Loop
   - adapt
   - compile
   - execute
   - audit
   - review
   - archive

Message:
```
SR-Workbench turns heterogeneous methods and datasets into a common,
auditable research operating layer.
```

### Panel C: What the Platform Enables

Show the resulting benefits.

Suggested bullets or icons:
- reproducible execution
- formula-aware dataset validation
- OOD-aware evaluation
- bounded method improvement
- extensible research workflow

Optional outputs:
- result.json
- canonical equation output
- audit report
- paper-ready experiment table

Message:
```
The platform supports reproducible symbolic regression experimentation
and provides a foundation for future automated improvement.
```

## Caption Draft

```text
From fragmented symbolic regression repositories and heterogeneous benchmark formats to a unified, auditable research substrate. SR-Workbench standardizes method execution, dataset structure, and recurring research operations, then places a lightweight audited evolution loop on top of this substrate. The figure highlights the paper's central claim: the main contribution is not a new symbolic regression model, but a research operating layer that reduces engineering entropy and enables reproducible, extensible SR experimentation.
```

## Design Notes

- Keep the figure conceptual rather than implementation-heavy.
- Avoid too many logos; 4--6 method names and 3 dataset families are enough.
- Use color to distinguish:
  - methods
  - datasets
  - skill layer
  - loop
- If space is limited, collapse the loop into:
  - `Adapt -> Execute -> Audit -> Review`

## What This Figure Must Make Clear

1. The problem is fragmentation, not just model quality.
2. The contribution is a unified substrate, not only wrappers.
3. The loop is bounded and audited, not a free-form autonomous scientist.
