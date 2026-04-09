# Figures and Tables Plan

## Figures

### Figure 1. Problem Definition and Contributions Overview
- Purpose: one-glance explanation of fragmentation -> SR-Workbench substrate -> audited loop
- Source note: see `paper/figures/problem_definition_contributions_figure.md`
- Suggested placement: end of Introduction or beginning of System Overview

### Figure 2. System Architecture
- Show:
  - unified method substrate
  - canonical dataset substrate
  - skill layer
  - audited evolution loop
- Suggested placement: System Overview and Method

### Figure 3. Audited Evolution Loop
- Show:
  - Adapt
  - Compile
  - Execute
  - Normalize
  - Audit
  - Review
  - Archive
- Emphasize frozen protocol boundaries

## Tables

### Table 1. Integrated Method Coverage
- Columns:
  - method
  - wrapper status
  - environment
  - equation output
  - predict support
  - notes

### Table 2. Canonical Dataset Schema
- Columns:
  - artifact
  - required
  - role
  - current validator support

### Table 3. Pilot End-to-End Results
- Current source:
  - `paper/sections/3_experiments.tex`
- Columns:
  - method
  - dataset
  - ID RMSE
  - OOD RMSE
  - ID NMSE
  - OOD NMSE

## Formal Caption for Figure 1

From fragmented symbolic regression repositories and heterogeneous benchmark formats to a unified, auditable research substrate. SR-Workbench standardizes method execution, dataset structure, and recurring research operations, then places a lightweight audited evolution loop on top of this substrate. The figure highlights the central claim of the paper: the main contribution is not a new symbolic regression model, but a research operating layer that reduces engineering entropy and enables reproducible, extensible symbolic regression experimentation.
