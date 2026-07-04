# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

《数学思维实践》CDIO二级项目 (Mathematical Thinking Practice — CDIO Level 2 Project).  
Course: CST4822A, Summer 2026. Team of 3.

Three independent mathematical modeling tasks, each structured as:  
**Mathematical Model Layer → Computational Implementation Layer → Visualization/Verification Layer**

## Repository Structure

- `docs/` — Design documents (任务书, 项目计划, per-task detailed design, report/practice templates)
- `task1/` — Linear Algebra: iterative solvers (Jacobi, Gauss-Seidel) for Ax=b (heat conduction)
- `task2/` — Discrete Math: three-way switch logic (XOR-based)
- `task3/` — Probability & Statistics: Bootstrap estimation
- `output/` (within each task dir) — Generated visualization PNGs

Source code does not yet exist at this stage — only the planning/design documents are in place.

## Tech Stack & Dependencies

- **Python 3** exclusively (no other languages or build systems)
- `numpy` — linear algebra, array operations
- `matplotlib` — all visualizations (DPI=300, unified style)
- `scipy` — optional, for sparse matrix utilities if needed
- Dependencies listed in root `requirements.txt`

## Key Decisions & Conventions (from 项目计划.md)

- **Task assignments**: 同学 A → Task 1, 同学 B → Task 3, 同学 C → Task 2 + report integration
- **Task 2 logic locked**: `L = S1 ⊕ S2 ⊕ S3` (3-way XOR, "odd number of switches ON → light ON")
- **Task 1 recommended scenario**: 1D steady-state heat conduction on a metal rod (两端固定温度的铁棍)
- **Task 3 recommended target**: median estimation with small real dataset (n≈20–30) + synthetic validation

## Visualization Standards

Every chart must have: number, title, and text explaining what it shows and how it supports the conclusion.  
Three chart categories per task:
1. **Structure** — matrix sparsity/heatmap (T1), state-space diagram (T2), sample distribution (T3)
2. **Process** — convergence curves (T1), state transition paths (T2), bootstrap distribution evolution (T3)
3. **Results** — solution comparison plots (T1), truth-table color matrix (T2), confidence interval plots (T3)

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run any task
python task1/heat1d.py
python task2/three_switch.py
python task3/bootstrap_estimate.py

# Generate all outputs for a task (each script self-contained)
python task1/heat1d.py          # produces output/*.png
python task2/three_switch.py    # produces output/*.png
```

There is no test framework, linter, or build step — each script is standalone, produces its own figures, and prints results to stdout.

## Academic Integrity Notes

- AI tools used must be cited in the report (参考资料 section).
- Code must be runnable and reproduce all figures/numbers.
- Each task report section requires original explanation of model, algorithm, chart, and analysis — placeholders from the templates must be replaced.
