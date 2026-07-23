# Research Index

Purpose: keep standalone app validation, calibration, and audit outputs in
dated, reviewable bundles.

## Layout
- `studies/YYYY-MM-DD_<slug>/`: app-side investigations.

## Study Rules
- Each study should include a short `README.md`.
- Durable outputs should live in `results/`.
- Local scratch artifacts should live in `artifacts/local/` when needed.
- Promote lasting conclusions into `Agent_Notes/DECISION_LOG.md` or the relevant
  runbook.

## Current Study Types
- Template residual blend calibration.
- Rolling best-ball roster calibration of current, heavier, and full joint
  template residual use, including missed-week and zero-active exposure.
- Full-scaled template production promotion checks, including the legacy 0.30
  rollback and frozen sequential-state comparisons.
- X-pruning availability leakage checks.
- ADP/name join audits.
- ILP runtime/performance checks.
- Sequential best-ball policy correctness, shortlist regret, and runtime gates.
- Sequential default-promotion studies with operational decision banks,
  policy-inert audit banks, and physical cross-slot/cross-round fixtures.
