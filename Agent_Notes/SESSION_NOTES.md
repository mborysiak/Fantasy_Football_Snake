# Session Notes Landing

Last updated: 2026-07-19

## Project Objective

Maintain and improve the Streamlit snake-draft best-ball app that consumes the
fantasy football simulation database, samples ADP/projection/weekly-template
outcomes, and solves roster recommendations with the ILP optimizer.

## Current Focus

- Current active workstream: DK-only Preview rollout of the non-clairvoyant sequential
  best-ball policy while preserving the legacy ILP fallback.
- The app consumes `app/Simulation.sqlite3`, copied from the modeling repo.
- Recent changes added weighted weekly-template sampling, centered
  variance-preserving template residual blending, broader x-pruning buffers,
  Streamlit cache support for weekly templates, ADP audit support, and stack
  bonus controls.

## Recent Durable Decisions

- Sample weekly templates with `template_sample_prob` when available, preserving
  all selected templates while making closer matches more prevalent.
- Blend season outcomes with both model and template residual context: model
  residuals preserve calibrated distribution shape, while centered template
  residuals add player-type context.
- Keep template residual strength at `0.30` unless a follow-up calibration study
  supports changing it.
- Use a wider x-pruning max-side buffer so sampled ADP rank inflation does not
  hide materially available fallers from the ILP.
- Cache weekly template profile reads by DB modified time to improve repeated
  app runs.
- Store reusable audit CSV outputs under `research/studies/` and reusable SQL
  snippets under `docs/runbooks/queries/`.
- Keep the sequential policy on the explicit 16-week template horizon, with
  disjoint construction/evaluation banks and candidate-consistent room state.
- Keep beam search out and retain 24 candidates throughout; the DK-only gate
  cleared the 24-versus-32 shortlist threshold.
- Keep legacy as the default until decision/audit stability passes on fresh
  opening states. Preview runtime is an accepted opt-in tradeoff.

## Key Links

- Module tracker: `MODULE_TRACKER.md`
- Decision log: `DECISION_LOG.md`
- Cross-repo context: `CROSS_REPO_CONTEXT.md`
- App DB contract: `../docs/data_contracts/simulation_sqlite_app_contract.md`
- ILP validation runbook: `../docs/runbooks/best_ball_ilp_validation.md`
- Research index: `../research/README.md`
- Sequential policy runbook: `../docs/runbooks/best_ball_sequential_policy.md`
- Latest chronological log: `Session_Notes/2026-07.md`

## Working Defaults

- Keep generated databases and calibration CSVs out of app-root clutter.
- Treat `zSim_Helper.py` as the core optimizer/runtime module.
- Treat `snake_draft_app.py` as UI/orchestration.
- When in doubt, preserve the app's current behavior and document experiments in
  `research/studies/`.
