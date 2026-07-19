# Best-Ball ILP Validation Runbook

Last updated: 2026-07-07

## Core Files

- `app/zSim_Helper.py`: data loading, sampling, optimizer setup, and ILP logic.
- `app/snake_draft_app.py`: Streamlit UI and orchestration.
- `app/Simulation.sqlite3`: generated database copied from the modeling repo.
- `docs/runbooks/best_ball_sequential_policy.md`: Preview policy methodology,
  validation gates, and limitations.

## Quick Syntax Check

```powershell
streamlitvenv\Scripts\python.exe -m py_compile app\zSim_Helper.py app\snake_draft_app.py
```

Use the active app environment if `streamlitvenv` is not available.

## App Smoke Check

```powershell
streamlitvenv\Scripts\streamlit.exe run app\snake_draft_app.py
```

Then confirm:
- the app loads the player pool
- draft controls render
- optimizer can run at least one recommendation
- weekly template controls do not error
- stack bonus controls render when enabled
- the DK-only Sequential best-ball policy (Preview) renders the four
  decision-stage finalists with EV, paired SE, pilot context, survival, and
  completed-room columns

## Best-Ball Runtime Checks

When changing template or residual logic, check:

- `template_sample_prob` is used when available
- sampled templates span the full pool but favor better matches
- template residuals are centered by pool before blending
- template residual strength remains documented in `Agent_Notes/DECISION_LOG.md`
- x-pruning buffer does not hide materially available fallers
- app text/help descriptions match current residual and template-sampling logic
- weekly template profile reads still use the DB-mtime cache path
- league-aware template joins do not duplicate template rows when multiple
  `Best_Ball_Weekly_Templates.league` slices exist

## Research Outputs

Store calibration and pruning audit outputs under:

```text
research/studies/YYYY-MM-DD_<slug>/results/
```

Do not leave reusable audit CSVs in `app/`.

For the non-clairvoyant policy checks, use:

```text
research/studies/2026-07-19_sequential_best_ball_policy/
```
