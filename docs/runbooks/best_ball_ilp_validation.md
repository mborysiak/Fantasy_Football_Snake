# Best-Ball ILP Validation Runbook

Last updated: 2026-07-29

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
- the DK-only Sequential best-ball policy (Preview) renders all completed
  decision candidates with raw EV, roster stack utility, immediate stack
  utility, stack-adjusted decision score, paired SE, survival, and room coverage

## Best-Ball Runtime Checks

When changing template or residual logic, check:

- `template_sample_prob` is used when available
- sampled templates span the full pool but favor better matches
- no donor probability exceeds 5%, pool probabilities sum to one, and effective
  sample sizes remain broad
- every donor season precedes the target year and the persisted recency
  multiplier equals `0.5 ** (template_season_gap / 12)`
- declared structural exclusions have zero pool uses while ordinary zero-active
  downside remains represented
- V2 current point samples are constant before template application, while the
  final weekly score bank retains variance from the sampled centered donor
  residual/path
- production reports `joint_centered_template_v2_v1`, adds the centered donor
  residual directly to the V2 point center, and is independent of the zeroed
  current residual quantiles
- older databases without a V2 handoff retain the legacy `full_scaled_v1`
  branch; V2 contexts reject legacy template-residual blend settings
- optional next-year samples apply `pred_appear_ny`, and no-appearance draws
  remain zero after weekly-template application
- the production template-residual method and fallback remain documented in
  `Agent_Notes/DECISION_LOG.md`
- x-pruning buffer does not hide materially available fallers
- app text/help descriptions match current residual and template-sampling logic
- weekly template profile reads still use the DB-mtime cache path
- league-aware template joins do not duplicate template rows when multiple
  `Best_Ball_Weekly_Templates.league` slices exist
- `player_key` exists and is non-null in every weekly template and current
  player-map row; current keys are unique by version/dataset/year/player
- canonical-key handoff audits join every current and historical V2 population
  row without falling back to display-name matching
- renamed projection/player-map rows resolve by `player_key`, weekly template
  caches remain keyed by `player_key`, and a selected row survives ILP pruning
- the current keyless `Avg_ADPs` schema enters only the explicit
  `legacy_normalized_name` path; relevant normalized-name collisions fail
  instead of being silently deduplicated
- every MyTeam and OtherTeam row resolves exactly once in the active player
  population; stale, blank, duplicate, conflicting, or ambiguous saved-state
  rows fail before simulation
- source audit fields report exactly 39 unavailable beta 2018 QB V2 diagnostic
  centers with quarantine-linked reasons; their active center remains the
  validated legacy OOS center, and no other unavailable center is present
- center-position mismatch audit contains only Cordarrelle Patterson 2019/2021
  template WR to locked RB and Ty Montgomery 2022 template RB to locked WR;
  every other mismatch is absent
- uncapped `year_exp` survives the database copy, values above ten remain in
  player/template rows, and the app uses the persisted pool mapping without
  re-capping tenure
- DK pick schedules remain straight serpentine
- NFFC 12-team pick schedules use 3RR: slot 12 starts `12, 13, 25, 48` and
  slot 1 starts `1, 24, 36, 37`

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
