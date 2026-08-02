# Session Notes Landing

Last updated: 2026-07-31

## Project Objective

Maintain and improve the Streamlit snake-draft best-ball app that consumes the
fantasy football simulation database, samples ADP/projection/weekly-template
outcomes, and solves roster recommendations with the ILP optimizer.

## Current Focus

- The modeling source now audits and omits incomplete market-only players only
  in the final sixth of a draft surface, while keeping them in canonical ADP
  and preserving full-room projection coverage. Snake does not fill these rows;
  its existing sequential aligned-pool gate remains authoritative. This source
  change has passed a disposable handoff replay but has not been promoted.
- Current active workstream: DK production Preview rollout of the
  non-clairvoyant sequential best-ball policy while preserving the legacy ILP
  fallback, plus a governed independently scored NFFC offense-only candidate.
  The full 2026 modeling refresh and staged DK/NFFC AppTest passed, but no NFFC
  database has been promoted to this app.
- The app consumes `app/Simulation.sqlite3`, copied from the modeling repo.
- Canonical projection contexts now refuse keyless `Avg_ADPs` slices. Every
  supported offensive ADP row must carry a complete unique player key, and the
  same keyed value feeds runtime availability sampling and the displayed board.
  Explicitly positioned NFFC kicker/team-defense rows are outside the
  offense-only validation. The live canonical cutover is complete: the DK
  source feed has 416 rows, and weekly runtime context uses 343 exact keyed ADP
  joins plus eight governed player-map fallbacks.
- The 2026-07-29 source refresh preserves the runtime contract while correcting
  V2 identity/scoring lineage and the formerly DK-scored beta weekly slice.
  Both leagues now contain 5,298 templates; 5,120 paired active-PPG values and
  5,147 paired weekly paths differ. The live production pools have complete
  unique keys for 351 DK players (56 QB/101 RB/143 WR/51 TE) and 328 beta
  players (50 QB/95 RB/133 WR/50 TE), including one stable Tetairoa McMillan
  identity and no duplicate truncated Amon-Ra St. Brown identity. Beta weekly
  context uses 238 exact keyed ADP joins plus 90 governed fallbacks.
- New audit-only template-center fields expose the 39 governed beta 2018 QB
  diagnostic fallbacks and the three permitted hybrid-position mismatches.
  Snake scoring behavior is unchanged. The live database is table-identical to
  the frozen staged source, and the live Streamlit AppTest completed with zero
  exceptions or rendered error elements.
- The 2026 DK V2 production handoff is active. Current point samples repeat
  the V2 center before template application; one sampled donor then supplies
  its pool-centered active-PPG residual and matching weekly path directly.
  `joint_centered_template_v2_v1` replaces scaling to the now-zero legacy
  model-residual spread. Optional next-year draws apply `pred_appear_ny`, and
  no-appearance paths remain zero.
- `SNAKE_SIMULATION_DB` can opt into a staged database. The governed NFFC
  candidate uses independent NFFC projections, canonical NFFC ADP, 1,509
  modern-era templates, and a 385-player 17-week map. The historical
  DK-cloned preview remains a wiring fixture only and is not a release input.
- NFFC draft turns use Third Round Reversal. The separate 30-round
  kicker/team-defense Championship roster contract remains unimplemented and
  is explicitly warned in the UI.
- Recent changes added weighted weekly-template sampling, centered joint
  template residual/path uncertainty, broader x-pruning buffers,
  Streamlit cache support for weekly templates, ADP audit support, and stack
  bonus controls.
- The copied beta/DK pools now use source-owned adaptive absolute-distance
  kernels with weak-match shrinkage and a 5% donor cap. Ordinary zero-active
  seasons remain downside donors; Bell's 2018 holdout is the only declared
  audit-only exclusion. Snake's existing residual blend is otherwise unchanged.

## Recent Durable Decisions

- Sample weekly templates with `template_sample_prob` when available, preserving
  all selected templates while making closer matches more prevalent.
- V2 production weekly templates use `joint_centered_template_v2_v1`: one
  sampled donor supplies both the directly applied centered active-PPG residual
  and the 16-week availability/scoring path. Older databases without a V2
  handoff retain the `full_scaled_v1` branch.
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
- Give every completed root candidate the disjoint 128-season operational
  decision score; the 64-season pilot remains diagnostic rather than a
  finalist gate.
- Rank sequential actions by the empirically weighted same-position replacement
  available at the next user pick, recalculating every turn without a 100%
  availability threshold. Allocate the 24-root screen by remaining roster need.
- The focused Puka-only round-two fixture now ranks Love first, Rice second,
  Olave fourth, and its three QBs 20th/22nd/24th. The refreshed gate still found
  four audit regrets just above 10 points, with a 12.13 maximum, so this remains
  a Preview limitation rather than a promotion pass.
- Sequential Preview is the fresh-session default for owner-directed field
  testing; Legacy remains available as fallback. The audit/runtime gate failures
  remain documented and the Preview label stays visible.
- Opponent-pick count mismatches are advisory rather than blocking. The Preview
  runs from the marked availability state, so any missed drafted player can be
  marked Other Team and the recommendation rerun.
- Sequential Preview uses default-on symmetric QB-WR/TE tournament utility at
  every pick and in the final rank. Raw EV, immediate stack value, average
  final-roster stack utility, and the combined decision score remain separate.

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
