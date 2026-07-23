# Simulation SQLite App Contract

Last updated: 2026-07-23

## Owner

`Fantasy_Football` owns source generation of `Simulation.sqlite3`.

## Consumer

This app consumes:

```text
app/Simulation.sqlite3
```

The file is generated/copied from the modeling repo and should not be hand-edited
as a durable fix.

The production/default Snake workflow uses the `dk` league slice. The app may
also expose an `nffc` slice when the selected database contains one, but NFFC
remains a setup preview and is outside the DK sequential-policy release gates.
Other league/version slices in the shared database belong to separate
applications and remain out of scope.

`SNAKE_SIMULATION_DB` may select an alternate database filename from the app
directory (or an absolute path). The default remains `Simulation.sqlite3`.

## NFFC Setup Preview

`Fantasy_Football/Scripts/Modeling/create_snake_nffc_preview.py` creates
`app/Simulation_nffc_preview.sqlite3` from the stable app database without
modifying the modeling source database. It clones the DK runtime rows under
NFFC-safe league keys and the reserved 3,000,000 template-ID range while
retaining the real NFFC `Avg_ADPs` slice.

The preview is valid for app wiring, selector, persistence, league-aware draft
order, and partial draft-flow tests only. Its cloned projections and weekly
profiles still reflect DK scoring/calibration. The 2026 NFFC Best Ball
Championship is also a 30-round format with kicker and team-defense roster
slots, which the current offense-only app does not yet model. Replace the cloned
rows with normal NFFC s3/s4 outputs and add the complete roster contract before
evaluating recommendation quality or using the app for a live NFFC draft.

For the NFFC slice, `FootballSimulation.calculate_snake_picks()` uses Third
Round Reversal: Round 1 is first-to-last, Rounds 2 and 3 are last-to-first,
Round 4 is first-to-last, and the draft alternates thereafter. DK retains its
existing straight serpentine schedule. The official contest reference is:
https://nfc.shgn.com/rules/2680.

## Best-Ball Weekly Tables

### `Best_Ball_Weekly_Player_Map`

Used for current-player projection and template-pool context.

`year_exp` is the source builder's uncapped template-matching tenure.
`source_year_exp` preserves the potentially capped compiled-model value, while
`year_exp_source` and `year_exp_uncapped_delta` make the reconstruction
auditable. Runtime template matching must use the persisted pool mapping and
must not re-cap `year_exp`.

Expected columns include:
- `player`, `pos`, `team`, `year`, `version`, `dataset`
- `pred_fp_per_game`
- residual quantile columns prefixed `pred_resid_`
- `avg_pick`
- `template_pool_key`

### `Best_Ball_Weekly_Template_Pools`

Used to select historical weekly templates for current players.

Expected columns include:
- `template_pool_key`
- `template_id`
- `pool_version`, `pool_dataset`
- `template_league`
- `template_distance`
- `match_rank`
- `template_sample_prob`

The app should sample with `template_sample_prob` when the column exists.
The source builder now uses a position-specific absolute-distance kernel,
shrinks toward uniform when no local donor is close, and caps one donor at 5%.
The intended behavior is to use all selected templates while giving genuinely
closer matches higher prevalence without allowing one season to dominate.

### `Best_Ball_Weekly_Templates`

Used to turn sampled season outcomes into week-level scores.

Historical `year_exp` and `year_exp_bucket` are uncapped. The corresponding
`year_exp_scaled` distance feature equals `year_exp / 10` without an upper
clip, so veteran seasons above year ten remain distinguishable.

Expected columns include:
- `league`
- `template_id`
- `template_local_id`
- `player`, `pos`, `season`
- `active_games`, `played_games`, `active_ppg`, `season_points`, `profile_total`
- `active_ppg_resid`
- `template_eligible`, `template_exclusion_reason`
- `week_1` through `week_16`
- `managed_week_1` through `managed_week_16`
- `played_week_1` through `played_week_16`

The `played_week_*` fields are additive 0/1 source-observation masks owned by
the modeling build. The Snake app does not currently use them for best-ball
scoring, and its weekly multiplier loader must continue selecting only columns
whose names begin with `week_`.
Their row sum equals `played_games`, which can exceed `active_games` for QBs
because short appearances are retained as participation evidence while the
existing greater-than-15-play performance-profile filter remains in place.
The `managed_week_*` fields retain those short-QB score profiles for the auction
app. Snake must continue selecting only columns whose names begin exactly with
`week_`, so neither `managed_week_*` nor `played_week_*` changes best-ball
scoring.

`template_eligible = 0` preserves a structurally non-transferable outcome for
audit while preventing pool use. Le'Veon Bell's 2018 contract holdout is the
current declared exclusion. Ordinary zero-active seasons remain eligible as
real downside outcomes. Runtime sampling follows the already-published pools;
it must not independently filter templates.

### `Best_Ball_ADP_Audit`

Optional review table for identifying draftable players with suspicious missing
or fallback ADP context.

## Sequential Player-Pool Coverage

The sequential policy simulates every room pick from the user's current pick
through the user's final pick. After already drafted players are removed, the
projection and ADP inputs must therefore retain at least
`last_adjusted_pick - current_adjusted_pick + 1` aligned players. The app
rejects smaller pools rather than allowing an exhausted room to create missing
or duplicated selections.

## Runtime Rules

- Preserve `template_pool_key` joins across player map, pools, and templates.
- When `Best_Ball_Weekly_Templates.league` exists, join pools to templates on
  both `template_id` and league context (`pool_version` to `league`).
- Best-ball table builds should preserve other league slices already present in
  `Simulation.sqlite3`.
- Treat `week_1` through `week_16` as multipliers.
- Do not reinterpret `played_week_*` as score multipliers. A value of `1`
  means the source weekly table contained a qualifying player-week row, not
  that the player necessarily had comprehensive snap-count coverage.
- Center each sampled donor's active-PPG residual within its published pool and
  scale it to the current player's model-residual standard deviation.
- The production `full_scaled_v1` path uses that scaled donor residual without
  adding an independent model-residual draw, then applies the same donor's
  `week_1` through `week_16` path. This keeps performance magnitude and weekly
  availability as one matched historical outcome.
- Keep `template_resid_blend=0.30` callable as a legacy rollback and validation
  comparator. Do not expose it as the production default.
- Do not use an uncentered template residual as a mean shift.
- Keep app logic tolerant of older DBs when practical, but update this contract
  when new columns become required.
- Preserve enough aligned projection/ADP rows for every supported sequential
  draft format.
