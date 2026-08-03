# Simulation SQLite App Contract

Last updated: 2026-08-03

## Owner

`Fantasy_Football` owns source generation of `Simulation.sqlite3`.

## Consumer

This app consumes:

```text
app/Simulation.sqlite3
```

The file is generated/copied from the modeling repo and should not be hand-edited
as a durable fix.

The production orchestrator compacts the staged Snake copy with SQLite
`VACUUM` before app smoke and promotion. It must retain the full source table
inventory and logical table content, report zero freelist pages afterward, and
remain at or below GitHub's 100 MiB blob limit. Byte identity with the source
database is neither expected nor required because compaction changes physical
page layout without changing rows or schemas.

The default Snake workflow uses the `dk` league slice. The app also exposes the
experimental, governed offense-only `nffc` scoring adapter when the selected
database contains one.
NFFC uses its own scoring, ADP, and weekly-template inputs rather than a renamed
or cloned DK slice. Other league/version slices in the shared database belong
to separate applications and remain out of scope.

`SNAKE_SIMULATION_DB` may select an alternate database filename from the app
directory (or an absolute path). The default remains `Simulation.sqlite3`.

## Governed NFFC Offensive Scoring Adapter

A publishable NFFC slice must contain all of the following under the same
prediction year and `nffc` league/version key:

- independently scored NFFC offensive rows in `Final_Predictions_Resid`;
- matching canonical player-map and template-pool rows;
- modern-era NFFC weekly templates with populated `week_1` through `week_17`;
- the current canonical NFFC `Avg_ADPs` feed, keyed by `player_key`.

The year/league selector lists only `Final_Predictions_Resid` slices with a
matching `Best_Ball_Weekly_Player_Map` slice. Annual source rebuilds replace
all older pool/map rows for the active league/dataset because template IDs are
regenerated from the current donor bank; an older map must never be joined to
the newer unversioned template table.

The NFFC weekly build uses 2021-and-later donors so its 17-week profiles do not
silently reinterpret older 16-game regular seasons. The old
`Simulation_nffc_preview.sqlite3`/DK-clone path may remain useful as a historical
wiring fixture, but it is not a governed NFFC production input.

This mode supports only `QB`, `RB`, `WR`, and `TE`. Kicker (`TK`) and
team-defense (`TDSP`) ADP entities are excluded before app identity validation,
and the optimizer has no kicker or defense roster slots. NFFC mode is therefore
not a complete implementation of an official NFFC contest.

For every NFFC selection,
`FootballSimulation.calculate_snake_picks()` currently uses Third Round
Reversal: Round 1 is first-to-last, Rounds 2 and 3 are last-to-first, Round 4 is
first-to-last, and the draft alternates thereafter. DK retains straight
serpentine order. The app does not currently expose the straight-snake schedule
used by NFFC25/NFFC50 formats, so those formats must not be represented by
selecting NFFC. The Championship 3RR reference used for the current schedule
implementation is https://nfc.shgn.com/rules/2680.

## Best-Ball Weekly Tables

### `Best_Ball_Weekly_Player_Map`

Used for current-player projection and template-pool context.

`year_exp` is the source builder's uncapped template-matching tenure.
`source_year_exp` preserves the potentially capped compiled-model value, while
`year_exp_source` and `year_exp_uncapped_delta` make the reconstruction
auditable. Runtime template matching must use the persisted pool mapping and
must not re-cap `year_exp`.

Expected columns include:
- canonical `player_key` and `player_key_match_method`
- `player`, `pos`, `team`, `year`, `version`, `dataset`
- current `pred_fp_per_game`, conditional next-year `pred_fp_per_game_ny`,
  and `pred_appear_ny`
- residual quantile columns prefixed `pred_resid_`
- V2 model/handoff provenance, `current_uncertainty_source`, and
  `independent_current_residual_draw_allowed`
- `avg_pick`
- `current_team_source`, which identifies `model_inputs`,
  `v2_player_season_features`, `canonical_avg_adps`, or `unassigned`
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
- `season`, `template_season_gap`, `template_recency_multiplier`
- `template_sample_prob`

The app should sample with `template_sample_prob` when the column exists.
The source builder now uses a position-specific absolute-distance kernel,
shrinks toward uniform when no local donor is close, applies a fixed 12-season
recency half-life, and caps one donor at 5%. The intended behavior is to use all
selected templates while giving genuinely closer and more recent matches higher
prevalence without allowing one season to dominate. The recency prior changes
sampling weight only; every donor must still precede the target year.

### `Best_Ball_Weekly_Templates`

Used to turn sampled season outcomes into week-level scores.

Historical `year_exp` and `year_exp_bucket` are uncapped. The corresponding
`year_exp_scaled` distance feature equals `year_exp / 10` without an upper
clip, so veteran seasons above year ten remain distinguishable.

Expected columns include:
- `league`
- `template_id`
- `template_local_id`
- canonical `player_key` and `player_key_match_method`
- `player`, `pos`, `season`
- `active_games`, `played_games`, `active_ppg`, `season_points`, `profile_total`
- `active_ppg_resid`
- `historical_pred_fp_per_game`, `v2_historical_pred_fp_per_game`,
  `v2_template_center_available`,
  `v2_template_center_unavailable_reason`,
  `v2_template_center_position`,
  `v2_template_center_position_mismatch`,
  `v2_template_center_position_mismatch_reason`,
  `historical_center_policy`, and `v2_recenter_promoted`
- beta/NFFC scoring-context lineage fields, including
  `projection_context_source`, `projection_context_scoring_hash`,
  `projection_context_run_id`, `scoring_context_available`, and the exact
  unavailable reason
- `template_eligible`, `template_exclusion_reason`
- `week_1` through the league horizon (`week_16` for DK and `week_17` for NFFC)
- `managed_week_1` through the same league horizon
- `played_week_1` through the same league horizon

Because league slices share one SQLite table, the NFFC publication adds the
`*_week_17` columns while DK rows may leave those columns null. The app removes
columns that are entirely null for the selected league and rejects a partially
populated selected horizon. A governed DK slice therefore remains 16 weeks and
a governed NFFC slice must resolve to exactly 17 populated weeks.

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

The center/context availability and position fields are source-owned audit
columns; Snake does not use them to recenter or filter donors. Missing beta
context is permitted only for the 39 governed 2018 QB rows whose historical
center remains auditable, whose unavailable reason is tied to the FFToday
vintage quarantine, and whose `template_eligible` flag is zero. The copied
database also audits three exact
hybrid-role position differences: Cordarrelle Patterson's 2019/2021 template
WR rows use locked RB centers, and Ty Montgomery's 2022 template RB row uses a
locked WR center. Every other unavailable center or position mismatch fails in
the source build.

### `Best_Ball_ADP_Audit`

Optional review table for identifying draftable players with suspicious missing
or fallback ADP context.

### `Avg_ADPs`

Canonical projection contexts require `Avg_ADPs.player_key` and join ADP
one-to-one by that key. Every supported offensive row (`QB`, `RB`, `WR`, or
`TE`) in the selected year/league slice must have a nonblank, unique
`player_key`; the app fails closed rather than reverting a canonical projection
to a normalized-display-name join.

For offensive rows, `Avg_ADPs.team` is the canonical identity's latest team.
The source weekly builder may use it to fill an otherwise unassigned current
team, but it must not override an assigned Model Inputs/V2 team and must retain
the choice in `Best_Ball_Weekly_Player_Map.current_team_source`. The Snake app
consumes the published map and does not perform its own team inference.

Formats may retain non-offensive draft entities such as NFFC kicker (`TK`) and
team-defense (`TDSP`) rows in `Avg_ADPs`. When those rows are present, the table
must publish `pos` so the offense-only Snake runtime can exclude them before
validating player keys. Such team units may use a separate
`draft_entity_key` and do not need an NFL player key. A missing key on an
offensive row still fails the entire selected slice.

An unmatched canonical projection row may use its governed player-map
`model_input_avg_pick`; a row lacking both direct keyed ADP and governed
player-map ADP fails closed. The normalized-display-name path remains available
only for genuinely legacy projection tables that do not publish any
`player_key`.

The live 2026 DK source slice contains 416 canonical keyed rows. Snake retains
the full aligned source population for draft-room simulation, joins production
players by `player_key`, and uses the resulting ADP consistently for both
availability sampling and the displayed draft board. The NFFC selector follows
the same key-only contract against the current canonical NFFC feed; it does not
reuse DK ADP. Provider/display labels do not participate in either canonical
join.

The source handoff may omit an incomplete market-only projection in the final
sixth of its draft surface, but it must retain that player in canonical
`Avg_ADPs` and record the omission in
`V2_Production_Eligibility_Audit`. Core projection players and keepers always
fail the source build; a new gap in the protected first five-sixths also fails
unless it has a separately reviewed annual exclusion. Snake does not invent
a projection for an omitted tail player; it requires the remaining aligned
projection/ADP pool to cover every simulated room pick through the user's final
pick.

## Current Validated Source Copy

The live V2 production pools retain 350 unique non-null DK player keys
(56 QB/100 RB/143 WR/51 TE), 383 NFFC offensive keys
(61 QB/110 RB/154 WR/58 TE), and 328 unique non-null beta player keys
(50 QB/95 RB/133 WR/50 TE).
Tetairoa McMillan's provisional and GSIS aliases share one stable key, and
truncated Amon-Ra St. Brown aliases no longer create a duplicate player.

The weekly player context has 342 exact keyed ADP joins and eight governed
player-map fallbacks for DK, plus 237 exact joins and 91 governed fallbacks for
beta. These partitions cover all 350/328 production players; there are no
unresolved or generic-default runtime rows.

The follow-up source build explicitly scores each weekly league, quarantines
the FFToday QB rows stored as 2018 that match the native 2019 vintage, and adds
the audit-only V2 center-availability/position columns above. It rebuilds 5,298
DK and 5,298 beta templates; 5,120 paired `active_ppg` values and 5,147 paired
weekly paths differ. These added columns do not change Snake's selected
`week_*` profile query or its DK runtime-scoring semantics.

The 2026-08-03 correction further removes mixed DK/beta matcher units from the
shared source database. Beta historical/current matcher context is now exact
beta V2 context; historical centers remain `legacy_validated_oos` for 2,696
rows and use `beta_scored_expert_fallback` for 2,602 rows. The 39 unavailable
2018 QBs are donor-ineligible. This is source provenance for Snake because the
app exposes only DK and NFFC; it does not change either supported runtime
scoring path.

The copied database passed SQLite integrity and foreign-key checks and is
table-identical to the frozen staged modeling source. Byte identity is not
required because equivalent SQLite files may have different physical layouts.
Snake now carries the canonical key through its runtime joins, displayed
player board, sampling caches, selection state, and optimizer masks; this
changes identity handling, not runtime-scoring semantics. The live Streamlit
AppTest completed with zero exceptions and zero rendered error elements.

## Sequential Player-Pool Coverage

The sequential policy simulates every room pick from the user's current pick
through the user's final pick. After already drafted players are removed, the
projection and ADP inputs must therefore retain at least
`last_adjusted_pick - current_adjusted_pick + 1` aligned players. The app
rejects smaller pools rather than allowing an exhausted room to create missing
or duplicated selections.

## Runtime Rules

- Treat `player_key` as the permanent player identity in both template and
  current-player-map rows. Generated copies require complete non-null keys,
  including stable provisional keys for players who have not played. Display
  names remain labels and must not become a fuzzy production join key.
- Require a fully populated, unique `Avg_ADPs.player_key` on every supported
  offensive row whenever the active projections use canonical keys. Filter
  explicitly positioned non-offensive draft entities before key validation;
  never use their separate draft-entity identifiers as player identities.
- Join `Final_Predictions_Resid` to `Best_Ball_Weekly_Player_Map` by
  `player_key` whenever both tables publish complete keys. Require one map row
  per key, matching positions, and complete projection-map coverage.
- Key weekly template profile dictionaries and tensor caches by `player_key`.
  A display-name change must not change the donor pool used by a player.
- Persist `PlayerKey` in saved draft-state files and use it for MyTeam,
  OtherTeam, exclusion, and selected-player masks. Old saved files and old
  databases may fall back to exact display-name identity, but unmatched or
  ambiguous selected rows must fail closed.
- Preserve the full aligned source population for opponent draft simulation.
  The ILP candidate reduction may retain its existing position-specific
  ADP/projection quotas, but it must force every already selected player into
  the retained pool and verify that none was pruned.
- Preserve `template_pool_key` joins across player map, pools, and templates.
- When `Best_Ball_Weekly_Templates.league` exists, join pools to templates on
  both `template_id` and league context (`pool_version` to `league`).
- Best-ball table builds preserve other league slices already present in
  `Simulation.sqlite3`, but replace every retained year for the rebuilt
  league/dataset's pool, map, summary, and player-audit surfaces.
- Treat the selected league's populated `week_*` horizon as multipliers:
  `week_1` through `week_16` for DK and `week_1` through `week_17` for NFFC.
  Drop only columns that are entirely null for the selected slice and reject a
  partial horizon.
- Do not reinterpret `played_week_*` as score multipliers. A value of `1`
  means the source weekly table contained a qualifying player-week row, not
  that the player necessarily had comprehensive snap-count coverage.
- For a V2 handoff, require zero current residual quantiles,
  `current_uncertainty_source = joint_weekly_template_only`, and
  `independent_current_residual_draw_allowed = 0`.
- Center each sampled donor's active-PPG residual within its published pool and
  add it directly to the current V2 point center. Apply that same donor's
  league-specific 16- or 17-week path. Do not scale to the model-residual
  standard deviation: the legacy spread is deliberately zero in V2 and scaling
  would collapse PPG variance.
- The production method is `joint_centered_template_v2_v1`. The former
  `full_scaled_v1` branch remains only for older databases without a V2
  production handoff; V2 rejects legacy template-residual blend settings.
- DK historical residuals remain on the validated legacy/preseason policy;
  NFFC uses its scoring-matched expert center. Beta uses validated legacy OOS
  centers where available and `beta_scored_expert_fallback` otherwise.
  `v2_historical_pred_fp_per_game` remains diagnostic and
  `v2_recenter_promoted = 0`.
- Treat the new V2 center-availability and position-mismatch fields as audit
  provenance. Do not use them to create an app-side fallback, cross-league
  center substitution, or template filter.
- `pred_fp_per_game_ny` is conditional on appearing. When the optional
  next-year blend is used, apply the separate `pred_appear_ny` Bernoulli draw;
  a no-appearance draw must remain a zero weekly path.
- Do not use an uncentered template residual as a mean shift.
- Keep app logic tolerant of older DBs when practical, but update this contract
  when new columns become required.
- Preserve enough aligned projection/ADP rows for every supported sequential
  draft format.
