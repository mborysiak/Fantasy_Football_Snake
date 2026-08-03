# Historical forced-pick replay: data inventory

## Scope conclusion

No pick-by-pick historical fantasy draft logs or completed historical fantasy
draft boards are present in either `Fantasy_Football` or
`Fantasy_Football_Snake`.  The similarly named `Team_Drafts` and
`Draft_Positions` tables in `Season_Stats_New.sqlite3` describe the NFL draft,
not fantasy drafts.  The existing frozen root boards and rolling-validation
rosters are current or synthetic artifacts.  Consequently this study is a
**synthetic historical ADP-board forced-pick replay**, not a replay of observed
draft rooms.

The two repositories contain enough governed projection, donor, identity, and
outcome provenance for a leakage-safe DK study over 2017--2025. The canonical
weekly source is not either local FastR copy. It is the exact season-specific
nflverse weekly CSV whose URI, payload SHA-256, and parsed row count are pinned
in `Projection_V2.sqlite3.source_manifest` by the active V2 foundation. The
payload is downloaded and verified only after the global freeze seal. A
missing, duplicate, or mismatched manifest receipt or payload fails closed.

## Source map

| Need | Source | Historical coverage / use |
| --- | --- | --- |
| Full projection universe and locked point center | `Fantasy_Football/Data/Databases/Projection_V2.sqlite3`, `locked_template_handoff` | 2017--2025 rolling OOS rows; use all keyed offensive rows, including players with no realized appearance |
| Preseason ADP and player context | `Projection_V2.sqlite3`, `player_season_features` and `player_season_market_values` | 2017--2025; `feature_cutoff_season = season - 1` and `preseason_source_season = season` are required. Only `fantasypros_best_ball_adp`, `draftkings_adp`, and `adp_average_dk` may enter the board |
| Weekly-path donor library | `Fantasy_Football/Data/Databases/Simulation.sqlite3`, `Best_Ball_Weekly_Templates` | DK 2008--2025; only rows with `donor season < origin` are eligible |
| Raw realized weekly statistics | nflverse `stats_player_week_{season}.csv`, pinned by `Projection_V2.sqlite3.source_manifest` | Target-season bytes may be downloaded/read only after every recommendation and downstream roster is frozen and sealed; URI, SHA-256, and row count must exactly match the active foundation receipt |
| Canonical outcome identity | `Projection_V2.sqlite3.player_identity` | Join raw `player_id` to one unique nonblank candidate `gsis_id`, then to `player_key`; no name or fuzzy fallback is permitted |
| Governed seasonal reconciliation | `Projection_V2.sqlite3.player_season_spine` | Apply the production fantasy-week and opportunity masks over the full governed horizon and reconcile configured points to `unconditional_season_points` before evaluating rosters |
| Prior study scaffolds | `2026-08-02_sequential_v2_bank_stability` and `2026-07-22_joint_template_blend_rolling_validation` | Nested D128/D256 banks, policy runner, strict-prior synthetic rooms, and DK best-ball scoring patterns |

`Best_Ball_Weekly_Templates` target-season rows are not a valid draft universe.
Across 2017--2025 they omit 190 player-seasons with ADP at or before 240,
including highly drafted injured or retired players.  Building the universe
from those rows would create outcome-conditioned survivorship.  The study must
instead build the target pool from the V2 handoff plus preseason feature/market
data and independently join outcomes.  An unmatched/no-appearance player has
an all-zero realized weekly path, subject to reconciliation against the
governed seasonal outcome data.

## Frozen experiment contract

- Origins: DK 2017--2025.
- States per origin: slots 1, 6, and 12; completed user-pick depths 0, 7, and
  14.  Each slot uses one nested neutral ADP-only draft trajectory.  No policy
  or target outcome may affect a prefix.
- Current actions: Sequential D128, Sequential D256, and the current Legacy
  action.  D128 is the exact prefix of D256.
- Downstream evaluation: force the union of those actions, then use the same
  Sequential continuation and the same 24 latent ADP rooms for every action.
  Adding a forced Legacy action must not alter D128 or D256 selection.
- Outcome: 16-week **configured-DK all-played** best-ball points (1 QB, 2 RB,
  3 WR, 1 TE, one RB/WR/TE flex), with missing weekly points treated as zero.
  Stack utility is not part of the score. This is not labeled official contest
  scoring: the repository scoring dictionaries intentionally leave two-point
  and individual return/special-teams touchdown components at zero.
- Primary contrast: paired D256 minus D128.  D256 minus Legacy is secondary.
  Resample whole seasons in the bootstrap.  The predeclared noninferiority
  margin is -0.25% of the paired baseline realized mean.

## Gates frozen before result inspection

1. Every weekly donor has `season < origin`; the target season never enters a
   donor pool.
2. The disposable decision database contains no actual target outcome field,
   target weekly score, or target realized template.
3. No target-season weekly payload is downloaded or read during
   state/action/roster freeze; the sealed roster JSONL hash is recorded first.
4. D128 is byte-for-byte the first 128 scenarios of D256; all compared actions
   share the same 24 rooms and downstream random fields.
5. Neutral state prefixes depend only on preseason ADP inputs and frozen random
   seeds, never on a policy recommendation or target outcome.
6. All candidate-room rollouts finish with legal 20-player rosters; every
   current action in the comparison union is covered in every room.
7. Every allowed-source ADP belongs to a confirmed identity or an explicit
   reviewed key redirect. The complete confirmed allowed-source population is
   retained; an arbitrary mean-rank/population cap cannot hide a reachable row.
   A direct same-key provider-position disagreement is retained as a sealed
   diagnostic, while a reviewed key redirect that changes position is fatal.
8. Raw outcomes join by exact GSIS ID. The opportunity-filtered full-horizon
   stream (weeks 1--16 before 2021 and 1--17 from 2021) reconciles numerically
   to the governed V2 seasonal target. The separately recorded all-played
   stream is sliced to weeks 1--16 only after that audit; a legitimate cameo
   may therefore be zero in the governed stream and nonzero in roster scoring.
9. Source URI, payload and scoring-code hashes, schema signatures, row counts,
   scoring hash, mapping counts, and origin-specific receipts remain identical
   when an origin is reopened and re-scored.
10. Promotion-style noninferiority is evaluated only from the season-cluster
   bootstrap lower bound against the frozen -0.25% margin.  Smoke or partial
   runs are ineligible to pass the frozen gates.
