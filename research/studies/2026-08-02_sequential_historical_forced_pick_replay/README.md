# Sequential Historical Forced-Pick Replay

This is the final historical evidence stage for the DK Sequential Preview. It
compares the current-pick action chosen by the exact production D128 prefix with
its nested D256 extension. Legacy is deliberately excluded: Sequential is the
production methodology, and GLPK is neither the oracle nor an approval gate.
The study does **not** reconstruct observed draft rooms. Every state is a
clearly labeled synthetic ADP board.

## Frozen design

- DK origins: 2017 through 2025.
- Draft slots: 1, 6, and 12 in a 12-team, 20-round straight snake.
- Completed user picks: 0, 7, and 14.
- One nested neutral board per origin/slot. Opponents take the next available
  player from one noisy historical-ADP priority order. Prior user picks take
  the highest-priority legal player from that same order. Projections, policy
  scores, and held-out outcomes never select the prefix.
- Sequential construction/pilot banks: 16/64. All 24 screened roots complete
  24 downstream rooms. D128 is the exact first 128 columns of D256; the second
  128 columns use a separately seeded extension.
- The unique D128/D256 actions are forced through one shared set of 24 latent
  ADP rooms using the same current Sequential downstream policy. This isolates
  current-pick quality.
- Every weekly-template donor season must be strictly earlier than its origin.
  Target-season weekly outcomes are the only evaluation data.
- The primary effect is mean held-out configured-DK all-played best-ball points
  for D256 minus D128.
  Uncertainty resamples complete origin-season clusters for 10,000 draws. D256
  passes noninferiority only when the 95% lower bound in points is at least
  `-0.0025 * observed paired D128 mean points`. Percent intervals are reported
  as diagnostics; their draw-varying denominator does not define the gate.

The default design and gate are constants in `run_study.py`. A CLI override is
allowed for diagnostics, but sets `frozen_design_exact=false` and cannot pass
the full frozen gate. The design records SHA-256 fingerprints for the runner,
historical-data adapter, production simulation helper, weekly-template builder,
and the V2 outcome scorer plus its scoring/configuration dependencies. Freeze
and score sealing fail if any contracted file changes mid-run. The lazy target
receipt must also publish the exact contracted scorer hash and the same V2
scoring hash used by the origin's decision surface.

The runner accepts only the canonical sibling model repository and its
maintained Python 3.12 interpreter. Runtime receipts seal the executable
content hash, Python/platform identity, and relevant package versions. The
historical builder, scorer, and config module paths must also resolve exactly
to their sealed code-contract paths.

## Leakage firewall

Execution has two explicit phases:

1. `freeze` assembles strict-prior origin databases, creates every ADP-only
   state, chooses all actions, runs all downstream rooms, and writes canonical
   player-key rosters to `frozen_states.jsonl`. Each state uses two fresh
   Python 3.12 children: Nested D128/D256 and the forced common Sequential
   rollout (162 isolated heavy jobs for the 81-state design). The
   parent only orchestrates and checkpoints. Children use unique hashed
   request/result envelopes under OS temporary storage; the parent rejects any
   changed state, seed, origin receipt, code/runtime, selected path, ADP room,
   or construction/evaluation bank. There is no in-process fallback. The
   builder and every child must confirm target outcomes remain unread. After
   all states exist, the parent reopens every origin and seals the artifact
   hashes in `freeze_manifest.json`.
2. `score` refuses to run unless the design, exact state set, per-state hashes,
   and sealed file hash match. Only then does it resolve the active foundation's
   nflverse target receipt, download and verify the exact pinned season payload,
   join by exact GSIS ID, reconcile the opportunity-filtered full-season stream,
   and call `score_rosters` against weeks 1--16 of the all-played stream.

This global two-pass boundary is intentional: no recommendation or downstream
roster can change after any target outcome becomes visible.
`freeze` is the CLI default; target outcomes require an explicit `--phase score`
(or an equally explicit diagnostic `--phase all`).

Freeze provenance hashes only the selected decision-time frames, policies, and
the disposable decision-only simulation database. It deliberately does not hash
the whole Projection V2 database, which can contain held-out target bytes.
Target tables, identities, aliases, payload, scoring code, and outcome frames
are authenticated independently by the canonical target receipt after the
global freeze seal.

The outcome label is intentionally **configured-DK all-played**, not official
contest-realized DK. The repository scoring dictionaries do not cover two-point
plays or individual return/special-teams touchdowns. The canonical target-source
receipt seals the payload URI/SHA-256/row count, scoring hash, scoring and mask
code hashes, mapped/unmapped counts, all-played and opportunity-excluded totals,
full governed horizon, Week-17 exclusion, and reconciliation diagnostics.

## Historical data adapter

`historical_data.py` supplies:

```python
open_origin(
    source_db: Path,
    origin_year: int,
    work_dir: Path,
    league="dk",
    strict_prior=True,
    smoke=False,
)
```

The yielded object exposes `db_path`, `set_year`, `pred_vers`, `league`,
`donor_years`, `receipt`, `source_fingerprint`, `target_outcome_fingerprint`,
`assert_decision_inputs_clean()`, `assert_target_outcomes_unread()`, and
`score_rosters(rosters)`. The disposable SQLite database contains only decision
inputs. Target outcomes are loaded lazily by `score_rosters` from the
manifest-pinned nflverse payload and are never materialized into that database.
Candidate outcomes require a unique nonblank `gsis_id`; no name-alias or fuzzy
scoring join is allowed.

## Commands

Run only with the modeling repository's maintained Python 3.12 environment.
The runner fails closed on a different interpreter; it never injects one
environment's `site-packages` into another Python process.

```powershell
$python = (Resolve-Path ..\Fantasy_Football\.venv_ff_312\Scripts\python.exe).Path
```

Inspect the preregistration without building an origin:

```powershell
& $python research/studies/2026-08-02_sequential_historical_forced_pick_replay/run_study.py --dry-run
```

Run a tiny, non-frozen integration smoke (one origin/state, two rooms, D8/D16):

```powershell
& $python research/studies/2026-08-02_sequential_historical_forced_pick_replay/run_study.py --smoke --phase freeze
```

Do not treat smoke output as evidence. The full replay is deliberately split so
the sealed freeze can be reviewed before any outcome access:

```powershell
$output = 'research/studies/2026-08-02_sequential_historical_forced_pick_replay/results_v5'
& $python research/studies/2026-08-02_sequential_historical_forced_pick_replay/run_study.py --phase freeze --output-dir $output
& $python research/studies/2026-08-02_sequential_historical_forced_pick_replay/run_study.py --phase score --output-dir $output
```

Checkpoints are resumable by default. `--no-resume` starts from an empty
in-memory checkpoint but still refuses a conflicting `design.json`. Use a new
output directory for a materially different diagnostic design. Before sealing,
the runner reopens every origin, including origins already complete in a prior
checkpoint, and rejects a changed source fingerprint, decision receipt, donor
set, or code contract. Target scoring revalidates and re-scores every origin;
old row counts alone never qualify a score checkpoint for reuse. A child crash
never merges a partial state: the parent checkpoint remains resumable and that
state restarts at its Nested child.

The sealed final output is `results_v5/`. Earlier `results*` directories are
retained as invalid audit history: they were bound to older hashes, reduced
smokes, or superseded designs and must not be resumed or cited as approval
evidence.

## Final result

The sealed 2026-08-03 replay completed 81 states across nine held-out origins
and 3,888 paired arm-room scores. D256 averaged 2,079.373 configured-DK
all-played points versus 2,078.467 for D128, a +0.907-point (+0.044%) mean
difference. The season-clustered 95% interval was [-3.160, 5.439] points
([-0.150%, 0.262%]), above the preregistered -5.196-point (-0.25%)
noninferiority margin. Every frozen design, global-freeze, source, identity,
physical-state, roster, isolation, exact-prefix, target-receipt, and scoring
gate passed. This approves D256 for DK Sequential Preview; it does not remove
the Preview label or broaden the result to observed draft-room behavior.

## Outputs

- `design.json`: preregistered inputs and gate, written before outcome scoring.
- `runner_receipt.json` and `origin_receipts.json`: code/runtime and data lineage.
- `frozen_states.jsonl`: actions plus 24 canonical-key rosters per unique action.
- `frozen_action_summary.csv`: compact human-reviewable freeze inventory.
- `freeze_manifest.json`: global freeze seal, hashes, and two-stage process-
  isolation gate. A successful freeze also replaces the in-progress summary
  with `freeze_complete_waiting_for_score` and no target results.
- `room_scores.csv`: target-season score for each arm/room.
- `score_receipts.json`: exact per-origin governed target-source receipt,
  content/scoring-code SHA256 values, receipt hash, row count, scoring horizon,
  and target-outcome fingerprint, all loaded lazily after the freeze seal.
- `state_metrics.csv`: arm means and paired state deltas.
- `score_manifest.json`: sealed hashes of room scores, state metrics, score
  receipts, the freeze artifact, and the code contract.
- `summary.json`: season-cluster interval and frozen gates.

Generated work databases live under the operating system's temporary directory
(`fantasy_football_historical_replay/`) and are removed when each origin context
closes. The study never writes a generated database under the repository, and
never edits `app/Simulation.sqlite3` or production helper code.
