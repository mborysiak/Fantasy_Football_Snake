# Sequential Best-Ball Policy Runbook

Last updated: 2026-08-03

## Release Status

`best_ball_policy` remains labeled Preview and is the fresh-session default.
DK uses the approved nested D256 decision bank with 24 rooms and 24 candidates.
The experimental governed NFFC offense-only adapter remains D128 against
independently scored NFFC projections, current canonical NFFC ADP, and 17-week
modern-era templates. Legacy `best_ball_ilp` remains available as a fallback
and diagnostic, but is not an approval oracle. The Preview is limited to the
current-pick recommendation; it does not present future-round recommendations
as if their players were fixed in advance.

Every Sequential or Legacy click runs in one fresh Python subprocess, forces
numeric thread counts to one, uses one inner worker, and has no automatic retry
or cross-method fallback. A failed worker leaves the draft selections unchanged.

## League-Aware Pick Schedule

All recommendation paths derive future user turns from
`FootballSimulation.my_picks`.

- `dk` retains straight serpentine order.
- `nffc` uses Third Round Reversal: Round 1 runs first-to-last, Rounds 2 and 3
  both run last-to-first, Round 4 returns first-to-last, and the order
  alternates from there.

For a 12-team NFFC draft, slot 12 therefore picks 12, 13, 25, 48, 49, and 72;
slot 1 picks 1, 24, 36, 37, 60, and 61. This schedule applies to draft-status
counts, opponent-turn validation, availability horizons, ILP turns, and
sequential rollouts.

This is the only NFFC schedule currently implemented. The app has no
straight-snake NFFC25/NFFC50 selector. Its NFFC player and roster scope is
limited to QB/RB/WR/TE, with no kicker or team defense, so the governed
offensive mode is not a complete implementation of an official NFFC contest.

## Methodology Boundary

The policy separates information used to make draft decisions from outcomes
used to score those decisions:

1. Generate construction seasons from the selected league's weekly-template
   model: 16 weeks for DK or 17 weeks for NFFC.
2. Generate noisy-ADP opponent priority orders for shared draft rooms.
3. Allocate a 24-player legal root screen across positions according to the
   remaining roster minimum deficits and maximum capacity.
4. Lock one candidate, remove that player from the room immediately, and then
   alternate opponent removals with the user's future picks.
5. At each user pick, estimate the best same-position replacement at the next
   user pick from all 1,000 conditional ADP samples. Add the incremental
   tournament utility of any QB-WR/TE pairing completed by the candidate,
   regardless of which side was drafted first. Choose based on the positive
   stack-adjusted utility drop-off from waiting, with no hard availability
   threshold. Recalculate after every opponent turn.
6. Score every completed roster on a separate 64-season pilot bank.
7. Rank every completed candidate on a separate decision bank: nested D256 for
   DK and D128 for NFFC offense-only. Keep raw best-ball EV visible and rank on
   a separate decision score equal to raw EV plus average final-roster stack
   utility.

Release studies may additionally rescore every completed candidate on a fourth
128-season audit bank. Its PPG columns are unique and disjoint from every
operational bank. Audit is diagnostic only: changing its seed cannot change the
candidate screen, rollout paths, pilot ranking, decision ranking, recommendation,
or downstream draft state.

DK D256 preserves the exact production D128 allocation as its first 128 columns
and adds a separately seeded, non-overlapping 128-column extension. Release
studies score D128 and D256 actions on common policy-inert R512 outcomes; R512
is research-only and never changes a production recommendation.

Construction, pilot, decision, and optional audit draw unique PPG scenario
columns from explicitly disjoint subsets of the 1,000 prediction columns. The
engine asserts that the intersections are empty. Template/profile draws also
use separate seeds. `evaluation_seed`, `decision_seed`, and `audit_seed` are
independent of the construction/draft seed. Changing audit alone changes only
audit scores; changing pilot or decision may change the recommendation but
never the candidate-specific rollout paths.

The pilot and decision banks are never passed to the rollout policy. Weekly best-ball
lineup selection may use realized weekly scores because that is the contest's
scoring rule; draft selection may not.

All root candidates share the same construction bank, opponent rooms, and
pilot/decision bank. Candidate differences are therefore paired. `Paired SE` is an
approximate two-way standard error of the difference versus the observed best,
with draft-room and evaluation-season components. It is diagnostic rather than
a formal posterior probability that a candidate is optimal.

## Explicit Horizons

- `sequential_template_16`: DK Preview sequential policy and weekly-template horizon.
- `sequential_template_17`: NFFC offense-only Preview and weekly-template horizon.
- `legacy_template_16`/`legacy_template_17`: legacy ILP when weekly templates
  are selected for DK/NFFC.
- `legacy_residual_17`: historical independent-residual configuration.

The app does not synthesize or pad a weekly horizon. DK consumes 16 populated
weeks. NFFC consumes 17 populated weeks from the modeling repository's
2021-and-later modern-era donor build; a partial horizon fails closed.

## Candidate Screen

The screen allocates candidates by remaining roster need. With only Puka Nacua
selected under the default construction, the 24 roots are 3 QB, 8 RB, 10 WR,
and 3 TE. Within each position it prioritizes empirical draft-now advantage,
low next-pick survival, and immediate marginal value. This prevents raw QB
points from consuming most of the root pool while preserving alternatives at
every open position.

## Sequential Stack Utility

The Preview uses a default-on, explicitly separate tournament-utility term for
same-team QB-WR/TE pairs. It is symmetric: drafting a QB after an earlier pass
catcher and drafting a pass catcher after an earlier QB create the same pair
value. The incremental utility participates in the current root screen, every
future rollout decision, next-pick replacement value, and final candidate rank.

The default utility is 20% of combined QB and pass-catcher projected PPG,
capped at 8 points per pair and 12 points per QB/team. The team cap gives later
double-stack additions diminishing value; if multiple QBs from one NFL team are
rostered, only the strongest QB stack on that team receives utility. The app
presents raw `Decision EV`,
average `Roster Stack`, immediate `Stack Now`, and the combined `Decision Score`
side by side. Raw EV is never relabeled as correlated forecast points.

Milestone A originally used 16 candidates. That study compared it with 32
candidates in a physical slot-six state (five round-one opponent selections
already removed) and a derived seventh-round state. In both observed states the
best 16-candidate option matched the 32-candidate best and empirical omission
regret was zero.

The Preview keeps 24 candidates at every draft depth. After replacement-aware
scoring and roster-need quotas, the DK-only release gate found one 24-versus-32
miss across 27 states: 8.25 points at slot 12/seed 2017/round 8. That clears the
fixed 10-point shortlist gate. The broader pool is an intentional
runtime-for-coverage tradeoff and is not adaptively reduced late. The isolated
current-V2 prelaunch and 33 additional clean fresh states found no positive
R512 advantage for the 32-candidate action over the production 24-candidate
action. The exact launch later failed closed when only a wide control worker
hit a native access violation, so those states remain diagnostic rather than a
completed confirmation.

The scarcity-aware and pure-greedy policies selected Jonathan Taylor in the
physical opening fixture, but scarcity changed their future draft paths. The
paired completed-roster result favored scarcity by 28.1 points with an
approximate paired SE of 11.6. This metric is generated by the study script.

## Runtime Baseline

On the 2026 DK database, pick slot 6, 12 teams, and 20 roster spots:

- five matched physical-fixture repeats, 24 rooms/iterations: sequential p50
  3.26 seconds versus frozen single-worker legacy p50 7.09 seconds;
- the sequential policy was faster in every matched repeat and therefore passes
  the current runtime gate;
- the primary physical opening run was 3.26 seconds and the derived mid-draft
  run was 4.57 seconds;
- 32 current candidates took 6.19 seconds opening and 8.85 seconds mid-draft.

Candidate rollouts dominate runtime. Template tensor packing is cached on the
simulation object; subsequent score-bank draws are vectorized and inexpensive.
Roster legality is vectorized, and conditional next-pick survival is precomputed
once per player/turn instead of recalculated inside every rollout.

With replacement-aware future choices and all 24 completed candidates receiving
the operational decision score, the 27-state DK matrix measured app-equivalent
p50 of 3.85 seconds in round 1, 2.25 in round 8, and 1.10 in round 15, versus
legacy at 2.98, 1.12, and 0.27. Across all states the Preview was 2.25 seconds
versus 1.12. Hidden audit work was excluded and cost about 0.26 seconds p50.
This still fails the no-slower-than-legacy promotion gate. The focused 50-room
Puka fixture took 9.73 seconds, with 8.97 seconds in candidate rollouts.

The final production smoke completed DK D256 at 24/24 rooms in 11.72 seconds
end to end without loading CVXOPT. Legacy completed 50/50 simulations in 7.76
seconds and loaded CVXOPT only inside its disposable worker. The isolated
six-state prelaunch measured primary parent p50/p90 of 12.74/12.90 seconds; the
33 clean primary arms in the later exact launch measured 8.64/12.80 seconds
across opening, mid-, and late-draft states. Both are comfortably below the
accepted parent-observed p90 SLA of 30 seconds. Runtime relative to Legacy is a
diagnostic, not a release gate.

Fresh Streamlit sessions initialize Sequential with the validated 24 rooms.
Custom room counts are under Advanced settings and show a warning; Legacy
initializes at 50 simulations with one inner worker.

## Validation

Run:

```powershell
python research/studies/2026-07-19_sequential_best_ball_policy/verify_milestone_a.py
python research/studies/2026-07-19_sequential_best_ball_policy/verify_replacement_policy.py
python research/studies/2026-07-19_sequential_best_ball_policy/run_milestone_a.py
python research/studies/2026-07-19_sequential_best_ball_release_gate/run_release_gate.py
python research/studies/2026-08-02_sequential_v2_bank_stability/run_study.py
.venv_snake_312\Scripts\python.exe research/studies/2026-08-02_sequential_v2_fresh_confirmation/run_study.py --fail-fast
..\Fantasy_Football\.venv_ff_312\Scripts\python.exe research/studies/2026-08-02_sequential_historical_forced_pick_replay/run_study.py --phase score --output-dir research/studies/2026-08-02_sequential_historical_forced_pick_replay/results_v5
```

The verifier checks frozen legacy source hashes, tensor scoring parity,
candidate-consistent availability, empirical replacement integration,
roster-need root quotas, global player removal, unique/legal roster paths,
mutual construction/pilot/decision/audit bank disjointness, decision-seed path
invariance, audit-seed recommendation invariance, non-blocking warnings for
opponent-pick count mismatches, and real-database smoke/regression runs.

## Known Limitations

- Opponents use noisy ADP priority, not roster-aware or opponent-specific
  behavior.
- Replacement values assume the independently sampled player ADP distributions
  used by the app; correlated room tactics are not yet modeled.
- The prior top-four decision gate excluded candidates that ranked well on
  broader independent scoring. Every completed candidate now receives the
  operational decision score. Replacement-aware rollout choices then fixed the
  focused Puka-only round-two failure: Love ranked first, Rice second, Olave
  fourth, and the three screened QBs ranked 20th, 22nd, and 24th. Independent
  bank noise remains: decision and audit winners agreed in 17 of 27 refreshed
  states, with four regrets above 10 points and a maximum of 12.13.
- On the current-V2 reused-seed compatibility matrix, D128 exact R512-winner
  agreement was 11/27 and D256 improved it to 17/27. D256 reduced cross-fitted
  reference score-gap mean/maximum from 1.400/10.823 to 0.107/4.303 and changed
  eight actions, seven positively on R512. The +1.2927-point conditional mean
  has a three-base-seed 95% interval of [0, +2.2599]; it touches zero and is not
  promotion-grade evidence of superiority.
- The exact fresh confirmation launch is not a formal pass: a non-production
  32-candidate control worker exited with Windows `0xC0000005` after 33 clean
  states. Those states nevertheless showed D256 +0.621 R512 points, lower
  regret, zero 24-versus-32 loss, and 12.80-second primary parent p90. The
  completed held-out historical replay is the approval-grade D256/D128 test.
- The app warns when the number of marked opponent picks does not match the
  snake schedule, but the Preview still runs with the marked availability state.
  An unmarked drafted player may therefore appear in recommendations until the
  user marks that player Other Team and reruns.
- The app deliberately uses fixed construction/draft and evaluation seeds so
  repeated clicks on an unchanged draft state are reproducible. The API exposes
  `evaluation_seed` for validation and research.
- The current Streamlit template cache is keyed by database modified time. A
  modeling-repo build ID is preferable to OneDrive-sensitive mtimes when the
  source database begins publishing one.
- Sequential stack utility is a tournament-objective proxy, not a calibrated
  joint weekly outcome model. Shared team-passing shocks or joint weekly player
  outcomes remain the appropriate future mechanism for measuring actual
  late-season correlation and tail probability.

## Release Decision

As of 2026-08-03, nested D256 is approved for DK Sequential Preview. The final
leakage-safe replay froze 81 states before outcome access and scored 3,888
paired arm-room results across held-out 2017-2025 seasons. D256 averaged
2,079.373 points versus 2,078.467 for D128: +0.907 points (+0.044%). The
season-clustered 95% interval was [-3.160, 5.439] points
([-0.150%, 0.262%]), above the preregistered -5.196-point (-0.25%)
noninferiority margin. Every frozen design, leakage, source, identity, roster,
isolation, and scoring gate passed.

Sequential is the production methodology independent of GLPK/Legacy results.
Legacy remains available for fallback and diagnostics, but it is not the
approval oracle and does not block D256. NFFC offense-only remains D128. Keep
the Preview label while live stability and modeling limitations are monitored.
