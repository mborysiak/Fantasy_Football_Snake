# Sequential Best-Ball Policy Runbook

Last updated: 2026-07-20

## Release Status

`best_ball_policy` is the DK-only default for field testing and remains labeled
Preview. The legacy `best_ball_ilp` remains available as the fallback. The
Preview is intentionally limited to the current-pick recommendation; it does
not present future-round recommendations as if their players were fixed in
advance.

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

NFFC remains setup-only beyond the pick schedule. The 2026 Best Ball
Championship is 30 rounds and includes kicker/team-defense slots, while the
current app is still offense-only and defaults to the DK roster construction.

## Methodology Boundary

The policy separates information used to make draft decisions from outcomes
used to score those decisions:

1. Generate 16 construction seasons from the 16-week weekly-template model.
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
7. Rank every completed candidate on a separate 128-season decision bank. Keep
   raw best-ball EV visible and rank on a separate decision score equal to raw
   EV plus average final-roster stack utility.

Release studies may additionally rescore every completed candidate on a fourth
128-season audit bank. Its PPG columns are unique and disjoint from every
operational bank. Audit is diagnostic only: changing its seed cannot change the
candidate screen, rollout paths, pilot ranking, decision ranking, recommendation,
or downstream draft state.

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

- `sequential_template_16`: Preview sequential policy and weekly-template horizon.
- `legacy_template_16`: legacy ILP when weekly templates are selected.
- `legacy_residual_17`: historical independent-residual configuration.

The Preview does not silently synthesize Week 17. Rebuilding weekly templates for
17 weeks remains separate modeling-repo work.

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
runtime-for-coverage tradeoff and is not adaptively reduced late.

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

## Validation

Run:

```powershell
python research/studies/2026-07-19_sequential_best_ball_policy/verify_milestone_a.py
python research/studies/2026-07-19_sequential_best_ball_policy/verify_replacement_policy.py
python research/studies/2026-07-19_sequential_best_ball_policy/run_milestone_a.py
python research/studies/2026-07-19_sequential_best_ball_release_gate/run_release_gate.py
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

Milestone A justified shipping the explicit Preview, retaining scarcity, and
skipping beam search. The DK-only 24-candidate gate cleared candidate coverage,
and the operational decision stage scores all completed candidates. Empirical
replacement-aware draft timing fixes the observed early-QB failure and improves
the maximum audit regret, while the refreshed matrix still fails the fixed
decision/audit stability and runtime gates. On 2026-07-19 the app owner chose
Preview as the fresh-session default for field testing; Legacy remains available
as fallback and the hidden audit bank remains policy-inert.
