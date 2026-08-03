# Current-V2 Sequential Decision-Bank Stability

## Question

Does the intended 24-room Sequential Preview become materially more stable when
its operational decision bank grows from 128 to 256 season scenarios and its
policy-inert reference bank grows from 128 to 512 scenarios?

This is a current-V2 DK compatibility study, not a complete promotion study.
It uses the existing 27-state matrix: pick slots 1/6/12, seeds
17/1017/2017, and completed-pick depths 0/7/14. These seeds previously informed
policy development, so a passing result still requires a separately frozen
fresh-seed confirmation before promotion.

## Frozen Design

- 12 teams, 20 rounds, DK, 2026.
- 24 noisy-ADP draft rooms.
- 24 roster-need-balanced root candidates.
- Default-on Sequential stack utility: 20%, cap 8 per pair, cap 12 per
  QB/team.
- One nested allocation from the 1,000 available scenario columns:
  - 16 construction;
  - 64 pilot/evaluation;
  - 256 decision superbank;
  - 512 policy-inert reference scenarios;
  - 152 columns intentionally unused.
- The control action uses the exact 128 columns selected by the production
  allocator. A separately seeded, non-overlapping 128-column extension forms
  the 256-scenario challenger bank. The historical 128-scenario audit view is
  the first 128 rows of the common 512-scenario reference bank.
- Construction rooms, candidate rosters, decision evidence, and reference
  evidence are common across actions. The reference bank never selects a
  downstream state.
- Rounds 8 and 15 are frozen from room zero of the deployed 128-scenario
  control trajectory. The estimand is therefore one-step recommendation quality
  in control-policy-reachable states.

The UI currently initializes a fresh Preview with 50 rooms even though the
helper constant and help text say 24. This study explicitly tests the intended
24-room setting; it becomes literally app-default-equivalent only after that
configuration mismatch is resolved.

## Primary Metrics

Both actions are scored on the same independent 512-scenario combined objective:
raw best-ball EV plus deterministic average final-roster stack utility.

The primary paired value effect is:

```text
reference value(challenger action) - reference value(control action)
```

Reference regret is cross-fitted. The first 256 reference scenarios choose an
empirical best action that is scored on the second 256; the halves are then
swapped and averaged. This avoids selecting and judging the reference winner
on the same simulations. Full-reference oracle regret, exact winner agreement,
raw-EV regret, rank correlation, and the legacy 128-scenario audit view are
diagnostics.

Aggregate uncertainty resamples the three base seeds. Each seed cluster contains
all three slots and all three draft depths because scenario/template seeds are
shared across slots. The bootstrap uses 10,000 draws and seed 20260802. With
only three clusters, its interval is a conditional compatibility diagnostic,
not promotion-grade significance.

## Frozen Gates

1. The exact frozen configuration is present, and the source database and
   relevant app code are SHA-256 fingerprinted.
2. All 27 states execute; every root completes all 24 legal rooms; physical
   state and final-pick pool coverage checks pass.
3. Control and challenger have identical candidates, quotas, ADP rooms,
   rollout paths, and first-128 decision tensors.
4. Construction, pilot, full decision, and reference columns are pairwise
   disjoint, with exact 128-of-256 decision nesting.
5. Challenger cross-fitted reference regret is at most 10 points in every
   state.
6. Challenger mean and maximum cross-fitted regret do not exceed control.
7. The trajectory-bootstrap 95% lower bound for the paired reference value
   effect is no worse than -0.25% of the control reference mean.
8. Preserve the existing operational runtime gate: challenger median runtime,
   excluding policy-inert reference work, is no slower than matched Legacy.

Failure of only gate 8 can support the larger bank methodologically while still
blocking full Preview promotion on latency.

## Run

```powershell
streamlitvenv\Scripts\python.exe `
  research\studies\2026-08-02_sequential_v2_bank_stability\run_study.py
```

Use the following only for a mechanical smoke; subset runs are explicitly
ineligible to pass the frozen-study gates:

```powershell
streamlitvenv\Scripts\python.exe `
  research\studies\2026-08-02_sequential_v2_bank_stability\run_study.py `
  --slots 6 --seeds 17 --completed-picks 0 --rooms 2 `
  --output-dir research\studies\2026-08-02_sequential_v2_bank_stability\artifacts\local\smoke
```

Durable results from the exact frozen design belong in `results/`.

## Results

The exact study completed all 27 states with zero errors and passed every
methodological compatibility gate. D256 reduced mean/maximum cross-fitted
reference score gap from 1.400/10.823 to 0.107/4.303 and reduced gaps above 10
from two to zero. It changed eight actions; seven improved on the common R512
bank. The paired conditional point estimate was +1.2927 combined-objective
points (+0.0579%), with a three-base-seed clustered 95% interval of
[0, +2.2599] points. The interval touches zero and the seeds were reused, so
this advances D256 to fresh-seed confirmation rather than proving superiority.

The latency gate failed: D256 median policy time was 13.452 seconds versus
3.614 seconds for Legacy. D256 added only 3.18% over D128's 13.038 seconds,
which locates the main bottleneck in candidate rollouts rather than the larger
decision bank. Preview remains unpromoted pending runtime work, fresh seeds,
current-V2 24-versus-32 coverage, and historical forced-pick validation.

- Full interpretation: [`results/findings.md`](results/findings.md)
- Aggregate receipt: [`results/summary.json`](results/summary.json)
- State evidence: [`results/state_metrics.csv`](results/state_metrics.csv)
- Source receipt: [`results/source_receipt.json`](results/source_receipt.json)
