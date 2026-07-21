# Sequential Best-Ball Default-Promotion Gate

## Purpose

Exercise the DK-only Sequential Preview across physical draft states. The
operational policy uses a 64-season pilot for diagnostics and a disjoint
128-season decision bank to rank all 24 completed candidates. A fourth
128-season audit bank is available only to this study and cannot change
recommendations or draft paths.

## Default Matrix

- League slice: `dk`
- Draft slots: 1, 6, 12
- Seeds: 17, 1017, 2017
- Completed user picks: 0, 7, 14 (current rounds 1, 8, 15)
- Preview/benchmark candidate pools: 24/32
- Draft rooms: 16
- Season banks: 16 construction, 64 pilot, 128 decision, 128 audit
- Decision candidates: all 24 completed roots

All four PPG-column banks are unique and mutually disjoint. Physical fixtures
include every opponent selection before the user's current pick. Later fixtures
follow the decision-stage winner; the hidden audit winner never selects a draft
path or a downstream validation state.

## Gates

- No observed 24-candidate shortlist omission regret above 10 points versus 32
  candidates.
- Decision-winner regret on the hidden audit bank no greater than 10 points.
- Every candidate-room rollout completes legally.
- App-equivalent Sequential runtime p50 no slower than matched legacy template
  ILP. Hidden audit time is excluded; decision-stage time is included.
- The DK pool can simulate every room pick through the user's final selection.

Run:

```powershell
python research/studies/2026-07-19_sequential_best_ball_release_gate/run_release_gate.py
```

Outputs are written to `results/state_metrics.csv` and
`results/release_gate_summary.json`. A failed gate retains Preview status; its
threshold is not changed after observing results.

## All-Candidate Result

The latest gate scores all 24 completed candidates on the operational decision
bank after replacement-aware future choices and roster-need root quotas. The app
owner chose Preview as the fresh-session default for field testing, with Legacy
retained as fallback.

- All 27 physical states completed with no errors and every candidate-room
  rollout legal and complete.
- The 24-player screen cleared the shortlist gate. Its only observed miss was
  8.25 points at slot 12/seed 2017/round 8.
- Decision and hidden-audit winners agreed in 17 of 27 states. Four states
  exceeded the fixed 10-point audit-regret gate, with a maximum of 12.13 at
  slot 6/seed 1017/round 1. This improves the prior 20.27 maximum but does not
  pass the predeclared threshold.
- Keeping 24 candidates throughout has the accepted runtime cost. App-equivalent
  p50 was 3.85 seconds in round 1, 2.25 in round 8, and 1.10 in round 15,
  versus legacy at 2.98, 1.12, and 0.27. Across all states Preview was 2.25
  seconds versus 1.12 for legacy.
- Hidden audit work cost about 0.26 seconds p50 per state and is not
  included in the app path or its runtime measurements.

## Follow-up Gate Work

1. Retain all-candidate decision scoring and empirical replacement-aware draft
   timing; the focused Puka-only state now has no QB in the top five.
2. Diagnose the remaining decision/audit variance with larger or repeated
   evaluation banks and fresh predeclared seeds.
3. Improve the noisy-ADP opponent policy with roster-aware behavior before
   adding more search complexity.
