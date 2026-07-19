# Sequential Best-Ball Default-Promotion Gate

## Purpose

Exercise the DK-only Sequential Preview across physical draft states before
considering a default flip. The operational policy uses a 64-season pilot to
select four finalists and a disjoint 128-season decision bank to rank them. A
fourth 128-season audit bank is available only to this study and cannot change
recommendations or draft paths.

## Default Matrix

- League slice: `dk`
- Draft slots: 1, 6, 12
- Seeds: 17, 1017, 2017
- Completed user picks: 0, 7, 14 (current rounds 1, 8, 15)
- Preview/benchmark candidate pools: 24/32
- Draft rooms: 16
- Season banks: 16 construction, 64 pilot, 128 decision, 128 audit
- Decision candidates: pilot top 4

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

## Observed Result

The DK-only default-promotion gate still fails. The app owner subsequently chose
Preview as the fresh-session default for field testing, with Legacy retained as
fallback. That product choice does not change the failed gate result.

- All 27 physical states completed with no errors and every candidate-room
  rollout legal and complete.
- The 24-player screen cleared the shortlist gate. Its only observed miss was
  7.56 points at slot 12/seed 1017/round 1, down from the previous 16-player
  screen's 10.71-point maximum.
- Decision and hidden-audit winners agreed in 18 of 27 states, up from the old
  pilot/confirmation comparison's 16 of 27. Two opening states exceeded the
  fixed 10-point audit-regret gate: 15.59 points at slot 6/seed 1017 and 15.39
  at slot 12/seed 1017. Every round-8 and round-15 state stayed within 10.
- Keeping 24 candidates throughout has the accepted runtime cost. App-equivalent
  p50 was 7.60 seconds in round 1, 4.48 in round 8, and 2.00 in round 15,
  versus legacy at 5.96, 2.00, and 0.64. Across all states Preview was 4.48
  seconds versus 2.00 for legacy.
- Hidden audit work cost only about 0.12 seconds p50 per state and is not
  included in the app path or its runtime measurements.

## Follow-up Gate Work

1. Treat the four decision finalists as an uncertainty tier in the Preview UI;
   the decision winner is not yet stable enough to justify a default flip.
2. Test additional predeclared decision-season counts or repeated audit seeds
   on a fresh tuning study before altering the fixed release gate.
3. Improve the noisy-ADP opponent policy with roster-aware behavior before
   adding more search complexity.
4. Repeat the frozen DK matrix with fresh seeds after methodology changes.
