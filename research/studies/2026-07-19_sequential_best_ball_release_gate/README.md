# Sequential Best-Ball Default-Promotion Gate

## Purpose

Exercise the Beta across physical draft states before considering a default
flip. This study adds an independent confirmation stage after the adaptive
pilot ranking and tests player-pool coverage, shortlist regret, confirmation
stability, completion, and matched runtime.

## Default Matrix

- League slices: `dk`, `beta`
- Draft slots: 1, 6, 12
- Seeds: 17, 1017, 2017
- Completed user picks: 0, 7, 14 (current rounds 1, 8, 15)
- Primary/wide candidate pools: 16/32
- Draft rooms: 16
- Season banks: 16 construction, 64 pilot, 128 confirmation
- Confirmation candidates: pilot top 4

All construction, pilot, and confirmation PPG columns are unique and mutually
disjoint. Physical fixtures include every opponent selection before the user's
current pick.

## Gates

- No observed primary-shortlist omission regret above 10 points versus 32
  candidates.
- Pilot winner confirmation regret no greater than 10 points.
- Every requested candidate-room rollout completes legally.
- Sequential runtime p50 no slower than matched legacy template ILP.
- Every configured league/format has enough modeled players to simulate every
  pick through the user's final selection.

Run:

```powershell
python research/studies/2026-07-19_sequential_best_ball_release_gate/run_release_gate.py
```

Outputs are written to `results/state_metrics.csv` and
`results/release_gate_summary.json`. A failed gate is evidence to retain Beta
status, not a reason to weaken the threshold after observing results.

## Observed Result

The default-promotion gate failed, so legacy remains the default.

- DK completed all 27 physical states with no execution errors and every
  candidate-room rollout legal and complete.
- The 16-player screen missed three 32-player winners. Regret was 10.71 points
  at slot 1/seed 2017/round 1, 7.56 at slot 12/seed 1017/round 1, and 2.18 at
  slot 12/seed 17/round 15. Only the first exceeded the fixed 10-point gate.
- Pilot and confirmation winners agreed in 16 of 27 states. Three opening
  states exceeded the 10-point confirmation-regret gate, with a maximum of
  18.01 points. All round-8 and round-15 states were within 10 points.
- Sequential was faster in all nine opening states (p50 5.17 seconds versus
  5.88 for legacy), but slower in every round-8 and round-15 state. Across all
  DK states its p50 was 3.10 seconds versus 1.97 for legacy.
- The `beta` slice has 180 modeled players. A 12-team, 20-round room needs 218
  to 240 still-available players at the tested opening picks, so all 27 Beta
  states were recorded as unsupported rather than simulated with missing
  players.

Downstream fixtures follow the pilot winner—the recommendation visible to the
deployed policy. The hidden confirmation winner is never allowed to select a
draft path or later validation state.

## Follow-up Gate Work

1. Deepen the upstream `beta` player table or explicitly define and validate a
   smaller Beta draft format.
2. Diagnose the three opening shortlist/confirmation failures. Test more pilot
   evaluation seasons and targeted position-screen coverage before simply
   increasing every rollout candidate.
3. Profile round-8 and round-15 rollouts; preserve the opening-round speed win
   while bringing the across-state p50 below legacy.
4. Repeat this frozen matrix with fresh seeds after changes. Do not promote on
   the tuning matrix alone.
