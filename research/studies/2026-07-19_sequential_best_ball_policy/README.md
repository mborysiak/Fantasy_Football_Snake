# Sequential Best-Ball Policy: Milestone A

## Question

Can the app replace clairvoyant, decision-independent draft availability with a
candidate-consistent sequential policy without making an interactive draft
workflow impractically slow?

## Scope

Milestone A began as a shadow engine and passed its initial release gate. It is
now exposed as the DK-only Streamlit default for field testing while remaining
labeled Preview. The existing GLPK path remains available as fallback. The new
engine:

- samples a small construction bank of season outcomes and uses only its mean
  marginal values when making draft decisions;
- assigns construction, pilot, decision, and optional audit unique, explicitly
  disjoint PPG scenario columns and exposes independent seeds;
- samples opponent priority orders from noisy ADP and removes every opponent
  and user pick from one shared room state;
- locks each current-pick candidate, completes the remaining draft with
  empirical next-pick replacement value plus symmetric incremental QB-WR/TE
  stack utility recalculated at every turn, and evaluates completed rosters on
  a separate season bank;
- allocates the 24-root screen across positions using remaining roster minimum
  deficits and maximum capacity;
- uses the same draft rooms and pilot seasons for all current candidates, then
  ranks every completed candidate on raw 128-season EV plus a separately
  reported average final-roster stack utility;
- limits the primary horizon to the 16 weeks available in the weekly-template
  tables and labels it `sequential_template_16`.

The legacy solver remains labeled `legacy_template_16` for template runs and
`legacy_residual_17` for its older residual-week configuration.

## Commands

Run invariant and real-database smoke checks:

```powershell
python research/studies/2026-07-19_sequential_best_ball_policy/verify_milestone_a.py
python research/studies/2026-07-19_sequential_best_ball_policy/verify_replacement_policy.py
```

Run the runtime and shortlist-quality study:

```powershell
python research/studies/2026-07-19_sequential_best_ball_policy/run_milestone_a.py
```

Generated outputs are written to `results/`.

## Gates

All correctness checks must pass:

- frozen legacy method source hashes remain unchanged;
- tensorized lineup and marginal scoring match the existing scorer;
- an owned or newly forced player cannot be selected by an opponent;
- every drafted player is removed globally and paths contain no duplicates;
- all completed rosters satisfy the configured construction ranges;
- changing pilot or decision outcomes cannot change a rollout path;
- changing the hidden audit seed cannot change a recommendation;
- opponent-pick count mismatches emit an advisory warning and continue with the
  marked availability state;
- vectorized legality results match the frozen loop reference;
- empirical replacement integration uses the full conditional survival
  distribution rather than a hard availability cutoff;
- root position quotas sum to the requested candidate pool;
- stack utility is symmetric to whether the QB or pass catcher is drafted first,
  respects pair/team caps, and remains separate from raw EV;
- the real-database smoke run completes every requested room.

The study reports rather than presupposes quality and runtime:

- wall time by engine section;
- completion rate;
- top-pick agreement between the primary and wide candidate sets;
- empirical shortlist omission regret on the shared evaluation design;
- paired standard errors versus the observed best candidate;
- greedy-only versus scarcity-aware policy differences.

The physical slot-six and derived mid-draft measurements found zero empirical
omission regret for 16 versus 32 candidates. Scarcity improved paired final
roster EV by 28.1 points (approximate paired SE 11.6), so it remains enabled.
After vectorized legality and precomputed survival, five matched runs produced a
3.26-second sequential p50 versus 7.09 seconds for legacy. Beam search and
adaptive candidate allocation were not added because the measured gate did not
justify their runtime cost.

The subsequent DK-only release study keeps 24 candidates throughout and gives
all 24 completed roots the independent decision score. Replacement-aware draft
timing and roster-need quotas reduced the focused Puka-only round-two root pool
from 15 QBs to 3: Love ranked first, Rice second, Olave fourth, and the QBs
ranked 20th/22nd/24th. The refreshed frozen matrix completed all 27 states and
maximum 24-versus-32 omission regret was 8.25, clearing the fixed shortlist
gate. Decision and hidden-audit winners agreed in 17 states; four states narrowly
exceeded 10 points of audit regret and the maximum fell from 20.27 to 12.13.
Candidate timing and coverage improved, while evaluation-bank stability remains
a separate limitation.

## Known Limitation

Opponent behavior is a noisy-ADP priority model, not a calibrated model of each
opponent's roster needs or correlated draft-room tactics. Milestone A fixes the
state-transition and information-set errors; it does not claim a complete
behavioral model of opponents.

The stack term is also a tournament-utility proxy rather than a joint weekly
forecast. It rewards correlated roster construction without claiming that the
individual weekly template draws themselves are correlated.
