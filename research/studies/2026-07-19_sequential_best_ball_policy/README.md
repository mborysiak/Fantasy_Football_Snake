# Sequential Best-Ball Policy: Milestone A

## Question

Can the app replace clairvoyant, decision-independent draft availability with a
candidate-consistent sequential policy without making an interactive draft
workflow impractically slow?

## Scope

Milestone A began as a shadow engine and passed its initial release gate. It is
now exposed as an explicit Streamlit Beta while the existing GLPK path remains
the default and fallback. The new engine:

- samples a small construction bank of season outcomes and uses only its mean
  marginal values when making draft decisions;
- assigns construction and evaluation unique, explicitly disjoint PPG scenario
  columns and exposes an independent evaluation seed;
- samples opponent priority orders from noisy ADP and removes every opponent
  and user pick from one shared room state;
- locks each current-pick candidate, completes the remaining draft with a
  greedy expected-value plus scarcity policy, and evaluates completed rosters
  on a separate season bank;
- uses the same draft rooms and evaluation seasons for all current candidates;
- limits the primary horizon to the 16 weeks available in the weekly-template
  tables and labels it `sequential_template_16`.

The legacy solver remains labeled `legacy_template_16` for template runs and
`legacy_residual_17` for its older residual-week configuration.

## Commands

Run invariant and real-database smoke checks:

```powershell
python research/studies/2026-07-19_sequential_best_ball_policy/verify_milestone_a.py
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
- changing a hidden evaluation bank cannot change a rollout path;
- vectorized legality results match the frozen loop reference;
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

## Known Limitation

Opponent behavior is a noisy-ADP priority model, not a calibrated model of each
opponent's roster needs or correlated draft-room tactics. Milestone A fixes the
state-transition and information-set errors; it does not claim a complete
behavioral model of opponents.
