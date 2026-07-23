# Full-Scaled Template Promotion Readout

## Decision

Promote `full_scaled_v1` as the Snake app's default weekly-template outcome
process. Retain `template_resid_blend=0.30` as a callable rollback and matched
validation comparator.

The production path now:

1. samples one historical donor season from the published player pool;
2. centers that donor's active-PPG residual within the pool;
3. scales the centered residual to the current player's model-residual
   standard deviation; and
4. applies the same donor's 16-week availability and scoring profile.

No independent model-residual draw is added at the production default.

## Outcome Evidence

The preceding strict-prior rolling replay covered 2,696 players and 1,296 legal
20-player rosters. In the recent 2020-2025 window, full scaled improved CRPS
from 90.7 to 86.2, MAE from 129.2 to 123.0, mean bias from +26.8 to +4.8, and
top-20% AUC from 0.548 to 0.569 versus the legacy 0.30 blend. It also improved
high-missed-week CRPS from 107.1 to 95.5.

## Focused Runtime Checks

- The default blend is 1.00 and the independent model-residual weight is zero.
- Holding the donor draw fixed, changing the independent model-residual sample
  column leaves full-scaled scores exactly unchanged.
- The legacy 0.30 fallback still responds to the model-residual sample.
- The blend survives `get_sim_config()` reconstruction used by worker
  processes.
- Invalid blend values are rejected.

## Frozen Sequential Checks

Both the legacy and full-scaled methods completed all nine slot/depth fixtures
with no execution errors, 100% room completion, and zero shortlist-omission
regret.

The older sequential-policy release gate remained globally false for both
methods because its sequential-versus-legacy runtime requirement and strict
audit-regret threshold were already missed by the 0.30 baseline. Full scaled
did not worsen those results: policy runtime p50 moved from 4.83 to 4.77
seconds and maximum audit regret improved from 13.47 to 10.82.

Across nine identical root-state method comparisons (three slots by three
seeds):

- mean recommendation-rank correlation was 0.837;
- mean Decision-EV rank correlation was 0.836;
- candidate-set Jaccard similarity was 0.797;
- the top-five lists shared 3.44 players on average;
- the exact top player agreed in 2 of 9 states; and
- runtime p50 was effectively unchanged: 8.064 seconds at 0.30 versus 8.069
  seconds at full scaled.

The broad board therefore remains recognizable, but the top action changes
materially. That is expected from replacing a mostly independent performance
draw with a joint donor magnitude/availability outcome and should be monitored
after material projection or template-pool updates.

## Regression Checks

- Syntax compilation passed.
- The Milestone A sequential correctness suite passed, including tensor
  scoring, legality, bank isolation, candidate-consistent state, seed
  invariance, and the real-database smoke check.
- The replacement-policy fixture passed after replacing its brittle exact
  first/second player assertion with the intended structural requirement that
  Love and Rice remain in the top five. All completion, position-quota, and
  no-early-QB gates remained intact.
