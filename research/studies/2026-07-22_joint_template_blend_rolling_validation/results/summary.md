# Joint Template Blend Rolling Validation Readout

## Design

- Held out 2017-2025 one origin at a time and restricted each target player's
  weekly-template donors to strictly earlier seasons.
- Used causal DK `Final_Validations_Resid` OOS point/residual forecasts for
  2,696 target players.
- Drafted 12 preseason-only noisy-ADP rooms per origin: 1,296 legal 20-player
  rosters under the production 2-3 QB, 5-7 RB, 7-9 WR, and 2-3 TE bounds.
- Drew 384 common-random-number outcome seasons per target player and retained
  the same donor weekly path across every blend comparison.
- Scored 1 QB, 2 RB, 3 WR, 1 TE, and one RB/WR/TE flex each week over the
  production 16-week template horizon.
- Compared the current centered/scaled 0.30 residual blend against 0.50/0.70
  scaled blends, 0.50/0.70/0.85 raw blends, full scaled-template residuals, full
  raw centered-template residuals, and model-residual/profile-only.
- Used seasons as bootstrap clusters. The primary window is 2020-2025.

## What "Heavier Template Use" Changes

Snake already uses the sampled historical donor's complete weekly profile at
every blend strength. Missed games and zero weeks therefore already come from
the template. The blend controls only the active-PPG residual applied to that
path:

- current 0.30 combines an independent calibrated model residual with 30% of a
  variance-scaled centered donor residual;
- full scaled uses only the same donor's centered residual, rescaled to the
  model residual standard deviation; and
- full raw uses only the donor's actual pool-centered residual without
  variance rescaling.

The full methods make performance magnitude and weekly availability one joint
historical outcome. The current method makes only the weekly shape fully
template-based.

## Recent Roster Calibration

| Method | Bias | MAE | CRPS | P10-P90 coverage | Width | Top-20% AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current 0.30 scaled | +26.8 | 129.2 | 90.7 | 76.6% | 384.2 | 0.548 |
| 0.70 scaled | +28.6 | 127.7 | 89.8 | 77.0% | 378.3 | 0.552 |
| Full scaled template | +4.8 | 123.0 | 86.2 | 76.3% | 372.9 | 0.569 |
| Full raw template | +22.6 | 124.3 | 87.2 | 80.8% | 415.6 | 0.566 |

Relative to current 0.30:

- 0.70 scaled improved CRPS by 0.87 (season-bootstrap 95% interval 0.11 to
  1.68) and MAE by 1.43 (0.29 to 2.64). It was the only larger blend whose
  recent improvement interval stayed entirely positive.
- Full scaled had the best point estimates: CRPS improved by 4.42 and MAE by
  6.21, roster bias fell from +26.8 to +4.8, and ranking improved. With only six
  recent season clusters, its CRPS interval was -1.72 to +10.79 and probability
  of improvement was 90%.
- Full raw improved CRPS by 3.42 and MAE by 4.86, with 92% probability of
  improvement. Its P10-P90 coverage was essentially exact at 80.8%, but the
  wider residual scale created +22.6 points of mean best-ball bias through the
  convex weekly maximum.
- Intermediate raw blends retained the model residual's positive historical
  mean while adding donor variance. They did not beat current CRPS until 0.85
  and are not attractive production candidates.

Across all nine origins, full raw had 80.6% coverage and the best CRPS among the
raw-scale choices. Full scaled had the best recent accuracy but its 2017
historical result collapses to a point forecast because 2017 residual-quantile
calibration is unavailable. All 268 current 2026 DK rows have complete residual
quantiles, so that historical fallback is not a current-runtime limitation.

## Missed-Week and Zero-Season Result

The weekly best-ball mechanism is behaving as intended:

- Recent rosters averaged 80.7 missing player-weeks across their 20 players.
- After weekly best-ball replacement, they averaged only 1.73 zero-valued
  lineup slots out of 128; 73.7% had at least one, but the overall lineup-slot
  rate was only 1.35%.
- Roster depth did not erase all fragility. Rosters with two or more completely
  zero-active players averaged 4.04 zero lineup slots, versus 1.54 for rosters
  with none.
- In the highest missed-week third (278 rosters), current 0.30 had CRPS 107.1,
  +117.7 bias, and 65.8% coverage. Full scaled improved those to 95.5, +89.4,
  and 70.5%; full raw improved them to 101.7, +107.8, and 71.2%.
- Among the 28 recent rosters with at least two zero-active players, full
  scaled reduced CRPS from 90.6 to 79.4 and MAE from 122.4 to 106.4. Full raw
  reduced CRPS to 85.1 and MAE to 112.2. This cell is directionally useful but
  too small to stand alone.

Zero weeks therefore are not "dragged through" the lineup as starter points.
They are replacement opportunities. Heavier joint residual use still helps
because it better prices how much useful production the rest of the donor path
contains when availability is poor.

## Recommendation

- Retire the assumption that 0.30 is the calibrated optimum.
- If making one conservative production change now, use 0.70 scaled: its gain
  is smaller, but it is the only heavier setting with a positive
  season-clustered 95% interval for both recent CRPS and MAE.
- Keep full scaled as the leading high-template candidate. It has materially
  better point accuracy, ranking, mean bias, and high-missed-week behavior, but
  its interval is too narrow and its six-season uncertainty still crosses zero.
- Keep full raw as the calibration reference: it gives the best nominal
  coverage and is fully coherent, but its raw residual variance inflates mean
  best-ball scores through lineup optionality.
- Before promoting full scaled, run the frozen sequential state/rank suite at
  0.30, 0.70, and full scaled. Require legal completion, acceptable runtime,
  and recommendation stability; do not choose the setting from one live board.

## Subsequent Production Decision

On 2026-07-23, `full_scaled_v1` was promoted after focused sampling invariants
and frozen sequential fixtures. The implementation and promotion evidence are
recorded in
`research/studies/2026-07-23_full_scaled_template_promotion/`. The legacy 0.30
scaled blend remains callable for rollback and matched comparison.
