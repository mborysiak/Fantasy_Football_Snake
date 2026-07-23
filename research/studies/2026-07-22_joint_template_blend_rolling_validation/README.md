# Joint Template Blend Rolling Validation

## Objective

Compare the Snake app's current centered/scaled 0.30 template-residual blend
with heavier and full-template alternatives under historical best-ball roster
scoring.

This is intentionally separate from the managed-auction validation:

- it uses the source template's `week_1` through `week_16` best-ball profiles;
- it scores complete 20-player rosters as 1 QB, 2 RB, 3 WR, 1 TE, and one
  RB/WR/TE flex every week; and
- a player's missed or zero week only hurts when roster depth cannot replace it
  in the weekly lineup.

## Design

- Hold out 2017-2025 one origin at a time.
- Use causal DK `Final_Validations_Resid` OOS forecasts for target players.
- Restrict every target's historical template donors to strictly earlier
  seasons.
- Draft 12 preseason-only synthetic 12-team rooms per origin from noisy
  historical ADP, producing 1,296 legal 20-player rosters.
- Draw 384 common-random-number outcome seasons per target player.
- Compare model-residual/profile-only, current 0.30, heavier 0.50/0.70 scaled
  blends, 0.50/0.70/0.85 raw-template blends, full scaled-template, and full
  raw centered-template residual methods.
- Preserve the same sampled donor weekly profile across every residual method.

Run from the Snake repository root:

```powershell
..\Fantasy_Football\.venv_ff_312\Scripts\python.exe research\studies\2026-07-22_joint_template_blend_rolling_validation\run_validation.py
```

Outputs are written to `results/`.
