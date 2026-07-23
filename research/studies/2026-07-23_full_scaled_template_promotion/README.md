# Full-Scaled Template Promotion

This study records the production promotion of the Snake app's joint weekly
template outcome process from the legacy 0.30 scaled blend to full scaled.

The production method samples one historical donor season, centers that
donor's active-PPG residual within the matched pool, rescales it to the current
model residual standard deviation, and applies the same donor's 16-week
availability and scoring shape. It does not add an independent model-residual
draw.

The prior 0.30 method remains callable through
`FootballSimulation(template_resid_blend=0.30)` for rollback and matched
validation.

Run the focused checks from the repository root:

```powershell
streamlitvenv\Scripts\python.exe research\studies\2026-07-23_full_scaled_template_promotion\verify_full_scaled_sampling.py
```

Frozen sequential release-gate outputs for both methods are stored under this
study's `results/` directory.

The root-board comparison runs both methods on identical initial opponent
picks for slots 1, 6, and 12 across three frozen seeds and saves the complete
recommendation boards:

```powershell
streamlitvenv\Scripts\python.exe research\studies\2026-07-23_full_scaled_template_promotion\compare_frozen_root_boards.py
```
