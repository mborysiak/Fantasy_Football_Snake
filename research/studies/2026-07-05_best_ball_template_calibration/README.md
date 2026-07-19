# Best-Ball Template Calibration

## Objective

Collect app-side calibration outputs for weekly template residual blending and
x-pruning availability leakage checks.

## Inputs

- `app/Simulation.sqlite3`
- `app/zSim_Helper.py`
- `app/snake_draft_app.py`

## Outputs

Durable CSV outputs are stored in `results/`.

Current outputs:
- `template_resid_blend_calibration_summary.csv`
- `template_resid_blend_calibration_by_pos.csv`
- `template_resid_blend_calibration_players.csv`
- `x_pruning_availability_leak_summary.csv`
- `x_pruning_availability_leak_by_round.csv`
- `x_pruning_max_side_leak_summary.csv`
- `x_pruning_max_side_leak_by_round.csv`
- `x_pruning_max_side_leak_by_pick_position.csv`

## Notes

Current conclusions promoted to app notes:
- keep template residual blend strength at `0.30`
- center and scale template residuals before blending with model residuals
- use a wider max-side pick buffer to reduce deterministic-rank availability
  leakage
