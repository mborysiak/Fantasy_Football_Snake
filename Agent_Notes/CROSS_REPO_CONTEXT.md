# Cross-Repo Context

Last updated: 2026-07-06

## Repos

- `Fantasy_Football`: owns modeling, table generation, and source
  `Data/Databases/Simulation.sqlite3`.
- `Fantasy_Football_Snake`: owns Streamlit app runtime, ILP logic, and app copy
  `app/Simulation.sqlite3`.

## Database Handoff

The modeling repo's best-ball weekly build writes the source DB and copies it to:

```text
app/Simulation.sqlite3
```

This app should treat that file as generated. Durable fixes to table contents
belong in the modeling repo generation scripts.

## App-Sensitive Tables

- `Final_Predictions_Resid`
- `Avg_ADPs`
- `Best_Ball_Weekly_Templates`
- `Best_Ball_Weekly_Template_Pools`
- `Best_Ball_Weekly_Player_Map`
- `Best_Ball_ADP_Audit`

## Coordination Rules

- If app code starts using a new column from `Simulation.sqlite3`, document it in
  `docs/data_contracts/simulation_sqlite_app_contract.md`.
- If the modeling repo changes a table schema, update app loading logic and the
  contract together.
- Calibration/audit outputs should live under `research/studies/`, not `app/`.
