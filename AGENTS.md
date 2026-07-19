# Working Agreement

## How To Work
- Keep app changes narrow and runtime-focused.
- Preserve the existing Streamlit and helper-module style unless a task calls for
  a larger cleanup.
- Ask only when ambiguity is consequential and cannot be resolved from local
  code or the modeling repo contract.
- Push back early on schema drift, calibration drift, unstable ADP logic,
  optimizer infeasibility, and performance regressions.

## Startup Reading Order
1. Read `Agent_Notes/SESSION_NOTES.md`.
2. Read the latest relevant monthly note under `Agent_Notes/Session_Notes/`.
3. Check `Agent_Notes/MODULE_TRACKER.md` and `Agent_Notes/DECISION_LOG.md`.
4. For DB/app coupling, read `docs/data_contracts/simulation_sqlite_app_contract.md`.
5. For optimizer/runtime changes, read `docs/runbooks/best_ball_ilp_validation.md`.

## Notes Policy
- `Agent_Notes/SESSION_NOTES.md` is the landing page for active state.
- `Agent_Notes/MODULE_TRACKER.md` stores durable module status and next steps.
- `Agent_Notes/DECISION_LOG.md` stores durable app/runtime decisions.
- `Agent_Notes/Session_Notes/YYYY-MM.md` stores short task receipts.
- Large study outputs belong under `research/`, not `app/`.

## Cross-Repo Policy
- `Fantasy_Football` owns source generation of `Simulation.sqlite3`.
- This repo consumes `app/Simulation.sqlite3`.
- If a table or column from `Simulation.sqlite3` changes, update this app and
  the data contract in the same task when possible.
- Do not hand-edit generated database contents as a durable fix; update the
  source generation path in `Fantasy_Football`.

## Verification
- For helper/app code changes, run focused syntax checks such as:

```powershell
streamlitvenv\Scripts\python.exe -m py_compile app\zSim_Helper.py app\snake_draft_app.py
```

- For optimizer behavior changes, run a small smoke path or document why it was
  not practical.
