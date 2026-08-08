# Fantasy Football Snake

Streamlit draft assistant for snake-draft best ball. The app consumes the
generated `app/Simulation.sqlite3` database owned by the sibling
`Fantasy_Football` modeling repository.

## Run

Use the pinned Python 3.12 environment so the one-shot simulation workers inherit
the validated NumPy/SciPy/CVXOPT stack:

```powershell
.venv_snake_312\Scripts\python.exe -m streamlit run app\snake_draft_app.py
```

Fresh sessions default to Sequential Preview. DK uses 24 rooms, 24 candidates,
and the approved nested D256 decision bank; NFFC offense-only remains D128.
Sequential and Legacy each run in a fresh subprocess with one inner worker and
no automatic retry or method fallback. Legacy remains available as a
fallback/diagnostic, not as the Sequential approval oracle.

## League Selector

- `DK` remains the default format.
- `NFFC` is an experimental offense-only scoring adapter available when the
  database contains the governed NFFC slice. That slice uses independently
  scored NFFC offensive projections, the current canonical composite NFFC ADP
  feed, and 17-week modern-era weekly templates. NFFC draft turns currently use
  Third Round Reversal.

The selector shows only year/league prediction slices backed by the current
weekly player map; older maps are removed when annual template IDs are rebuilt.
Saved draft CSVs record the selected league and restore it on upload. CSVs from
before the league field was added remain valid and load with DK as the default.
Uploaded team-count and draft-position settings remain the active control
baseline across reruns while still allowing the user to edit either control.
Imported My Team and Other Team selections remain the data-editor baseline, so
subsequent checkbox or sidebar interactions do not clear the loaded draft.

The NFFC mode is intentionally offense-only: it supports `QB`, `RB`, `WR`, and
`TE`, but not kicker or team defense. It is therefore not a complete
implementation of an official NFFC contest. The app also does not currently
offer the straight-snake schedule used by NFFC25/NFFC50 formats; selecting NFFC
always selects the existing Third Round Reversal behavior. The official $150
Best Ball Championship is a 30-round format with `TK` and `TDSP` roster slots;
this app's offense-only roster size remains configurable and defaults to 20, so
it does not enforce that contest composition. See the
[official NFFC rules](https://nfc.shgn.com/rules/2680).

See
[`docs/data_contracts/simulation_sqlite_app_contract.md`](docs/data_contracts/simulation_sqlite_app_contract.md)
for the database contract and
[`docs/runbooks/best_ball_ilp_validation.md`](docs/runbooks/best_ball_ilp_validation.md)
for refresh and runtime checks.
 
