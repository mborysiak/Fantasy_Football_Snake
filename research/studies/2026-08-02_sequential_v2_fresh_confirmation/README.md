# Sequential V2 Fresh Confirmation

## Frozen question

On untouched current-V2 DK random streams, does the nested D256 decision bank
confirm the D128 compatibility result, remain adequately covered by the
24-candidate root screen relative to 32 candidates, and satisfy the accepted
24-room usability SLA?

## Preregistered design

- DK 2026, 12 teams, 20 rounds, slots 1/6/12.
- Nine untouched base seeds: 20017, 21017, 22017, 23017, 24017, 25017,
  26017, 27017, and 28017.
- A SHA-256-derived independent stream for every `(base seed, slot)` pair.
- Completed-pick depths 0/7/14, for 81 paired states.
- 24 shared noisy-ADP rooms; primary root screen 24; wide screen 32.
- C16, E64, nested D128/D256, and policy-inert R512, all disjoint.
- The exact production D128 allocation is the first half of D256. The D256
  extension uses the prior validated separately seeded nonoverlapping allocator.
- Primary and wide arms use identical random streams. Candidate sets are not
  assumed nested. Paths and score tensors must match exactly for every overlap;
  selected actions from both arms are scored on the same R512 stream.
- Before any paired subtraction, each worker proves that every pilot, D256,
  and R512 payload has rooms exactly `0..23`, exact configured matrix shapes,
  one finite stack value per room, and that every full policy path has the same
  canonical `room_idx` order. The controller requires both alignment receipts.
- Every primary and wide policy arm runs in its own fresh subprocess. The
  controller uses a sealed OS-temp JSON job and accepts only a versioned,
  hash-sealed JSON result. There is no worker pool and no retry. A timeout,
  native exit, missing result, source/runtime mismatch, or failed final arm
  invalidates the launch.
- Before primary starts, the controller atomically writes `in_progress.json`
  with the state and both sealed job hashes; it advances that journal before
  wide starts. The marker is removed only after both arms validate and the
  matching complete CSV row, source receipt, and summary are re-read from the
  committed checkpoint. Any stale marker blocks both resume and retry, so a
  native controller/worker crash cannot be laundered as a clean interruption.
- Workers start with `PYTHONHASHSEED=1` and OpenBLAS, OMP, MKL, NumExpr,
  Accelerate, and BLIS thread counts fixed at one. The source receipt records
  Python/package identity and SHA-256 hashes for the high-risk Python, NumPy,
  SciPy, pandas, CVXOPT/GLPK, and vendor-BLAS native binaries.
- Both arms materialize R512 so a primary-only selected action remains
  scoreable when the screens are nonnested. The SLA subtracts reference-bank
  generation, reference scoring, and reference-column allocation from the
  timed primary policy section.
- Rounds 8 and 15 come only from room zero of the primary-24 D256 opening
  action. Neither R512 nor the 32-candidate arm may select a downstream state.
- Legacy is excluded entirely from the confirmation process. Its timing does
  not affect a promotion gate, and exclusion avoids executing GLPK solves in
  otherwise Sequential-only evidence. The helper now lazy-loads CVXOPT/GLPK
  only if a Legacy matrix or solve path is called.
- Aggregate uncertainty clusters the three depths by `(base seed, slot)` and
  resamples the nine trajectories independently within each fixed slot.

The nine confirmation seeds must not be used for a smoke, partial run, tuning,
or gate changes. Mechanical smokes use non-confirmation seeds and cannot pass
the exact-design gate. Final-seed evidence is accepted only in this study's
canonical `results/` directory, so an interrupted/stale journal cannot be
bypassed by relaunching the same seeds into a different output directory.

Prior launches and the old continuous-process prelaunch stress were
invalidated after Python-level impossible errors and Windows native access
violations. No output from seeds 3017 or 4017 is promotion evidence. The old
`artifacts/local/prelaunch_stress` directory is quarantined and is never read
by this harness. The complete abort history and reset rationale are in
`ABORTED.md`. The final confirmation is re-frozen on the nine untouched seeds
above. Before any final seed is touched, the final harness and helper hashes
must pass the isolated-arm stress at the full 24-room configuration on
non-confirmation seeds.

The final application runtime is the dedicated Python 3.12 environment at
`.venv_snake_312`. Python 3.12 continuous-process evidence remains failed
closed; the final launch cannot proceed unless the new isolated-arm prelaunch
stress completes cleanly under that exact runtime and native-binary
fingerprint.

## Frozen gates

1. Exact design, seed-manifest, source fingerprints, and unchanged sources.
2. All 81 states complete; physical-state, identity, pool, room, bank, path,
   canonical row-alignment, and overlapping-tensor contracts pass.
3. D128 is the exact production prefix of D256; C/E/D256/R512 are disjoint.
4. D256 cross-fitted R512 gap is at most 10 points in every state.
5. D256 mean and maximum cross-fitted gaps are no worse than D128.
6. Mean D256-minus-D128 R512 value is positive and the stratified
   trajectory-bootstrap 95% lower bound is at least -0.25%.
7. The positive R512 advantage of the D256-selected wide action over the
   primary action is at most 10 points in every state.
8. Primary-24 internal policy runtime p90, excluding R512 work and
   prehydration, is reported as a diagnostic and is not a release gate.
9. Primary-24 parent-observed end-to-end worker runtime p90, including process
   startup, native/source attestation, R512 work, and JSON transport, is
   strictly below 30 seconds.

Passing confirms D256 for the Preview workflow; it does not remove the Preview
label or replace the later historical forced-pick/opponent-sensitivity stage.

## Commands

Non-confirmation mechanical smoke:

```powershell
.venv_snake_312\Scripts\python.exe `
  research\studies\2026-08-02_sequential_v2_fresh_confirmation\run_study.py `
  --base-seeds 1217 --slots 6 --completed-picks 0 --rooms 2 `
  --output-dir research\studies\2026-08-02_sequential_v2_fresh_confirmation\artifacts\local\smoke
```

Full-size isolated-arm prelaunch stress (six states and twelve fresh policy
workers, under the final source/runtime/native hashes, before confirmation
seeds are touched):

```powershell
.venv_snake_312\Scripts\python.exe `
  research\studies\2026-08-02_sequential_v2_fresh_confirmation\run_study.py `
  --base-seeds 12017,13017 --slots 1,6,12 --completed-picks 0 `
  --rooms 24 --fail-fast `
  --output-dir research\studies\2026-08-02_sequential_v2_fresh_confirmation\artifacts\local\prelaunch_arm_isolation_v2
```

The confirmation refuses to enter its state loop unless that stress contains
exactly six clean opening states, its contract gates pass, and its full source
and runtime/native fingerprints match the confirmation process. Every state
must attest clean primary and wide worker exits, exact shared banks, and exact
full path plus room/value/stack tensor hashes for all overlapping candidates.
The stress controller must be non-resumed. A checkpoint with any failed state
cannot be resumed as evidence.

Frozen confirmation:

```powershell
.venv_snake_312\Scripts\python.exe `
  research\studies\2026-08-02_sequential_v2_fresh_confirmation\run_study.py `
  --fail-fast
```

A successful checkpoint is written only after both fresh arms return and pass
all envelope, bank, path, tensor, and selected-trajectory checks. A failed arm
writes an invalidating error receipt with its traceback/exit code and then the
exact launch terminates; it is never retried. An interrupted exact run may
resume only with `--resume --fail-fast` and an unchanged source/design/runtime
guard. A checkpoint containing any failed state or a stale `in_progress.json`
cannot be resumed as evidence.
