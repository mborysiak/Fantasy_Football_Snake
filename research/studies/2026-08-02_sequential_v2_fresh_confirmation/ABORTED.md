# Aborted Launches

None of the outputs described here are confirmation evidence.

- A launch containing base seed 3017 stopped after an impossible transient
  `int(tuple)` error in the harness. The full 3017 cluster was invalidated.
  A later full-size rerun of the failing state completed cleanly, so the error
  was not reproducible.
- A clean replacement launch beginning with base seed 4017 stopped at
  slot 12, round 1 after an impossible `method_descriptor - float` exception
  inside a float-only utility calculation. The same state passed in a clean
  process, both alone and after six Legacy/GLPK calls. The cause was therefore
  not reproducible, but the entire process and all 4017 outputs were treated as
  unsafe and invalidated.

Legacy is timing-only and invokes the native CVXOPT/GLPK solver without
affecting the promotion gates. Its solves were removed entirely from the final
confirmation process. The helper now lazy-loads CVXOPT/GLPK only on Legacy
matrix/solve paths.

- The old final-hash Python 3.9 prelaunch stress completed three opening states
  for seed 3017 and then terminated natively before the next state checkpoint.
  Windows Error Reporting recorded `python3.9.exe`, exception `0xc0000005`,
  fault offset zero, and `StackHash_ac46`. NumPy, SciPy, and CVXOPT OpenBLAS
  binaries plus `cvxopt.glpk` were mapped in that process. There was no
  resource-exhaustion, WHEA, or Python traceback establishing a model error.
- A continuous-process Python 3.12 stress also terminated natively after one
  clean state, in `python312.dll`. A single-state 3.12 control had matched the
  Python 3.9 action and non-timing metrics, so changing Python alone did not
  remove the process-lifetime failure mode.

All artifacts under `artifacts/local/prelaunch_stress` are quarantined. They
must not be copied, resumed, or cited as stress or confirmation evidence. The
first isolated-arm replacement was also invalidated before release because its
seeds (3017 and 4017) had already appeared in failed process-lifetime work and
its source/database fingerprint no longer matched the final application.

The final replacement attestation uses untouched non-confirmation seeds 12017
and 13017 under `artifacts/local/prelaunch_arm_isolation_v2`. It is accepted only if all
six non-confirmation states complete as twelve fresh, one-shot primary/wide
workers under the exact final source, helper, runtime, environment, and native
binary hashes. A worker failure has no retry path. An atomic in-progress
journal survives any controller/worker native termination and permanently
blocks resume of that output unless both arm results had already validated and
the matching complete checkpoint was durably committed and re-read.

The final design still uses the untouched seeds 20017 through 28017. None may
run until the replacement isolated-arm stress completes cleanly in the
dedicated `.venv_snake_312` Python 3.12 application runtime.
