"""One-shot subprocess boundary for Snake simulation runs."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import pandas as pd


REQUEST_SCHEMA = "snake-simulation-request-v1"
RESULT_SCHEMA = "snake-simulation-result-v1"
DEFAULT_TIMEOUT_SECONDS = 60.0
RESULT_ATTR_KEYS = ("sequential_future_picks",)
WORKER_ENVIRONMENT = {
    "PYTHONHASHSEED": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}


def _package_versions():
    versions = {}
    for package in ("numpy", "pandas", "scipy", "cvxopt", "streamlit"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


class SimulationWorkerError(RuntimeError):
    """A one-shot worker failed; callers must not retry or switch methods."""


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _stable_hash(payload) -> str:
    encoded = json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seal(payload, hash_field):
    sealed = _json_safe(dict(payload))
    sealed.pop(hash_field, None)
    sealed[hash_field] = _stable_hash(sealed)
    return sealed


def _validate_seal(payload, hash_field):
    claimed = str(payload.get(hash_field, ""))
    unsigned = dict(payload)
    unsigned.pop(hash_field, None)
    if not claimed or claimed != _stable_hash(unsigned):
        raise SimulationWorkerError(f"Invalid {hash_field} seal.")
    return payload


def _write_json_atomic(path: Path, payload) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _encode_frame(frame: pd.DataFrame) -> str:
    return frame.to_json(orient="split", double_precision=15)


def _decode_frame(payload: str) -> pd.DataFrame:
    return pd.read_json(io.StringIO(payload), orient="split")


def _result_attrs(frame: pd.DataFrame) -> dict:
    """Return the small, JSON-safe display attrs allowed across the worker."""
    return {
        key: _json_safe(frame.attrs[key])
        for key in RESULT_ATTR_KEYS
        if key in frame.attrs
    }


def build_request(
    sim,
    to_add,
    to_drop,
    *,
    num_iters,
    scoring_mode,
    current_pick_ev=False,
    ev_shortlist_size=8,
    weekly_score_mode="residual",
):
    db_path = sim.get_db_path()
    if not db_path:
        raise SimulationWorkerError(
            "Simulation isolation requires a file-backed SQLite database."
        )
    request = {
        "schema_version": REQUEST_SCHEMA,
        "database_path": str(Path(db_path).resolve()),
        "simulation_config": sim.get_sim_config(),
        "selection": {
            "to_add": list(map(str, to_add)),
            "to_drop": list(map(str, to_drop)),
        },
        "run": {
            "num_iters": int(num_iters),
            "scoring_mode": str(scoring_mode),
            "current_pick_ev": bool(current_pick_ev),
            "ev_shortlist_size": int(ev_shortlist_size),
            "weekly_score_mode": str(weekly_score_mode),
            "parallel_workers": 1,
        },
    }
    if request["run"]["scoring_mode"] not in {
        "best_ball_policy",
        "best_ball_ilp",
    }:
        raise SimulationWorkerError(
            "Only Sequential and Legacy runs use the isolated worker."
        )
    return _seal(request, "request_sha256")


def _worker_execute(request):
    from zSim_Helper import FootballSimulation

    _validate_seal(request, "request_sha256")
    if request.get("schema_version") != REQUEST_SCHEMA:
        raise SimulationWorkerError("Unsupported worker request schema.")
    db_path = Path(request["database_path"]).resolve()
    if not db_path.is_file():
        raise SimulationWorkerError(f"Simulation database does not exist: {db_path}")
    connection = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
    try:
        sim = FootballSimulation(
            conn=connection,
            **request["simulation_config"],
        )
        run = request["run"]
        selection = request["selection"]
        started = time.perf_counter()
        frame = sim.run_sim(
            to_add=selection["to_add"],
            to_drop=selection["to_drop"],
            num_iters=run["num_iters"],
            scoring_mode=run["scoring_mode"],
            current_pick_ev=run["current_pick_ev"],
            ev_shortlist_size=run["ev_shortlist_size"],
            weekly_score_mode=run["weekly_score_mode"],
            parallel_workers=1,
        )
        worker_seconds = time.perf_counter() - started
    finally:
        connection.close()
    timings = dict(getattr(frame, "attrs", {}).get("timings", {}))
    timings.update({
        "execution_isolation": "fresh_subprocess_per_run",
        "worker_pid": int(os.getpid()),
        "worker_python_executable": str(Path(sys.executable).resolve()),
        "worker_python_version": sys.version,
        "worker_package_versions": _package_versions(),
        "worker_cvxopt_loaded": "cvxopt" in sys.modules,
        "worker_seconds": float(worker_seconds),
        "parallel_workers": 1,
    })
    return {
        "schema_version": RESULT_SCHEMA,
        "status": "ok",
        "request_sha256": request["request_sha256"],
        "frame_json": _encode_frame(frame),
        "timings": timings,
        "frame_attrs": _result_attrs(frame),
    }


def worker_main(request_path: Path, result_path: Path) -> int:
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        result = _worker_execute(request)
        exit_code = 0
    except BaseException as exc:  # Preserve a durable Python failure envelope.
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        exit_code = 1
    _write_json_atomic(result_path, _seal(result, "result_sha256"))
    return exit_code


def run_isolated_simulation(
    sim,
    to_add,
    to_drop,
    *,
    num_iters,
    scoring_mode,
    current_pick_ev=False,
    ev_shortlist_size=8,
    weekly_score_mode="residual",
    timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
    worker_path=None,
):
    request = build_request(
        sim,
        to_add,
        to_drop,
        num_iters=num_iters,
        scoring_mode=scoring_mode,
        current_pick_ev=current_pick_ev,
        ev_shortlist_size=ev_shortlist_size,
        weekly_score_mode=weekly_score_mode,
    )
    worker_path = Path(worker_path or __file__).resolve()
    environment = os.environ.copy()
    environment.update(WORKER_ENVIRONMENT)
    parent_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="snake_simulation_worker_") as tmp:
        tmp_path = Path(tmp)
        request_path = tmp_path / "request.json"
        result_path = tmp_path / "result.json"
        _write_json_atomic(request_path, request)
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(worker_path),
                    "--request",
                    str(request_path),
                    "--result",
                    str(result_path),
                ],
                cwd=worker_path.parent,
                env=environment,
                capture_output=True,
                text=True,
                timeout=float(timeout_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SimulationWorkerError(
                f"Simulation worker exceeded {float(timeout_seconds):.0f} seconds. "
                "The run was not retried and your draft selections are unchanged."
            ) from exc
        parent_seconds = time.perf_counter() - parent_started
        if not result_path.is_file():
            raise SimulationWorkerError(
                f"Simulation worker exited {completed.returncode} without a result. "
                "The run was not retried and your draft selections are unchanged."
            )
        result = _validate_seal(
            json.loads(result_path.read_text(encoding="utf-8")),
            "result_sha256",
        )
    if result.get("schema_version") != RESULT_SCHEMA:
        raise SimulationWorkerError("Unsupported simulation-worker result schema.")
    if result.get("status") != "ok" or completed.returncode != 0:
        message = result.get("error_message") or completed.stderr.strip()
        raise SimulationWorkerError(
            f"Simulation worker failed without retry: {message}. "
            "Your draft selections are unchanged."
        )
    if result.get("request_sha256") != request["request_sha256"]:
        raise SimulationWorkerError("Simulation worker returned a mismatched request.")
    frame = _decode_frame(result["frame_json"])
    timings = dict(result.get("timings", {}))
    timings["parent_end_to_end_seconds"] = float(parent_seconds)
    timings["worker_return_code"] = int(completed.returncode)
    frame.attrs["timings"] = timings
    frame_attrs = result.get("frame_attrs", {})
    if not isinstance(frame_attrs, dict):
        raise SimulationWorkerError("Simulation worker returned invalid frame attrs.")
    for key in RESULT_ATTR_KEYS:
        if key in frame_attrs:
            frame.attrs[key] = frame_attrs[key]
    return frame


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    raise SystemExit(worker_main(arguments.request, arguments.result))
