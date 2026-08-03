import sqlite3
import sys
from pathlib import Path

import pytest


APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from simulation_worker import (  # noqa: E402
    SimulationWorkerError,
    build_request,
    run_isolated_simulation,
)


class FakeSimulation:
    def __init__(self, db_path):
        self.db_path = db_path

    def get_db_path(self):
        return str(self.db_path)

    def get_sim_config(self):
        return {"set_year": 2026, "league": "dk"}


def test_worker_request_forces_one_inner_worker(tmp_path):
    db_path = tmp_path / "simulation.sqlite3"
    sqlite3.connect(db_path).close()
    request = build_request(
        FakeSimulation(db_path),
        ["my-player"],
        ["other-player"],
        num_iters=24,
        scoring_mode="best_ball_policy",
    )

    assert request["run"]["parallel_workers"] == 1
    assert request["selection"] == {
        "to_add": ["my-player"],
        "to_drop": ["other-player"],
    }
    assert request["request_sha256"]


def test_native_exit_is_not_retried_and_keeps_input_state(tmp_path):
    db_path = tmp_path / "simulation.sqlite3"
    sqlite3.connect(db_path).close()
    marker = tmp_path / "launches.txt"
    worker = tmp_path / "crash_worker.py"
    worker.write_text(
        "from pathlib import Path\n"
        f"p = Path({str(marker)!r})\n"
        "p.write_text(p.read_text() + 'x' if p.exists() else 'x')\n"
        "import os\n"
        "os._exit(7)\n",
        encoding="utf-8",
    )
    my_team = ["my-player"]
    other_team = ["other-player"]

    with pytest.raises(SimulationWorkerError, match="was not retried"):
        run_isolated_simulation(
            FakeSimulation(db_path),
            my_team,
            other_team,
            num_iters=24,
            scoring_mode="best_ball_policy",
            worker_path=worker,
            timeout_seconds=10,
        )

    assert marker.read_text(encoding="utf-8") == "x"
    assert my_team == ["my-player"]
    assert other_team == ["other-player"]
