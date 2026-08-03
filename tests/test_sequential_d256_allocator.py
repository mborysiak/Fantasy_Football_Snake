import sys
from pathlib import Path

import numpy as np


APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from zSim_Helper import (  # noqa: E402
    FootballSimulation,
    SEQUENTIAL_DECISION_EXTENSION_SEED_OFFSET,
    SEQUENTIAL_DECISION_SAMPLES_BY_LEAGUE,
)


def test_d256_keeps_the_exact_d128_prefix_and_independent_extension():
    excluded = np.arange(80, dtype=np.int64)
    seed = 20260719 + 404

    d128 = FootballSimulation.select_additional_policy_ppg_columns(
        1000,
        excluded,
        128,
        seed,
        "Decision",
    )
    d256 = FootballSimulation.select_additional_policy_ppg_columns(
        1000,
        excluded,
        256,
        seed,
        "Decision",
    )

    assert np.array_equal(d128, d256[:128])
    assert len(np.unique(d256)) == 256
    assert not np.intersect1d(excluded, d256).size

    extension_pool = np.setdiff1d(
        np.setdiff1d(np.arange(1000, dtype=np.int64), excluded),
        d128,
        assume_unique=True,
    )
    expected_extension = np.random.default_rng(
        seed + SEQUENTIAL_DECISION_EXTENSION_SEED_OFFSET
    ).choice(extension_pool, size=128, replace=False)
    assert np.array_equal(d256[128:], expected_extension)


def test_release_decision_bank_sizes_are_league_specific():
    assert SEQUENTIAL_DECISION_SAMPLES_BY_LEAGUE == {"dk": 256, "nffc": 128}
