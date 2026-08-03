import sys
from pathlib import Path

APP_DIR = Path(__file__).parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from zSim_Helper import SEQUENTIAL_DRAFT_ROOMS
from snake_draft_app import (
    resolve_legacy_simulation_default,
    resolve_sequential_room_default,
)


def test_fresh_session_opens_at_validated_room_constant():
    assert SEQUENTIAL_DRAFT_ROOMS == 24
    assert (
        resolve_sequential_room_default(None, "best_ball_policy", 50)
        == SEQUENTIAL_DRAFT_ROOMS
    )


def test_loaded_policy_settings_keep_saved_room_count():
    loaded = {"NumIters": 32}
    assert resolve_sequential_room_default(loaded, "best_ball_policy", 32) == 32


def test_loaded_nonpolicy_settings_do_not_leak_sim_count_into_rooms():
    loaded = {"NumIters": 200}
    assert (
        resolve_sequential_room_default(loaded, "best_ball_ilp", 200)
        == SEQUENTIAL_DRAFT_ROOMS
    )


def test_legacy_defaults_to_50_and_only_keeps_legacy_saved_counts():
    assert resolve_legacy_simulation_default(None, "best_ball_policy", 24) == 50
    assert (
        resolve_legacy_simulation_default(
            {"NumIters": 24},
            "best_ball_policy",
            24,
        )
        == 50
    )
    assert (
        resolve_legacy_simulation_default(
            {"NumIters": 80},
            "best_ball_ilp",
            80,
        )
        == 80
    )
