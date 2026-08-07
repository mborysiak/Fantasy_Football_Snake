import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from snake_draft_app import get_sequential_future_recommendations  # noqa: E402
from zSim_Helper import FootballSimulation  # noqa: E402


def test_app_import_tolerates_stale_helper_without_future_floor_constant():
    probe = (
        "import sys; "
        f"sys.path.insert(0, {str(APP_DIR)!r}); "
        "import zSim_Helper; "
        "del zSim_Helper.SEQUENTIAL_FUTURE_MIN_AVAILABILITY_PCT; "
        "import snake_draft_app; "
        "assert snake_draft_app.SEQUENTIAL_FUTURE_MIN_AVAILABILITY_PCT == 10.0"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_future_summary_separates_room_availability_from_roster_legality():
    summary = FootballSimulation.summarize_sequential_future_picks(
        player_names=np.array(["Current", "Next A", "Next B", "Later"]),
        player_positions=np.array(["WR", "RB", "WR", "QB"]),
        adjusted_picks=[6, 19, 30],
        current_round_num=1,
        room_details=[
            {
                "path": np.array([0, 1, 3]),
                "room_available_by_pick": [
                    np.array([1, 2, 3]),
                    np.array([2, 3]),
                ],
                "available_by_pick": [
                    np.array([1, 2]),
                    np.array([3]),
                ],
            },
            {
                "path": np.array([0, 2, 3]),
                "room_available_by_pick": [
                    np.array([1, 2, 3]),
                    np.array([1, 3]),
                ],
                "available_by_pick": [
                    np.array([1, 2]),
                    np.array([3]),
                ],
            },
        ],
    )

    assert summary["completed_rooms"] == 2
    assert [turn["pick"] for turn in summary["turns"]] == [19, 30]
    next_rows = {row["player"]: row for row in summary["turns"][0]["rows"]}
    assert next_rows["Next A"] == {
        "player": "Next A",
        "pos": "RB",
        "selected_rooms": 1,
        "available_rooms": 2,
        "legal_rooms": 2,
        "completed_rooms": 2,
        "avg_draft_now_edge": None,
        "expected_edge": None,
    }
    later_row = summary["turns"][1]["rows"][0]
    assert later_row["player"] == "Later"
    assert later_row["selected_rooms"] == 2
    assert later_row["available_rooms"] == 2
    assert later_row["legal_rooms"] == 2


def test_unconditional_future_summary_equal_weights_current_choices():
    summary = FootballSimulation.summarize_sequential_unconditional_future_picks(
        player_names=np.array(["Current A", "Current B", "Next A", "Next B"]),
        player_positions=np.array(["WR", "RB", "WR", "RB"]),
        adjusted_picks=[6, 19],
        current_round_num=1,
        branch_room_details={
            "Current A": [
                {
                    "path": np.array([0, 2]),
                    "room_available_by_pick": [np.array([2, 3])],
                    "available_by_pick": [np.array([2, 3])],
                    "decision_metrics_by_pick": [
                        {"selected_idx": 2, "draft_now_edge": 3.0}
                    ],
                },
                {
                    "path": np.array([0, 2]),
                    "room_available_by_pick": [np.array([2, 3])],
                    "available_by_pick": [np.array([2, 3])],
                    "decision_metrics_by_pick": [
                        {"selected_idx": 2, "draft_now_edge": 5.0}
                    ],
                },
            ],
            "Current B": [
                {
                    "path": np.array([1, 3]),
                    "room_available_by_pick": [np.array([2, 3])],
                    "available_by_pick": [np.array([2, 3])],
                    "decision_metrics_by_pick": [
                        {"selected_idx": 3, "draft_now_edge": 10.0}
                    ],
                }
            ],
        },
    )

    assert summary["weighting"] == "equal_current_choice"
    assert summary["current_choices"] == ["Current A", "Current B"]
    rows = {row["player"]: row for row in summary["turns"][0]["rows"]}
    assert rows["Next A"]["selection_rate"] == pytest.approx(0.5)
    assert rows["Next A"]["availability_rate"] == pytest.approx(1.0)
    assert rows["Next A"]["pick_if_available"] == pytest.approx(0.5)
    assert rows["Next A"]["avg_draft_now_edge"] == pytest.approx(4.0)
    assert rows["Next A"]["expected_edge"] == pytest.approx(2.0)
    assert rows["Next B"]["selection_rate"] == pytest.approx(0.5)
    assert rows["Next B"]["avg_draft_now_edge"] == pytest.approx(10.0)
    assert rows["Next B"]["expected_edge"] == pytest.approx(5.0)


def _candidate_summary(current_player, next_counts):
    return {
        "completed_rooms": 10,
        "turns": [
            {
                "future_offset": 1,
                "round": 2,
                "pick": 19,
                "completed_rooms": 10,
                "rows": [
                    {
                        "player": player,
                        "pos": pos,
                        "selected_rooms": selected,
                        "available_rooms": available,
                        "legal_rooms": legal,
                        "completed_rooms": 10,
                        "avg_draft_now_edge": 4.0,
                        "expected_edge": 0.4 * selected,
                    }
                    for player, pos, selected, available, legal in next_counts
                ],
            },
            {
                "future_offset": 2,
                "round": 3,
                "pick": 30,
                "completed_rooms": 10,
                "rows": [
                    {
                        "player": f"{current_player} later",
                        "pos": "QB",
                        "selected_rooms": 6,
                        "available_rooms": 8,
                        "legal_rooms": 7,
                        "completed_rooms": 10,
                        "avg_draft_now_edge": 2.0,
                        "expected_edge": 1.2,
                    }
                ],
            },
        ],
    }


def test_future_display_is_unconditional_then_conditional_by_current_choice():
    results = pd.DataFrame({
        "player": ["Current A", "Current B"],
        "DecisionEV": [2000.0, 1999.0],
        "RecommendationRank": [1, 2],
    })
    results.attrs["sequential_future_picks"] = {
        "recommended_min_availability_pct": 10.0,
        "unconditional": {
            "weighting": "equal_current_choice",
            "current_choices": ["Current A", "Current B"],
            "turns": [
                {
                    "future_offset": 1,
                    "round": 2,
                    "pick": 19,
                    "rows": [
                        {
                            "player": "Overall Next",
                            "pos": "WR",
                            "selection_rate": 0.7,
                            "availability_rate": 0.8,
                            "pick_if_available": 0.875,
                            "legality_rate": 0.75,
                            "avg_draft_now_edge": 5.0,
                            "expected_edge": 3.5,
                            "selected_rooms": 14,
                            "completed_rooms": 20,
                            "completed_branches": 2,
                        }
                    ],
                },
                {
                    "future_offset": 2,
                    "round": 3,
                    "pick": 30,
                    "rows": [
                        {
                            "player": "Overall Later",
                            "pos": "QB",
                            "selection_rate": 0.6,
                            "availability_rate": 0.75,
                            "pick_if_available": 0.8,
                            "legality_rate": 0.7,
                            "avg_draft_now_edge": 2.0,
                            "expected_edge": 1.2,
                            "selected_rooms": 12,
                            "completed_rooms": 20,
                            "completed_branches": 2,
                        }
                    ],
                },
            ],
        },
        "candidates": {
            "Current A": _candidate_summary(
                "Current A",
                [
                    ("Next A1", "WR", 6, 8, 8),
                    ("Next A2", "RB", 3, 5, 4),
                    ("Filtered", "TE", 1, 0, 0),
                ],
            ),
            "Current B": _candidate_summary(
                "Current B",
                [
                    ("Next B1", "RB", 7, 9, 9),
                    ("Next B2", "WR", 2, 4, 3),
                ],
            ),
        },
    }

    display = get_sequential_future_recommendations(
        results,
        current_candidate_limit=2,
        per_candidate_limit=3,
    )

    assert display["recommended_player"] == "Current A"
    assert display["unconditional_choices"] == ["Current A", "Current B"]
    assert list(display["future_rounds"]) == [1, 2]
    next_round = display["future_rounds"][1]["data"]
    assert next_round.iloc[0]["Player"] == "Overall Next"
    assert next_round.iloc[0]["Pick Rate"] == pytest.approx(70.0)
    assert next_round.iloc[0]["Available"] == pytest.approx(80.0)
    assert next_round.iloc[0]["Pick If Available"] == pytest.approx(87.5)
    assert next_round.iloc[0]["Draft-Now Edge"] == pytest.approx(5.0)
    assert next_round.iloc[0]["Avail-Adjusted Edge"] == pytest.approx(3.5)
    conditional = display["conditional_next"]
    assert conditional.groupby("Current Pick").size().to_dict() == {
        "Current A": 2,
        "Current B": 2,
    }
    assert "Filtered" not in conditional["Next Pick"].tolist()

    transport_payload = results.attrs.pop("sequential_future_picks")
    results["_sequential_future_picks_json"] = [
        json.dumps(transport_payload),
        None,
    ]
    transport_display = get_sequential_future_recommendations(
        results,
        current_candidate_limit=2,
        per_candidate_limit=3,
    )
    assert transport_display["recommended_player"] == "Current A"
    assert list(transport_display["future_rounds"]) == [1, 2]
