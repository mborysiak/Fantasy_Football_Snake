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
    }
    later_row = summary["turns"][1]["rows"][0]
    assert later_row["player"] == "Later"
    assert later_row["selected_rooms"] == 2
    assert later_row["available_rooms"] == 2
    assert later_row["legal_rooms"] == 2


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
                    }
                ],
            },
        ],
    }


def test_future_display_uses_top_current_branch_and_conditional_top_three():
    results = pd.DataFrame({
        "player": ["Current A", "Current B"],
        "DecisionEV": [2000.0, 1999.0],
        "RecommendationRank": [1, 2],
    })
    results.attrs["sequential_future_picks"] = {
        "recommended_min_availability_pct": 10.0,
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
    assert list(display["future_rounds"]) == [1, 2]
    next_round = display["future_rounds"][1]["data"]
    assert next_round.iloc[0]["Player"] == "Next A1"
    assert next_round.iloc[0]["Pick Rate"] == pytest.approx(60.0)
    assert next_round.iloc[0]["Available"] == pytest.approx(80.0)
    assert next_round.iloc[0]["Pick If Available"] == pytest.approx(75.0)
    conditional = display["conditional_next"]
    assert conditional.groupby("Current Pick").size().to_dict() == {
        "Current A": 2,
        "Current B": 2,
    }
    assert "Filtered" not in conditional["Next Pick"].tolist()
