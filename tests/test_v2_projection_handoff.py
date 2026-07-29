import numpy as np
import pandas as pd

from app.zSim_Helper import (
    BASE_PRED_COL,
    PREDICTION_HORIZON_COL,
    FootballSimulation,
)


CURRENT_RESID_COLS = [
    "pred_resid_5",
    "pred_resid_10",
    "pred_resid_25",
    "pred_resid_75",
    "pred_resid_90",
    "pred_resid_95",
]
NEXT_RESID_COLS = [f"{column}_ny" for column in CURRENT_RESID_COLS]


def _v2_simulation():
    sim = FootballSimulation.__new__(FootballSimulation)
    sim.uses_v2_joint_template = True
    sim.player_data = pd.DataFrame(
        {
            "player": ["Absent", "Appears"],
            "pos": ["RB", "WR"],
            "team": ["A", "B"],
            "pred_fp_per_game": [10.0, 12.0],
            "pred_fp_per_game_ny": [11.0, 13.0],
            "pred_appear_ny": [0.0, 1.0],
            **{column: [0.0, 0.0] for column in CURRENT_RESID_COLS},
            **{column: [0.0, 0.0] for column in NEXT_RESID_COLS},
        }
    )
    return sim


def test_v2_current_draws_repeat_point_center():
    sim = _v2_simulation()

    draws = sim.trunc_normal_dist("pred_fp_per_game", num_options=20)

    assert np.array_equal(
        draws.to_numpy(),
        np.repeat([[10.0], [12.0]], 20, axis=1),
    )


def test_v2_next_draws_apply_appearance_after_conditional_draw():
    sim = _v2_simulation()

    draws = sim.trunc_normal_dist("pred_fp_per_game_ny", num_options=20)

    assert np.array_equal(draws.iloc[0].to_numpy(), np.zeros(20))
    assert np.array_equal(draws.iloc[1].to_numpy(), np.full(20, 13.0))


def test_v2_template_path_uses_raw_centered_residual_only_for_current():
    sim = _v2_simulation()
    sim.weekly_template_profiles = {"Appears": np.array([[1.0, 0.5]])}
    sim.weekly_template_week_cols = ["week_1", "week_2"]
    sim.weekly_template_cum_probs = {"Appears": np.array([1.0])}
    sim.weekly_template_centered_active_ppg_resids = {
        "Appears": np.array([2.0])
    }
    sim.weekly_template_active_ppg_resid_sds = {"Appears": 2.0}

    current = pd.DataFrame(
        {
            "player": ["Appears"],
            "pos": ["WR"],
            "team": ["B"],
            0: [12.0],
            1: [12.0],
            BASE_PRED_COL: [12.0],
            PREDICTION_HORIZON_COL: ["pred_fp_per_game"],
        }
    )
    next_absent = current.copy()
    next_absent[[0, 1]] = 0.0
    next_absent[BASE_PRED_COL] = 13.0
    next_absent[PREDICTION_HORIZON_COL] = "pred_fp_per_game_ny"

    current_scores = sim.sample_template_weekly_scores(current, num_weeks=2)
    absent_scores = sim.sample_template_weekly_scores(
        next_absent,
        num_weeks=2,
    )

    assert np.allclose(current_scores, [[14.0, 7.0]])
    assert np.array_equal(absent_scores, [[0.0, 0.0]])
