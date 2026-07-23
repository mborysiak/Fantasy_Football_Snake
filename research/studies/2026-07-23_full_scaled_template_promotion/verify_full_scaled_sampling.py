"""Focused runtime checks for the full-scaled weekly-template promotion."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.zSim_Helper import (  # noqa: E402
    LEGACY_TEMPLATE_RESID_BLEND,
    TEMPLATE_RESID_BLEND,
    TEMPLATE_RESID_METHOD_VERSION,
    FootballSimulation,
)


POSITION_RANGES = {
    'QB': (2, 3),
    'RB': (5, 7),
    'WR': (7, 9),
    'TE': (2, 3),
}
POS_REQUIRE = {
    position: maximum
    for position, (_, maximum) in POSITION_RANGES.items()
}


def make_sim(conn, blend=TEMPLATE_RESID_BLEND):
    return FootballSimulation(
        conn=conn,
        set_year=2026,
        pos_require_start=POS_REQUIRE,
        num_teams=12,
        num_rounds=20,
        my_pick_position=6,
        pred_vers='final_ensemble',
        league='dk',
        position_ranges=POSITION_RANGES,
        template_resid_blend=blend,
    )


def main():
    db_path = REPO_ROOT / 'app' / 'Simulation.sqlite3'
    conn = sqlite3.connect(db_path)
    try:
        full = make_sim(conn)
        legacy = make_sim(conn, LEGACY_TEMPLATE_RESID_BLEND)

        assert np.isclose(full.template_resid_blend, 1.0)
        assert np.isclose(full.model_resid_blend, 0.0)
        assert full.template_resid_method_version == TEMPLATE_RESID_METHOD_VERSION
        assert np.isclose(legacy.template_resid_blend, 0.30)
        assert legacy.model_resid_blend > 0

        with full.temp_seed(20260723):
            predictions = full.get_predictions(
                'pred_fp_per_game',
                num_options=64,
            )
        predictions = predictions.head(48).reset_index(drop=True)
        low_columns = np.zeros(128, dtype=np.int64)
        high_columns = np.full(128, 63, dtype=np.int64)

        full_low = full.sample_template_weekly_score_bank(
            predictions,
            num_scenarios=128,
            seed=20260724,
            ppg_column_indices=low_columns,
        )
        full_high = full.sample_template_weekly_score_bank(
            predictions,
            num_scenarios=128,
            seed=20260724,
            ppg_column_indices=high_columns,
        )
        assert np.array_equal(full_low, full_high), (
            "Full-scaled output changed with the independent model-residual "
            "column even though its production weight should be zero."
        )

        legacy.set_weekly_template_profile_cache(
            full.weekly_template_week_cols,
            full.weekly_template_profiles,
            full.weekly_template_cum_probs,
            full.weekly_template_active_ppg_resids,
            full.weekly_template_centered_active_ppg_resids,
            full.weekly_template_active_ppg_resid_sds,
        )
        legacy_low = legacy.sample_template_weekly_score_bank(
            predictions,
            num_scenarios=128,
            seed=20260724,
            ppg_column_indices=low_columns,
        )
        legacy_high = legacy.sample_template_weekly_score_bank(
            predictions,
            num_scenarios=128,
            seed=20260724,
            ppg_column_indices=high_columns,
        )
        assert not np.array_equal(legacy_low, legacy_high), (
            "Legacy 0.30 fallback did not respond to the model-residual draw."
        )

        round_trip = FootballSimulation(
            conn=conn,
            **full.get_sim_config(),
        )
        assert np.isclose(round_trip.template_resid_blend, 1.0)
        assert np.isclose(round_trip.model_resid_blend, 0.0)

        try:
            make_sim(conn, 1.01)
        except ValueError:
            invalid_blend_rejected = True
        else:
            invalid_blend_rejected = False
        assert invalid_blend_rejected

        results = {
            'default_template_resid_blend': full.template_resid_blend,
            'default_model_resid_blend': full.model_resid_blend,
            'method_version': full.template_resid_method_version,
            'legacy_template_resid_blend': legacy.template_resid_blend,
            'legacy_model_resid_blend': legacy.model_resid_blend,
            'full_ignores_independent_model_draw': True,
            'legacy_fallback_uses_independent_model_draw': True,
            'worker_config_round_trip': True,
            'invalid_blend_rejected': True,
            'players_checked': int(len(predictions)),
            'scenarios_checked': int(len(low_columns)),
        }
        results_dir = STUDY_DIR / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / 'focused_sampling_checks.json').write_text(
            json.dumps(results, indent=2),
            encoding='utf-8',
        )
        print(json.dumps(results, indent=2))
    finally:
        conn.close()


if __name__ == '__main__':
    main()
