"""Compare legacy and full-scaled recommendations on identical root states."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.zSim_Helper import FootballSimulation  # noqa: E402


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
BLENDS = (0.30, 1.00)
SLOTS = (1, 6, 12)
SEEDS = (17, 1017, 2017)


def make_sim(conn, pick_slot, blend):
    return FootballSimulation(
        conn=conn,
        set_year=2026,
        pos_require_start=POS_REQUIRE,
        num_teams=12,
        num_rounds=20,
        my_pick_position=pick_slot,
        pred_vers='final_ensemble',
        league='dk',
        position_ranges=POSITION_RANGES,
        template_resid_blend=blend,
    )


def initial_opponent_picks(sim, seed):
    picks_before_user = max(int(sim.my_picks[0]) - 1, 0)
    if picks_before_user == 0:
        return []
    with sim.temp_seed(seed):
        sim.get_predictions('pred_fp_per_game', num_options=1000)
        adp_samples = sim.get_adp_samples(num_options=1000)
    adp_matrix = adp_samples.iloc[:, 2:].to_numpy(dtype=np.float32)
    draft_orders, _ = sim.build_sequential_draft_orders(
        adp_matrix,
        1,
        seed=seed + 303,
    )
    return adp_samples.player.to_numpy()[
        draft_orders[0, :picks_before_user]
    ].tolist()


def run_board(sim, opponents, seed):
    return sim.run_sim_best_ball_policy(
        [],
        opponents,
        num_iters=16,
        construction_samples=16,
        evaluation_samples=64,
        decision_samples=128,
        decision_candidate_count=24,
        audit_samples=0,
        candidate_pool_size=24,
        seed=seed,
        evaluation_seed=seed + 202,
        decision_seed=seed + 404,
        audit_seed=seed + 505,
    )


def main():
    results_dir = STUDY_DIR / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(REPO_ROOT / 'app' / 'Simulation.sqlite3')
    boards = {}
    board_records = []
    try:
        for seed in SEEDS:
            for slot in SLOTS:
                fixture_opponents = None
                for blend in BLENDS:
                    sim = make_sim(conn, slot, blend)
                    opponents = initial_opponent_picks(sim, seed)
                    if fixture_opponents is None:
                        fixture_opponents = opponents
                    assert opponents == fixture_opponents
                    board = run_board(sim, opponents, seed)
                    assert (board.PolicyCompletedRooms == 16).all()
                    boards[(seed, slot, blend)] = board.copy()
                    keep_columns = [
                        column for column in [
                            'player',
                            'pos',
                            'RecommendationRank',
                            'DecisionCandidate',
                            'DecisionEV',
                            'DecisionSE',
                            'PolicyCompletedRooms',
                        ]
                        if column in board.columns
                    ]
                    output = board[keep_columns].copy()
                    output.insert(0, 'template_resid_blend', blend)
                    output.insert(0, 'pick_slot', slot)
                    output.insert(0, 'seed', seed)
                    board_records.append(output)

        comparisons = []
        for seed in SEEDS:
            for slot in SLOTS:
                legacy = boards[(seed, slot, 0.30)].set_index('player')
                full = boards[(seed, slot, 1.00)].set_index('player')
                shared = legacy.index.intersection(full.index)
                union = legacy.index.union(full.index)
                legacy_top = str(legacy.RecommendationRank.idxmin())
                full_top = str(full.RecommendationRank.idxmin())
                legacy_top_five = set(
                    legacy.nsmallest(5, 'RecommendationRank').index
                )
                full_top_five = set(
                    full.nsmallest(5, 'RecommendationRank').index
                )
                comparisons.append({
                    'seed': seed,
                    'pick_slot': slot,
                    'legacy_top': legacy_top,
                    'full_scaled_top': full_top,
                    'top_exact_agreement': legacy_top == full_top,
                    'top_five_overlap': len(
                        legacy_top_five & full_top_five
                    ),
                    'candidate_jaccard': float(len(shared) / len(union)),
                    'shared_candidate_count': int(len(shared)),
                    'recommendation_rank_spearman': float(
                        legacy.loc[shared, 'RecommendationRank'].corr(
                            full.loc[shared, 'RecommendationRank'],
                            method='spearman',
                        )
                    ),
                    'decision_ev_spearman': float(
                        legacy.loc[shared, 'DecisionEV'].corr(
                            full.loc[shared, 'DecisionEV'],
                            method='spearman',
                        )
                    ),
                    'legacy_runtime_seconds': float(
                        boards[(seed, slot, 0.30)].attrs[
                            'timings'
                        ]['sections']['total']
                    ),
                    'full_scaled_runtime_seconds': float(
                        boards[(seed, slot, 1.00)].attrs[
                            'timings'
                        ]['sections']['total']
                    ),
                })

        board_frame = pd.concat(board_records, ignore_index=True)
        comparison_frame = pd.DataFrame(comparisons)
        board_frame.to_csv(
            results_dir / 'frozen_root_boards.csv',
            index=False,
        )
        comparison_frame.to_csv(
            results_dir / 'frozen_root_comparison.csv',
            index=False,
        )
        summary = {
            'states': int(len(comparison_frame)),
            'all_rooms_complete': True,
            'top_exact_agreement_rate': float(
                comparison_frame.top_exact_agreement.mean()
            ),
            'mean_top_five_overlap': float(
                comparison_frame.top_five_overlap.mean()
            ),
            'mean_candidate_jaccard': float(
                comparison_frame.candidate_jaccard.mean()
            ),
            'mean_recommendation_rank_spearman': float(
                comparison_frame.recommendation_rank_spearman.mean()
            ),
            'mean_decision_ev_spearman': float(
                comparison_frame.decision_ev_spearman.mean()
            ),
            'legacy_runtime_p50': float(
                comparison_frame.legacy_runtime_seconds.median()
            ),
            'full_scaled_runtime_p50': float(
                comparison_frame.full_scaled_runtime_seconds.median()
            ),
        }
        (results_dir / 'frozen_root_summary.json').write_text(
            json.dumps(summary, indent=2),
            encoding='utf-8',
        )
        print(json.dumps(summary, indent=2))
        print(comparison_frame.to_string(index=False))
    finally:
        conn.close()


if __name__ == '__main__':
    main()
