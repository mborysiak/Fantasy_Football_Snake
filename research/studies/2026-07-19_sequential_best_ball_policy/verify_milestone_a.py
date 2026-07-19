"""Correctness checks for the Milestone A sequential best-ball policy."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.zSim_Helper import FootballSimulation  # noqa: E402


FROZEN_LEGACY_SOURCE_HASHES = {
    'sample_template_weekly_scores': (
        'acddf80b795b8b90a133e23ff8d74f0836d98eded2f369036d72d67b76bc2ae7'
    ),
    'simulate_opponent_draft_availability': (
        '51f875b4dc9778c72b7fa8a9f9e6599949db9afa891d3b43664ccd4bc370c0eb'
    ),
    'run_best_ball_ilp_iteration_chunk': (
        'fed5ca53297fc044c7864b6aeb26205628107aab91a70bd23dc0997d05bc2a19'
    ),
    'run_sim_best_ball_ilp': (
        'ba7c7763b3aa8a266c59529009c344f32e9ede057d3e93d7568850e5fc7ab684'
    ),
}


def method_source_hashes(path: Path) -> dict[str, str]:
    source = path.read_text(encoding='utf-8').replace('\r\n', '\n')
    source_lines = source.splitlines()
    tree = ast.parse(source)
    hashes = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name in FROZEN_LEGACY_SOURCE_HASHES
        ):
            start_line = min(
                [node.lineno]
                + [decorator.lineno for decorator in node.decorator_list]
            ) - 1
            payload = (
                '\n'.join(source_lines[start_line:node.end_lineno]).strip()
                + '\n'
            )
            hashes[node.name] = hashlib.sha256(payload.encode()).hexdigest()
    return hashes


def make_sim_stub() -> FootballSimulation:
    sim = FootballSimulation.__new__(FootballSimulation)
    sim.position_ranges = {
        'QB': (1, 1),
        'RB': (1, 2),
        'WR': (1, 2),
        'TE': (1, 1),
    }
    sim.num_rounds = 5
    return sim


def verify_tensor_scoring() -> None:
    rng = np.random.default_rng(11)
    num_scenarios, num_players, num_weeks = 5, 18, 16
    positions = np.array(['QB'] * 3 + ['RB'] * 5 + ['WR'] * 6 + ['TE'] * 4)
    bank = rng.uniform(
        0,
        30,
        size=(num_scenarios, num_players, num_weeks),
    ).astype(np.float32)
    selected = np.array([0, 3, 4, 8, 9, 10, 14])
    candidates = np.array([1, 5, 11, 15])

    scenario_values, _ = FootballSimulation.marginal_best_ball_values_bank(
        bank,
        positions,
        selected,
        candidates,
    )
    bank_roster_scores = FootballSimulation.best_ball_roster_scores_bank(
        bank,
        positions,
        selected,
    )
    for scenario_idx in range(num_scenarios):
        expected_marginals = FootballSimulation.marginal_best_ball_values(
            bank[scenario_idx, selected],
            positions[selected],
            bank[scenario_idx, candidates],
            positions[candidates],
        ) * num_weeks
        np.testing.assert_allclose(
            scenario_values[scenario_idx],
            expected_marginals,
            rtol=1e-5,
            atol=1e-4,
        )
        expected_score = FootballSimulation.best_ball_total_score(
            selected,
            bank[scenario_idx],
            positions,
        )
        np.testing.assert_allclose(
            bank_roster_scores[scenario_idx],
            expected_score,
            rtol=1e-5,
            atol=1e-4,
        )


def verify_candidate_consistent_state() -> None:
    # A is our locked pick. Opponents must skip it and draft B/C, not A/B.
    remaining = np.ones(8, dtype=bool)
    remaining[0] = False
    pointer, drafted = FootballSimulation.advance_sequential_opponents(
        remaining,
        np.arange(8),
        0,
        2,
    )
    assert pointer == 3
    assert drafted.tolist() == [1, 2]
    assert not remaining[[0, 1, 2]].any()


def reference_legal_candidates(
    remaining,
    player_positions,
    selected_indices,
    picks_left,
    pos_ranges,
):
    positions = tuple(pos_ranges)
    current_counts = {
        pos: int(np.sum(player_positions[selected_indices] == pos))
        for pos in positions
    }
    future_slots = picks_left - 1
    legal = []
    for candidate_idx in np.flatnonzero(remaining):
        candidate_pos = player_positions[candidate_idx]
        if candidate_pos not in pos_ranges:
            continue
        counts = current_counts.copy()
        counts[candidate_pos] += 1
        if counts[candidate_pos] > pos_ranges[candidate_pos][1]:
            continue
        minimum_deficit = sum(
            max(pos_ranges[pos][0] - counts[pos], 0)
            for pos in positions
        )
        maximum_capacity = sum(
            max(pos_ranges[pos][1] - counts[pos], 0)
            for pos in positions
        )
        if minimum_deficit <= future_slots <= maximum_capacity:
            legal.append(candidate_idx)
    return np.asarray(legal, dtype=np.int64)


def verify_vectorized_legality() -> None:
    sim = FootballSimulation.__new__(FootballSimulation)
    pos_ranges = {
        'QB': (2, 3),
        'RB': (5, 7),
        'WR': (7, 9),
        'TE': (2, 3),
    }
    sim.position_ranges = pos_ranges
    rng = np.random.default_rng(83)
    player_positions = rng.choice(
        np.array(['QB', 'RB', 'WR', 'TE', 'K']),
        size=180,
        p=[0.12, 0.31, 0.36, 0.16, 0.05],
    )
    for _ in range(100):
        selected_indices = rng.choice(
            len(player_positions),
            size=int(rng.integers(0, 16)),
            replace=False,
        )
        remaining = rng.random(len(player_positions)) > 0.25
        remaining[selected_indices] = False
        picks_left = int(rng.integers(1, 21))
        expected = reference_legal_candidates(
            remaining,
            player_positions,
            selected_indices,
            picks_left,
            pos_ranges,
        )
        actual = sim.sequential_legal_candidate_indices(
            remaining,
            player_positions,
            selected_indices,
            picks_left,
            pos_ranges=pos_ranges,
        )
        np.testing.assert_array_equal(actual, expected)


def verify_disjoint_bank_columns() -> None:
    construction, evaluation = (
        FootballSimulation.select_disjoint_policy_ppg_columns(
            1000,
            16,
            64,
            construction_seed=118,
            evaluation_seed=219,
        )
    )
    assert len(construction) == len(np.unique(construction)) == 16
    assert len(evaluation) == len(np.unique(evaluation)) == 64
    assert np.intersect1d(construction, evaluation).size == 0

    rng = np.random.default_rng(91)
    adp_matrix = rng.integers(1, 241, size=(25, 100))
    adjusted_picks = [6, 19, 30, 43]
    survival_table = FootballSimulation.build_sequential_survival_table(
        adp_matrix,
        adjusted_picks,
    )
    for pick_idx in range(len(adjusted_picks) - 1):
        direct = FootballSimulation.sequential_survival_probabilities(
            np.arange(adp_matrix.shape[0]),
            adp_matrix,
            adjusted_picks[pick_idx],
            adjusted_picks[pick_idx + 1],
        )
        np.testing.assert_allclose(survival_table[pick_idx], direct)


def verify_rollout_boundaries() -> None:
    sim = make_sim_stub()
    positions = np.array([
        'QB', 'QB', 'RB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'TE',
    ])
    rng = np.random.default_rng(3)
    construction_bank = rng.uniform(0, 20, (4, len(positions), 3)).astype(np.float32)
    adp_matrix = np.tile(
        np.arange(1, len(positions) + 1)[:, None],
        (1, 10),
    ).astype(np.float32)
    draft_order = np.array([0, 2, 5, 1, 3, 6, 4, 7, 9, 8])
    base_remaining = np.ones(len(positions), dtype=bool)
    args = (
        [],
        8,
        [1, 3, 5, 7, 9],
        base_remaining,
        draft_order,
        construction_bank,
        positions,
        adp_matrix,
        sim.position_ranges,
    )
    roster_one, details_one, success_one = (
        sim.complete_sequential_best_ball_rollout(*args)
    )

    # Evaluation outcomes are deliberately not an argument to the policy.
    hidden_evaluation_bank = rng.uniform(0, 100, (8, len(positions), 3))
    assert hidden_evaluation_bank.shape[0] == 8
    roster_two, details_two, success_two = (
        sim.complete_sequential_best_ball_rollout(*args)
    )
    assert success_one and success_two
    np.testing.assert_array_equal(roster_one, roster_two)
    np.testing.assert_array_equal(details_one['path'], details_two['path'])
    assert len(roster_one) == len(set(roster_one)) == sim.num_rounds

    opponent_picks = {
        int(player_idx)
        for turn in details_one['opponent_picks_by_turn']
        for player_idx in turn
    }
    assert not set(details_one['path']) & opponent_picks
    counts = {
        pos: int(np.sum(positions[roster_one] == pos))
        for pos in sim.position_ranges
    }
    assert all(
        minimum <= counts[pos] <= maximum
        for pos, (minimum, maximum) in sim.position_ranges.items()
    )


def verify_real_database(db_path: Path) -> dict:
    conn = sqlite3.connect(db_path)
    try:
        sim = FootballSimulation(
            conn=conn,
            set_year=2026,
            pos_require_start={'QB': 3, 'RB': 7, 'WR': 9, 'TE': 3},
            num_teams=12,
            num_rounds=20,
            my_pick_position=6,
            pred_vers='final_ensemble',
            league='dk',
            position_ranges={
                'QB': (2, 3),
                'RB': (5, 7),
                'WR': (7, 9),
                'TE': (2, 3),
            },
        )
        results = sim.run_sim_best_ball_policy(
            [],
            [],
            num_iters=2,
            construction_samples=4,
            evaluation_samples=8,
            candidate_pool_size=4,
            seed=17,
            evaluation_seed=1001,
        )
        alternate_evaluation = sim.run_sim_best_ball_policy(
            [],
            [],
            num_iters=2,
            construction_samples=4,
            evaluation_samples=8,
            candidate_pool_size=4,
            seed=17,
            evaluation_seed=2002,
        )
    finally:
        conn.close()

    assert len(results) == 4
    assert (results.PolicyCompletedRooms == 2).all()
    assert results.attrs['timings']['horizon_label'] == 'sequential_template_16'
    primary_banks = results.attrs['scenario_banks']
    alternate_banks = alternate_evaluation.attrs['scenario_banks']
    assert primary_banks['construction_ppg_columns'] == (
        alternate_banks['construction_ppg_columns']
    )
    assert primary_banks['evaluation_ppg_columns'] != (
        alternate_banks['evaluation_ppg_columns']
    )
    assert not (
        set(primary_banks['construction_ppg_columns'])
        & set(primary_banks['evaluation_ppg_columns'])
    )
    assert results.attrs['policy_paths'] == alternate_evaluation.attrs['policy_paths']

    primary_ev = results.set_index('player').CurrentPickEV.sort_index()
    alternate_ev = (
        alternate_evaluation.set_index('player').CurrentPickEV.sort_index()
    )
    assert not np.allclose(primary_ev, alternate_ev)
    for root_player, room_paths in results.attrs['policy_paths'].items():
        assert len(room_paths) == 2
        for room_path in room_paths:
            path = room_path['path']
            opponents = {
                player
                for turn in room_path['opponent_picks_by_turn']
                for player in turn
            }
            assert path[0] == root_player
            assert len(path) == len(set(path)) == 20
            assert not set(path) & opponents

    return {
        'candidate_count': len(results),
        'completed_rooms_per_candidate': results.PolicyCompletedRooms.tolist(),
        'construction_evaluation_columns_disjoint': True,
        'evaluation_seed_path_invariance': True,
        'total_seconds': results.attrs['timings']['sections']['total'],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--db',
        type=Path,
        default=REPO_ROOT / 'app' / 'Simulation.sqlite3',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=STUDY_DIR / 'results' / 'verification.json',
    )
    args = parser.parse_args()

    actual_hashes = method_source_hashes(REPO_ROOT / 'app' / 'zSim_Helper.py')
    assert actual_hashes == FROZEN_LEGACY_SOURCE_HASHES
    verify_tensor_scoring()
    verify_candidate_consistent_state()
    verify_vectorized_legality()
    verify_disjoint_bank_columns()
    verify_rollout_boundaries()
    smoke = verify_real_database(args.db)

    output = {
        'status': 'pass',
        'legacy_source_hashes': actual_hashes,
        'checks': [
            'legacy_methods_frozen',
            'tensor_scoring_parity',
            'vectorized_legality_parity',
            'construction_evaluation_columns_disjoint',
            'candidate_consistent_availability',
            'no_duplicate_or_cross-drafted_players',
            'legal_final_construction',
            'evaluation_seed_path_invariance',
            'real_database_smoke',
        ],
        'real_database_smoke': smoke,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
