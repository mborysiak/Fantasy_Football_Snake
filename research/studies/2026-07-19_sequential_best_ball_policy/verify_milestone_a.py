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

from app.zSim_Helper import (  # noqa: E402
    FootballSimulation,
    SEQUENTIAL_CANDIDATE_POOL_SIZE,
    SEQUENTIAL_DECISION_CANDIDATES,
    SEQUENTIAL_STACK_BONUS_PCT,
    SEQUENTIAL_STACK_PAIR_CAP,
    SEQUENTIAL_STACK_TEAM_CAP,
)


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
    sim.use_stack_bonus = False
    sim.stack_bonus_pct = SEQUENTIAL_STACK_BONUS_PCT
    sim.stack_pair_cap = SEQUENTIAL_STACK_PAIR_CAP
    sim.stack_team_cap = SEQUENTIAL_STACK_TEAM_CAP
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
    decision = FootballSimulation.select_additional_policy_ppg_columns(
        1000,
        np.concatenate([construction, evaluation]),
        128,
        seed=421,
        bank_name='Decision',
    )
    audit = FootballSimulation.select_additional_policy_ppg_columns(
        1000,
        np.concatenate([construction, evaluation, decision]),
        128,
        seed=522,
        bank_name='Audit',
    )
    banks = [construction, evaluation, decision, audit]
    assert all(len(bank) == len(np.unique(bank)) for bank in banks)
    assert len(decision) == len(audit) == 128
    assert all(
        np.intersect1d(left, right).size == 0
        for idx, left in enumerate(banks)
        for right in banks[idx + 1:]
    )

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


def verify_empirical_replacement_scoring() -> None:
    values = np.array([30.0, 20.0, 15.0, 10.0], dtype=np.float32)
    positions = np.array(['QB', 'QB', 'RB', 'RB'])
    survival = np.array([0.5, 1.0, 0.0, 1.0], dtype=np.float32)
    replacement = FootballSimulation.expected_sequential_replacement_values(
        values,
        positions,
        survival,
    )
    np.testing.assert_allclose(replacement, [25.0, 25.0, 10.0, 10.0])
    (
        policy_scores,
        returned_survival,
        scored_replacement,
        draft_now_advantage,
        _,
    ) = FootballSimulation.sequential_policy_scores(
        np.arange(4),
        values,
        positions,
        np.zeros((4, 2), dtype=np.float32),
        current_pick=1,
        next_pick=2,
        survival_probabilities=survival,
    )
    np.testing.assert_allclose(returned_survival, survival)
    np.testing.assert_allclose(scored_replacement, replacement)
    np.testing.assert_allclose(draft_now_advantage, [5.0, 0.0, 5.0, 0.0])
    assert policy_scores[0] > 0 and policy_scores[1] == 0

    player_positions = np.array(['QB'] * 40 + ['RB'] * 60 + ['WR'] * 80 + ['TE'] * 30)
    quotas = FootballSimulation.sequential_root_position_quotas(
        np.arange(len(player_positions)),
        player_positions,
        np.array([100]),
        {
            'QB': (2, 3),
            'RB': (5, 7),
            'WR': (7, 9),
            'TE': (2, 3),
        },
        24,
    )
    assert quotas == {'QB': 3, 'RB': 8, 'WR': 10, 'TE': 3}
    assert sum(quotas.values()) == 24


def verify_sequential_stack_utility() -> None:
    positions = np.array(['QB', 'QB', 'WR', 'WR', 'TE', 'RB', 'QB'])
    teams = np.array(['A', 'B', 'A', 'B', 'A', 'A', 'A'])
    ppg = np.array([20.0, 18.0, 15.0, 14.0, 8.0, 12.0, 10.0])
    kwargs = {
        'player_positions': positions,
        'player_teams': teams,
        'player_ppg': ppg,
        'bonus_pct': SEQUENTIAL_STACK_BONUS_PCT,
        'pair_cap': SEQUENTIAL_STACK_PAIR_CAP,
        'team_cap': SEQUENTIAL_STACK_TEAM_CAP,
    }

    qb_after_catcher = FootballSimulation.sequential_stack_marginal_utilities(
        [2],
        [0],
        **kwargs,
    )
    catcher_after_qb = FootballSimulation.sequential_stack_marginal_utilities(
        [0],
        [2],
        **kwargs,
    )
    np.testing.assert_allclose(qb_after_catcher, [7.0])
    np.testing.assert_allclose(catcher_after_qb, [7.0])

    second_catcher = FootballSimulation.sequential_stack_marginal_utilities(
        [0, 2],
        [4, 3, 5],
        **kwargs,
    )
    np.testing.assert_allclose(second_catcher, [5.0, 0.0, 0.0])
    full_stack = FootballSimulation.sequential_stack_roster_utility(
        [0, 2, 4],
        **kwargs,
    )
    assert full_stack == SEQUENTIAL_STACK_TEAM_CAP
    backup_qb = FootballSimulation.sequential_stack_marginal_utilities(
        [0, 2],
        [6],
        **kwargs,
    )
    np.testing.assert_allclose(backup_qb, [0.0])


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
            use_stack_bonus=True,
            stack_bonus_pct=SEQUENTIAL_STACK_BONUS_PCT,
            stack_pair_cap=SEQUENTIAL_STACK_PAIR_CAP,
            stack_team_cap=SEQUENTIAL_STACK_TEAM_CAP,
        )
        with sim.temp_seed(17):
            sim.get_predictions('pred_fp_per_game', num_options=1000)
            adp_samples = sim.get_adp_samples(num_options=1000)
        adp_matrix = adp_samples.iloc[:, 2:].to_numpy(dtype=np.float32)
        draft_orders, _ = sim.build_sequential_draft_orders(
            adp_matrix,
            1,
            seed=320,
        )
        initial_opponents = adp_samples.player.to_numpy()[
            draft_orders[0, :5]
        ].tolist()

        common = {
            'to_add': [],
            'to_drop': initial_opponents,
            'num_iters': 2,
            'construction_samples': 4,
            'evaluation_samples': 8,
            'decision_samples': 12,
            'decision_candidate_count': 4,
            'audit_samples': 10,
            'candidate_pool_size': 4,
            'seed': 17,
        }
        results = sim.run_sim_best_ball_policy(
            **common,
            evaluation_seed=1001,
            decision_seed=3003,
            audit_seed=5005,
        )
        alternate_evaluation = sim.run_sim_best_ball_policy(
            **common,
            evaluation_seed=2002,
            decision_seed=3003,
            audit_seed=5005,
        )
        alternate_decision = sim.run_sim_best_ball_policy(
            **common,
            evaluation_seed=1001,
            decision_seed=4004,
            audit_seed=5005,
        )
        alternate_audit = sim.run_sim_best_ball_policy(
            **common,
            evaluation_seed=1001,
            decision_seed=3003,
            audit_seed=6006,
        )
        incomplete_state = sim.run_sim_best_ball_policy(
            [],
            [],
            num_iters=1,
            construction_samples=2,
            evaluation_samples=2,
            decision_samples=2,
            candidate_pool_size=2,
            seed=17,
        )
        assert len(incomplete_state) == 2
        incomplete_warnings = incomplete_state.attrs['warnings']
        assert len(incomplete_warnings) == 1
        assert "5 opponent picks should be marked" in incomplete_warnings[0]
        assert "will continue" in incomplete_warnings[0]
    finally:
        conn.close()

    assert len(results) == 4
    assert (results.PolicyCompletedRooms == 2).all()
    assert results.attrs['timings']['horizon_label'] == 'sequential_template_16'
    assert results.attrs['timings']['release_stage'] == 'preview'
    assert results.attrs['warnings'] == []
    assert SEQUENTIAL_DECISION_CANDIDATES == SEQUENTIAL_CANDIDATE_POOL_SIZE == 24
    assert int(results.DecisionCandidate.sum()) == len(results) == 4
    assert int(results.DecisionEV.notna().sum()) == len(results)
    assert int(results.AuditEV.notna().sum()) == len(results)
    assert results.ExpectedReplacementValue.notna().all()
    assert results.DraftNowAdvantage.notna().all()
    assert results.AverageStackUtility.gt(0).all()
    np.testing.assert_allclose(
        results.StackAdjustedDecisionScore,
        results.DecisionEV + results.AverageStackUtility,
        atol=1e-3,
    )
    assert results.attrs['timings']['stack_bonus_enabled'] is True
    assert sum(results.attrs['root_position_quotas'].values()) == len(results)
    assert results.iloc[0].player == results.attrs['decision_top_player']
    primary_banks = results.attrs['scenario_banks']
    alternate_banks = alternate_evaluation.attrs['scenario_banks']
    assert primary_banks['construction_ppg_columns'] == (
        alternate_banks['construction_ppg_columns']
    )
    assert primary_banks['evaluation_ppg_columns'] != (
        alternate_banks['evaluation_ppg_columns']
    )
    bank_names = [
        'construction_ppg_columns',
        'evaluation_ppg_columns',
        'decision_ppg_columns',
        'audit_ppg_columns',
    ]
    assert all(
        not (set(primary_banks[left]) & set(primary_banks[right]))
        for idx, left in enumerate(bank_names)
        for right in bank_names[idx + 1:]
    )
    assert results.attrs['policy_paths'] == alternate_evaluation.attrs['policy_paths']
    assert results.attrs['policy_paths'] == alternate_decision.attrs['policy_paths']
    assert results.attrs['policy_paths'] == alternate_audit.attrs['policy_paths']

    primary_ev = results.set_index('player').CurrentPickEV.sort_index()
    alternate_ev = (
        alternate_evaluation.set_index('player').CurrentPickEV.sort_index()
    )
    assert not np.allclose(primary_ev, alternate_ev)

    decision_banks = alternate_decision.attrs['scenario_banks']
    assert primary_banks['construction_ppg_columns'] == (
        decision_banks['construction_ppg_columns']
    )
    assert primary_banks['evaluation_ppg_columns'] == (
        decision_banks['evaluation_ppg_columns']
    )
    assert primary_banks['decision_ppg_columns'] != (
        decision_banks['decision_ppg_columns']
    )
    alternate_decision_pilot_ev = (
        alternate_decision.set_index('player').CurrentPickEV.sort_index()
    )
    np.testing.assert_allclose(primary_ev, alternate_decision_pilot_ev)
    assert results.attrs['decision_candidates'] == (
        alternate_decision.attrs['decision_candidates']
    )
    primary_decision = (
        results.set_index('player').DecisionEV.dropna().sort_index()
    )
    alternate_decision_ev = (
        alternate_decision.set_index('player')
        .DecisionEV.dropna()
        .sort_index()
    )
    assert not np.allclose(primary_decision, alternate_decision_ev)

    audit_banks = alternate_audit.attrs['scenario_banks']
    for bank_name in bank_names[:-1]:
        assert primary_banks[bank_name] == audit_banks[bank_name]
    assert primary_banks['audit_ppg_columns'] != (
        audit_banks['audit_ppg_columns']
    )
    alternate_audit_pilot = (
        alternate_audit.set_index('player').CurrentPickEV.sort_index()
    )
    alternate_audit_decision = (
        alternate_audit.set_index('player').DecisionEV.dropna().sort_index()
    )
    np.testing.assert_allclose(primary_ev, alternate_audit_pilot)
    np.testing.assert_allclose(primary_decision, alternate_audit_decision)
    assert results.player.tolist() == alternate_audit.player.tolist()
    assert results.attrs['decision_top_player'] == (
        alternate_audit.attrs['decision_top_player']
    )
    primary_audit = results.set_index('player').AuditEV.dropna().sort_index()
    alternate_audit_ev = (
        alternate_audit.set_index('player').AuditEV.dropna().sort_index()
    )
    assert not np.allclose(primary_audit, alternate_audit_ev)

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
        'all_policy_bank_columns_disjoint': True,
        'evaluation_seed_path_invariance': True,
        'decision_seed_path_invariance': True,
        'audit_seed_recommendation_invariance': True,
        'incomplete_physical_state_warned': True,
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
    verify_empirical_replacement_scoring()
    verify_sequential_stack_utility()
    verify_rollout_boundaries()
    smoke = verify_real_database(args.db)

    output = {
        'status': 'pass',
        'legacy_source_hashes': actual_hashes,
        'checks': [
            'legacy_methods_frozen',
            'tensor_scoring_parity',
            'vectorized_legality_parity',
            'all_policy_bank_columns_disjoint',
            'empirical_replacement_value_scoring',
            'roster_need_root_position_quotas',
            'symmetric_sequential_stack_utility',
            'candidate_consistent_availability',
            'no_duplicate_or_cross-drafted_players',
            'legal_final_construction',
            'evaluation_seed_path_invariance',
            'decision_seed_path_invariance',
            'audit_seed_recommendation_invariance',
            'incomplete_physical_state_warned',
            'real_database_smoke',
        ],
        'real_database_smoke': smoke,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
