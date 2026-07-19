"""Profile runtime and shortlist quality for the Milestone A policy."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np


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
    pos: maximum for pos, (_, maximum) in POSITION_RANGES.items()
}


def make_sim(conn, args) -> FootballSimulation:
    return FootballSimulation(
        conn=conn,
        set_year=args.year,
        pos_require_start=POS_REQUIRE,
        num_teams=args.teams,
        num_rounds=args.rounds,
        my_pick_position=args.pick_slot,
        pred_vers='final_ensemble',
        league=args.league,
        position_ranges=POSITION_RANGES,
    )


def run_policy(
    sim,
    args,
    to_add,
    to_drop,
    label,
    pool_size,
    scarcity,
    urgency,
    seed=None,
    evaluation_seed=None,
):
    policy_seed = args.seed if seed is None else int(seed)
    holdout_seed = (
        args.evaluation_seed
        if evaluation_seed is None
        else int(evaluation_seed)
    )
    results = sim.run_sim_best_ball_policy(
        to_add,
        to_drop,
        num_iters=args.rooms,
        construction_samples=args.construction_samples,
        evaluation_samples=args.evaluation_samples,
        candidate_pool_size=pool_size,
        scarcity_weight=scarcity,
        urgency_weight=urgency,
        seed=policy_seed,
        evaluation_seed=holdout_seed,
    )
    if label is not None:
        results.to_csv(args.output_dir / f'{label}.csv', index=False)
    return results


def comparison_metrics(primary, wide) -> dict:
    primary_players = set(primary.player)
    wide_by_player = wide.set_index('player')
    covered = wide_by_player.loc[
        wide_by_player.index.intersection(primary_players)
    ]
    excluded = wide_by_player.loc[
        ~wide_by_player.index.isin(primary_players)
    ]
    wide_best = float(wide.CurrentPickEV.max())
    covered_best = float(covered.CurrentPickEV.max())
    excluded_best = (
        float(excluded.CurrentPickEV.max()) if len(excluded) else np.nan
    )
    return {
        'primary_top_player': str(primary.iloc[0].player),
        'wide_top_player': str(wide.iloc[0].player),
        'top_pick_agreement': bool(primary.iloc[0].player == wide.iloc[0].player),
        'wide_candidate_count': int(len(wide)),
        'primary_candidate_count': int(len(primary)),
        'wide_best_ev': wide_best,
        'best_primary_candidate_ev_in_wide_run': covered_best,
        'empirical_shortlist_omission_regret': wide_best - covered_best,
        'best_excluded_candidate_ev': excluded_best,
        'best_excluded_regret_vs_wide_best': (
            wide_best - excluded_best if len(excluded) else np.nan
        ),
    }


def paired_policy_metrics(primary, comparator) -> dict:
    primary_player = str(primary.iloc[0].player)
    comparator_player = str(comparator.iloc[0].player)
    primary_matrix = primary.attrs['candidate_value_matrices'][primary_player]
    comparator_matrix = comparator.attrs['candidate_value_matrices'][
        comparator_player
    ]
    common_rooms = np.intersect1d(
        primary_matrix['rooms'],
        comparator_matrix['rooms'],
    )
    primary_locs = np.searchsorted(primary_matrix['rooms'], common_rooms)
    comparator_locs = np.searchsorted(
        comparator_matrix['rooms'],
        common_rooms,
    )
    paired_differences = (
        primary_matrix['values'][primary_locs]
        - comparator_matrix['values'][comparator_locs]
    )
    return {
        'primary_top_player': primary_player,
        'comparator_top_player': comparator_player,
        'common_rooms': int(len(common_rooms)),
        'evaluation_samples': int(paired_differences.shape[1]),
        'primary_policy_ev': float(
            primary_matrix['values'][primary_locs].mean()
        ),
        'comparator_policy_ev': float(
            comparator_matrix['values'][comparator_locs].mean()
        ),
        'paired_ev_difference': float(paired_differences.mean()),
        'paired_se': FootballSimulation.approximate_two_way_se(
            paired_differences
        ),
    }


def derive_initial_opponent_picks(sim, args) -> list[str]:
    picks_before_user = max(int(sim.my_picks[0]) - 1, 0)
    if picks_before_user == 0:
        return []

    with sim.temp_seed(args.seed):
        sim.get_predictions('pred_fp_per_game', num_options=1000)
        adp_samples = sim.get_adp_samples(num_options=1000)
    adp_matrix = adp_samples.iloc[:, 2:].to_numpy(dtype=np.float32)
    draft_orders, _ = sim.build_sequential_draft_orders(
        adp_matrix,
        1,
        seed=args.seed + 303,
    )
    player_names = adp_samples.player.to_numpy()
    return player_names[draft_orders[0, :picks_before_user]].tolist()


def derive_mid_draft_state(
    primary_results,
    initial_opponent_picks,
) -> tuple[list[str], list[str]]:
    best_player = str(primary_results.iloc[0].player)
    room_path = primary_results.attrs['policy_paths'][best_player][0]
    to_add = room_path['path'][:7]
    later_opponent_picks = [
        player
        for turn in room_path['opponent_picks_by_turn'][:6]
        for player in turn
    ]
    to_drop = list(dict.fromkeys(
        list(initial_opponent_picks) + later_opponent_picks
    ))
    return to_add, to_drop


def run_legacy(sim, args, to_add, to_drop, seed):
    with sim.temp_seed(seed):
        return sim.run_sim_best_ball_ilp(
            to_add,
            to_drop,
            args.legacy_iters,
            num_weeks=16,
            weekly_score_mode='template',
            current_pick_ev=False,
            parallel_workers=1,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--db',
        type=Path,
        default=REPO_ROOT / 'app' / 'Simulation.sqlite3',
    )
    parser.add_argument('--year', type=int, default=2026)
    parser.add_argument('--league', default='dk')
    parser.add_argument('--teams', type=int, default=12)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--pick-slot', type=int, default=6)
    parser.add_argument('--rooms', type=int, default=24)
    parser.add_argument('--construction-samples', type=int, default=16)
    parser.add_argument('--evaluation-samples', type=int, default=64)
    parser.add_argument('--primary-pool', type=int, default=16)
    parser.add_argument('--wide-pool', type=int, default=32)
    parser.add_argument('--legacy-iters', type=int, default=24)
    parser.add_argument('--runtime-repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--evaluation-seed', type=int)
    parser.add_argument('--skip-mid-draft', action='store_true')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=STUDY_DIR / 'results',
    )
    args = parser.parse_args()
    if args.evaluation_seed is None:
        args.evaluation_seed = args.seed + 202
    args.output_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(args.db)
    try:
        sim = make_sim(conn, args)
        early_to_drop = derive_initial_opponent_picks(sim, args)
        early_primary = run_policy(
            sim,
            args,
            [],
            early_to_drop,
            'early_primary',
            args.primary_pool,
            0.50,
            0.25,
        )
        early_greedy = run_policy(
            sim,
            args,
            [],
            early_to_drop,
            'early_greedy_only',
            args.primary_pool,
            0.0,
            0.0,
        )
        early_wide = run_policy(
            sim,
            args,
            [],
            early_to_drop,
            'early_wide',
            args.wide_pool,
            0.50,
            0.25,
        )

        legacy = run_legacy(
            sim,
            args,
            [],
            early_to_drop,
            args.seed,
        )
        legacy.to_csv(args.output_dir / 'legacy_template_16.csv', index=False)

        policy_runtime_seconds = [
            float(early_primary.attrs['timings']['sections']['total'])
        ]
        legacy_runtime_seconds = [
            float(legacy.attrs['timings']['sections']['total'])
        ]
        for repeat_idx in range(1, max(1, args.runtime_repeats)):
            repeat_seed = args.seed + (1000 * repeat_idx)
            repeated_policy = run_policy(
                sim,
                args,
                [],
                early_to_drop,
                None,
                args.primary_pool,
                0.50,
                0.25,
                seed=repeat_seed,
                evaluation_seed=repeat_seed + 202,
            )
            repeated_legacy = run_legacy(
                sim,
                args,
                [],
                early_to_drop,
                repeat_seed,
            )
            policy_runtime_seconds.append(
                float(repeated_policy.attrs['timings']['sections']['total'])
            )
            legacy_runtime_seconds.append(
                float(repeated_legacy.attrs['timings']['sections']['total'])
            )

        policy_runtime_p50 = float(np.median(policy_runtime_seconds))
        legacy_runtime_p50 = float(np.median(legacy_runtime_seconds))

        summary = {
            'configuration': {
                'year': args.year,
                'league': args.league,
                'teams': args.teams,
                'rounds': args.rounds,
                'pick_slot': args.pick_slot,
                'draft_rooms': args.rooms,
                'construction_samples': args.construction_samples,
                'evaluation_samples': args.evaluation_samples,
                'primary_pool': args.primary_pool,
                'wide_pool': args.wide_pool,
                'seed': args.seed,
                'evaluation_seed': args.evaluation_seed,
                'runtime_repeats': max(1, args.runtime_repeats),
            },
            'fixture': {
                'label': 'physical_slot_6_round_1',
                'initial_opponent_picks': early_to_drop,
                'initial_opponent_pick_count': len(early_to_drop),
            },
            'bank_invariants': {
                'construction_evaluation_ppg_columns_disjoint': bool(
                    early_primary.attrs['scenario_banks']['disjoint']
                ),
                'construction_ppg_column_count': len(
                    early_primary.attrs['scenario_banks'][
                        'construction_ppg_columns'
                    ]
                ),
                'evaluation_ppg_column_count': len(
                    early_primary.attrs['scenario_banks'][
                        'evaluation_ppg_columns'
                    ]
                ),
            },
            'early': {
                'primary_vs_wide': comparison_metrics(early_primary, early_wide),
                'scarcity_top_player': str(early_primary.iloc[0].player),
                'greedy_only_top_player': str(early_greedy.iloc[0].player),
                'scarcity_vs_greedy_top_agreement': bool(
                    early_primary.iloc[0].player == early_greedy.iloc[0].player
                ),
                'scarcity_vs_greedy_paired': paired_policy_metrics(
                    early_primary,
                    early_greedy,
                ),
                'primary_completion_rate': float(
                    early_primary.PolicyCompletedRooms.sum()
                    / (len(early_primary) * args.rooms)
                ),
                'timings': {
                    'primary': early_primary.attrs['timings'],
                    'greedy_only': early_greedy.attrs['timings'],
                    'wide': early_wide.attrs['timings'],
                },
            },
            'legacy': {
                'horizon_label': 'legacy_template_16',
                'requested_iters': args.legacy_iters,
                'timings': legacy.attrs.get('timings', {}),
            },
            'matched_runtime': {
                'policy_seconds': policy_runtime_seconds,
                'legacy_seconds': legacy_runtime_seconds,
                'policy_p50_seconds': policy_runtime_p50,
                'legacy_p50_seconds': legacy_runtime_p50,
                'policy_to_legacy_p50_ratio': (
                    policy_runtime_p50 / legacy_runtime_p50
                    if legacy_runtime_p50 > 0
                    else np.nan
                ),
                'runtime_gate_policy_no_slower_than_legacy': bool(
                    policy_runtime_p50 <= legacy_runtime_p50
                ),
            },
        }

        if not args.skip_mid_draft:
            mid_to_add, mid_to_drop = derive_mid_draft_state(
                early_primary,
                early_to_drop,
            )
            mid_primary = run_policy(
                sim,
                args,
                mid_to_add,
                mid_to_drop,
                'mid_primary',
                args.primary_pool,
                0.50,
                0.25,
            )
            mid_wide = run_policy(
                sim,
                args,
                mid_to_add,
                mid_to_drop,
                'mid_wide',
                args.wide_pool,
                0.50,
                0.25,
            )
            summary['mid'] = {
                'to_add': mid_to_add,
                'opponents_removed': len(mid_to_drop),
                'primary_vs_wide': comparison_metrics(mid_primary, mid_wide),
                'primary_completion_rate': float(
                    mid_primary.PolicyCompletedRooms.sum()
                    / (len(mid_primary) * args.rooms)
                ),
                'timings': {
                    'primary': mid_primary.attrs['timings'],
                    'wide': mid_wide.attrs['timings'],
                },
            }

    finally:
        conn.close()

    output_path = args.output_dir / 'milestone_a_summary.json'
    output_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
