"""Run the physical-fixture default-promotion gate for the sequential policy."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
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
    pos: maximum for pos, (_, maximum) in POSITION_RANGES.items()
}


def parse_csv_values(raw, value_type=str):
    return [value_type(value.strip()) for value in raw.split(',') if value.strip()]


def make_sim(conn, args, league, pick_slot):
    return FootballSimulation(
        conn=conn,
        set_year=args.year,
        pos_require_start=POS_REQUIRE,
        num_teams=args.teams,
        num_rounds=args.rounds,
        my_pick_position=pick_slot,
        pred_vers='final_ensemble',
        league=league,
        position_ranges=POSITION_RANGES,
        template_resid_blend=args.template_resid_blend,
    )


def derive_initial_opponent_picks(sim, seed):
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


def player_pool_coverage(sim, to_add, to_drop):
    adjusted_picks = sim.calculate_adjusted_picks(len(to_add))
    if not adjusted_picks:
        return True, 0, 0
    excluded_opponents = set(to_drop) - set(to_add)
    modeled_pool = set(sim.player_data.player) - excluded_opponents
    available_undrafted = len(modeled_pool - set(to_add))
    required = int(adjusted_picks[-1] - adjusted_picks[0] + 1)
    return available_undrafted >= required, available_undrafted, required


def run_policy(
    sim,
    args,
    to_add,
    to_drop,
    seed,
    candidate_pool_size,
    include_audit,
):
    return sim.run_sim_best_ball_policy(
        to_add,
        to_drop,
        num_iters=args.rooms,
        construction_samples=args.construction_samples,
        evaluation_samples=args.evaluation_samples,
        decision_samples=args.decision_samples,
        decision_candidate_count=args.decision_candidates,
        audit_samples=args.audit_samples if include_audit else 0,
        candidate_pool_size=candidate_pool_size,
        seed=seed,
        evaluation_seed=seed + 202,
        decision_seed=seed + 404,
        audit_seed=seed + 505,
    )


def run_legacy(sim, args, to_add, to_drop, seed):
    with sim.temp_seed(seed):
        return sim.run_sim_best_ball_ilp(
            to_add,
            to_drop,
            args.rooms,
            num_weeks=16,
            weekly_score_mode='template',
            current_pick_ev=False,
            parallel_workers=1,
        )


def derive_state(primary, initial_opponents, completed_picks):
    if completed_picks == 0:
        return [], list(initial_opponents)
    # Later fixtures follow the deployed decision-stage recommendation. The
    # hidden audit bank cannot choose a path or downstream validation state.
    root_player = str(primary.attrs['decision_top_player'])
    room_path = primary.attrs['policy_paths'][root_player][0]
    to_add = room_path['path'][:completed_picks]
    later_opponents = [
        player
        for turn in room_path['opponent_picks_by_turn'][:completed_picks]
        for player in turn
    ]
    to_drop = list(dict.fromkeys(list(initial_opponents) + later_opponents))
    return to_add, to_drop


def shortlist_metrics(primary, wide):
    primary_players = set(primary.player)
    wide_by_player = wide.set_index('player')
    covered = wide_by_player.loc[
        wide_by_player.index.intersection(primary_players)
    ]
    wide_best = float(wide.CurrentPickEV.max())
    covered_best = float(covered.CurrentPickEV.max())
    return {
        'primary_pilot_top': str(
            primary.loc[primary.PilotRank.idxmin(), 'player']
        ),
        'wide_pilot_top': str(
            wide.loc[wide.PilotRank.idxmin(), 'player']
        ),
        'pilot_top_agreement': bool(
            primary.loc[primary.PilotRank.idxmin(), 'player']
            == wide.loc[wide.PilotRank.idxmin(), 'player']
        ),
        'shortlist_omission_regret': wide_best - covered_best,
    }


def audit_metrics(primary, regret_threshold):
    decision_top = str(primary.attrs['decision_top_player'])
    audit_top = str(primary.attrs['audit_top_player'])
    audit_rows = primary[primary.AuditEV.notna()].set_index('player')
    decision_audit_ev = float(
        audit_rows.loc[decision_top, 'AuditEV']
    )
    best_audit_ev = float(audit_rows.AuditEV.max())
    regret = best_audit_ev - decision_audit_ev
    return {
        'decision_top': decision_top,
        'audit_top': audit_top,
        'decision_audit_exact_agreement': bool(
            decision_top == audit_top
        ),
        'decision_audit_regret': regret,
        'decision_audit_stable': bool(regret <= regret_threshold),
    }


def physical_state_is_valid(sim, to_add, to_drop, completed_picks):
    if len(to_add) != completed_picks:
        return False
    if set(to_add) & set(to_drop):
        return False
    adjusted = sim.calculate_adjusted_picks(completed_picks)
    if not adjusted:
        return True
    expected_opponents = int(adjusted[0] - 1 - completed_picks)
    return len(to_drop) == expected_opponents


def summarize(records, args):
    frame = pd.DataFrame(records)
    completed = frame[frame.status == 'complete'].copy()
    unsupported = frame[frame.status == 'unsupported'].copy()
    errors = frame[frame.status == 'error'].copy()

    def league_summary(league):
        league_frame = frame[frame.league == league]
        league_completed = league_frame[league_frame.status == 'complete']
        summary = {
            'configured_states': int(len(league_frame)),
            'completed_states': int(len(league_completed)),
            'unsupported_states': int(
                np.sum(league_frame.status == 'unsupported')
            ),
            'error_states': int(np.sum(league_frame.status == 'error')),
        }
        if len(league_completed):
            summary.update({
                'max_shortlist_omission_regret': float(
                    league_completed.shortlist_omission_regret.max()
                ),
                'max_decision_audit_regret': float(
                    league_completed.decision_audit_regret.max()
                ),
                'audit_exact_agreement_rate': float(
                    league_completed.decision_audit_exact_agreement.mean()
                ),
                'completion_rate': float(
                    league_completed.completion_rate.mean()
                ),
                'policy_runtime_p50': float(
                    league_completed.policy_seconds.median()
                ),
                'legacy_runtime_p50': float(
                    league_completed.legacy_seconds.median()
                ),
            })
        summary['promotion_ready'] = bool(
            len(league_completed) == len(league_frame)
            and len(league_frame) > 0
            and (league_completed.shortlist_omission_regret <= args.regret_threshold).all()
            and (league_completed.decision_audit_regret <= args.regret_threshold).all()
            and (league_completed.completion_rate == 1.0).all()
            and league_completed.policy_seconds.median()
            <= league_completed.legacy_seconds.median()
        )
        return summary

    league_summaries = {
        league: league_summary(league)
        for league in sorted(frame.league.unique())
    }
    gates = {
        'all_formats_supported': bool(len(unsupported) == 0),
        'no_execution_errors': bool(len(errors) == 0),
        'zero_high_regret_shortlist_omissions': bool(
            len(completed) > 0
            and (
                completed.shortlist_omission_regret
                <= args.regret_threshold
            ).all()
        ),
        'decision_audit_stable': bool(
            len(completed) > 0
            and (
                completed.decision_audit_regret
                <= args.regret_threshold
            ).all()
        ),
        'all_rooms_complete': bool(
            len(completed) > 0
            and (completed.completion_rate == 1.0).all()
        ),
        'runtime_p50_no_slower_than_legacy': bool(
            len(completed) > 0
            and completed.policy_seconds.median()
            <= completed.legacy_seconds.median()
        ),
    }
    return {
        'configuration': {
            'leagues': sorted(frame.league.unique().tolist()),
            'slots': sorted(frame.pick_slot.unique().tolist()),
            'seeds': sorted(frame.seed.unique().tolist()),
            'completed_pick_depths': sorted(
                frame.completed_picks.unique().tolist()
            ),
            'rooms': args.rooms,
            'construction_samples': args.construction_samples,
            'evaluation_samples': args.evaluation_samples,
            'decision_samples': args.decision_samples,
            'decision_candidates': args.decision_candidates,
            'audit_samples': args.audit_samples,
            'template_resid_blend': args.template_resid_blend,
            'regret_threshold': args.regret_threshold,
        },
        'state_counts': {
            'configured': int(len(frame)),
            'completed': int(len(completed)),
            'unsupported': int(len(unsupported)),
            'errors': int(len(errors)),
        },
        'gates': gates,
        'promotion_ready': bool(all(gates.values())),
        'by_league': league_summaries,
        'unsupported_reasons': sorted(
            unsupported.message.dropna().unique().tolist()
        ),
        'error_messages': sorted(errors.message.dropna().unique().tolist()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--db',
        type=Path,
        default=REPO_ROOT / 'app' / 'Simulation.sqlite3',
    )
    parser.add_argument('--year', type=int, default=2026)
    parser.add_argument('--teams', type=int, default=12)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--leagues', default='dk')
    parser.add_argument('--slots', default='1,6,12')
    parser.add_argument('--seeds', default='17,1017,2017')
    parser.add_argument('--completed-picks', default='0,7,14')
    parser.add_argument('--rooms', type=int, default=16)
    parser.add_argument('--construction-samples', type=int, default=16)
    parser.add_argument('--evaluation-samples', type=int, default=64)
    parser.add_argument('--decision-samples', type=int, default=128)
    parser.add_argument('--decision-candidates', type=int, default=24)
    parser.add_argument('--audit-samples', type=int, default=128)
    parser.add_argument('--primary-pool', type=int, default=24)
    parser.add_argument('--wide-pool', type=int, default=32)
    parser.add_argument('--template-resid-blend', type=float, default=1.0)
    parser.add_argument('--regret-threshold', type=float, default=10.0)
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=STUDY_DIR / 'results',
    )
    args = parser.parse_args()
    leagues = parse_csv_values(args.leagues)
    if leagues != ['dk']:
        parser.error("The Snake release gate supports only the DK league slice.")
    slots = parse_csv_values(args.slots, int)
    seeds = parse_csv_values(args.seeds, int)
    completed_depths = parse_csv_values(args.completed_picks, int)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    conn = sqlite3.connect(args.db)
    try:
        for league in leagues:
            for pick_slot in slots:
                for seed in seeds:
                    sim = make_sim(conn, args, league, pick_slot)
                    coverage, available, required = player_pool_coverage(
                        sim,
                        [],
                        [],
                    )
                    if not coverage:
                        message = (
                            f"{league} has {available} modeled undrafted players "
                            f"for {required} required picks at slot {pick_slot}."
                        )
                        for completed_picks in completed_depths:
                            records.append({
                                'league': league,
                                'pick_slot': pick_slot,
                                'seed': seed,
                                'completed_picks': completed_picks,
                                'current_round': completed_picks + 1,
                                'status': 'unsupported',
                                'message': message,
                            })
                        print(message, flush=True)
                        continue

                    initial_opponents = derive_initial_opponent_picks(sim, seed)
                    early_primary = None
                    for completed_picks in completed_depths:
                        state_start = time.perf_counter()
                        try:
                            if completed_picks == 0:
                                to_add, to_drop = [], initial_opponents
                            else:
                                to_add, to_drop = derive_state(
                                    early_primary,
                                    initial_opponents,
                                    completed_picks,
                                )
                            if not physical_state_is_valid(
                                sim,
                                to_add,
                                to_drop,
                                completed_picks,
                            ):
                                raise ValueError("Derived draft state is not physical.")
                            state_coverage = player_pool_coverage(
                                sim,
                                to_add,
                                to_drop,
                            )
                            if not state_coverage[0]:
                                raise ValueError(
                                    f"State has {state_coverage[1]} modeled players "
                                    f"for {state_coverage[2]} required picks."
                                )

                            primary = run_policy(
                                sim,
                                args,
                                to_add,
                                to_drop,
                                seed,
                                args.primary_pool,
                                True,
                            )
                            if completed_picks == 0:
                                early_primary = primary
                            wide = run_policy(
                                sim,
                                args,
                                to_add,
                                to_drop,
                                seed,
                                args.wide_pool,
                                False,
                            )
                            legacy = run_legacy(
                                sim,
                                args,
                                to_add,
                                to_drop,
                                seed,
                            )
                            metrics = shortlist_metrics(primary, wide)
                            metrics.update(
                                audit_metrics(
                                    primary,
                                    args.regret_threshold,
                                )
                            )
                            primary_sections = primary.attrs['timings']['sections']
                            audit_seconds = float(
                                primary_sections.get('audit_bank', 0)
                                + primary_sections.get('audit_scoring', 0)
                            )
                            records.append({
                                'league': league,
                                'pick_slot': pick_slot,
                                'seed': seed,
                                'completed_picks': completed_picks,
                                'current_round': completed_picks + 1,
                                'template_resid_blend': args.template_resid_blend,
                                'status': 'complete',
                                'message': '',
                                'to_add_count': len(to_add),
                                'to_drop_count': len(to_drop),
                                'physical_state_valid': True,
                                'completion_rate': float(
                                    primary.PolicyCompletedRooms.sum()
                                    / (len(primary) * args.rooms)
                                ),
                                'policy_seconds': float(
                                    primary_sections['total'] - audit_seconds
                                ),
                                'audit_seconds': audit_seconds,
                                'study_policy_seconds': float(
                                    primary_sections['total']
                                ),
                                'wide_seconds': float(
                                    wide.attrs['timings']['sections']['total']
                                ),
                                'legacy_seconds': float(
                                    legacy.attrs['timings']['sections']['total']
                                ),
                                'state_wall_seconds': (
                                    time.perf_counter() - state_start
                                ),
                                **metrics,
                            })
                            print(
                                f"complete {league} slot={pick_slot} seed={seed} "
                                f"round={completed_picks + 1}",
                                flush=True,
                            )
                        except Exception as exc:
                            records.append({
                                'league': league,
                                'pick_slot': pick_slot,
                                'seed': seed,
                                'completed_picks': completed_picks,
                                'current_round': completed_picks + 1,
                                'template_resid_blend': args.template_resid_blend,
                                'status': 'error',
                                'message': f"{type(exc).__name__}: {exc}",
                            })
                            print(
                                f"error {league} slot={pick_slot} seed={seed} "
                                f"round={completed_picks + 1}: {exc}",
                                flush=True,
                            )
                            if completed_picks == 0:
                                break

                    pd.DataFrame(records).to_csv(
                        args.output_dir / 'state_metrics.csv',
                        index=False,
                    )
    finally:
        conn.close()

    metrics_path = args.output_dir / 'state_metrics.csv'
    pd.DataFrame(records).to_csv(metrics_path, index=False)
    summary = summarize(records, args)
    summary_path = args.output_dir / 'release_gate_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
