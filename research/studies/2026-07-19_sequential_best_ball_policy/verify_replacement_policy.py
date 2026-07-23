"""Regression fixture for replacement-aware sequential draft timing."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--db',
        type=Path,
        default=REPO_ROOT / 'app' / 'Simulation.sqlite3',
    )
    parser.add_argument('--rooms', type=int, default=50)
    parser.add_argument(
        '--output',
        type=Path,
        default=STUDY_DIR / 'results' / 'replacement_policy_fixture.json',
    )
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        sim = FootballSimulation(
            conn=conn,
            set_year=2026,
            pos_require_start={'QB': 3, 'RB': 7, 'WR': 9, 'TE': 3},
            num_teams=12,
            num_rounds=20,
            my_pick_position=3,
            pred_vers='final_ensemble',
            league='dk',
            position_ranges=POSITION_RANGES,
        )
        sim.load_weekly_template_profiles()
        player_data = sim.player_data.reset_index()
        adp_order = player_data[['player', 'avg_pick']].sort_values('avg_pick')
        my_team = ['Puka Nacua']
        opponent_picks = (
            adp_order.loc[~adp_order.player.isin(my_team)]
            .head(20)
            .player.tolist()
        )
        results = sim.run_sim_best_ball_policy(
            my_team,
            opponent_picks,
            num_iters=args.rooms,
        )
    finally:
        conn.close()

    position_by_player = dict(zip(player_data.player, player_data.pos))
    root_position_counts = dict(Counter(results.pos))
    top_rows = []
    for rank, row in results.head(10).iterrows():
        top_rows.append({
            'rank': int(rank + 1),
            'player': str(row.player),
            'position': str(row.pos),
            'decision_ev': float(row.DecisionEV),
            'decision_ev_vs_best': float(row.DecisionEVVsBest),
            'survive_next': float(row.SurviveNext),
            'expected_replacement_value': float(row.ExpectedReplacementValue),
            'draft_now_advantage': float(row.DraftNowAdvantage),
        })

    quarterback_rows = []
    for rank, row in results.reset_index(drop=True).iterrows():
        if row.pos != 'QB':
            continue
        quarterback_rows.append({
            'rank': int(rank + 1),
            'player': str(row.player),
            'decision_ev': float(row.DecisionEV),
            'decision_ev_vs_best': float(row.DecisionEVVsBest),
            'survive_next': float(row.SurviveNext),
        })

    next_pick_positions = {}
    for root_player in ('Jeremiyah Love', 'Rashee Rice', 'Chris Olave', 'Josh Allen'):
        if root_player not in results.attrs['policy_paths']:
            continue
        counts = Counter(
            position_by_player[room_path['path'][1]]
            for room_path in results.attrs['policy_paths'][root_player]
        )
        next_pick_positions[root_player] = dict(counts)

    gates = {
        'all_candidate_rooms_complete': bool(
            (results.PolicyCompletedRooms == args.rooms).all()
        ),
        'root_screen_qbs_no_more_than_roster_max': bool(
            root_position_counts.get('QB', 0) <= POSITION_RANGES['QB'][1]
        ),
        'love_and_rice_remain_top_five': (
            {'Jeremiyah Love', 'Rashee Rice'}
            <= set(results.player.head(5))
        ),
        'no_qb_in_top_five': bool((results.head(5).pos != 'QB').all()),
    }
    assert all(gates.values()), gates

    output = {
        'status': 'pass',
        'configuration': {
            'year': 2026,
            'league': 'dk',
            'pick_slot': 3,
            'current_pick': 22,
            'rooms': args.rooms,
            'my_team': my_team,
            'opponent_picks': opponent_picks,
        },
        'root_position_quotas': results.attrs['root_position_quotas'],
        'root_position_counts': root_position_counts,
        'top_recommendations': top_rows,
        'quarterback_recommendations': quarterback_rows,
        'next_pick_position_counts': next_pick_positions,
        'timings': results.attrs['timings']['sections'],
        'gates': gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
