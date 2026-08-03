"""Compare nested 128- and 256-season Sequential decision banks on current V2."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import platform
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.zSim_Helper import (  # noqa: E402
    SEQUENTIAL_STACK_BONUS_PCT,
    SEQUENTIAL_STACK_PAIR_CAP,
    SEQUENTIAL_STACK_TEAM_CAP,
    FootballSimulation,
)


POSITION_RANGES = {
    'QB': (2, 3),
    'RB': (5, 7),
    'WR': (7, 9),
    'TE': (2, 3),
}
POS_REQUIRE = {
    pos: maximum for pos, (_, maximum) in POSITION_RANGES.items()
}
BOOTSTRAP_SEED = 20260802
BOOTSTRAP_DRAWS = 10_000
REGRET_THRESHOLD = 10.0
VALUE_NONINFERIORITY_PCT = -0.25
DECISION_EXTENSION_SEED_OFFSET = 1_000_003
FROZEN_SLOTS = [1, 6, 12]
FROZEN_SEEDS = [17, 1017, 2017]
FROZEN_COMPLETED_PICKS = [0, 7, 14]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def parse_csv_values(raw, value_type=str):
    return [value_type(value.strip()) for value in raw.split(',') if value.strip()]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def git_head():
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def stable_json_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


class NestedBankFootballSimulation(FootballSimulation):
    """Allocate one D256 superbank and one common R512 reference bank."""

    def __init__(
        self,
        *args,
        study_control_decision_samples,
        study_decision_superbank_samples,
        study_reference_samples,
        **kwargs,
    ):
        self.study_control_decision_samples = int(
            study_control_decision_samples
        )
        self.study_decision_superbank_samples = int(
            study_decision_superbank_samples
        )
        self.study_reference_samples = int(study_reference_samples)
        self.study_decision_superbank = np.zeros(0, dtype=np.int64)
        self.study_reference_bank = np.zeros(0, dtype=np.int64)
        super().__init__(*args, **kwargs)

    def select_additional_policy_ppg_columns(
        self,
        num_columns,
        excluded_columns,
        samples,
        seed,
        bank_name,
    ):
        samples = int(samples)
        if samples <= 0:
            return np.zeros(0, dtype=np.int64)

        excluded = np.unique(np.asarray(excluded_columns, dtype=np.int64))
        all_columns = np.arange(int(num_columns), dtype=np.int64)
        if bank_name == 'Decision':
            if (
                self.study_decision_superbank_samples
                < self.study_control_decision_samples
            ):
                raise ValueError(
                    "Decision superbank is smaller than the production control."
                )
            self.study_decision_superbank = (
                super().select_additional_policy_ppg_columns(
                    num_columns,
                    excluded,
                    self.study_decision_superbank_samples,
                    seed,
                    bank_name,
                )
            )
            if samples > len(self.study_decision_superbank):
                raise ValueError("Requested decision samples exceed the superbank.")
            return self.study_decision_superbank[:samples].copy()

        if bank_name == 'Audit':
            if len(self.study_decision_superbank) == 0:
                raise RuntimeError("Reference allocation requires a decision superbank.")
            reference_exclusions = np.unique(
                np.concatenate([excluded, self.study_decision_superbank])
            )
            available = np.setdiff1d(
                all_columns,
                reference_exclusions,
                assume_unique=True,
            )
            if self.study_reference_samples > len(available):
                raise ValueError(
                    "Reference superbank exceeds remaining prediction columns."
                )
            rng = np.random.default_rng(seed)
            self.study_reference_bank = rng.choice(
                available,
                size=self.study_reference_samples,
                replace=False,
            ).astype(np.int64)
            if samples > len(self.study_reference_bank):
                raise ValueError("Requested audit samples exceed the reference bank.")
            return self.study_reference_bank[:samples].copy()

        return super().select_additional_policy_ppg_columns(
            num_columns,
            excluded,
            samples,
            seed,
            bank_name,
        )


def make_sim(conn, args, pick_slot, *, legacy=False):
    kwargs = {
        'conn': conn,
        'set_year': args.year,
        'pos_require_start': POS_REQUIRE,
        'num_teams': args.teams,
        'num_rounds': args.rounds,
        'my_pick_position': pick_slot,
        'pred_vers': 'final_ensemble',
        'league': 'dk',
        'position_ranges': POSITION_RANGES,
        'template_resid_blend': 1.0,
    }
    if legacy:
        return FootballSimulation(
            **kwargs,
            use_stack_bonus=True,
            stack_bonus_pct=0.25,
            stack_pair_cap=12.0,
            stack_team_cap=18.0,
        )
    return NestedBankFootballSimulation(
        **kwargs,
        use_stack_bonus=True,
        stack_bonus_pct=SEQUENTIAL_STACK_BONUS_PCT,
        stack_pair_cap=SEQUENTIAL_STACK_PAIR_CAP,
        stack_team_cap=SEQUENTIAL_STACK_TEAM_CAP,
        study_decision_superbank_samples=args.expanded_decision_samples,
        study_control_decision_samples=args.control_decision_samples,
        study_reference_samples=args.reference_samples,
    )


def frozen_design_exact(args):
    return bool(
        args.slot_values == FROZEN_SLOTS
        and args.seed_values == FROZEN_SEEDS
        and args.completed_pick_values == FROZEN_COMPLETED_PICKS
        and args.year == 2026
        and args.teams == 12
        and args.rounds == 20
        and args.rooms == 24
        and args.candidates == 24
        and args.construction_samples == 16
        and args.evaluation_samples == 64
        and args.control_decision_samples == 128
        and args.expanded_decision_samples == 256
        and args.reference_samples == 512
    )


def identity_name_maps(sim):
    identity_col = sim.identity_column(sim.player_data)
    frame = sim.player_data[[identity_col, 'player']].copy()
    if frame[identity_col].duplicated().any():
        raise ValueError("Player identities are not unique in the runtime pool.")
    if frame.player.duplicated().any():
        duplicates = sorted(frame.loc[frame.player.duplicated(False), 'player'].unique())
        raise ValueError(
            "Policy paths expose ambiguous display names: " + ', '.join(duplicates)
        )
    name_to_id = dict(zip(frame.player.astype(str), frame[identity_col].astype(str)))
    id_to_name = dict(zip(frame[identity_col].astype(str), frame.player.astype(str)))
    return name_to_id, id_to_name


def derive_initial_opponent_picks(sim, seed):
    picks_before_user = max(int(sim.my_picks[0]) - 1, 0)
    if picks_before_user == 0:
        return []
    with sim.temp_seed(seed):
        ppg_samples = sim.get_predictions('pred_fp_per_game', num_options=1000)
        adp_samples = sim.get_adp_samples(num_options=1000)
    ppg_ids = sim.identity_values(ppg_samples, validate_unique=True)
    adp_ids = sim.identity_values(adp_samples, validate_unique=True)
    adp_values = adp_samples[sim.sample_value_columns(adp_samples)].copy()
    adp_values.index = adp_ids
    adp_values = adp_values.reindex(ppg_ids)
    if adp_values.isna().any().any():
        raise ValueError("Could not align ADP samples for the initial fixture.")
    draft_orders, _ = sim.build_sequential_draft_orders(
        adp_values.to_numpy(dtype=np.float32),
        1,
        seed=seed + 303,
    )
    return ppg_ids[draft_orders[0, :picks_before_user]].astype(str).tolist()


def derive_state_from_control_path(
    room_path,
    initial_opponents,
    completed_picks,
    name_to_id,
):
    if completed_picks == 0:
        return [], list(initial_opponents)
    to_add = [name_to_id[name] for name in room_path['path'][:completed_picks]]
    later_opponents = [
        name_to_id[name]
        for turn in room_path['opponent_picks_by_turn'][:completed_picks]
        for name in turn
    ]
    to_drop = list(dict.fromkeys(list(initial_opponents) + later_opponents))
    return to_add, to_drop


def player_pool_coverage(sim, to_add, to_drop):
    adjusted_picks = sim.calculate_adjusted_picks(len(to_add))
    if not adjusted_picks:
        return True, 0, 0
    player_ids = set(sim.identity_values(sim.player_data, validate_unique=True))
    excluded_opponents = set(to_drop) - set(to_add)
    available_undrafted = len((player_ids - excluded_opponents) - set(to_add))
    required = int(adjusted_picks[-1] - adjusted_picks[0] + 1)
    return available_undrafted >= required, available_undrafted, required


def physical_state_is_valid(sim, to_add, to_drop, completed_picks):
    if len(to_add) != completed_picks or set(to_add) & set(to_drop):
        return False
    sim.validate_selection_coverage(to_add, to_drop)
    adjusted = sim.calculate_adjusted_picks(completed_picks)
    if not adjusted:
        return True
    expected_opponents = int(adjusted[0] - 1 - completed_picks)
    return len(to_drop) == expected_opponents


def run_policy(sim, args, to_add, to_drop, seed, decision_samples, audit_samples):
    return sim.run_sim_best_ball_policy(
        to_add,
        to_drop,
        num_iters=args.rooms,
        construction_samples=args.construction_samples,
        evaluation_samples=args.evaluation_samples,
        decision_samples=decision_samples,
        decision_candidate_count=args.candidates,
        audit_samples=audit_samples,
        candidate_pool_size=args.candidates,
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


def adjusted_matrices(result, bank_name):
    matrices = result.attrs[f'{bank_name}_value_matrices']
    adjusted = {}
    raw = {}
    rooms = {}
    for player, matrix in matrices.items():
        raw[player] = np.asarray(matrix['values'], dtype=np.float64)
        stack = np.asarray(matrix['stack_utilities'], dtype=np.float64).reshape(-1, 1)
        adjusted[player] = raw[player] + stack
        rooms[player] = np.asarray(matrix['rooms'], dtype=np.int64)
    return adjusted, raw, rooms


def top_player(matrices, scenario_slice=slice(None)):
    if not matrices:
        raise ValueError("No candidate matrices were available for ranking.")
    scores = {
        player: float(values[:, scenario_slice].mean())
        for player, values in matrices.items()
    }
    winner = max(scores, key=scores.get)
    return winner, scores


def two_way_components(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        return np.nan, np.nan, np.nan
    num_rooms, num_seasons = values.shape
    room_component = (
        float(np.var(values.mean(axis=1), ddof=1) / num_rooms)
        if num_rooms > 1
        else 0.0
    )
    season_component = (
        float(np.var(values.mean(axis=0), ddof=1) / num_seasons)
        if num_seasons > 1
        else 0.0
    )
    return room_component, season_component, float(
        np.sqrt(room_component + season_component)
    )


def cross_fitted_regret(reference_matrices, action):
    first_top, first_scores = top_player(reference_matrices, slice(0, 256))
    second_top, second_scores = top_player(reference_matrices, slice(256, 512))
    regret = 0.5 * (
        second_scores[first_top]
        - second_scores[action]
        + first_scores[second_top]
        - first_scores[action]
    )
    return float(regret), first_top, second_top


def rank_diagnostics(decision_scores, reference_scores):
    players = list(decision_scores)
    decision = pd.Series({p: decision_scores[p] for p in players})
    reference = pd.Series({p: reference_scores[p] for p in players})
    rank_corr = float(decision.rank().corr(reference.rank(), method='pearson'))
    decision_top5 = set(decision.nlargest(min(5, len(decision))).index)
    reference_top5 = set(reference.nlargest(min(5, len(reference))).index)
    return rank_corr, len(decision_top5 & reference_top5)


def assert_bank_and_path_contracts(baseline, expanded, baseline_sim, expanded_sim):
    baseline_banks = baseline.attrs['scenario_banks']
    expanded_banks = expanded.attrs['scenario_banks']
    if baseline_banks['construction_ppg_columns'] != expanded_banks['construction_ppg_columns']:
        raise AssertionError("Construction columns differ across arms.")
    if baseline_banks['evaluation_ppg_columns'] != expanded_banks['evaluation_ppg_columns']:
        raise AssertionError("Pilot columns differ across arms.")
    baseline_decision = np.asarray(
        baseline_banks['decision_ppg_columns'], dtype=np.int64
    )
    expanded_decision = np.asarray(
        expanded_banks['decision_ppg_columns'], dtype=np.int64
    )
    reference = np.asarray(expanded_banks['audit_ppg_columns'], dtype=np.int64)
    construction = np.asarray(expanded_banks['construction_ppg_columns'], dtype=np.int64)
    evaluation = np.asarray(expanded_banks['evaluation_ppg_columns'], dtype=np.int64)
    if not np.array_equal(baseline_decision, expanded_decision[:len(baseline_decision)]):
        raise AssertionError("The 128-scenario control is not nested in D256.")
    banks = [construction, evaluation, expanded_decision, reference]
    for left in range(len(banks)):
        for right in range(left + 1, len(banks)):
            if np.intersect1d(banks[left], banks[right]).size:
                raise AssertionError("Study scenario banks overlap.")
    if not np.array_equal(
        expanded_sim.study_decision_superbank,
        expanded_decision,
    ):
        raise AssertionError("Expanded decision superbank receipt mismatch.")
    if not np.array_equal(expanded_sim.study_reference_bank, reference):
        raise AssertionError("Reference superbank receipt mismatch.")
    if not np.array_equal(
        baseline_sim.study_decision_superbank,
        expanded_sim.study_decision_superbank,
    ):
        raise AssertionError("Decision superbanks differ across arms.")
    if baseline.attrs['decision_candidates'] != expanded.attrs['decision_candidates']:
        raise AssertionError("Candidate sets differ across arms.")
    if baseline.attrs['root_position_quotas'] != expanded.attrs['root_position_quotas']:
        raise AssertionError("Root position quotas differ across arms.")
    if baseline.attrs['draft_room_adp_columns'] != expanded.attrs['draft_room_adp_columns']:
        raise AssertionError("ADP room columns differ across arms.")
    if baseline.attrs['policy_paths'] != expanded.attrs['policy_paths']:
        raise AssertionError("Candidate rollout paths differ across arms.")

    baseline_adjusted, _, baseline_rooms = adjusted_matrices(baseline, 'decision')
    expanded_adjusted, _, expanded_rooms = adjusted_matrices(expanded, 'decision')
    if set(baseline_adjusted) != set(expanded_adjusted):
        raise AssertionError("Decision-scored candidates differ across arms.")
    for player in baseline_adjusted:
        if not np.array_equal(baseline_rooms[player], expanded_rooms[player]):
            raise AssertionError(f"Completed rooms differ for {player}.")
        if not np.array_equal(
            baseline_adjusted[player],
            expanded_adjusted[player][:, :baseline_adjusted[player].shape[1]],
        ):
            raise AssertionError(f"First-128 decision scores differ for {player}.")
    return True


def policy_seconds(result):
    sections = result.attrs['timings']['sections']
    audit_seconds = float(
        sections.get('audit_bank', 0.0) + sections.get('audit_scoring', 0.0)
    )
    return float(sections['total'] - audit_seconds), audit_seconds


def compare_state(
    baseline,
    expanded,
    baseline_sim,
    expanded_sim,
    args,
    pick_slot,
    seed,
    completed_picks,
    to_add,
    to_drop,
    id_to_name,
):
    assert_bank_and_path_contracts(
        baseline,
        expanded,
        baseline_sim,
        expanded_sim,
    )
    if len(baseline) != args.candidates or len(expanded) != args.candidates:
        raise AssertionError("The root screen did not retain every configured candidate.")
    if not (baseline.PolicyCompletedRooms == args.rooms).all():
        raise AssertionError("The control has an incomplete candidate room.")
    if not (expanded.PolicyCompletedRooms == args.rooms).all():
        raise AssertionError("The challenger has an incomplete candidate room.")

    d256_adjusted, d256_raw, _ = adjusted_matrices(expanded, 'decision')
    reference_adjusted, reference_raw, _ = adjusted_matrices(expanded, 'audit')
    control_action, d128_scores = top_player(d256_adjusted, slice(0, 128))
    challenger_action, d256_scores = top_player(d256_adjusted)
    if control_action != baseline.attrs['decision_top_player']:
        raise AssertionError("Nested D128 action differs from the control run.")
    if challenger_action != expanded.attrs['decision_top_player']:
        raise AssertionError("D256 action differs from the expanded run.")

    reference_top, reference_scores = top_player(reference_adjusted)
    reference_raw_top, reference_raw_scores = top_player(reference_raw)
    legacy_audit_top, legacy_audit_scores = top_player(
        reference_adjusted,
        slice(0, 128),
    )
    control_cf_regret, ref_first_top, ref_second_top = cross_fitted_regret(
        reference_adjusted,
        control_action,
    )
    challenger_cf_regret, _, _ = cross_fitted_regret(
        reference_adjusted,
        challenger_action,
    )
    control_full_regret = reference_scores[reference_top] - reference_scores[control_action]
    challenger_full_regret = (
        reference_scores[reference_top] - reference_scores[challenger_action]
    )
    control_raw_regret = (
        reference_raw_scores[reference_raw_top]
        - reference_raw_scores[control_action]
    )
    challenger_raw_regret = (
        reference_raw_scores[reference_raw_top]
        - reference_raw_scores[challenger_action]
    )
    legacy_audit_regret = (
        legacy_audit_scores[legacy_audit_top]
        - legacy_audit_scores[control_action]
    )
    value_delta_matrix = (
        reference_adjusted[challenger_action]
        - reference_adjusted[control_action]
    )
    room_var, season_var, delta_se = two_way_components(value_delta_matrix)
    value_delta = float(value_delta_matrix.mean())
    control_reference_value = float(reference_adjusted[control_action].mean())
    challenger_reference_value = float(reference_adjusted[challenger_action].mean())
    d128_corr, d128_top5 = rank_diagnostics(d128_scores, reference_scores)
    d256_corr, d256_top5 = rank_diagnostics(d256_scores, reference_scores)
    baseline_seconds, baseline_audit_seconds = policy_seconds(baseline)
    expanded_seconds, expanded_audit_seconds = policy_seconds(expanded)
    state_payload = {'to_add': list(to_add), 'to_drop': list(to_drop)}
    return {
        'league': 'dk',
        'pick_slot': int(pick_slot),
        'seed': int(seed),
        'trajectory_id': f'{pick_slot}:{seed}',
        'completed_picks': int(completed_picks),
        'current_round': int(completed_picks + 1),
        'state_hash': stable_json_hash(state_payload),
        'to_add_keys': json.dumps(list(to_add)),
        'to_drop_keys': json.dumps(list(to_drop)),
        'to_add_names': json.dumps([id_to_name[key] for key in to_add]),
        'to_drop_names': json.dumps([id_to_name[key] for key in to_drop]),
        'candidate_count': int(len(expanded)),
        'rooms': int(args.rooms),
        'all_rooms_complete': True,
        'bank_contracts_pass': True,
        'path_invariance_pass': True,
        'control_action': control_action,
        'challenger_action': challenger_action,
        'action_agreement': bool(control_action == challenger_action),
        'reference_top': reference_top,
        'reference_half_1_top': ref_first_top,
        'reference_half_2_top': ref_second_top,
        'control_reference_exact': bool(control_action == reference_top),
        'challenger_reference_exact': bool(challenger_action == reference_top),
        'legacy_audit_128_top': legacy_audit_top,
        'legacy_audit_128_exact': bool(control_action == legacy_audit_top),
        'legacy_audit_128_regret': float(legacy_audit_regret),
        'control_crossfit_regret': float(control_cf_regret),
        'challenger_crossfit_regret': float(challenger_cf_regret),
        'control_full_reference_regret': float(control_full_regret),
        'challenger_full_reference_regret': float(challenger_full_regret),
        'control_raw_reference_regret': float(control_raw_regret),
        'challenger_raw_reference_regret': float(challenger_raw_regret),
        'control_reference_value': control_reference_value,
        'challenger_reference_value': challenger_reference_value,
        'challenger_minus_control_value': value_delta,
        'value_delta_room_variance_component': room_var,
        'value_delta_season_variance_component': season_var,
        'value_delta_approx_se': delta_se,
        'control_rank_correlation_vs_reference': d128_corr,
        'challenger_rank_correlation_vs_reference': d256_corr,
        'control_top5_overlap_reference': d128_top5,
        'challenger_top5_overlap_reference': d256_top5,
        'control_policy_seconds': baseline_seconds,
        'control_audit_seconds': baseline_audit_seconds,
        'challenger_policy_seconds': expanded_seconds,
        'challenger_audit_seconds': expanded_audit_seconds,
    }


def bootstrap_seed_effect(frame):
    complete = frame[frame.status == 'complete'].copy()
    cluster_ids = sorted(complete.seed.unique())
    if not cluster_ids:
        return {
            'draws': BOOTSTRAP_DRAWS,
            'clusters': 0,
            'mean_delta': None,
            'ci95': [None, None],
            'mean_delta_pct': None,
            'ci95_pct': [None, None],
        }
    clusters = {
        cluster_id: complete[complete.seed == cluster_id]
        for cluster_id in cluster_ids
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    raw_draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    pct_draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for idx in range(BOOTSTRAP_DRAWS):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        delta = np.concatenate([
            clusters[cluster].challenger_minus_control_value.to_numpy()
            for cluster in sampled
        ])
        control = np.concatenate([
            clusters[cluster].control_reference_value.to_numpy()
            for cluster in sampled
        ])
        raw_draws[idx] = delta.mean()
        pct_draws[idx] = 100.0 * delta.mean() / control.mean()
    mean_delta = float(complete.challenger_minus_control_value.mean())
    mean_delta_pct = float(
        100.0 * mean_delta / complete.control_reference_value.mean()
    )
    return {
        'draws': BOOTSTRAP_DRAWS,
        'seed': BOOTSTRAP_SEED,
        'cluster_unit': 'base_seed',
        'clusters': len(cluster_ids),
        'mean_delta': mean_delta,
        'ci95': np.quantile(raw_draws, [0.025, 0.975]).tolist(),
        'mean_delta_pct': mean_delta_pct,
        'ci95_pct': np.quantile(pct_draws, [0.025, 0.975]).tolist(),
    }


def runtime_summary(frame, column):
    complete = frame[frame.status == 'complete']
    if complete.empty:
        return {}
    result = {
        'p50': float(complete[column].median()),
        'p90': float(complete[column].quantile(0.90)),
    }
    result['by_round'] = {
        str(int(round_number)): {
            'p50': float(group[column].median()),
            'p90': float(group[column].quantile(0.90)),
        }
        for round_number, group in complete.groupby('current_round')
    }
    return result


def summarize(records, args, receipt):
    frame = pd.DataFrame(records)
    complete = frame[frame.status == 'complete'].copy()
    effect = bootstrap_seed_effect(frame)
    quality = {}
    if not complete.empty:
        for label in ['control', 'challenger']:
            quality[label] = {
                'crossfit_regret_mean': float(
                    complete[f'{label}_crossfit_regret'].mean()
                ),
                'crossfit_regret_max': float(
                    complete[f'{label}_crossfit_regret'].max()
                ),
                'crossfit_regret_above_10': int(
                    (complete[f'{label}_crossfit_regret'] > REGRET_THRESHOLD).sum()
                ),
                'full_reference_regret_mean': float(
                    complete[f'{label}_full_reference_regret'].mean()
                ),
                'full_reference_regret_max': float(
                    complete[f'{label}_full_reference_regret'].max()
                ),
                'reference_exact_rate': float(
                    complete[f'{label}_reference_exact'].mean()
                ),
                'rank_correlation_mean': float(
                    complete[f'{label}_rank_correlation_vs_reference'].mean()
                ),
                'top5_overlap_mean': float(
                    complete[f'{label}_top5_overlap_reference'].mean()
                ),
            }
        quality['action_agreement_rate'] = float(complete.action_agreement.mean())
        quality['legacy_audit_128_exact_rate'] = float(
            complete.legacy_audit_128_exact.mean()
        )
        quality['legacy_audit_128_regret_max'] = float(
            complete.legacy_audit_128_regret.max()
        )

    exact_design = frozen_design_exact(args)
    execution_ok = len(complete) == args.expected_states
    contracts_ok = bool(
        execution_ok
        and complete.all_rooms_complete.all()
        and complete.bank_contracts_pass.all()
        and complete.path_invariance_pass.all()
    )
    challenger_regret_ok = bool(
        execution_ok
        and (complete.challenger_crossfit_regret <= REGRET_THRESHOLD).all()
    )
    regret_nonworse = bool(
        execution_ok
        and complete.challenger_crossfit_regret.mean()
        <= complete.control_crossfit_regret.mean()
        and complete.challenger_crossfit_regret.max()
        <= complete.control_crossfit_regret.max()
    )
    value_noninferior = bool(
        execution_ok
        and effect['ci95_pct'][0] is not None
        and effect['ci95_pct'][0] >= VALUE_NONINFERIORITY_PCT
    )
    challenger_runtime = runtime_summary(frame, 'challenger_policy_seconds')
    legacy_runtime = runtime_summary(frame, 'legacy_seconds')
    runtime_gate = bool(
        execution_ok
        and challenger_runtime
        and legacy_runtime
        and challenger_runtime['p50'] <= legacy_runtime['p50']
    )
    gates = {
        'frozen_design_exact': exact_design,
        'all_states_complete': execution_ok,
        'legality_completion_bank_and_path_contracts': contracts_ok,
        'challenger_crossfit_regret_at_most_10': challenger_regret_ok,
        'challenger_regret_nonworse_than_control': regret_nonworse,
        'reference_value_noninferior_at_minus_0_25_pct': value_noninferior,
        'challenger_runtime_p50_no_slower_than_legacy': runtime_gate,
    }
    return {
        'completed_at_utc': utc_now(),
        'configuration': {
            'year': args.year,
            'league': 'dk',
            'teams': args.teams,
            'rounds': args.rounds,
            'slots': args.slot_values,
            'seeds': args.seed_values,
            'completed_picks': args.completed_pick_values,
            'rooms': args.rooms,
            'candidates': args.candidates,
            'construction_samples': args.construction_samples,
            'evaluation_samples': args.evaluation_samples,
            'control_decision_samples': args.control_decision_samples,
            'expanded_decision_samples': args.expanded_decision_samples,
            'reference_samples': args.reference_samples,
            'reference_halves': [256, 256],
            'stack_bonus_pct': SEQUENTIAL_STACK_BONUS_PCT,
            'stack_pair_cap': SEQUENTIAL_STACK_PAIR_CAP,
            'stack_team_cap': SEQUENTIAL_STACK_TEAM_CAP,
            'regret_threshold': REGRET_THRESHOLD,
            'value_noninferiority_pct': VALUE_NONINFERIORITY_PCT,
        },
        'source_receipt': receipt,
        'state_counts': {
            'configured': args.expected_states,
            'complete': int(len(complete)),
            'errors': int((frame.status == 'error').sum()),
        },
        'quality': quality,
        'paired_reference_effect': effect,
        'runtime': {
            'control': runtime_summary(frame, 'control_policy_seconds'),
            'challenger': challenger_runtime,
            'legacy': legacy_runtime,
            'challenger_vs_control_p50_ratio': (
                float(
                    complete.challenger_policy_seconds.median()
                    / complete.control_policy_seconds.median()
                )
                if not complete.empty
                else None
            ),
        },
        'gates': gates,
        'compatibility_methodology_pass': bool(all(
            value for key, value in gates.items() if 'runtime' not in key
        )),
        'frozen_study_all_gates_pass': bool(all(gates.values())),
        'promotion_ready': False,
        'promotion_blockers': [
            'fresh_seed_confirmation_not_run',
            'current_v2_24_vs_32_shortlist_gate_not_run',
            *(
                []
                if runtime_gate
                else ['challenger_runtime_p50_slower_than_legacy']
            ),
        ],
        'errors': frame.loc[frame.status == 'error', [
            'pick_slot', 'seed', 'completed_picks', 'message'
        ]].to_dict('records'),
    }


def build_source_receipt(args, conn):
    db_path = args.db.resolve()
    helper_path = REPO_ROOT / 'app' / 'zSim_Helper.py'
    ui_path = REPO_ROOT / 'app' / 'snake_draft_app.py'
    quick_check = conn.execute('PRAGMA quick_check').fetchone()[0]
    probe = make_sim(conn, args, args.slot_values[0])
    if not probe.uses_v2_joint_template:
        raise ValueError("The frozen study requires the V2 joint-template path.")
    if probe.template_resid_method_version != 'joint_centered_template_v2_v1':
        raise ValueError(
            "Unexpected template residual method: "
            f"{probe.template_resid_method_version}"
        )
    return {
        'created_at_utc': utc_now(),
        'git_head': git_head(),
        'database': {
            'path': str(db_path),
            'size_bytes': db_path.stat().st_size,
            'sha256': sha256_file(db_path),
            'quick_check': quick_check,
            'page_count': conn.execute('PRAGMA page_count').fetchone()[0],
            'freelist_count': conn.execute('PRAGMA freelist_count').fetchone()[0],
        },
        'code': {
            'zSim_Helper.py': sha256_file(helper_path),
            'snake_draft_app.py': sha256_file(ui_path),
            'run_study.py': sha256_file(Path(__file__)),
        },
        'runtime': {
            'python': sys.version,
            'platform': platform.platform(),
            'numpy': np.__version__,
            'pandas': pd.__version__,
        },
        'model_contract': {
            'league': probe.league,
            'player_count': int(len(probe.player_data)),
            'identity_column': probe.identity_column(probe.player_data),
            'uses_v2_joint_template': bool(probe.uses_v2_joint_template),
            'template_resid_method_version': probe.template_resid_method_version,
            'weekly_horizon': 16,
        },
    }


def prehydrate_weekly_templates(sim):
    start = time.perf_counter()
    sim.load_weekly_template_profiles()
    return float(time.perf_counter() - start)


def write_checkpoint(records, args, receipt):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(args.output_dir / 'state_metrics.csv', index=False)
    (args.output_dir / 'source_receipt.json').write_text(
        json.dumps(receipt, indent=2),
        encoding='utf-8',
    )
    if records:
        summary = summarize(records, args, receipt)
        (args.output_dir / 'summary.json').write_text(
            json.dumps(summary, indent=2),
            encoding='utf-8',
        )


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
    parser.add_argument('--slots', default='1,6,12')
    parser.add_argument('--seeds', default='17,1017,2017')
    parser.add_argument('--completed-picks', default='0,7,14')
    parser.add_argument('--rooms', type=int, default=24)
    parser.add_argument('--candidates', type=int, default=24)
    parser.add_argument('--construction-samples', type=int, default=16)
    parser.add_argument('--evaluation-samples', type=int, default=64)
    parser.add_argument('--control-decision-samples', type=int, default=128)
    parser.add_argument('--expanded-decision-samples', type=int, default=256)
    parser.add_argument('--reference-samples', type=int, default=512)
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=STUDY_DIR / 'results',
    )
    args = parser.parse_args()
    args.slot_values = parse_csv_values(args.slots, int)
    args.seed_values = parse_csv_values(args.seeds, int)
    args.completed_pick_values = parse_csv_values(args.completed_picks, int)
    args.expected_states = (
        len(args.slot_values)
        * len(args.seed_values)
        * len(args.completed_pick_values)
    )
    if args.control_decision_samples > args.expanded_decision_samples:
        parser.error("Control decision samples must be a prefix of the expanded bank.")
    if args.reference_samples != 512:
        parser.error("The frozen cross-fit requires exactly 512 reference scenarios.")
    total_samples = (
        args.construction_samples
        + args.evaluation_samples
        + args.expanded_decision_samples
        + args.reference_samples
    )
    if total_samples > 1000:
        parser.error("The configured disjoint banks exceed 1,000 columns.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    conn = sqlite3.connect(args.db)
    receipt = build_source_receipt(args, conn)
    print(json.dumps({'source_receipt': receipt}, indent=2), flush=True)
    state_index = 0
    try:
        for pick_slot in args.slot_values:
            for seed in args.seed_values:
                fixture_sim = make_sim(conn, args, pick_slot)
                name_to_id, id_to_name = identity_name_maps(fixture_sim)
                initial_opponents = derive_initial_opponent_picks(fixture_sim, seed)
                control_room_path = None
                for completed_picks in args.completed_pick_values:
                    state_index += 1
                    state_start = time.perf_counter()
                    try:
                        if completed_picks == 0:
                            to_add, to_drop = [], list(initial_opponents)
                        elif control_room_path is None:
                            raise RuntimeError("The opening control trajectory is unavailable.")
                        else:
                            to_add, to_drop = derive_state_from_control_path(
                                control_room_path,
                                initial_opponents,
                                completed_picks,
                                name_to_id,
                            )
                        if not physical_state_is_valid(
                            fixture_sim,
                            to_add,
                            to_drop,
                            completed_picks,
                        ):
                            raise ValueError("Derived draft state is not physical.")
                        coverage = player_pool_coverage(
                            fixture_sim,
                            to_add,
                            to_drop,
                        )
                        if not coverage[0]:
                            raise ValueError(
                                f"State has {coverage[1]} modeled players for "
                                f"{coverage[2]} required picks."
                            )

                        baseline_sim = make_sim(conn, args, pick_slot)
                        expanded_sim = make_sim(conn, args, pick_slot)
                        legacy_sim = make_sim(conn, args, pick_slot, legacy=True)
                        hydration_seconds = {
                            'control': prehydrate_weekly_templates(baseline_sim),
                            'challenger': prehydrate_weekly_templates(expanded_sim),
                            'legacy': prehydrate_weekly_templates(legacy_sim),
                        }
                        runners = {
                            'baseline': lambda: run_policy(
                                baseline_sim,
                                args,
                                to_add,
                                to_drop,
                                seed,
                                args.control_decision_samples,
                                0,
                            ),
                            'expanded': lambda: run_policy(
                                expanded_sim,
                                args,
                                to_add,
                                to_drop,
                                seed,
                                args.expanded_decision_samples,
                                args.reference_samples,
                            ),
                            'legacy': lambda: run_legacy(
                                legacy_sim,
                                args,
                                to_add,
                                to_drop,
                                seed,
                            ),
                        }
                        orders = [
                            ['baseline', 'expanded', 'legacy'],
                            ['expanded', 'legacy', 'baseline'],
                            ['legacy', 'baseline', 'expanded'],
                        ]
                        outputs = {}
                        for label in orders[(state_index - 1) % len(orders)]:
                            outputs[label] = runners[label]()
                        baseline = outputs['baseline']
                        expanded = outputs['expanded']
                        legacy = outputs['legacy']
                        record = compare_state(
                            baseline,
                            expanded,
                            baseline_sim,
                            expanded_sim,
                            args,
                            pick_slot,
                            seed,
                            completed_picks,
                            to_add,
                            to_drop,
                            id_to_name,
                        )
                        legacy_sections = legacy.attrs['timings']['sections']
                        record.update({
                            'status': 'complete',
                            'message': '',
                            'physical_state_valid': True,
                            'pool_available': coverage[1],
                            'pool_required': coverage[2],
                            'legacy_seconds': float(legacy_sections['total']),
                            'control_hydration_seconds': hydration_seconds['control'],
                            'challenger_hydration_seconds': hydration_seconds['challenger'],
                            'legacy_hydration_seconds': hydration_seconds['legacy'],
                            'state_wall_seconds': float(
                                time.perf_counter() - state_start
                            ),
                        })
                        records.append(record)
                        if completed_picks == 0:
                            control_name = str(baseline.attrs['decision_top_player'])
                            control_room_path = copy.deepcopy(
                                baseline.attrs['policy_paths'][control_name][0]
                            )
                        print(
                            f"complete {state_index}/{args.expected_states} "
                            f"slot={pick_slot} seed={seed} "
                            f"round={completed_picks + 1} "
                            f"control={record['control_action']} "
                            f"challenger={record['challenger_action']} "
                            f"delta={record['challenger_minus_control_value']:.3f}",
                            flush=True,
                        )
                    except Exception as exc:
                        records.append({
                            'league': 'dk',
                            'pick_slot': pick_slot,
                            'seed': seed,
                            'trajectory_id': f'{pick_slot}:{seed}',
                            'completed_picks': completed_picks,
                            'current_round': completed_picks + 1,
                            'status': 'error',
                            'message': f'{type(exc).__name__}: {exc}',
                            'state_wall_seconds': float(
                                time.perf_counter() - state_start
                            ),
                        })
                        print(
                            f"error {state_index}/{args.expected_states} "
                            f"slot={pick_slot} seed={seed} "
                            f"round={completed_picks + 1}: {type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        if completed_picks == 0:
                            break
                    finally:
                        write_checkpoint(records, args, receipt)
                        gc.collect()
    finally:
        conn.close()

    summary = summarize(records, args, receipt)
    write_checkpoint(records, args, receipt)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
