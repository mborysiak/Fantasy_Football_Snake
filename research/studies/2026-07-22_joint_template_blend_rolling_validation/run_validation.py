"""Rolling best-ball validation of centered weekly-template residual blends."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


SNAKE_ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = SNAKE_ROOT.parent / "Fantasy_Football"
MODEL_SCRIPTS = MODEL_ROOT / "Scripts"
if str(MODEL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MODEL_SCRIPTS))

from Modeling import s4_Best_Ball_Weekly as builder  # noqa: E402


builder.LEAGUE = "dk"
RESULTS = Path(__file__).resolve().parent / "results"
ORIGIN_START = 2017
RECENT_START = 2020
ROOMS_PER_ORIGIN = 12
TEAMS_PER_ROOM = 12
ROSTER_SIZE = 20
SCENARIOS = 384
BOOTSTRAP_REPEATS = 1000
ROOT_SEED = 20260723
WEEK_COLS = [f"week_{week}" for week in builder.WEEKS]
RESID_COLS = [
    "pred_resid_5",
    "pred_resid_10",
    "pred_resid_25",
    "pred_resid_75",
    "pred_resid_90",
    "pred_resid_95",
]
POSITION_RANGES = {
    "QB": (2, 3),
    "RB": (5, 7),
    "WR": (7, 9),
    "TE": (2, 3),
}
METHODS = {
    "model_residual_profile": {"kind": "scaled_blend", "blend": 0.00},
    "current_030_scaled": {"kind": "scaled_blend", "blend": 0.30},
    "heavier_050_scaled": {"kind": "scaled_blend", "blend": 0.50},
    "heavier_070_scaled": {"kind": "scaled_blend", "blend": 0.70},
    "heavier_050_raw": {"kind": "raw_blend", "blend": 0.50},
    "heavier_070_raw": {"kind": "raw_blend", "blend": 0.70},
    "heavier_085_raw": {"kind": "raw_blend", "blend": 0.85},
    "template_100_scaled": {"kind": "scaled_blend", "blend": 1.00},
    "template_100_raw": {"kind": "raw_template", "blend": 1.00},
}


def binary_auc(outcome, score):
    outcome = np.asarray(outcome, dtype=int)
    score = np.asarray(score, dtype=float)
    positive = outcome == 1
    negative = outcome == 0
    if positive.sum() == 0 or negative.sum() == 0:
        return np.nan
    ranks = rankdata(score, method="average")
    return float(
        (
            ranks[positive].sum()
            - positive.sum() * (positive.sum() + 1) / 2
        )
        / (positive.sum() * negative.sum())
    )


def empirical_crps(samples, observed):
    samples = np.sort(np.asarray(samples, dtype=float))
    count = len(samples)
    coefficients = 2 * np.arange(1, count + 1) - count - 1
    return float(
        np.abs(samples - observed).mean()
        - np.sum(coefficients * samples) / (count * count)
    )


def load_oos_forecasts(max_season):
    select_cols = ", ".join(RESID_COLS)
    forecasts = builder.dm.read(
        f"""
        SELECT player,
               CAST(season AS INTEGER) season,
               pos,
               pred_fp_per_game production_oos_pred_fp_per_game,
               y_act production_validation_y_act,
               resid_calibration_available,
               {select_cols}
        FROM Final_Validations_Resid
        WHERE version='dk'
              AND model_spec_asof_year={builder.YEAR}
              AND data_oos=1
              AND season BETWEEN {ORIGIN_START} AND {max_season}
        """,
        "Validations",
    )
    forecasts = builder.clean_player_names(forecasts)
    if forecasts.duplicated(["player", "season", "pos"]).any():
        raise ValueError("Duplicate DK final-validation forecast rows.")
    forecasts[RESID_COLS] = forecasts[RESID_COLS].fillna(0)
    forecasts["resid_calibration_available"] = (
        forecasts["resid_calibration_available"].fillna(0).astype(int)
    )
    return forecasts


def build_target_templates(templates, forecasts):
    targets = templates[
        templates.season.between(ORIGIN_START, forecasts.season.max())
    ].copy()
    targets = targets.rename(
        columns={
            "historical_pred_fp_per_game": "builder_historical_pred_fp_per_game",
            "historical_projection_source": "builder_historical_projection_source",
        }
    ).merge(
        forecasts,
        on=["player", "season", "pos"],
        how="inner",
        validate="one_to_one",
    )
    targets["historical_pred_fp_per_game"] = targets[
        "production_oos_pred_fp_per_game"
    ]
    targets["historical_projection_source"] = "final_validations_resid_oos"
    targets = builder.add_projection_buckets(
        targets,
        value_col="historical_pred_fp_per_game",
        group_cols=["season", "pos"],
    )
    targets["match_projection_rank_pct"] = targets["projection_rank_pct"]
    targets["match_projection_ppg_scaled"] = (
        targets["historical_pred_fp_per_game"]
        .clip(lower=0)
        .div(builder.PROJECTION_PPG_SCALE)
    )
    targets["projection_x_exp"] = (
        targets["match_projection_rank_pct"] * targets["year_exp_scaled"]
    )
    targets["market_projection_gap"] = (
        targets["adp_rank_pct"] - targets["match_projection_rank_pct"]
    )
    targets["actual_missing_weeks"] = (
        len(builder.WEEKS) - targets["active_games"].clip(0, len(builder.WEEKS))
    )
    targets["actual_zero_active"] = targets["active_games"].eq(0).astype(int)
    return targets.reset_index(drop=True)


def donor_distances(target, donors):
    distances = np.zeros(len(donors), dtype=float)
    target_qb_bucket = getattr(target, "qb_team_rank_bucket", "non_qb")
    target_qb_value = builder.QB_RANK_DISTANCE_ORDER.get(target_qb_bucket, 2)
    qb_distance = (
        donors.qb_team_rank_bucket.map(builder.QB_RANK_DISTANCE_ORDER)
        .fillna(2)
        .sub(target_qb_value)
        .abs()
        .to_numpy(dtype=float)
        if target.pos == "QB"
        else np.zeros(len(donors), dtype=float)
    )
    for feature, weight in builder.MATCH_FEATURE_WEIGHTS[target.pos].items():
        if feature == "qb_team_rank_distance":
            feature_distance = qb_distance
        else:
            donor_values = pd.to_numeric(
                donors[feature], errors="coerce"
            ).fillna(builder.MATCH_FILL_VALUE).to_numpy(dtype=float)
            target_value = pd.to_numeric(
                pd.Series(
                    [getattr(target, feature, builder.MATCH_FILL_VALUE)]
                ),
                errors="coerce",
            ).fillna(builder.MATCH_FILL_VALUE).iloc[0]
            feature_distance = np.abs(donor_values - float(target_value))
        distances += float(weight) * feature_distance
    return distances


def select_donor_pool(target, eligible_donors):
    distances = donor_distances(target, eligible_donors)
    tie_rng = np.random.default_rng(
        builder.stable_seed(target.player, target.pos, target.season, "snake_roll")
    )
    order = np.lexsort((tie_rng.random(len(eligible_donors)), distances))
    selected_index = order[: min(builder.MAX_TEMPLATE_POOL_SIZE, len(order))]
    selected = eligible_donors.iloc[selected_index].copy()
    selected_distances = distances[selected_index]

    distance_min = float(selected_distances.min())
    bandwidth = builder.TEMPLATE_KERNEL_BANDWIDTH[target.pos]
    local_weights = np.exp(-(selected_distances - distance_min) / bandwidth)
    local_probabilities = local_weights / local_weights.sum()
    local_fraction = max(
        builder.TEMPLATE_MIN_LOCAL_WEIGHT,
        np.exp(-distance_min / builder.TEMPLATE_LOCAL_DISTANCE_SCALE),
    )
    probabilities = (
        min(float(local_fraction), 1.0) * local_probabilities
        + (1 - min(float(local_fraction), 1.0)) / len(selected)
    )
    probabilities = builder.cap_probability_vector(
        probabilities,
        builder.TEMPLATE_MAX_SAMPLE_PROBABILITY,
    )
    residuals = selected.active_ppg_resid.to_numpy(dtype=float)
    residual_mean = float(np.sum(probabilities * residuals))
    centered_residuals = residuals - residual_mean
    residual_sd = float(
        np.sqrt(np.sum(probabilities * np.square(centered_residuals)))
    )
    return {
        "profiles": selected[WEEK_COLS].to_numpy(dtype=np.float32),
        "centered_residuals": centered_residuals.astype(np.float32),
        "probabilities": probabilities.astype(np.float64),
        "residual_sd": residual_sd,
        "raw_residual_mean": residual_mean,
        "zero_active_probability": float(
            probabilities[selected.active_games.eq(0).to_numpy()].sum()
        ),
        "expected_active_games": float(
            np.sum(probabilities * selected.active_games.to_numpy(dtype=float))
        ),
    }


def build_donor_pools(templates, targets):
    eligible = templates[templates.template_eligible.eq(1)].copy()
    pools = {}
    total = len(targets)
    for number, target in enumerate(targets.itertuples(index=False), start=1):
        donors = eligible[
            eligible.pos.eq(target.pos) & eligible.season.lt(target.season)
        ].reset_index(drop=True)
        if len(donors) < builder.MIN_TEMPLATE_POOL_SIZE:
            raise ValueError(
                f"Only {len(donors)} prior donors for "
                f"{target.player} {target.season}."
            )
        pools[(int(target.season), target.player, target.pos)] = (
            select_donor_pool(target, donors)
        )
        if number % 300 == 0 or number == total:
            print(f"Built {number}/{total} strict-prior target pools")
    return pools


def residual_knots(row):
    q5, q10, q25, q75, q90, q95 = [
        float(getattr(row, column)) for column in RESID_COLS
    ]
    knots = np.array(
        [
            (2 * q5) - q10,
            q5,
            q10,
            q25,
            q75,
            q90,
            q95,
            (2 * q95) - q90,
        ],
        dtype=float,
    )
    return np.maximum.accumulate(knots)


def interpolate_residuals(knots, uniforms):
    probabilities = np.array(
        [0.00, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 1.00]
    )
    indices = np.searchsorted(probabilities, uniforms, side="right") - 1
    indices = np.clip(indices, 0, len(probabilities) - 2)
    left_probability = probabilities[indices]
    right_probability = probabilities[indices + 1]
    weight = (
        (uniforms - left_probability)
        / (right_probability - left_probability)
    )
    return knots[indices] + weight * (knots[indices + 1] - knots[indices])


def build_season_score_banks(season_targets, pools):
    player_count = len(season_targets)
    sampled_profiles = np.empty(
        (SCENARIOS, player_count, len(builder.WEEKS)),
        dtype=np.float32,
    )
    sampled_model_residuals = np.empty(
        (SCENARIOS, player_count),
        dtype=np.float32,
    )
    sampled_template_residuals = np.empty(
        (SCENARIOS, player_count),
        dtype=np.float32,
    )
    sampled_scaled_template_residuals = np.empty(
        (SCENARIOS, player_count),
        dtype=np.float32,
    )
    pool_audit = []

    for player_index, target in enumerate(
        season_targets.itertuples(index=False)
    ):
        key = (int(target.season), target.player, target.pos)
        pool = pools[key]
        rng = np.random.default_rng(
            builder.stable_seed(
                ROOT_SEED,
                target.season,
                target.player,
                target.pos,
                "outcomes",
            )
        )
        donor_uniforms = rng.random(SCENARIOS)
        donor_indices = np.searchsorted(
            np.cumsum(pool["probabilities"]),
            donor_uniforms,
            side="right",
        )
        donor_indices = np.minimum(
            donor_indices,
            len(pool["probabilities"]) - 1,
        )
        sampled_profiles[:, player_index, :] = pool["profiles"][donor_indices]
        template_residuals = pool["centered_residuals"][donor_indices]
        sampled_template_residuals[:, player_index] = template_residuals

        knots = residual_knots(target)
        model_uniforms = rng.random(SCENARIOS)
        model_residuals = interpolate_residuals(knots, model_uniforms)
        sampled_model_residuals[:, player_index] = model_residuals
        model_grid = interpolate_residuals(
            knots,
            (np.arange(4096, dtype=float) + 0.5) / 4096,
        )
        model_sd = float(np.std(model_grid))
        if pool["residual_sd"] > 1e-6 and model_sd > 1e-6:
            scaled_template_residuals = (
                template_residuals * model_sd / pool["residual_sd"]
            )
        else:
            scaled_template_residuals = np.zeros_like(template_residuals)
        sampled_scaled_template_residuals[:, player_index] = (
            scaled_template_residuals
        )
        pool_audit.append(
            {
                "season": int(target.season),
                "player": target.player,
                "pos": target.pos,
                "pool_raw_residual_mean": pool["raw_residual_mean"],
                "pool_residual_sd": pool["residual_sd"],
                "model_residual_sd": model_sd,
                "pool_zero_active_probability": pool[
                    "zero_active_probability"
                ],
                "pool_expected_active_games": pool["expected_active_games"],
                "resid_calibration_available": int(
                    target.resid_calibration_available
                ),
            }
        )

    point_forecast = season_targets[
        "historical_pred_fp_per_game"
    ].to_numpy(dtype=np.float32)
    banks = {}
    for method, specification in METHODS.items():
        if specification["kind"] == "raw_template":
            residual = sampled_template_residuals
        elif specification["kind"] == "raw_blend":
            blend = float(specification["blend"])
            residual = (
                np.sqrt(1 - blend**2) * sampled_model_residuals
                + blend * sampled_template_residuals
            )
        else:
            blend = float(specification["blend"])
            residual = (
                np.sqrt(1 - blend**2) * sampled_model_residuals
                + blend * sampled_scaled_template_residuals
            )
        sampled_ppg = np.maximum(
            point_forecast[None, :] + residual,
            0,
        ).astype(np.float32)
        banks[method] = sampled_profiles * sampled_ppg[:, :, None]
    return banks, pd.DataFrame(pool_audit)


def valid_pick_positions(counts, slots_remaining):
    deficits = {
        pos: max(POSITION_RANGES[pos][0] - counts[pos], 0)
        for pos in POSITION_RANGES
    }
    if sum(deficits.values()) >= slots_remaining:
        return {
            pos
            for pos, deficit in deficits.items()
            if deficit > 0 and counts[pos] < POSITION_RANGES[pos][1]
        }
    return {
        pos
        for pos in POSITION_RANGES
        if counts[pos] < POSITION_RANGES[pos][1]
    }


def draft_rooms(season_targets, season):
    base_adp = pd.to_numeric(
        season_targets.avg_pick,
        errors="coerce",
    )
    fallback_order = (
        season_targets.historical_pred_fp_per_game.rank(
            method="first",
            ascending=False,
        )
        + 240
    )
    base_adp = base_adp.where(
        base_adp.between(1, TEAMS_PER_ROOM * ROSTER_SIZE * 1.5),
        fallback_order,
    ).to_numpy(dtype=float)
    positions = season_targets.pos.to_numpy()
    roster_records = []

    for room in range(ROOMS_PER_ORIGIN):
        rng = np.random.default_rng(
            builder.stable_seed(ROOT_SEED, season, room, "draft")
        )
        room_rank = base_adp + rng.normal(
            0,
            np.maximum(4.0, 0.08 * base_adp),
        )
        available = np.ones(len(season_targets), dtype=bool)
        teams = [[] for _ in range(TEAMS_PER_ROOM)]
        counts = [
            {pos: 0 for pos in POSITION_RANGES}
            for _ in range(TEAMS_PER_ROOM)
        ]

        for round_number in range(ROSTER_SIZE):
            order = (
                range(TEAMS_PER_ROOM)
                if round_number % 2 == 0
                else range(TEAMS_PER_ROOM - 1, -1, -1)
            )
            for team in order:
                slots_remaining = ROSTER_SIZE - len(teams[team])
                allowed_positions = valid_pick_positions(
                    counts[team],
                    slots_remaining,
                )
                eligible = np.flatnonzero(
                    available & np.isin(positions, list(allowed_positions))
                )
                if len(eligible) == 0:
                    raise ValueError(
                        f"No legal historical draft pick: {season=} "
                        f"{room=} {team=} {round_number=}."
                    )
                chosen = int(eligible[np.argmin(room_rank[eligible])])
                teams[team].append(chosen)
                counts[team][positions[chosen]] += 1
                available[chosen] = False

        for team, roster in enumerate(teams):
            if len(roster) != ROSTER_SIZE:
                raise ValueError("Historical roster did not reach 20 players.")
            roster_records.append(
                {
                    "season": season,
                    "room": room,
                    "team": team,
                    "roster_indices": roster,
                }
            )
    return roster_records


def best_ball_score(weekly_bank, positions):
    """Score [scenario, player, week] as a DK best-ball roster."""
    scenarios, _, weeks = weekly_bank.shape
    score = np.zeros((scenarios, weeks), dtype=np.float32)
    flex_options = []
    starter_counts = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}

    for pos, starter_count in starter_counts.items():
        values = weekly_bank[:, positions == pos, :]
        values = np.sort(values, axis=1)[:, ::-1, :]
        score += values[:, :starter_count, :].sum(axis=1)
        if pos != "QB":
            if values.shape[1] > starter_count:
                flex_options.append(values[:, starter_count, :])
            else:
                flex_options.append(np.zeros((scenarios, weeks)))
    score += np.maximum.reduce(flex_options)
    return score.sum(axis=1)


def actual_lineup_zero_slots(weekly_scores, positions):
    """Count zero-valued weekly lineup slots after best-ball replacement."""
    selected_values = []
    flex_options = []
    starter_counts = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
    for pos, starter_count in starter_counts.items():
        values = weekly_scores[positions == pos]
        values = np.sort(values, axis=0)[::-1]
        selected_values.extend(values[:starter_count])
        if pos != "QB":
            if len(values) > starter_count:
                flex_options.append(values[starter_count])
            else:
                flex_options.append(np.zeros(weekly_scores.shape[1]))
    selected_values.append(np.maximum.reduce(flex_options))
    return int(np.sum(np.asarray(selected_values) <= 0))


def score_rosters(targets, pools):
    records = []
    pool_audits = []
    for season, season_targets in targets.groupby("season", sort=True):
        season_targets = season_targets.reset_index(drop=True)
        banks, pool_audit = build_season_score_banks(season_targets, pools)
        pool_audits.append(pool_audit)
        actual_weekly = (
            season_targets[WEEK_COLS].to_numpy(dtype=np.float32)
            * season_targets.active_ppg.to_numpy(dtype=np.float32)[:, None]
        )
        positions = season_targets.pos.to_numpy()
        rosters = draft_rooms(season_targets, int(season))

        for roster in rosters:
            indices = np.asarray(roster["roster_indices"], dtype=int)
            roster_positions = positions[indices]
            actual_bank = actual_weekly[indices][None, :, :]
            actual_score = float(
                best_ball_score(actual_bank, roster_positions)[0]
            )
            actual_zero_slots = actual_lineup_zero_slots(
                actual_weekly[indices],
                roster_positions,
            )
            roster_common = {
                "season": int(season),
                "room": int(roster["room"]),
                "team": int(roster["team"]),
                "roster_id": (
                    f"{int(season)}_{int(roster['room'])}_{int(roster['team'])}"
                ),
                "actual_score": actual_score,
                "actual_zero_active_players": int(
                    season_targets.iloc[indices].actual_zero_active.sum()
                ),
                "actual_missing_player_weeks": int(
                    season_targets.iloc[indices].actual_missing_weeks.sum()
                ),
                "actual_zero_lineup_slots": actual_zero_slots,
                "uncalibrated_residual_players": int(
                    (
                        season_targets.iloc[indices]
                        .resid_calibration_available.eq(0)
                    ).sum()
                ),
                "roster_players": "|".join(
                    season_targets.iloc[indices].player.tolist()
                ),
            }
            for method, bank in banks.items():
                predicted_scores = best_ball_score(
                    bank[:, indices, :],
                    roster_positions,
                )
                q10, q50, q90 = np.quantile(
                    predicted_scores,
                    [0.10, 0.50, 0.90],
                )
                records.append(
                    {
                        **roster_common,
                        "method": method,
                        "predicted_mean": float(predicted_scores.mean()),
                        "predicted_q10": float(q10),
                        "predicted_q50": float(q50),
                        "predicted_q90": float(q90),
                        "crps": empirical_crps(
                            predicted_scores,
                            actual_score,
                        ),
                        "covered_80": int(q10 <= actual_score <= q90),
                    }
                )
        print(
            f"Scored {len(rosters)} rosters for {int(season)} "
            f"across {len(METHODS)} methods"
        )
    return pd.DataFrame(records), pd.concat(pool_audits, ignore_index=True)


def safe_spearman(left, right):
    result = spearmanr(left, right, nan_policy="omit")
    return float(result.statistic) if np.isfinite(result.statistic) else np.nan


def summarize(group):
    error = group.predicted_mean - group.actual_score
    if "season" in group.columns:
        season_threshold = group.groupby("season").actual_score.transform(
            lambda values: values.quantile(0.80)
        )
    else:
        season_threshold = pd.Series(
            group.actual_score.quantile(0.80),
            index=group.index,
        )
    top_actual = group.actual_score.ge(season_threshold).astype(int)
    return pd.Series(
        {
            "n": len(group),
            "actual_score_mean": group.actual_score.mean(),
            "predicted_score_mean": group.predicted_mean.mean(),
            "bias": error.mean(),
            "mae": error.abs().mean(),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "crps": group.crps.mean(),
            "coverage_80": group.covered_80.mean(),
            "below_q10": group.actual_score.lt(group.predicted_q10).mean(),
            "above_q90": group.actual_score.gt(group.predicted_q90).mean(),
            "mean_interval_width": (
                group.predicted_q90 - group.predicted_q10
            ).mean(),
            "score_spearman": safe_spearman(
                group.predicted_mean,
                group.actual_score,
            ),
            "top_quintile_auc": binary_auc(
                top_actual,
                group.predicted_mean,
            ),
            "mean_zero_active_players": (
                group.actual_zero_active_players.mean()
            ),
            "mean_missing_player_weeks": (
                group.actual_missing_player_weeks.mean()
            ),
            "share_any_zero_lineup_slot": (
                group.actual_zero_lineup_slots.gt(0).mean()
            ),
            "mean_zero_lineup_slots": group.actual_zero_lineup_slots.mean(),
        }
    )


def grouped_summary(frame, group_cols):
    return (
        frame.groupby(group_cols, observed=True, sort=True)
        .apply(summarize, include_groups=False)
        .reset_index()
    )


def add_exposure_groups(predictions):
    predictions = predictions.copy()
    predictions["zero_active_group"] = pd.cut(
        predictions.actual_zero_active_players,
        bins=[-1, 0, 1, np.inf],
        labels=["0", "1", "2+"],
    )
    recent_current = predictions[
        predictions.season.ge(RECENT_START)
        & predictions.method.eq("current_030_scaled")
    ]
    missing_cutoffs = recent_current.actual_missing_player_weeks.quantile(
        [0.33, 0.67]
    ).to_numpy()
    predictions["missing_week_group"] = pd.cut(
        predictions.actual_missing_player_weeks,
        bins=[
            -np.inf,
            missing_cutoffs[0],
            missing_cutoffs[1],
            np.inf,
        ],
        labels=["low", "mid", "high"],
        include_lowest=True,
    )
    return predictions, missing_cutoffs


def metric_values(frame):
    error = frame.predicted_mean - frame.actual_score
    return {
        "crps": frame.crps.mean(),
        "mae": error.abs().mean(),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "absolute_bias": abs(error.mean()),
        "coverage_error": abs(frame.covered_80.mean() - 0.80),
        "interval_width": (
            frame.predicted_q90 - frame.predicted_q10
        ).mean(),
    }


def bootstrap_comparisons(recent):
    baseline = recent[recent.method.eq("current_030_scaled")]
    seasons = np.array(sorted(recent.season.unique()))
    rng = np.random.default_rng(ROOT_SEED + 99)
    rows = []
    for method in METHODS:
        if method == "current_030_scaled":
            continue
        candidate = recent[recent.method.eq(method)]
        point_candidate = metric_values(candidate)
        point_baseline = metric_values(baseline)
        draws = {metric: [] for metric in point_candidate}
        for _ in range(BOOTSTRAP_REPEATS):
            sampled_seasons = rng.choice(
                seasons,
                size=len(seasons),
                replace=True,
            )
            candidate_draw = pd.concat(
                [
                    candidate[candidate.season.eq(season)]
                    for season in sampled_seasons
                ],
                ignore_index=True,
            )
            baseline_draw = pd.concat(
                [
                    baseline[baseline.season.eq(season)]
                    for season in sampled_seasons
                ],
                ignore_index=True,
            )
            candidate_metrics = metric_values(candidate_draw)
            baseline_metrics = metric_values(baseline_draw)
            for metric in draws:
                if metric == "interval_width":
                    improvement = (
                        candidate_metrics[metric]
                        - baseline_metrics[metric]
                    )
                else:
                    improvement = (
                        baseline_metrics[metric]
                        - candidate_metrics[metric]
                    )
                draws[metric].append(improvement)
        for metric, values in draws.items():
            if metric == "interval_width":
                point_improvement = (
                    point_candidate[metric] - point_baseline[metric]
                )
            else:
                point_improvement = (
                    point_baseline[metric] - point_candidate[metric]
                )
            rows.append(
                {
                    "candidate": method,
                    "baseline": "current_030_scaled",
                    "metric": metric,
                    "candidate_value": point_candidate[metric],
                    "baseline_value": point_baseline[metric],
                    "improvement": point_improvement,
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                    "probability_better": float(np.mean(np.asarray(values) > 0)),
                }
            )
    return pd.DataFrame(rows)


def main():
    started = time.perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)
    max_season = builder.get_daily_max_template_season()
    print(f"Loading DK historical inputs through {max_season}")
    projections = builder.load_historical_projection_context(max_season)
    weekly = builder.load_weekly_points(max_season)
    templates = builder.build_weekly_templates(projections, weekly)
    forecasts = load_oos_forecasts(max_season)
    targets = build_target_templates(templates, forecasts)
    pools = build_donor_pools(templates, targets)
    predictions, pool_audit = score_rosters(targets, pools)
    predictions, missing_cutoffs = add_exposure_groups(predictions)

    predictions["period"] = np.where(
        predictions.season.ge(RECENT_START),
        "recent_2020_2025",
        "early_2017_2019",
    )
    summary = pd.concat(
        [
            grouped_summary(predictions, ["method"]).assign(
                period="all_2017_2025"
            ),
            grouped_summary(
                predictions[predictions.season.ge(RECENT_START)],
                ["method"],
            ).assign(period="recent_2020_2025"),
        ],
        ignore_index=True,
    )
    recent = predictions[predictions.season.ge(RECENT_START)].copy()
    summary_zero = grouped_summary(
        recent,
        ["method", "zero_active_group"],
    )
    summary_missing = grouped_summary(
        recent,
        ["method", "missing_week_group"],
    )
    summary_season = grouped_summary(
        predictions,
        ["method", "season"],
    )
    bootstrap = bootstrap_comparisons(recent)

    predictions.to_csv(RESULTS / "roster_predictions.csv", index=False)
    pool_audit.to_csv(RESULTS / "target_pool_audit.csv", index=False)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    summary_zero.to_csv(
        RESULTS / "summary_by_zero_active_exposure.csv",
        index=False,
    )
    summary_missing.to_csv(
        RESULTS / "summary_by_missing_week_exposure.csv",
        index=False,
    )
    summary_season.to_csv(RESULTS / "summary_by_season.csv", index=False)
    bootstrap.to_csv(RESULTS / "season_bootstrap.csv", index=False)

    metadata = {
        "league": "dk",
        "origin_start": ORIGIN_START,
        "recent_start": RECENT_START,
        "max_season": int(max_season),
        "rooms_per_origin": ROOMS_PER_ORIGIN,
        "teams_per_room": TEAMS_PER_ROOM,
        "roster_size": ROSTER_SIZE,
        "scenarios": SCENARIOS,
        "target_players": int(len(targets)),
        "rosters": int(predictions.roster_id.nunique()),
        "prediction_rows": int(len(predictions)),
        "methods": list(METHODS),
        "missing_week_group_cutoffs": missing_cutoffs.tolist(),
        "future_donor_rows": 0,
        "runtime_seconds": time.perf_counter() - started,
    }
    (RESULTS / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(
        summary[
            [
                "period",
                "method",
                "n",
                "bias",
                "mae",
                "crps",
                "coverage_80",
                "mean_interval_width",
                "score_spearman",
                "top_quintile_auc",
                "share_any_zero_lineup_slot",
            ]
        ].round(4).to_string(index=False)
    )
    print(
        bootstrap[
            bootstrap.metric.isin(
                ["crps", "mae", "absolute_bias", "coverage_error"]
            )
        ].round(4).to_string(index=False)
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
