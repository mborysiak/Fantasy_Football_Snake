#%%

# core packages
import pandas as pd
import numpy as np
import copy
from collections import Counter
import contextlib
import sqlite3
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# scipy imports
import scipy.stats as stats

# linear optimization
from cvxopt import matrix, spmatrix
from cvxopt.glpk import ilp

import cvxopt
cvxopt.glpk.options['msg_lev'] = 'GLP_MSG_OFF'

BASE_PRED_COL = 'base_pred_fp_per_game'
TEMPLATE_RESID_BLEND = 0.30
MODEL_RESID_BLEND = np.sqrt(1 - TEMPLATE_RESID_BLEND**2)
MIN_RESID_SD = 1e-6
X_PICK_BUFFER = 6
SEQUENTIAL_POLICY_HORIZON = 16
SEQUENTIAL_CONSTRUCTION_SAMPLES = 16
SEQUENTIAL_EVALUATION_SAMPLES = 64
SEQUENTIAL_DECISION_SAMPLES = 128
SEQUENTIAL_DECISION_CANDIDATES = 4
SEQUENTIAL_DRAFT_ROOMS = 24
SEQUENTIAL_CANDIDATE_POOL_SIZE = 24
SEQUENTIAL_SCARCITY_WEIGHT = 0.50
SEQUENTIAL_URGENCY_WEIGHT = 0.25
SEQUENTIAL_POLICY_SEED = 20260719

class FootballSimulation:

    def __init__(self, conn, set_year, pos_require_start, num_teams, num_rounds, my_pick_position,
                 pred_vers='final_ensemble', league='dk', use_ownership=0, position_ranges=None,
                 use_stack_bonus=False, stack_bonus_pct=0.25, stack_pair_cap=12.0,
                 stack_team_cap=18.0):

        self.set_year = set_year
        self.pos_require_start = pos_require_start
        self.pred_vers = pred_vers
        self.league = league
        self.use_ownership = use_ownership
        self.position_ranges = position_ranges
        self.use_stack_bonus = bool(use_stack_bonus)
        self.stack_bonus_pct = float(stack_bonus_pct or 0)
        self.stack_pair_cap = float(stack_pair_cap or 0)
        self.stack_team_cap = float(stack_team_cap or 0)
        self.conn = conn
        self.num_teams = num_teams
        self.num_rounds = num_rounds
        self.my_pick_position = my_pick_position
        
        # Calculate my picks for snake draft
        self.my_picks = self.calculate_snake_picks()
        self.weekly_template_profiles = None
        self.weekly_template_week_cols = None
        self.weekly_template_cum_probs = None
        self.weekly_template_active_ppg_resids = None
        self.weekly_template_centered_active_ppg_resids = None
        self.weekly_template_active_ppg_resid_sds = None
        self.weekly_template_tensor_cache = {}

        player_data = self.get_model_predictions()

        # join in ADP data to player data
        self.player_data = self.join_adp(player_data)

    def get_model_predictions(self):
        df = pd.read_sql_query(f'''SELECT player, 
                                          pos, 
                                          pred_fp_per_game, 
                                          pred_fp_per_game_ny,
                                          pred_resid_5,
                                          pred_resid_10,
                                          pred_resid_25,
                                          pred_resid_75,
                                          pred_resid_90,
                                          pred_resid_95,
                                          pred_resid_5_ny,
                                          pred_resid_10_ny,
                                          pred_resid_25_ny,
                                          pred_resid_75_ny,
                                          pred_resid_90_ny,
                                          pred_resid_95_ny
                                   FROM Final_Predictions_Resid
                                   WHERE year={self.set_year}
                                         AND dataset='{self.pred_vers}'
                                         AND version='{self.league}'
                                         
                                ''', self.conn)

        if len(df) == 0:
            raise ValueError(
                f"No Final_Predictions_Resid rows found for "
                f"year={self.set_year}, dataset={self.pred_vers}, version={self.league}."
            )

        team_df = pd.read_sql_query(f'''
            SELECT player,
                   pos,
                   team,
                   avg_pick model_input_avg_pick,
                   year_exp model_input_year_exp
            FROM Best_Ball_Weekly_Player_Map
            WHERE year={self.set_year}
                  AND dataset='{self.pred_vers}'
                  AND version='{self.league}'
        ''', self.conn)
        if len(team_df) > 0:
            team_df = team_df.drop_duplicates(['player', 'pos'])
            df = pd.merge(df, team_df, how='left', on=['player', 'pos'])
        else:
            df['team'] = ''
            df['model_input_avg_pick'] = np.nan
            df['model_input_year_exp'] = np.nan
        df['team'] = df['team'].fillna('').astype(str).str.strip()

        resid_cols = [c for c in df.columns if c.startswith('pred_resid_')]
        df[resid_cols] = df[resid_cols].fillna(0)
        df['pred_fp_per_game_ny'] = df.pred_fp_per_game_ny.fillna(df.pred_fp_per_game)
        df['pred_p10'] = np.maximum(0, df.pred_fp_per_game + df.pred_resid_10)
        df['pred_p90'] = np.maximum(0, df.pred_fp_per_game + df.pred_resid_90)

        return df
    

    @staticmethod
    def player_join_key(values):
        return (
            pd.Series(values)
            .fillna('')
            .astype(str)
            .str.lower()
            .str.replace(r'[^a-z0-9]+', '', regex=True)
        )


    def join_adp(self, df):

        # add ADP data to the dataframe 
        adp_data = pd.read_sql_query(f'''SELECT player adp_player, 
                                                Years_of_Experience as years_of_experience,
                                                avg_pick,
                                                std_dev adp_std_dev,
                                                min_pick adp_min_pick,
                                                max_pick adp_max_pick
                                         FROM Avg_ADPs
                                         WHERE year={self.set_year}
                                               AND league='{self.league}'
                                               ''', 
                                        self.conn)

        df = df.copy()
        df['_player_join_key'] = self.player_join_key(df['player']).values
        adp_data['_player_join_key'] = self.player_join_key(adp_data['adp_player']).values
        adp_data = (
            adp_data.sort_values(['_player_join_key', 'avg_pick'])
            .drop_duplicates('_player_join_key')
        )

        df = pd.merge(df, adp_data, how='left', on='_player_join_key')
        model_input_avg_pick = pd.to_numeric(
            df.get('model_input_avg_pick', np.nan), errors='coerce'
        )
        model_pick_fallback = df['avg_pick'].isna() & model_input_avg_pick.notna()
        df['avg_pick'] = df['avg_pick'].combine_first(model_input_avg_pick)
        df.loc[model_pick_fallback, 'adp_std_dev'] = (
            df.loc[model_pick_fallback, 'adp_std_dev']
            .fillna(0.2 * df.loc[model_pick_fallback, 'avg_pick'])
        )
        df.loc[model_pick_fallback, 'adp_min_pick'] = (
            df.loc[model_pick_fallback, 'adp_min_pick']
            .fillna(0.8 * df.loc[model_pick_fallback, 'avg_pick'])
        )
        df.loc[model_pick_fallback, 'adp_max_pick'] = (
            df.loc[model_pick_fallback, 'adp_max_pick']
            .fillna(1.2 * df.loc[model_pick_fallback, 'avg_pick'])
        )
        if 'model_input_year_exp' in df.columns:
            df['years_of_experience'] = df['years_of_experience'].combine_first(
                df['model_input_year_exp']
            )
        df = df.drop(columns=['_player_join_key', 'adp_player'], errors='ignore')
        df = df.fillna({'avg_pick': 240, 'adp_std_dev': 20, 'adp_min_pick': 200, 'adp_max_pick': 250})
        df.loc[df.adp_std_dev < 0.1, 'adp_std_dev'] = 0.2 * df.loc[df.adp_std_dev < 0.1, 'avg_pick']
        df.loc[df.adp_min_pick > df.avg_pick, 'adp_min_pick'] = df.loc[df.adp_min_pick > df.avg_pick, 'avg_pick'] * 0.8
        df.loc[df.adp_max_pick < df.avg_pick, 'adp_max_pick'] = df.loc[df.adp_max_pick < df.avg_pick, 'avg_pick'] * 1.2        
        return df
    
    # ...existing code...
    

    @staticmethod
    def _df_shuffle(df):
        '''
        Input: A dataframe to be shuffled, row-by-row indepedently.
        Return: The same dataframe whose columns have been shuffled for each row.
        '''
        # store the index before converting to numpy
        idx = df.player
        df = df.drop('player', axis=1).values

        # shuffle each row separately, inplace, and convert o df
        _ = [np.random.shuffle(i) for i in df]

        return pd.DataFrame(df, index=idx).reset_index()
    

    @staticmethod
    def trunc_normal_vectorized(mean_vals, sdevs, min_scs, max_scs, num_samples=50):
        """Fully vectorized truncated normal distribution generation for all players at once"""
        
        # Convert inputs to numpy arrays for vectorized operations
        mean_vals = np.asarray(mean_vals).reshape(-1, 1)  # Shape: (num_players, 1)
        sdevs = np.asarray(sdevs).reshape(-1, 1)
        min_scs = np.asarray(min_scs).reshape(-1, 1)
        max_scs = np.asarray(max_scs).reshape(-1, 1)
        
        # Calculate standardized bounds (vectorized)
        lower_bounds = (min_scs - mean_vals) / sdevs
        upper_bounds = (max_scs - mean_vals) / sdevs
        
        # Get CDF values for bounds (vectorized)
        norm_cdf_lower = stats.norm.cdf(lower_bounds)
        norm_cdf_upper = stats.norm.cdf(upper_bounds)
        
        # Generate uniform random samples for all players at once
        # Shape: (num_players, num_samples)
        num_players = len(mean_vals)
        uniform_samples = np.random.uniform(0, 1, (num_players, num_samples))
        
        # Transform uniform to truncated normal using vectorized operations
        # Broadcasting will handle the shape differences automatically
        scaled_uniform = norm_cdf_lower + uniform_samples * (norm_cdf_upper - norm_cdf_lower)
        
        # Convert to standard normal using inverse CDF (vectorized)
        standard_normal_samples = stats.norm.ppf(scaled_uniform)
        
        # Transform to desired mean and scale (vectorized)
        samples = mean_vals + sdevs * standard_normal_samples
        
        return samples

    @staticmethod
    def residual_quantile_vectorized(mean_vals, resid_vals, num_samples=50):
        """Sample player outcomes from residual percentile knots."""
        probs = np.array([0.00, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 1.00])
        mean_vals = np.asarray(mean_vals, dtype=np.float32).reshape(-1, 1)
        resid_vals = np.asarray(resid_vals, dtype=np.float32)

        q5, q10, q25, q75, q90, q95 = resid_vals.T
        q0 = (2 * q5) - q10
        q100 = (2 * q95) - q90
        resid_knots = np.column_stack([q0, q5, q10, q25, q75, q90, q95, q100])
        resid_knots = np.maximum.accumulate(resid_knots, axis=1)

        num_players = len(mean_vals)
        uniform_samples = np.random.uniform(0, 1, (num_players, num_samples)).astype(np.float32)
        knot_idx = np.searchsorted(probs, uniform_samples, side='right') - 1
        knot_idx = np.clip(knot_idx, 0, len(probs) - 2)

        prob_left = probs[knot_idx]
        prob_right = probs[knot_idx + 1]
        resid_left = np.take_along_axis(resid_knots, knot_idx, axis=1)
        resid_right = np.take_along_axis(resid_knots, knot_idx + 1, axis=1)

        weight = (uniform_samples - prob_left) / (prob_right - prob_left)
        sampled_resid = resid_left + (weight * (resid_right - resid_left))
        samples = mean_vals + sampled_resid

        return np.maximum(samples, 0)

    def trunc_normal_dist(self, col, num_options=50):
        
        if col=='pred_fp_per_game':
            mean_col = 'pred_fp_per_game'
            resid_dist_cols = [
                'pred_resid_5', 'pred_resid_10', 'pred_resid_25',
                'pred_resid_75', 'pred_resid_90', 'pred_resid_95'
            ]
            return pd.DataFrame(
                self.residual_quantile_vectorized(
                    self.player_data[mean_col].values,
                    self.player_data[resid_dist_cols].values,
                    num_options
                )
            )
        elif col == 'pred_fp_per_game_ny':
            mean_col = 'pred_fp_per_game_ny'
            resid_dist_cols = [
                'pred_resid_5_ny', 'pred_resid_10_ny', 'pred_resid_25_ny',
                'pred_resid_75_ny', 'pred_resid_90_ny', 'pred_resid_95_ny'
            ]
            return pd.DataFrame(
                self.residual_quantile_vectorized(
                    self.player_data[mean_col].values,
                    self.player_data[resid_dist_cols].values,
                    num_options
                )
            )
        elif col=='adp':
            cols = ['avg_pick', 'adp_std_dev', 'adp_min_pick', 'adp_max_pick']
        else:
            raise ValueError(f"Unknown distribution column: {col}")

        # Fully vectorized approach: process all players at once
        data_values = self.player_data[cols].values
        mean_vals, sdevs, min_scs, max_scs = data_values.T
        
        # Get vectorized samples for all players
        samples = self.trunc_normal_vectorized(mean_vals, sdevs, min_scs, max_scs, num_options)
        
        return pd.DataFrame(samples)
    

    def get_predictions(self, pred_label, num_options=500):

        labels = self.player_data[['player', 'pos', 'team']]
        predictions = self.trunc_normal_dist(pred_label, num_options)
        predictions = pd.concat([labels, predictions], axis=1)
        predictions[BASE_PRED_COL] = self.player_data[pred_label].values

        return predictions
    
    def get_adp_samples(self, num_options=500):
        labels = self.player_data[['player', 'pos']]
        adp = self.trunc_normal_dist('adp', num_options).astype('int64')
        adp = pd.concat([labels, adp], axis=1)
        return adp

    @staticmethod
    def sample_value_columns(df):
        return [c for c in df.columns if c not in ('player', 'pos', 'team', BASE_PRED_COL)]

    @staticmethod
    def read_weekly_template_profile_cache(conn, set_year, league, pred_vers):
        template_cols = pd.read_sql_query(
            "SELECT * FROM Best_Ball_Weekly_Templates LIMIT 0",
            conn,
        ).columns
        week_cols = sorted(
            [c for c in template_cols if c.startswith('week_')],
            key=lambda c: int(c.split('_')[1]),
        )
        if len(week_cols) == 0:
            raise ValueError("Best_Ball_Weekly_Templates has no week_* columns.")

        pool_cols = pd.read_sql_query(
            "SELECT * FROM Best_Ball_Weekly_Template_Pools LIMIT 0",
            conn,
        ).columns
        sample_prob_select = (
            "p.template_sample_prob"
            if "template_sample_prob" in pool_cols
            else "NULL AS template_sample_prob"
        )
        active_ppg_resid_select = (
            "t.active_ppg_resid"
            if "active_ppg_resid" in template_cols
            else "(t.active_ppg - t.historical_pred_fp_per_game) AS active_ppg_resid"
        )
        template_join = (
            "ON p.template_id = t.template_id AND p.pool_version = t.league"
            if "league" in template_cols
            else "ON p.template_id = t.template_id"
        )

        week_select = ', '.join([f't.{c}' for c in week_cols])
        profiles = pd.read_sql_query(
            f'''
            SELECT m.player,
                   p.template_id,
                   {sample_prob_select},
                   {active_ppg_resid_select},
                   {week_select}
            FROM Best_Ball_Weekly_Player_Map m
            INNER JOIN Best_Ball_Weekly_Template_Pools p
                    ON m.template_pool_key = p.template_pool_key
            INNER JOIN Best_Ball_Weekly_Templates t
                    {template_join}
            WHERE m.year = {set_year}
                  AND m.version = '{league}'
                  AND m.dataset = '{pred_vers}'
            ORDER BY m.player, p.match_rank
            ''',
            conn,
        )

        if len(profiles) == 0:
            raise ValueError(
                f"No weekly template profiles found for "
                f"year={set_year}, version={league}, dataset={pred_vers}."
            )

        weekly_template_profiles = {}
        weekly_template_cum_probs = {}
        weekly_template_active_ppg_resids = {}
        weekly_template_centered_active_ppg_resids = {}
        weekly_template_active_ppg_resid_sds = {}
        for player, group in profiles.groupby('player', sort=False):
            weekly_template_profiles[player] = group[week_cols].to_numpy(dtype=np.float32)
            active_ppg_resids = group['active_ppg_resid'].fillna(0).to_numpy(dtype=np.float32)
            weekly_template_active_ppg_resids[player] = active_ppg_resids
            probs = group['template_sample_prob'].to_numpy(dtype=np.float64)
            if (
                len(probs) == 0
                or np.any(~np.isfinite(probs))
                or np.any(probs < 0)
                or probs.sum() <= 0
            ):
                probs = np.repeat(1 / len(group), len(group))
            else:
                probs = probs / probs.sum()
            cum_probs = np.cumsum(probs)
            cum_probs[-1] = 1.0
            weekly_template_cum_probs[player] = cum_probs
            resid_mean = float(np.sum(probs * active_ppg_resids))
            centered_resids = active_ppg_resids - resid_mean
            resid_sd = float(np.sqrt(np.sum(probs * np.square(centered_resids))))
            weekly_template_centered_active_ppg_resids[player] = centered_resids.astype(
                np.float32
            )
            weekly_template_active_ppg_resid_sds[player] = resid_sd

        return (
            week_cols,
            weekly_template_profiles,
            weekly_template_cum_probs,
            weekly_template_active_ppg_resids,
            weekly_template_centered_active_ppg_resids,
            weekly_template_active_ppg_resid_sds,
        )

    def set_weekly_template_profile_cache(
        self,
        week_cols,
        weekly_template_profiles,
        weekly_template_cum_probs,
        weekly_template_active_ppg_resids,
        weekly_template_centered_active_ppg_resids=None,
        weekly_template_active_ppg_resid_sds=None,
    ):
        self.weekly_template_week_cols = list(week_cols)
        self.weekly_template_profiles = weekly_template_profiles
        self.weekly_template_cum_probs = weekly_template_cum_probs
        self.weekly_template_active_ppg_resids = weekly_template_active_ppg_resids
        if weekly_template_centered_active_ppg_resids is None:
            weekly_template_centered_active_ppg_resids = {}
            weekly_template_active_ppg_resid_sds = {}
            for player, active_ppg_resids in weekly_template_active_ppg_resids.items():
                active_ppg_resids = np.asarray(active_ppg_resids, dtype=np.float32)
                cum_probs = np.asarray(weekly_template_cum_probs[player], dtype=np.float64)
                probs = np.diff(np.insert(cum_probs, 0, 0.0))
                resid_mean = float(np.sum(probs * active_ppg_resids))
                centered_resids = active_ppg_resids - resid_mean
                weekly_template_centered_active_ppg_resids[player] = centered_resids.astype(
                    np.float32
                )
                weekly_template_active_ppg_resid_sds[player] = float(
                    np.sqrt(np.sum(probs * np.square(centered_resids)))
                )
        self.weekly_template_centered_active_ppg_resids = (
            weekly_template_centered_active_ppg_resids
        )
        self.weekly_template_active_ppg_resid_sds = weekly_template_active_ppg_resid_sds
        return len(week_cols)

    def load_weekly_template_profiles(self):
        if self.weekly_template_profiles is not None:
            return len(self.weekly_template_week_cols)

        cache = self.read_weekly_template_profile_cache(
            self.conn,
            self.set_year,
            self.league,
            self.pred_vers,
        )
        return self.set_weekly_template_profile_cache(*cache)

    def sample_template_weekly_scores(self, predictions, num_weeks):
        template_weeks = self.load_weekly_template_profiles()
        num_weeks = min(num_weeks, template_weeks)

        players = predictions.player.values
        score_matrix = predictions[self.sample_value_columns(predictions)].values.astype(np.float32)
        ppg_col = np.random.randint(0, score_matrix.shape[1])
        sampled_ppg = score_matrix[:, ppg_col]
        if BASE_PRED_COL in predictions.columns:
            base_ppg = predictions[BASE_PRED_COL].to_numpy(dtype=np.float32)
        else:
            base_ppg = score_matrix.mean(axis=1)
        model_resid_sds = np.std(score_matrix - base_ppg[:, None], axis=1)

        missing_players = [p for p in players if p not in self.weekly_template_profiles]
        if missing_players:
            missing_preview = ', '.join(missing_players[:10])
            raise ValueError(
                f"Missing weekly template profiles for {len(missing_players)} players: "
                f"{missing_preview}"
            )

        weekly_scores = np.empty((len(players), num_weeks), dtype=np.float32)
        for idx, player in enumerate(players):
            profiles = self.weekly_template_profiles[player]
            cum_probs = self.weekly_template_cum_probs[player]
            centered_active_ppg_resids = self.weekly_template_centered_active_ppg_resids[player]
            template_resid_sd = self.weekly_template_active_ppg_resid_sds[player]
            template_idx = np.searchsorted(cum_probs, np.random.random(), side='right')
            model_resid = sampled_ppg[idx] - base_ppg[idx]
            scaled_template_resid = 0.0
            # Centering removes pool bias; scaling keeps the model residual variance calibrated.
            if (
                np.isfinite(template_resid_sd)
                and np.isfinite(model_resid_sds[idx])
                and template_resid_sd > MIN_RESID_SD
                and model_resid_sds[idx] > MIN_RESID_SD
            ):
                scaled_template_resid = (
                    centered_active_ppg_resids[template_idx]
                    * (model_resid_sds[idx] / template_resid_sd)
                )
            blended_ppg = (
                base_ppg[idx]
                + (MODEL_RESID_BLEND * model_resid)
                + (TEMPLATE_RESID_BLEND * scaled_template_resid)
            )
            weekly_scores[idx] = max(blended_ppg, 0) * profiles[template_idx, :num_weeks]

        return weekly_scores

    def _template_tensor_cache_entry(self, players, num_weeks):
        """Pack ragged per-player template pools into arrays for fast sampling."""
        template_weeks = self.load_weekly_template_profiles()
        num_weeks = min(int(num_weeks), template_weeks)
        player_key = tuple(players)
        cache_key = (player_key, num_weeks)
        cached = self.weekly_template_tensor_cache.get(cache_key)
        if cached is not None:
            return cached

        missing_players = [
            player for player in players
            if player not in self.weekly_template_profiles
        ]
        if missing_players:
            missing_preview = ', '.join(missing_players[:10])
            raise ValueError(
                f"Missing weekly template profiles for {len(missing_players)} players: "
                f"{missing_preview}"
            )

        num_players = len(players)
        max_templates = max(
            len(self.weekly_template_cum_probs[player]) for player in players
        )
        profiles = np.zeros(
            (num_players, max_templates, num_weeks),
            dtype=np.float32,
        )
        cum_probs = np.ones((num_players, max_templates), dtype=np.float64)
        centered_resids = np.zeros(
            (num_players, max_templates),
            dtype=np.float32,
        )
        template_resid_sds = np.empty(num_players, dtype=np.float32)

        for player_idx, player in enumerate(players):
            player_profiles = np.asarray(
                self.weekly_template_profiles[player],
                dtype=np.float32,
            )
            player_cum_probs = np.asarray(
                self.weekly_template_cum_probs[player],
                dtype=np.float64,
            )
            player_resids = np.asarray(
                self.weekly_template_centered_active_ppg_resids[player],
                dtype=np.float32,
            )
            pool_size = len(player_cum_probs)
            profiles[player_idx, :pool_size] = player_profiles[:, :num_weeks]
            cum_probs[player_idx, :pool_size] = player_cum_probs
            centered_resids[player_idx, :pool_size] = player_resids
            template_resid_sds[player_idx] = (
                self.weekly_template_active_ppg_resid_sds[player]
            )

        cached = (
            profiles,
            cum_probs,
            centered_resids,
            template_resid_sds,
            num_weeks,
        )
        self.weekly_template_tensor_cache[cache_key] = cached
        return cached

    def sample_template_weekly_score_bank(
        self,
        predictions,
        num_scenarios,
        num_weeks=SEQUENTIAL_POLICY_HORIZON,
        seed=None,
        ppg_column_indices=None,
    ):
        """Sample common-random-number weekly scores as [scenario, player, week]."""
        if num_scenarios <= 0:
            raise ValueError("num_scenarios must be positive.")

        players = predictions.player.to_numpy()
        (
            profiles,
            cum_probs,
            centered_resids,
            template_resid_sds,
            num_weeks,
        ) = self._template_tensor_cache_entry(players, num_weeks)

        score_matrix = predictions[
            self.sample_value_columns(predictions)
        ].to_numpy(dtype=np.float32)
        if score_matrix.shape[1] == 0:
            raise ValueError("Predictions contain no sampled score columns.")

        rng = np.random.default_rng(seed)
        if ppg_column_indices is None:
            ppg_cols = rng.integers(
                0,
                score_matrix.shape[1],
                size=num_scenarios,
            )
        else:
            ppg_cols = np.asarray(ppg_column_indices, dtype=np.int64)
            if len(ppg_cols) != num_scenarios:
                raise ValueError(
                    "ppg_column_indices must contain one index per scenario."
                )
            if np.any(ppg_cols < 0) or np.any(ppg_cols >= score_matrix.shape[1]):
                raise ValueError("ppg_column_indices contains an out-of-range index.")
        sampled_ppg = score_matrix[:, ppg_cols].T
        if BASE_PRED_COL in predictions.columns:
            base_ppg = predictions[BASE_PRED_COL].to_numpy(dtype=np.float32)
        else:
            base_ppg = score_matrix.mean(axis=1)
        model_resid_sds = np.std(score_matrix - base_ppg[:, None], axis=1)

        template_draws = rng.random((num_scenarios, len(players), 1))
        template_indices = np.sum(
            template_draws >= cum_probs[None, :, :],
            axis=2,
        )
        player_indices = np.arange(len(players))[None, :]
        sampled_profiles = profiles[player_indices, template_indices]
        sampled_template_resids = centered_resids[
            player_indices,
            template_indices,
        ]

        scale_is_valid = (
            np.isfinite(template_resid_sds)
            & np.isfinite(model_resid_sds)
            & (template_resid_sds > MIN_RESID_SD)
            & (model_resid_sds > MIN_RESID_SD)
        )
        resid_scales = np.zeros(len(players), dtype=np.float32)
        resid_scales[scale_is_valid] = (
            model_resid_sds[scale_is_valid]
            / template_resid_sds[scale_is_valid]
        )
        scaled_template_resids = sampled_template_resids * resid_scales[None, :]
        model_resids = sampled_ppg - base_ppg[None, :]
        blended_ppg = (
            base_ppg[None, :]
            + (MODEL_RESID_BLEND * model_resids)
            + (TEMPLATE_RESID_BLEND * scaled_template_resids)
        )
        blended_ppg = np.maximum(blended_ppg, 0).astype(np.float32)

        return sampled_profiles * blended_ppg[:, :, None]

    def sample_ilp_weekly_scores(self, predictions, num_weeks, weekly_score_mode='residual'):
        if weekly_score_mode == 'template':
            return self.sample_template_weekly_scores(predictions, num_weeks)

        score_matrix = predictions[self.sample_value_columns(predictions)].values.astype(np.float32)
        week_cols = np.random.randint(0, score_matrix.shape[1], size=num_weeks)
        return score_matrix[:, week_cols]
    
    def init_select_cnts(self):
        
        player_selections = {}
        for p in self.player_data.player:
            # Initialize counts and availability for each round
            round_counts = {f'round_{i+1}_count': 0 for i in range(self.num_rounds)}
            round_availability = {f'round_{i+1}_available': 0 for i in range(self.num_rounds)}
            player_selections[p] = {
                'total_counts': 0, 
                'total_available_count': 0,
                **round_counts,
                **round_availability
            }
        return player_selections


    def add_players(self, to_add):
        
        h_player_add = {}
        open_pos_require = copy.deepcopy(self.pos_require_start)
        df_add = self.player_data[self.player_data.player.isin(to_add)]
        for player, pos in df_add[['player', 'pos']].values:
            h_player_add[f'{player}'] = -1
            open_pos_require[pos] -= 1

        return h_player_add, open_pos_require


    @staticmethod
    def drop_players(df, to_drop):
        if not to_drop:  # Early return for empty list
            return df
        # Convert to set for O(1) lookups if it's a list
        to_drop_set = set(to_drop) if isinstance(to_drop, (list, tuple)) else to_drop
        return df[~df.player.isin(to_drop_set)].reset_index(drop=True)


    @staticmethod
    def player_matrix_mapping(df):
        idx_player_map = {}
        player_idx_map = {}
        for i, row in df.iterrows():
            idx_player_map[i] = {
                'player': row.player,
                'pos': row.pos
            }

            player_idx_map[f'{row.player}'] = i

        return idx_player_map, player_idx_map


    def create_position_constraint_matrix(self, predictions, pos_require, num_rounds):
        """Create position constraint matrix for equality constraints (exactly meet position requirements)"""
        num_players = len(predictions)
        
        # Remove FLEX from position requirements for now
        positions = [pos for pos in pos_require.keys() if pos != 'FLEX']
        num_positions = len(positions)
        
        if num_positions == 0 or num_rounds == 0:
            # If no position constraints, return empty matrices
            return np.zeros((0, num_players * num_rounds)), np.zeros((0, 1))
        
        A_position = np.zeros((num_positions, num_players * num_rounds))
        
        # Vectorized position assignment - avoid iterrows()
        player_positions = predictions['pos'].values
        
        for pos_idx, position in enumerate(positions):
            # Find all players with this position
            player_mask = player_positions == position
            player_indices = np.where(player_mask)[0]
            
            # For each player with this position, set coefficient to 1 across all rounds
            for player_idx in player_indices:
                var_indices = player_idx * num_rounds + np.arange(num_rounds)
                A_position[pos_idx, var_indices] = 1
        
        # Create b vector with exact position requirements
        b_position = np.array([pos_require[pos] for pos in positions]).reshape(-1, 1)
        
        return A_position, b_position
    
    def create_availability_constraints(self, adp_sample, predictions, adjusted_picks):
        """Create availability constraints for player-round combinations
        For the current round (first in adjusted_picks), all players are available.
        For future rounds, use ADP-based availability constraints.
        """
        num_players = len(predictions)
        num_rounds = len(adjusted_picks)
        
        if num_rounds == 0:
            return np.zeros((0, num_players * num_rounds)), np.zeros((0, 1))
        
        # Pre-allocate availability vector directly - avoid creating large identity matrix
        total_constraints = num_players * num_rounds
        h_availability = np.ones(total_constraints, dtype=np.float64)
        
        # Vectorized availability calculation - only update future rounds
        if num_rounds > 1:
            adp_array = adp_sample  # Already numpy array
            picks_array = np.array(adjusted_picks)
            
            # Use broadcasting to compute availability for all player-round combinations at once
            # Shape: (num_players, num_rounds-1) for rounds 1 to num_rounds-1
            player_indices = np.arange(num_players)
            for round_idx in range(1, num_rounds):
                pick_num = picks_array[round_idx]
                # Calculate flat indices for this round
                flat_indices = player_indices * num_rounds + round_idx
                # Set availability based on ADP
                h_availability[flat_indices] = (adp_array >= pick_num).astype(np.float64)
        
        return None, h_availability.reshape(-1, 1)  # Return None for G since we'll use pre-allocated identity
    
    def create_round_selection_constraints(self, num_players, num_rounds):
        """Create constraints to ensure exactly 1 player is selected per round"""
        if num_rounds == 0:
            return np.zeros((0, num_players * num_rounds)), np.zeros((0, 1))
        
        # Vectorized approach: create block matrix
        A_rounds = np.zeros((num_rounds, num_players * num_rounds))
        
        # For each round, set coefficients to 1 for all players in that round
        for round_idx in range(num_rounds):
            # All variables for this round: player_idx * num_rounds + round_idx
            var_indices = np.arange(num_players) * num_rounds + round_idx
            A_rounds[round_idx, var_indices] = 1
        
        b_rounds = np.ones((num_rounds, 1))  # Exactly 1 player per round
        
        return A_rounds, b_rounds
    
    def create_player_uniqueness_constraints(self, num_players, num_rounds):
        """Create constraints to ensure each player is selected at most once across all rounds"""
        if num_rounds == 0:
            return np.zeros((0, num_players * num_rounds)), np.zeros((0, 1))
        
        # Vectorized approach: create block diagonal-like structure
        G_players = np.zeros((num_players, num_players * num_rounds))
        
        # For each player, set coefficients to 1 for all rounds they could be selected
        for player_idx in range(num_players):
            # All variables for this player: player_idx * num_rounds + round_idx for all rounds
            var_indices = player_idx * num_rounds + np.arange(num_rounds)
            G_players[player_idx, var_indices] = 1
        
        h_players = np.ones((num_players, 1))  # At most 1 selection per player
        
        return G_players, h_players

    # Removed old constraint methods - using new matrix-based approach

    @staticmethod
    def sample_c_points(data, max_entries, num_avg_pts, num_rounds):
        """Create objective function coefficients for player-round variables"""
        labels = data[['player', 'pos']]
        
        # Pre-generate random column indices to avoid repeated calls
        col_indices = np.random.randint(2, max_entries+2, size=num_avg_pts)
        current_points = -1 * data.iloc[:, col_indices].mean(axis=1)
        
        # Vectorized expansion: repeat each player's points for all rounds
        expanded_points = np.repeat(current_points.values, num_rounds)
        
        return labels, expanded_points

    @staticmethod
    def _top_sum(pos_scores, num_starters):
        if pos_scores.shape[0] == 0 or num_starters <= 0:
            return np.zeros(pos_scores.shape[1], dtype=np.float32)

        k = min(num_starters, pos_scores.shape[0])
        return np.partition(pos_scores, -k, axis=0)[-k:].sum(axis=0)

    @staticmethod
    def _next_best(pos_scores, num_starters):
        if pos_scores.shape[0] <= num_starters:
            return np.zeros(pos_scores.shape[1], dtype=np.float32)

        sorted_scores = np.sort(pos_scores, axis=0)
        return sorted_scores[-num_starters - 1]

    @staticmethod
    def _pos_lineup_state(pos_scores, num_starters, num_weeks):
        if pos_scores.shape[0] == 0 or num_starters <= 0:
            zeros = np.zeros(num_weeks, dtype=np.float32)
            return zeros, zeros, zeros

        k = min(num_starters, pos_scores.shape[0])
        top_k = np.partition(pos_scores, -k, axis=0)[-k:]
        top_sum = top_k.sum(axis=0).astype(np.float32)

        if pos_scores.shape[0] >= num_starters:
            starter_threshold = top_k.min(axis=0).astype(np.float32)
        else:
            starter_threshold = np.zeros(num_weeks, dtype=np.float32)

        if pos_scores.shape[0] > num_starters:
            next_best = np.partition(pos_scores, -num_starters - 1, axis=0)[-num_starters - 1].astype(np.float32)
        else:
            next_best = np.zeros(num_weeks, dtype=np.float32)

        return top_sum, starter_threshold, next_best

    @classmethod
    def best_ball_weekly_scores(cls, scores, positions):
        """Score selected players as DraftKings best ball weekly lineups."""
        if len(scores) == 0:
            return np.zeros(0, dtype=np.float32)

        scores = np.asarray(scores, dtype=np.float32)
        positions = np.asarray(positions)
        num_weeks = scores.shape[1]
        weekly_score = np.zeros(num_weeks, dtype=np.float32)

        qb = scores[positions == 'QB']
        rb = scores[positions == 'RB']
        wr = scores[positions == 'WR']
        te = scores[positions == 'TE']

        weekly_score += cls._top_sum(qb, 1)
        weekly_score += cls._top_sum(rb, 2)
        weekly_score += cls._top_sum(wr, 3)
        weekly_score += cls._top_sum(te, 1)

        flex = np.maximum.reduce([
            cls._next_best(rb, 2),
            cls._next_best(wr, 3),
            cls._next_best(te, 1),
        ])
        weekly_score += flex

        return weekly_score

    @classmethod
    def marginal_best_ball_values(cls, selected_scores, selected_positions, candidate_scores, candidate_positions):
        """Estimate each candidate's marginal best-ball lineup value."""
        candidate_scores = np.asarray(candidate_scores, dtype=np.float32)
        candidate_positions = np.asarray(candidate_positions)
        if candidate_scores.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)

        num_weeks = candidate_scores.shape[1]
        selected_scores = np.asarray(selected_scores, dtype=np.float32)
        selected_positions = np.asarray(selected_positions)
        if selected_scores.shape[0] == 0:
            selected_scores = np.empty((0, num_weeks), dtype=np.float32)
            selected_positions = np.array([], dtype=object)

        starter_counts = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1}
        states = {}
        for pos, starter_count in starter_counts.items():
            pos_scores = selected_scores[selected_positions == pos]
            states[pos] = cls._pos_lineup_state(pos_scores, starter_count, num_weeks)

        rb_next = states['RB'][2]
        wr_next = states['WR'][2]
        te_next = states['TE'][2]
        base_flex = np.maximum.reduce([rb_next, wr_next, te_next])

        values = np.zeros(candidate_scores.shape[0], dtype=np.float32)
        for pos in starter_counts:
            pos_mask = candidate_positions == pos
            if not np.any(pos_mask):
                continue

            pos_candidate_scores = candidate_scores[pos_mask]
            _, pos_threshold, pos_next = states[pos]
            starter_delta = np.maximum(pos_candidate_scores - pos_threshold, 0)

            if pos == 'QB':
                weekly_delta = starter_delta
            else:
                new_pos_next = np.maximum(pos_next, np.minimum(pos_candidate_scores, pos_threshold))
                shape = pos_candidate_scores.shape
                flex_rb = new_pos_next if pos == 'RB' else np.broadcast_to(rb_next, shape)
                flex_wr = new_pos_next if pos == 'WR' else np.broadcast_to(wr_next, shape)
                flex_te = new_pos_next if pos == 'TE' else np.broadcast_to(te_next, shape)
                new_flex = np.maximum.reduce([flex_rb, flex_wr, flex_te])
                weekly_delta = starter_delta + (new_flex - base_flex)

            values[pos_mask] = weekly_delta.mean(axis=1)

        return values

    @staticmethod
    def _bank_pos_lineup_state(
        pos_scores,
        num_starters,
        num_scenarios,
        num_weeks,
    ):
        """Return lineup totals, starter threshold, and next-best for a score bank."""
        if pos_scores.shape[1] == 0 or num_starters <= 0:
            zeros = np.zeros((num_scenarios, num_weeks), dtype=np.float32)
            return zeros, zeros, zeros

        num_players = pos_scores.shape[1]
        k = min(num_starters, num_players)
        top_k = np.partition(pos_scores, -k, axis=1)[:, -k:, :]
        top_sum = top_k.sum(axis=1).astype(np.float32)

        if num_players >= num_starters:
            starter_threshold = top_k.min(axis=1).astype(np.float32)
        else:
            starter_threshold = np.zeros(
                (num_scenarios, num_weeks),
                dtype=np.float32,
            )

        if num_players > num_starters:
            next_best = np.partition(
                pos_scores,
                -num_starters - 1,
                axis=1,
            )[:, -num_starters - 1, :].astype(np.float32)
        else:
            next_best = np.zeros(
                (num_scenarios, num_weeks),
                dtype=np.float32,
            )

        return top_sum, starter_threshold, next_best

    @classmethod
    def marginal_best_ball_values_bank(
        cls,
        score_bank,
        player_positions,
        selected_indices,
        candidate_indices,
    ):
        """Return scenario-level and mean marginal values for many candidates."""
        score_bank = np.asarray(score_bank, dtype=np.float32)
        player_positions = np.asarray(player_positions)
        selected_indices = np.asarray(selected_indices, dtype=np.int64)
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        if score_bank.ndim != 3:
            raise ValueError("score_bank must have shape [scenario, player, week].")

        num_scenarios, _, num_weeks = score_bank.shape
        if len(candidate_indices) == 0:
            empty = np.zeros((num_scenarios, 0), dtype=np.float32)
            return empty, np.zeros(0, dtype=np.float32)

        starter_counts = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1}
        states = {}
        for pos, starter_count in starter_counts.items():
            pos_selected = selected_indices[
                player_positions[selected_indices] == pos
            ]
            pos_scores = score_bank[:, pos_selected, :]
            states[pos] = cls._bank_pos_lineup_state(
                pos_scores,
                starter_count,
                num_scenarios,
                num_weeks,
            )

        rb_next = states['RB'][2]
        wr_next = states['WR'][2]
        te_next = states['TE'][2]
        base_flex = np.maximum.reduce([rb_next, wr_next, te_next])

        candidate_positions = player_positions[candidate_indices]
        scenario_values = np.zeros(
            (num_scenarios, len(candidate_indices)),
            dtype=np.float32,
        )
        for pos in starter_counts:
            candidate_locs = np.where(candidate_positions == pos)[0]
            if len(candidate_locs) == 0:
                continue

            pos_candidates = candidate_indices[candidate_locs]
            pos_scores = score_bank[:, pos_candidates, :]
            _, pos_threshold, pos_next = states[pos]
            starter_delta = np.maximum(
                pos_scores - pos_threshold[:, None, :],
                0,
            )

            if pos == 'QB':
                weekly_delta = starter_delta
            else:
                new_pos_next = np.maximum(
                    pos_next[:, None, :],
                    np.minimum(pos_scores, pos_threshold[:, None, :]),
                )
                flex_rb = (
                    new_pos_next
                    if pos == 'RB'
                    else np.broadcast_to(rb_next[:, None, :], pos_scores.shape)
                )
                flex_wr = (
                    new_pos_next
                    if pos == 'WR'
                    else np.broadcast_to(wr_next[:, None, :], pos_scores.shape)
                )
                flex_te = (
                    new_pos_next
                    if pos == 'TE'
                    else np.broadcast_to(te_next[:, None, :], pos_scores.shape)
                )
                new_flex = np.maximum.reduce([flex_rb, flex_wr, flex_te])
                weekly_delta = (
                    starter_delta
                    + new_flex
                    - base_flex[:, None, :]
                )

            scenario_values[:, candidate_locs] = weekly_delta.sum(axis=2)

        return scenario_values, scenario_values.mean(axis=0)

    @classmethod
    def best_ball_roster_scores_bank(
        cls,
        score_bank,
        player_positions,
        selected_indices,
    ):
        """Score one roster in every season scenario from a weekly score bank."""
        score_bank = np.asarray(score_bank, dtype=np.float32)
        player_positions = np.asarray(player_positions)
        selected_indices = np.asarray(selected_indices, dtype=np.int64)
        if score_bank.ndim != 3:
            raise ValueError("score_bank must have shape [scenario, player, week].")

        num_scenarios, _, num_weeks = score_bank.shape
        weekly_score = np.zeros(
            (num_scenarios, num_weeks),
            dtype=np.float32,
        )
        starter_counts = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1}
        states = {}
        for pos, starter_count in starter_counts.items():
            pos_selected = selected_indices[
                player_positions[selected_indices] == pos
            ]
            state = cls._bank_pos_lineup_state(
                score_bank[:, pos_selected, :],
                starter_count,
                num_scenarios,
                num_weeks,
            )
            states[pos] = state
            weekly_score += state[0]

        weekly_score += np.maximum.reduce([
            states['RB'][2],
            states['WR'][2],
            states['TE'][2],
        ])
        return weekly_score.sum(axis=1)

    def sequential_legal_candidate_indices(
        self,
        remaining,
        player_positions,
        selected_indices,
        picks_left,
        pos_ranges=None,
    ):
        """Candidates that can still lead to a legal final roster construction."""
        remaining = np.asarray(remaining, dtype=bool)
        player_positions = np.asarray(player_positions)
        selected_indices = np.asarray(selected_indices, dtype=np.int64)
        if picks_left <= 0:
            return np.zeros(0, dtype=np.int64)

        if pos_ranges is None:
            selected_mask = np.zeros(len(player_positions), dtype=bool)
            selected_mask[selected_indices] = True
            pos_ranges = self.best_ball_position_ranges(
                player_positions,
                selected_mask,
            )

        candidate_indices = np.flatnonzero(remaining)
        if len(candidate_indices) == 0:
            return np.zeros(0, dtype=np.int64)

        positions = np.asarray(tuple(pos_ranges), dtype=object)
        minimums = np.asarray(
            [pos_ranges[pos][0] for pos in positions],
            dtype=np.int64,
        )
        maximums = np.asarray(
            [pos_ranges[pos][1] for pos in positions],
            dtype=np.int64,
        )
        selected_positions = player_positions[selected_indices]
        current_counts = np.sum(
            selected_positions[:, None] == positions[None, :],
            axis=0,
        ).astype(np.int64)
        candidate_position_matrix = (
            player_positions[candidate_indices, None] == positions[None, :]
        )
        recognized_positions = candidate_position_matrix.any(axis=1)
        candidate_counts = (
            current_counts[None, :]
            + candidate_position_matrix.astype(np.int64)
        )
        future_slots = picks_left - 1
        minimum_deficits = np.maximum(
            minimums[None, :] - candidate_counts,
            0,
        ).sum(axis=1)
        maximum_capacity = np.maximum(
            maximums[None, :] - candidate_counts,
            0,
        ).sum(axis=1)
        legal_mask = (
            recognized_positions
            & np.all(candidate_counts <= maximums[None, :], axis=1)
            & (minimum_deficits <= future_slots)
            & (future_slots <= maximum_capacity)
        )
        return candidate_indices[legal_mask]

    @staticmethod
    def build_sequential_draft_orders(
        adp_matrix,
        num_rooms,
        seed=None,
    ):
        """Sample noisy-ADP opponent priority orders shared by all root candidates."""
        adp_matrix = np.asarray(adp_matrix, dtype=np.float64)
        if adp_matrix.ndim != 2 or adp_matrix.shape[1] == 0:
            raise ValueError("adp_matrix must have shape [player, sample].")
        if num_rooms <= 0:
            raise ValueError("num_rooms must be positive.")

        rng = np.random.default_rng(seed)
        adp_cols = rng.integers(0, adp_matrix.shape[1], size=num_rooms)
        orders = np.empty((num_rooms, adp_matrix.shape[0]), dtype=np.int64)
        for room_idx, adp_col in enumerate(adp_cols):
            noisy_adp = (
                np.maximum(adp_matrix[:, adp_col], 1.0)
                + rng.uniform(-0.01, 0.01, size=adp_matrix.shape[0])
            )
            orders[room_idx] = np.argsort(noisy_adp, kind='mergesort')
        return orders, adp_cols

    @staticmethod
    def advance_sequential_opponents(
        remaining,
        draft_order,
        order_pointer,
        num_opponent_picks,
    ):
        """Remove the next available players from one opponent priority order."""
        drafted = []
        for _ in range(max(0, int(num_opponent_picks))):
            while (
                order_pointer < len(draft_order)
                and not remaining[draft_order[order_pointer]]
            ):
                order_pointer += 1
            if order_pointer >= len(draft_order):
                break

            chosen_idx = int(draft_order[order_pointer])
            remaining[chosen_idx] = False
            drafted.append(chosen_idx)
            order_pointer += 1

        return order_pointer, np.asarray(drafted, dtype=np.int64)

    @staticmethod
    def sequential_survival_probabilities(
        candidate_indices,
        adp_matrix,
        current_pick,
        next_pick,
    ):
        if next_pick is None:
            return np.ones(len(candidate_indices), dtype=np.float32)
        candidate_adp = adp_matrix[candidate_indices]
        survives_current = np.sum(candidate_adp >= current_pick, axis=1)
        survives_next = np.sum(candidate_adp >= next_pick, axis=1)
        return np.divide(
            survives_next,
            survives_current,
            out=np.zeros(len(candidate_indices), dtype=np.float32),
            where=survives_current > 0,
        ).astype(np.float32)

    @classmethod
    def build_sequential_survival_table(cls, adp_matrix, adjusted_picks):
        """Precompute conditional next-pick survival for every player and turn."""
        adp_matrix = np.asarray(adp_matrix)
        survival_table = np.ones(
            (len(adjusted_picks), adp_matrix.shape[0]),
            dtype=np.float32,
        )
        for pick_idx in range(len(adjusted_picks) - 1):
            all_players = np.arange(adp_matrix.shape[0], dtype=np.int64)
            survival_table[pick_idx] = cls.sequential_survival_probabilities(
                all_players,
                adp_matrix,
                adjusted_picks[pick_idx],
                adjusted_picks[pick_idx + 1],
            )
        return survival_table

    @classmethod
    def sequential_policy_scores(
        cls,
        candidate_indices,
        immediate_values,
        player_positions,
        adp_matrix,
        current_pick,
        next_pick,
        scarcity_weight=SEQUENTIAL_SCARCITY_WEIGHT,
        urgency_weight=SEQUENTIAL_URGENCY_WEIGHT,
        survival_probabilities=None,
    ):
        """Combine expected marginal value with the cost of waiting one turn."""
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        immediate_values = np.asarray(immediate_values, dtype=np.float32)
        if survival_probabilities is None:
            survival = cls.sequential_survival_probabilities(
                candidate_indices,
                adp_matrix,
                current_pick,
                next_pick,
            )
        else:
            survival = np.asarray(
                survival_probabilities,
                dtype=np.float32,
            )[candidate_indices]
        urgency = immediate_values * (1.0 - survival)
        future_opportunity = immediate_values * survival
        scarcity_cliff = np.zeros(len(candidate_indices), dtype=np.float32)
        candidate_positions = player_positions[candidate_indices]

        for pos in ('QB', 'RB', 'WR', 'TE'):
            pos_locs = np.where(candidate_positions == pos)[0]
            if len(pos_locs) == 0:
                continue
            best_future = float(np.max(future_opportunity[pos_locs]))
            scarcity_cliff[pos_locs] = np.maximum(
                immediate_values[pos_locs] - best_future,
                0,
            )

        policy_score = (
            immediate_values
            + (float(scarcity_weight) * scarcity_cliff)
            + (float(urgency_weight) * urgency)
        )
        return policy_score, survival, scarcity_cliff, urgency

    @staticmethod
    def select_sequential_root_candidates(
        candidate_indices,
        immediate_values,
        policy_scores,
        survival,
        player_positions,
        candidate_pool_size,
    ):
        """Build a broad, position-diverse root set without evaluating every player."""
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        candidate_pool_size = min(
            max(1, int(candidate_pool_size)),
            len(candidate_indices),
        )
        if len(candidate_indices) <= candidate_pool_size:
            return candidate_indices, {
                int(idx): ('all_legal',) for idx in candidate_indices
            }

        reason_sets = {int(idx): set() for idx in candidate_indices}
        quota = max(4, candidate_pool_size // 3)
        rankings = {
            'top_immediate': np.argsort(-immediate_values),
            'top_screen': np.argsort(-policy_scores),
            'low_survival': np.argsort(survival),
        }
        for reason, ranking in rankings.items():
            for loc in ranking[:quota]:
                reason_sets[int(candidate_indices[loc])].add(reason)

        candidate_positions = player_positions[candidate_indices]
        protected = []
        for pos in ('QB', 'RB', 'WR', 'TE'):
            pos_locs = np.where(candidate_positions == pos)[0]
            if len(pos_locs) == 0:
                continue
            best_locs = pos_locs[np.argsort(-policy_scores[pos_locs])[:2]]
            for loc in best_locs:
                idx = int(candidate_indices[loc])
                protected.append(idx)
                reason_sets[idx].add(f'top_{pos.lower()}')

        loc_by_idx = {
            int(candidate_idx): loc
            for loc, candidate_idx in enumerate(candidate_indices)
        }
        chosen = []
        for candidate_idx in protected:
            if candidate_idx not in chosen and len(chosen) < candidate_pool_size:
                chosen.append(candidate_idx)

        reason_candidates = [
            int(candidate_idx)
            for candidate_idx in candidate_indices
            if reason_sets[int(candidate_idx)]
        ]
        reason_candidates.sort(
            key=lambda idx: policy_scores[loc_by_idx[idx]],
            reverse=True,
        )
        for candidate_idx in reason_candidates:
            if candidate_idx not in chosen and len(chosen) < candidate_pool_size:
                chosen.append(candidate_idx)

        if len(chosen) < candidate_pool_size:
            for loc in np.argsort(-policy_scores):
                candidate_idx = int(candidate_indices[loc])
                if candidate_idx not in chosen:
                    chosen.append(candidate_idx)
                if len(chosen) == candidate_pool_size:
                    break

        reasons = {
            candidate_idx: tuple(sorted(reason_sets[candidate_idx]))
            for candidate_idx in chosen
        }
        return np.asarray(chosen, dtype=np.int64), reasons

    @classmethod
    def best_ball_total_score(cls, selected_indices, weekly_scores, player_positions):
        if len(selected_indices) == 0:
            return 0

        weekly_score = cls.best_ball_weekly_scores(
            weekly_scores[selected_indices],
            player_positions[selected_indices]
        )
        return float(weekly_score.sum())

    @staticmethod
    def open_position_mask(player_positions, pos_require_adjusted):
        return np.array([
            pos_require_adjusted.get(pos, 0) > 0
            for pos in player_positions
        ])

    def get_candidate_indices(self, selected_set, player_positions, pos_require_adjusted, adp_sample, pick_num, use_adp):
        candidate_mask = np.ones(len(player_positions), dtype=bool)
        if selected_set:
            candidate_mask[list(selected_set)] = False

        candidate_mask &= self.open_position_mask(player_positions, pos_require_adjusted)

        if use_adp:
            candidate_mask &= adp_sample >= pick_num

        return np.where(candidate_mask)[0]

    @staticmethod
    def select_rollout_candidates(current_candidates, immediate_values, adp_sample, player_positions, candidate_pool_size):
        if len(current_candidates) <= candidate_pool_size:
            return current_candidates

        pool_parts = [
            current_candidates[np.argsort(-immediate_values)[:candidate_pool_size]],
            current_candidates[np.argsort(adp_sample[current_candidates])[:candidate_pool_size]],
        ]

        per_pos_quota = max(2, candidate_pool_size // 8)
        current_positions = player_positions[current_candidates]
        for pos in ['QB', 'RB', 'WR', 'TE']:
            pos_locs = np.where(current_positions == pos)[0]
            if len(pos_locs) == 0:
                continue

            pos_candidates = current_candidates[pos_locs]
            pos_values = immediate_values[pos_locs]
            pool_parts.append(pos_candidates[np.argsort(-pos_values)[:per_pos_quota]])
            pool_parts.append(pos_candidates[np.argsort(adp_sample[pos_candidates])[:per_pos_quota]])

        return np.unique(np.concatenate(pool_parts))

    def complete_greedy_best_ball_path(
        self,
        selected_indices,
        pos_require_adjusted,
        start_round_idx,
        adjusted_picks,
        weekly_scores,
        player_positions,
        adp_sample,
    ):
        selected_indices = list(selected_indices)
        selected_set = set(selected_indices)
        pos_require_adjusted = copy.deepcopy(pos_require_adjusted)
        path_picks = []
        path_available = []
        num_weeks = weekly_scores.shape[1]

        for round_idx in range(start_round_idx, len(adjusted_picks)):
            candidate_indices = self.get_candidate_indices(
                selected_set,
                player_positions,
                pos_require_adjusted,
                adp_sample,
                adjusted_picks[round_idx],
                use_adp=round_idx > 0
            )

            if len(candidate_indices) == 0:
                return None, None, False

            selected_scores = (
                weekly_scores[selected_indices]
                if selected_indices else np.empty((0, num_weeks), dtype=np.float32)
            )
            selected_positions = (
                player_positions[selected_indices]
                if selected_indices else np.array([], dtype=object)
            )
            values = self.marginal_best_ball_values(
                selected_scores,
                selected_positions,
                weekly_scores[candidate_indices],
                player_positions[candidate_indices]
            )

            pick_idx = candidate_indices[int(np.argmax(values))]
            path_picks.append(pick_idx)
            path_available.append(candidate_indices)

            selected_indices.append(pick_idx)
            selected_set.add(pick_idx)
            pick_pos = player_positions[pick_idx]
            if pick_pos in pos_require_adjusted and pos_require_adjusted[pick_pos] > 0:
                pos_require_adjusted[pick_pos] -= 1

        return path_picks, path_available, True

    def order_path_by_availability(
        self,
        path_picks,
        initial_selected_indices,
        pos_require_adjusted,
        adjusted_picks,
        player_positions,
        adp_sample,
    ):
        """Order a completed roster path by who is least likely to last to the next pick."""
        remaining_picks = list(path_picks)
        selected_set = set(initial_selected_indices)
        pos_require_remaining = copy.deepcopy(pos_require_adjusted)
        ordered_picks = []
        ordered_available = []

        for round_idx, pick_num in enumerate(adjusted_picks):
            available_indices = self.get_candidate_indices(
                selected_set,
                player_positions,
                pos_require_remaining,
                adp_sample,
                pick_num,
                use_adp=round_idx > 0
            )
            available_set = set(available_indices)
            feasible_picks = [idx for idx in remaining_picks if idx in available_set]
            if len(feasible_picks) == 0:
                return path_picks, ordered_available, False

            pick_idx = min(feasible_picks, key=lambda idx: adp_sample[idx])
            ordered_picks.append(pick_idx)
            ordered_available.append(available_indices)

            remaining_picks.remove(pick_idx)
            selected_set.add(pick_idx)
            pick_pos = player_positions[pick_idx]
            if pick_pos in pos_require_remaining and pos_require_remaining[pick_pos] > 0:
                pos_require_remaining[pick_pos] -= 1

        return ordered_picks, ordered_available, True

    @staticmethod
    def solve_ilp(c, G, h, A, b):
    
        (status, x) = ilp(c, G, h, A=A, b=b, B=set(range(0, len(c))))

        return status, x


    # Removed old tally method - using inline tracking in run_sim

    
    def final_results(self, player_selections, success_trials):
        # Calculate percentages for total and by round
        for k, _ in player_selections.items():
            total_available = player_selections[k]['total_available_count']
            if total_available > 0:
                player_selections[k]['pct_selected_when_available'] = 100 * player_selections[k]['total_counts'] / total_available
            else:
                player_selections[k]['pct_selected_when_available'] = 0
            
            # Calculate round-specific percentages
            for i in range(self.num_rounds):
                round_available = player_selections[k][f'round_{i+1}_available']
                round_count = player_selections[k][f'round_{i+1}_count']
                if round_available > 0:
                    player_selections[k][f'round_{i+1}_pct'] = 100 * round_count / round_available
                else:
                    player_selections[k][f'round_{i+1}_pct'] = 0
                
        results = pd.DataFrame(player_selections).T
        
        # Rename columns for clarity
        column_renames = {
            'total_counts': 'TotalSelectionCounts',
            'total_available_count': 'TotalAvailableCount', 
            'pct_selected_when_available': 'PctSelectedWhenAvailable'
        }
        
        # Add round-specific column renames
        for i in range(self.num_rounds):
            column_renames[f'round_{i+1}_count'] = f'Round{i+1}Count'
            column_renames[f'round_{i+1}_available'] = f'Round{i+1}Available'
            column_renames[f'round_{i+1}_pct'] = f'Round{i+1}Pct'
        
        results = results.rename(columns=column_renames)
        results = results.sort_values(by='TotalSelectionCounts', ascending=False)
        results = results.reset_index().rename(columns={'index': 'player'})
        results['PctSelected'] = 100*np.round(results.TotalSelectionCounts / success_trials, 3)
        return results


    @staticmethod
    @contextlib.contextmanager
    def temp_seed(seed):
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(state)

    @staticmethod
    def best_ball_slot_specs():
        return [
            ('QB', ('QB',), 1),
            ('RB', ('RB',), 2),
            ('WR', ('WR',), 3),
            ('TE', ('TE',), 1),
            ('FLEX', ('RB', 'WR', 'TE'), 1),
        ]

    @staticmethod
    def default_best_ball_position_ranges():
        return {
            'QB': (2, 3),
            'RB': (5, 7),
            'WR': (7, 9),
            'TE': (2, 3),
        }

    def best_ball_position_ranges(self, player_positions=None, selected_mask=None):
        pos_ranges = self.position_ranges or self.default_best_ball_position_ranges()
        if player_positions is None or selected_mask is None:
            return pos_ranges

        adjusted_ranges = {}
        for pos, (min_count, max_count) in pos_ranges.items():
            current_count = int(np.sum((player_positions == pos) & selected_mask))
            adjusted_ranges[pos] = (min_count, max(max_count, current_count))

        return adjusted_ranges

    def build_best_ball_ilp_model(self, predictions, to_add_set, adjusted_picks, num_weeks):
        player_names = predictions.player.values
        player_positions = predictions.pos.values
        if 'team' in predictions.columns:
            player_teams = predictions.team.fillna('').astype(str).str.strip().values
        else:
            player_teams = np.array([''] * len(predictions), dtype=object)
        player_ppg = (
            predictions[self.sample_value_columns(predictions)]
            .mean(axis=1)
            .fillna(0)
            .to_numpy(dtype=np.float64)
        )
        num_players = len(predictions)
        num_rounds = len(adjusted_picks)

        selected_mask = np.array([p in to_add_set for p in player_names])
        pos_ranges = self.best_ball_position_ranges(player_positions, selected_mask)
        draft_pool_indices = np.where(~selected_mask)[0]
        num_draftable = len(draft_pool_indices)

        full_x_count = num_draftable * num_rounds
        x_pick_buffer = X_PICK_BUFFER
        pick_nums = np.array(adjusted_picks, dtype=np.float64)
        adp_bounds = (
            self.player_data[['player', 'adp_min_pick', 'adp_max_pick']]
            .drop_duplicates('player')
            .set_index('player')
            .reindex(player_names)
        )
        total_draft_picks = self.num_teams * self.num_rounds
        adp_min_pick = adp_bounds.adp_min_pick.fillna(1).to_numpy(dtype=np.float64)
        adp_max_pick = adp_bounds.adp_max_pick.fillna(total_draft_picks + x_pick_buffer).to_numpy(dtype=np.float64)

        valid_x_mask = np.zeros((num_draftable, num_rounds), dtype=bool)
        if num_rounds > 0:
            valid_x_mask[:, 0] = True
        if num_rounds > 1 and num_draftable > 0:
            draft_min = adp_min_pick[draft_pool_indices].reshape(-1, 1) - x_pick_buffer
            draft_max = adp_max_pick[draft_pool_indices].reshape(-1, 1) + x_pick_buffer
            future_picks = pick_nums[1:].reshape(1, -1)
            valid_x_mask[:, 1:] = (future_picks >= draft_min) & (future_picks <= draft_max)

        for round_idx in range(num_rounds):
            if not valid_x_mask[:, round_idx].any():
                valid_x_mask[:, round_idx] = True

        x_count = int(valid_x_mask.sum())
        y_offset = x_count
        start_offset = y_offset + num_players
        next_var = start_offset

        lineup_positions = ('QB', 'RB', 'WR', 'TE')
        lineup_player_indices = np.where(np.isin(player_positions, lineup_positions))[0]
        start_var_by_player_week = np.full((num_players, num_weeks), -1, dtype=int)
        objective_start_entries = []

        for week in range(num_weeks):
            for player_idx in lineup_player_indices:
                var_idx = next_var
                next_var += 1
                start_var_by_player_week[player_idx, week] = var_idx
                objective_start_entries.append((var_idx, player_idx, week))

        stack_pair_entries = []
        stack_score_entries = []
        if (
            self.use_stack_bonus
            and self.stack_bonus_pct > 0
            and self.stack_pair_cap > 0
            and self.stack_team_cap > 0
        ):
            qb_indices = np.where(player_positions == 'QB')[0]
            pass_catcher_indices = np.where(np.isin(player_positions, ('WR', 'TE')))[0]
            stack_pairs_by_qb_team = {}

            for qb_idx in qb_indices:
                qb_team = player_teams[qb_idx]
                if qb_team == '':
                    continue

                same_team_receivers = [
                    rec_idx for rec_idx in pass_catcher_indices
                    if player_teams[rec_idx] == qb_team
                ]
                for rec_idx in same_team_receivers:
                    raw_bonus = self.stack_bonus_pct * (player_ppg[qb_idx] + player_ppg[rec_idx])
                    pair_bonus = min(self.stack_pair_cap, raw_bonus)
                    if pair_bonus <= 0:
                        continue

                    pair_var_idx = next_var
                    next_var += 1
                    stack_pair_entries.append(
                        (pair_var_idx, qb_idx, rec_idx, qb_team, float(pair_bonus))
                    )
                    stack_pairs_by_qb_team.setdefault((qb_idx, qb_team), []).append(
                        (pair_var_idx, float(pair_bonus))
                    )

            for (qb_idx, qb_team), pair_entries in stack_pairs_by_qb_team.items():
                score_var_idx = next_var
                next_var += 1
                stack_score_entries.append((score_var_idx, qb_idx, qb_team, pair_entries))

        num_vars = next_var
        start_var_indices = start_var_by_player_week[start_var_by_player_week >= 0]
        x_var_indices = np.full((num_draftable, num_rounds), -1, dtype=int)
        if x_count:
            x_var_indices[valid_x_mask] = np.arange(x_count)
        y_var_indices = y_offset + np.arange(num_players)

        a_i = []
        a_j = []
        a_v = []
        b_vals = []

        def add_a_row(coeffs, rhs):
            row_idx = len(b_vals)
            for col_idx, value in coeffs:
                if value != 0:
                    a_i.append(row_idx)
                    a_j.append(int(col_idx))
                    a_v.append(float(value))
            b_vals.append(float(rhs))

        # Exactly one pick in each remaining round.
        for round_idx in range(num_rounds):
            round_vars = [
                (x_var_indices[draft_idx, round_idx], 1.0)
                for draft_idx in range(num_draftable)
                if x_var_indices[draft_idx, round_idx] >= 0
            ]
            add_a_row(
                round_vars,
                1.0
            )

        # Link draft assignment variables to final roster variables.
        for draft_idx, player_idx in enumerate(draft_pool_indices):
            coeffs = [(y_var_indices[player_idx], 1.0)]
            coeffs.extend(
                (x_var_indices[draft_idx, round_idx], -1.0)
                for round_idx in range(num_rounds)
                if x_var_indices[draft_idx, round_idx] >= 0
            )
            add_a_row(coeffs, 0.0)

        # Already drafted players are fixed on the final roster.
        for player_idx in np.where(selected_mask)[0]:
            add_a_row([(y_var_indices[player_idx], 1.0)], 1.0)

        # Final roster size is fixed, while position counts are constrained by ranges below.
        add_a_row([(y_var_indices[player_idx], 1.0) for player_idx in range(num_players)], self.num_rounds)

        # Weekly best-ball lineup: 1 QB plus 2 RB, 3 WR, 1 TE, and one RB/WR/TE flex.
        for week in range(num_weeks):
            qb_vars = [
                (start_var_by_player_week[player_idx, week], 1.0)
                for player_idx in np.where(player_positions == 'QB')[0]
                if start_var_by_player_week[player_idx, week] >= 0
            ]
            skill_vars = [
                (start_var_by_player_week[player_idx, week], 1.0)
                for player_idx in np.where(np.isin(player_positions, ('RB', 'WR', 'TE')))[0]
                if start_var_by_player_week[player_idx, week] >= 0
            ]
            add_a_row(qb_vars, 1.0)
            add_a_row(skill_vars, 7.0)

        g_i = []
        g_j = []
        g_v = []
        h_static = []
        availability_row_start = None

        def add_g_row(coeffs, rhs):
            row_idx = len(h_static)
            for col_idx, value in coeffs:
                if value != 0:
                    g_i.append(row_idx)
                    g_j.append(int(col_idx))
                    g_v.append(float(value))
            h_static.append(float(rhs))

        # A player can only start if rostered.
        for player_idx in range(num_players):
            y_idx = y_var_indices[player_idx]
            for week in range(num_weeks):
                start_idx = start_var_by_player_week[player_idx, week]
                if start_idx < 0:
                    continue
                add_g_row([(start_idx, 1.0), (y_idx, -1.0)], 0.0)

        # Positional minimums create the non-QB flex structure with the weekly skill total above.
        for week in range(num_weeks):
            for pos, min_count in {'RB': 2, 'WR': 3, 'TE': 1}.items():
                pos_start_vars = [
                    (start_var_by_player_week[player_idx, week], -1.0)
                    for player_idx in np.where(player_positions == pos)[0]
                    if start_var_by_player_week[player_idx, week] >= 0
                ]
                add_g_row(pos_start_vars, -min_count)

        # Position construction ranges for the final roster.
        for pos, (min_count, max_count) in pos_ranges.items():
            pos_players = np.where(player_positions == pos)[0]
            pos_coeffs = [(y_var_indices[player_idx], 1.0) for player_idx in pos_players]
            add_g_row(pos_coeffs, max_count)
            add_g_row([(col_idx, -1.0) for col_idx, _ in pos_coeffs], -min_count)

        # Final roster indicators are continuous, but linked to binary draft variables.
        # Keep explicit bounds so y remains a valid 0/1 roster indicator.
        for player_idx in range(num_players):
            y_idx = y_var_indices[player_idx]
            add_g_row([(y_idx, 1.0)], 1.0)
            add_g_row([(y_idx, -1.0)], 0.0)

        # Stack bonus variables are continuous. Pair variables are forced to 1
        # when both same-team QB/WR-TE players are rostered, then a capped score
        # variable turns the pair bonuses into objective points.
        for pair_var_idx, qb_idx, rec_idx, _, _ in stack_pair_entries:
            add_g_row([(pair_var_idx, 1.0), (y_var_indices[qb_idx], -1.0)], 0.0)
            add_g_row([(pair_var_idx, 1.0), (y_var_indices[rec_idx], -1.0)], 0.0)
            add_g_row(
                [
                    (y_var_indices[qb_idx], 1.0),
                    (y_var_indices[rec_idx], 1.0),
                    (pair_var_idx, -1.0),
                ],
                1.0,
            )
            add_g_row([(pair_var_idx, -1.0)], 0.0)

        for score_var_idx, _, _, pair_entries in stack_score_entries:
            add_g_row([(score_var_idx, 1.0)], self.stack_team_cap)
            stack_score_coeffs = [(score_var_idx, 1.0)]
            stack_score_coeffs.extend(
                (pair_var_idx, -pair_bonus)
                for pair_var_idx, pair_bonus in pair_entries
            )
            add_g_row(stack_score_coeffs, 0.0)
            add_g_row([(score_var_idx, -1.0)], 0.0)

        availability_row_start = len(h_static)
        availability_entries = []
        for draft_idx in range(num_draftable):
            for round_idx in range(num_rounds):
                x_idx = x_var_indices[draft_idx, round_idx]
                if x_idx < 0:
                    continue
                add_g_row([(x_idx, 1.0)], 1.0)
                availability_entries.append((draft_idx, round_idx))

        # Weekly start variables are relaxed continuous variables, so they need explicit lower bounds.
        for var_idx in start_var_indices:
            add_g_row([(var_idx, -1.0)], 0.0)

        A = spmatrix(a_v, a_i, a_j, (len(b_vals), num_vars), tc='d')
        b = matrix(b_vals, tc='d')
        G = spmatrix(g_v, g_i, g_j, (len(h_static), num_vars), tc='d')
        h_template = np.array(h_static, dtype=np.float64)
        binary_vars = set(range(x_count))
        objective_start_entries = np.array(objective_start_entries, dtype=np.int64)
        objective_var_indices = objective_start_entries[:, 0]
        objective_player_indices = objective_start_entries[:, 1]
        objective_week_indices = objective_start_entries[:, 2]
        availability_entries = np.array(availability_entries, dtype=np.int64).reshape(-1, 2)
        availability_slice_end = availability_row_start + len(availability_entries)

        return {
            'A': A,
            'b': b,
            'G': G,
            'h_template': h_template,
            'availability_row_start': availability_row_start,
            'availability_slice_end': availability_slice_end,
            'num_vars': num_vars,
            'num_players': num_players,
            'num_rounds': num_rounds,
            'num_draftable': num_draftable,
            'x_count': x_count,
            'full_x_count': full_x_count,
            'x_pruned_count': full_x_count - x_count,
            'x_pick_buffer': x_pick_buffer,
            'start_offset': start_offset,
            'start_var_indices': start_var_indices,
            'start_var_by_player_week': start_var_by_player_week,
            'start_var_count': int(len(start_var_indices)),
            'pos_ranges': pos_ranges,
            'draft_pool_indices': draft_pool_indices,
            'x_var_indices': x_var_indices,
            'valid_x_mask': valid_x_mask,
            'availability_entries': availability_entries,
            'future_pick_nums': np.array(adjusted_picks[1:], dtype=np.float64),
            'player_names': player_names,
            'player_positions': player_positions,
            'player_teams': player_teams,
            'objective_var_indices': objective_var_indices,
            'objective_player_indices': objective_player_indices,
            'objective_week_indices': objective_week_indices,
            'stack_pair_entries': stack_pair_entries,
            'stack_score_var_indices': np.array(
                [entry[0] for entry in stack_score_entries],
                dtype=np.int64,
            ),
            'stack_pair_count': int(len(stack_pair_entries)),
            'stack_score_count': int(len(stack_score_entries)),
            'stack_bonus_pct': float(self.stack_bonus_pct if self.use_stack_bonus else 0),
            'stack_pair_cap': float(self.stack_pair_cap if self.use_stack_bonus else 0),
            'stack_team_cap': float(self.stack_team_cap if self.use_stack_bonus else 0),
            'c_template': np.zeros(num_vars, dtype=np.float64),
            'binary_vars': binary_vars,
        }

    @staticmethod
    def best_ball_ilp_objective(model, weekly_scores):
        c_vals = model['c_template'].copy()
        c_vals[model['objective_var_indices']] = -weekly_scores[
            model['objective_player_indices'],
            model['objective_week_indices']
        ]
        stack_score_var_indices = model.get('stack_score_var_indices', np.array([], dtype=np.int64))
        if len(stack_score_var_indices) > 0:
            c_vals[stack_score_var_indices] = -1.0
        return matrix(c_vals, tc='d')

    @staticmethod
    def best_ball_ilp_availability(model, adp_sample=None, adjusted_picks=None, availability=None):
        num_draftable = model['num_draftable']
        num_rounds = model['num_rounds']
        if availability is None:
            draft_pool_indices = model['draft_pool_indices']
            availability = np.ones((num_draftable, num_rounds), dtype=np.float64)

            if num_rounds > 1:
                availability[:, 1:] = (
                    adp_sample[draft_pool_indices].reshape(-1, 1) >= model['future_pick_nums'].reshape(1, -1)
                ).astype(np.float64)
        else:
            availability = np.asarray(availability, dtype=np.float64)

        h_vals = model['h_template'].copy()
        availability_entries = model['availability_entries']
        if len(availability_entries) > 0:
            h_vals[model['availability_row_start']:model['availability_slice_end']] = availability[
                availability_entries[:, 0],
                availability_entries[:, 1]
            ]
        return availability, matrix(h_vals, tc='d')

    def simulate_opponent_draft_availability(
        self,
        model,
        num_full_players,
        model_full_indices,
        selected_full_indices,
        full_adp_sample,
        adjusted_picks,
    ):
        """Simulate opponent picks from sampled ADP order, then project availability onto ILP players."""
        num_draftable = model['num_draftable']
        num_rounds = model['num_rounds']
        availability = np.ones((num_draftable, num_rounds), dtype=np.float64)

        full_adp_sample = np.asarray(full_adp_sample, dtype=np.float64)
        draft_order = np.argsort(
            np.maximum(full_adp_sample, 1.0) + np.random.uniform(-0.01, 0.01, size=len(full_adp_sample)),
            kind='mergesort',
        )
        draft_order_idx = 0

        remaining = np.ones(num_full_players, dtype=bool)
        if len(selected_full_indices) > 0:
            remaining[selected_full_indices] = False

        for round_idx in range(1, num_rounds):
            prev_pick = adjusted_picks[round_idx - 1]
            cur_pick = adjusted_picks[round_idx]

            for _ in range(prev_pick + 1, cur_pick):
                while draft_order_idx < len(draft_order) and not remaining[draft_order[draft_order_idx]]:
                    draft_order_idx += 1

                if draft_order_idx >= len(draft_order):
                    break

                chosen_idx = draft_order[draft_order_idx]
                remaining[chosen_idx] = False
                draft_order_idx += 1

            availability[:, round_idx] = remaining[model_full_indices].astype(np.float64)

        return availability

    def solve_best_ball_ilp(self, model, weekly_scores, adp_sample, adjusted_picks, availability=None):
        availability, h = self.best_ball_ilp_availability(model, adp_sample, adjusted_picks, availability)
        c = self.best_ball_ilp_objective(model, weekly_scores)
        status, x = ilp(
            c,
            model['G'],
            h,
            A=model['A'],
            b=model['b'],
            B=model['binary_vars'],
        )
        return status, x, availability

    def solve_forced_current_pick_best_ball_ilp(
        self,
        model,
        weekly_scores,
        adp_sample,
        adjusted_picks,
        availability,
        forced_player_name,
    ):
        if forced_player_name not in set(model['player_names']):
            return None

        player_idx = int(np.where(model['player_names'] == forced_player_name)[0][0])
        draft_pool_matches = np.where(model['draft_pool_indices'] == player_idx)[0]
        if len(draft_pool_matches) == 0:
            return None

        forced_draft_idx = int(draft_pool_matches[0])
        forced_availability = np.zeros_like(availability)
        forced_availability[:, 1:] = availability[:, 1:]
        forced_availability[forced_draft_idx, 0] = 1.0

        status, x, _ = self.solve_best_ball_ilp(
            model,
            weekly_scores,
            adp_sample,
            adjusted_picks,
            forced_availability,
        )
        if status != 'optimal':
            return None

        c = np.array(self.best_ball_ilp_objective(model, weekly_scores))[:, 0]
        x_vals = np.array(x)[:, 0]
        objective_value = -float(np.dot(c, x_vals))
        return objective_value

    def filter_best_ball_ilp_pool(self, ppg_pred, ppg_pred_ny, adp_samples, to_add_set, pos_pool_multiplier=8):
        pool = ppg_pred[['player', 'pos']].copy()
        pool['proj_mean'] = ppg_pred[self.sample_value_columns(ppg_pred)].mean(axis=1).values
        pool['adp_mean'] = adp_samples[self.sample_value_columns(adp_samples)].mean(axis=1).values
        pos_ranges = self.best_ball_position_ranges()

        keep_players = set(pool.loc[pool.player.isin(to_add_set), 'player'])

        for pos, (_, max_count) in pos_ranges.items():

            pos_pool = pool[pool.pos == pos].copy()
            if len(pos_pool) == 0:
                continue

            quota = min(len(pos_pool), max(max_count * pos_pool_multiplier, max_count + 12))
            core_quota = max(max_count + 2, int(np.ceil(quota * 0.25)))

            pos_pool['adp_rank'] = pos_pool.adp_mean.rank(method='first', ascending=True)
            pos_pool['proj_rank'] = pos_pool.proj_mean.rank(method='first', ascending=False)
            pos_pool['blend_rank'] = (0.55 * pos_pool.adp_rank) + (0.45 * pos_pool.proj_rank)

            pos_keep = set(pos_pool.nsmallest(core_quota, 'adp_rank').player)
            pos_keep.update(pos_pool.nsmallest(core_quota, 'proj_rank').player)

            late_cutoff = pos_pool.adp_mean.quantile(0.60)
            real_late_pool = pos_pool[
                (pos_pool.adp_mean >= late_cutoff) &
                (pos_pool.adp_mean < 230)
            ]
            late_pool = real_late_pool if len(real_late_pool) >= core_quota else pos_pool[pos_pool.adp_mean >= late_cutoff]
            pos_keep.update(late_pool.nsmallest(core_quota, 'proj_rank').player)

            if len(pos_keep) < quota:
                fill_pool = pos_pool[~pos_pool.player.isin(pos_keep)]
                fill_count = quota - len(pos_keep)
                pos_keep.update(fill_pool.nsmallest(fill_count, 'blend_rank').player)

            keep_players.update(pos_keep)

        keep_mask = ppg_pred.player.isin(keep_players)
        ppg_pred = ppg_pred[keep_mask].reset_index(drop=True)
        adp_samples = adp_samples[keep_mask].reset_index(drop=True)
        if ppg_pred_ny is not None:
            ppg_pred_ny = ppg_pred_ny[keep_mask].reset_index(drop=True)

        return ppg_pred, ppg_pred_ny, adp_samples

    def get_db_path(self):
        db_info = self.conn.execute("PRAGMA database_list").fetchall()
        for _, name, path in db_info:
            if name == 'main' and path:
                return path
        return None

    def get_sim_config(self):
        return {
            'set_year': self.set_year,
            'pos_require_start': self.pos_require_start,
            'num_teams': self.num_teams,
            'num_rounds': self.num_rounds,
            'my_pick_position': self.my_pick_position,
            'pred_vers': self.pred_vers,
            'league': self.league,
            'use_ownership': self.use_ownership,
            'position_ranges': self.position_ranges,
            'use_stack_bonus': self.use_stack_bonus,
            'stack_bonus_pct': self.stack_bonus_pct,
            'stack_pair_cap': self.stack_pair_cap,
            'stack_team_cap': self.stack_team_cap,
        }

    @staticmethod
    def merge_player_selection_counts(target, source):
        for player, counts in source.items():
            if player not in target:
                target[player] = copy.deepcopy(counts)
                continue
            for key, value in counts.items():
                target[player][key] += value
        return target

    @staticmethod
    def best_ball_model_summary(model, num_weeks):
        return {
            'num_players': int(model.get('num_players', 0)),
            'num_draftable': int(model.get('num_draftable', 0)),
            'num_rounds': int(model.get('num_rounds', 0)),
            'num_weeks': int(num_weeks),
            'num_vars': int(model.get('num_vars', 0)),
            'binary_vars': int(len(model.get('binary_vars', []))),
            'start_var_count': int(model.get('start_var_count', 0)),
            'x_count': int(model.get('x_count', 0)),
            'full_x_count': int(model.get('full_x_count', 0)),
            'x_pruned_count': int(model.get('x_pruned_count', 0)),
            'x_pick_buffer': int(model.get('x_pick_buffer', 0)),
            'stack_pair_count': int(model.get('stack_pair_count', 0)),
            'stack_score_count': int(model.get('stack_score_count', 0)),
            'stack_bonus_pct': float(model.get('stack_bonus_pct', 0)),
            'stack_pair_cap': float(model.get('stack_pair_cap', 0)),
            'stack_team_cap': float(model.get('stack_team_cap', 0)),
        }

    def run_best_ball_ilp_iteration_chunk(
        self,
        ppg_pred,
        ppg_pred_ny,
        adp_matrix,
        full_adp_matrix,
        adjusted_picks,
        model,
        model_full_indices,
        selected_full_indices,
        num_full_players,
        num_iters,
        to_add,
        next_year_frac,
        weekly_score_mode,
        num_weeks,
        collect_scenarios=False,
        seed=None,
    ):
        if seed is not None:
            np.random.seed(seed)

        timings = Counter()
        status_counts = Counter()
        failed_exception_count = 0
        player_selections = self.init_select_cnts()
        success_trials = 0
        scenario_records = []
        to_add_count = len(to_add)

        for iter_idx in range(num_iters):
            iter_start = time.perf_counter()
            predictions = ppg_pred_ny if (next_year_frac > 0 and np.random.random() < next_year_frac) else ppg_pred

            t0 = time.perf_counter()
            weekly_scores = self.sample_ilp_weekly_scores(
                predictions,
                num_weeks,
                weekly_score_mode=weekly_score_mode,
            )
            timings['weekly_score_sampling'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            adp_col = np.random.randint(0, full_adp_matrix.shape[1])
            adp_sample = adp_matrix[:, adp_col]
            full_adp_sample = full_adp_matrix[:, adp_col]
            availability = self.simulate_opponent_draft_availability(
                model,
                num_full_players,
                model_full_indices,
                selected_full_indices,
                full_adp_sample,
                adjusted_picks,
            )
            timings['opponent_draft_availability'] += time.perf_counter() - t0

            try:
                t0 = time.perf_counter()
                status, x, availability = self.solve_best_ball_ilp(
                    model,
                    weekly_scores,
                    adp_sample,
                    adjusted_picks,
                    availability,
                )
                timings['ilp_solve'] += time.perf_counter() - t0
                status_counts[status] += 1
            except Exception as e:
                timings['ilp_solve'] += time.perf_counter() - t0
                timings['iteration_total'] += time.perf_counter() - iter_start
                failed_exception_count += 1
                print(f"Best-ball ILP failed in iteration {iter_idx}: {e}")
                continue

            if status != 'optimal':
                timings['iteration_total'] += time.perf_counter() - iter_start
                continue

            t0 = time.perf_counter()
            player_names = model['player_names']
            draft_pool_indices = model['draft_pool_indices']
            num_rounds = model['num_rounds']
            x_solution = np.array(x)[:, 0]
            draft_solution = np.zeros((model['num_draftable'], num_rounds), dtype=np.float64)
            valid_x_mask = model['valid_x_mask']
            draft_solution[valid_x_mask] = x_solution[model['x_var_indices'][valid_x_mask]]

            available_positions = np.where((availability > 0.5) & valid_x_mask)
            for avail_idx in range(len(available_positions[0])):
                draft_idx = available_positions[0][avail_idx]
                round_idx = available_positions[1][avail_idx]
                player_name = player_names[draft_pool_indices[draft_idx]]
                round_num = round_idx + to_add_count + 1
                player_selections[player_name][f'round_{round_num}_available'] += 1
                player_selections[player_name]['total_available_count'] += 1

            selected_positions = np.where(draft_solution > 0.5)
            for selected_idx in range(len(selected_positions[0])):
                draft_idx = selected_positions[0][selected_idx]
                round_idx = selected_positions[1][selected_idx]
                player_name = player_names[draft_pool_indices[draft_idx]]
                round_num = round_idx + to_add_count + 1
                player_selections[player_name][f'round_{round_num}_count'] += 1
                player_selections[player_name]['total_counts'] += 1

            success_trials += 1
            if collect_scenarios:
                scenario_records.append({
                    'weekly_scores': weekly_scores.copy(),
                    'adp_sample': adp_sample.copy(),
                    'availability': availability.copy(),
                })
            timings['result_accounting'] += time.perf_counter() - t0
            timings['iteration_total'] += time.perf_counter() - iter_start

        return {
            'player_selections': player_selections,
            'success_trials': success_trials,
            'failed_exception_count': failed_exception_count,
            'status_counts': dict(status_counts),
            'scenario_records': scenario_records,
            'timings': dict(timings),
        }

    def get_current_pick_ev_shortlist(
        self,
        results,
        to_add,
        ev_shortlist_size,
        ev_wait_candidate_size=2,
    ):
        current_round = len(to_add) + 1
        count_col = f'Round{current_round}Count'
        available_col = f'Round{current_round}Available'
        if count_col not in results.columns or available_col not in results.columns:
            return []

        current_available = results[results[available_col] > 0].copy()
        shortlist = (
            current_available[current_available[count_col] > 0]
            .nlargest(ev_shortlist_size, count_col)
            .player
            .tolist()
        )

        next_count_col = f'Round{current_round + 1}Count'
        if ev_wait_candidate_size > 0 and next_count_col in results.columns:
            wait_candidates = (
                current_available[current_available[next_count_col] > 0]
                .nlargest(ev_wait_candidate_size, next_count_col)
                .player
                .tolist()
            )
            shortlist.extend(wait_candidates)

        return list(dict.fromkeys(shortlist))

    def solve_current_pick_ev_records(self, scenario_records, model, adjusted_picks, shortlist):
        ev_values_by_player = {player_name: [] for player_name in shortlist}
        complete_scenarios = 0
        failed_scenarios = 0

        for scenario in scenario_records:
            scenario_evs = {}
            for player_name in shortlist:
                ev = self.solve_forced_current_pick_best_ball_ilp(
                    model,
                    scenario['weekly_scores'],
                    scenario['adp_sample'],
                    adjusted_picks,
                    scenario['availability'],
                    player_name,
                )
                if ev is None:
                    scenario_evs = None
                    break
                scenario_evs[player_name] = ev

            if scenario_evs is None:
                failed_scenarios += 1
                continue

            complete_scenarios += 1
            for player_name, ev in scenario_evs.items():
                ev_values_by_player[player_name].append(ev)

        return {
            'ev_values_by_player': ev_values_by_player,
            'complete_scenarios': complete_scenarios,
            'failed_scenarios': failed_scenarios,
        }

    @staticmethod
    def merge_ev_outputs(outputs, shortlist):
        ev_values_by_player = {player_name: [] for player_name in shortlist}
        complete_scenarios = 0
        failed_scenarios = 0

        for output in outputs:
            complete_scenarios += output.get('complete_scenarios', 0)
            failed_scenarios += output.get('failed_scenarios', 0)
            for player_name in shortlist:
                ev_values_by_player[player_name].extend(
                    output.get('ev_values_by_player', {}).get(player_name, [])
                )

        return {
            'ev_values_by_player': ev_values_by_player,
            'complete_scenarios': complete_scenarios,
            'failed_scenarios': failed_scenarios,
        }

    def apply_current_pick_ev_results(self, results, ev_output, shortlist):
        ev_records = {}
        for player_name in shortlist:
            ev_values = ev_output['ev_values_by_player'].get(player_name, [])
            if len(ev_values) == 0:
                continue
            ev_records[player_name] = {
                'CurrentPickEV': float(np.mean(ev_values)),
                'CurrentPickEVSamples': len(ev_values),
                'CurrentPickEVFailedScenarios': int(ev_output.get('failed_scenarios', 0)),
            }

        if len(ev_records) == 0:
            return results

        best_ev = max(v['CurrentPickEV'] for v in ev_records.values())
        for player_name, ev_data in ev_records.items():
            ev_data['CurrentPickEVVsBest'] = ev_data['CurrentPickEV'] - best_ev

        ev_df = pd.DataFrame.from_dict(ev_records, orient='index').reset_index().rename(columns={'index': 'player'})
        return pd.merge(results, ev_df, how='left', on='player')

    def add_current_pick_ev(
        self,
        results,
        scenario_records,
        model,
        adjusted_picks,
        to_add,
        ev_shortlist_size,
        ev_wait_candidate_size=2,
    ):
        if len(scenario_records) == 0:
            return results

        shortlist = self.get_current_pick_ev_shortlist(
            results,
            to_add,
            ev_shortlist_size,
            ev_wait_candidate_size=ev_wait_candidate_size,
        )
        if len(shortlist) == 0:
            return results

        ev_output = self.solve_current_pick_ev_records(
            scenario_records,
            model,
            adjusted_picks,
            shortlist,
        )
        return self.apply_current_pick_ev_results(results, ev_output, shortlist)

    @staticmethod
    def _split_work(total_items, num_workers):
        num_workers = min(max(1, int(num_workers)), max(1, int(total_items)))
        base_size = total_items // num_workers
        remainder = total_items % num_workers
        return [
            base_size + (1 if idx < remainder else 0)
            for idx in range(num_workers)
            if base_size + (1 if idx < remainder else 0) > 0
        ]

    def run_parallel_best_ball_ilp_chunks(
        self,
        db_path,
        ppg_pred,
        ppg_pred_ny,
        adp_samples,
        full_player_names,
        full_adp_matrix,
        adjusted_picks,
        to_add,
        next_year_frac,
        weekly_score_mode,
        num_weeks,
        collect_scenarios,
        parallel_workers,
    ):
        chunk_sizes = self._split_work(self.num_iters, parallel_workers)
        seeds = np.random.randint(1, 2**31 - 1, size=len(chunk_sizes)).tolist()
        sim_config = self.get_sim_config()
        to_add_list = list(to_add)

        payloads = []
        for chunk_idx, chunk_size in enumerate(chunk_sizes):
            payloads.append({
                'db_path': db_path,
                'sim_config': sim_config,
                'ppg_pred': ppg_pred,
                'ppg_pred_ny': ppg_pred_ny,
                'adp_samples': adp_samples,
                'full_player_names': full_player_names,
                'full_adp_matrix': full_adp_matrix,
                'adjusted_picks': adjusted_picks,
                'to_add': to_add_list,
                'next_year_frac': next_year_frac,
                'weekly_score_mode': weekly_score_mode,
                'num_weeks': num_weeks,
                'collect_scenarios': collect_scenarios,
                'num_iters': chunk_size,
                'seed': seeds[chunk_idx],
            })

        outputs = []
        with ProcessPoolExecutor(max_workers=len(payloads)) as executor:
            futures = [executor.submit(_best_ball_ilp_base_worker, payload) for payload in payloads]
            for future in as_completed(futures):
                outputs.append(future.result())

        player_selections = self.init_select_cnts()
        status_counts = Counter()
        timings = Counter()
        scenario_records = []
        success_trials = 0
        failed_exception_count = 0

        for output in outputs:
            self.merge_player_selection_counts(player_selections, output['player_selections'])
            status_counts.update(output.get('status_counts', {}))
            timings.update(output.get('timings', {}))
            scenario_records.extend(output.get('scenario_records', []))
            success_trials += output.get('success_trials', 0)
            failed_exception_count += output.get('failed_exception_count', 0)

        return {
            'player_selections': player_selections,
            'success_trials': success_trials,
            'failed_exception_count': failed_exception_count,
            'status_counts': dict(status_counts),
            'scenario_records': scenario_records,
            'timings': dict(timings),
            'worker_count': len(payloads),
        }

    def run_parallel_current_pick_ev(
        self,
        db_path,
        ppg_pred,
        adp_samples,
        scenario_records,
        adjusted_picks,
        shortlist,
        to_add,
        parallel_workers,
    ):
        if len(scenario_records) == 0 or len(shortlist) == 0:
            return {
                'ev_values_by_player': {player_name: [] for player_name in shortlist},
                'complete_scenarios': 0,
                'failed_scenarios': 0,
            }

        chunk_sizes = self._split_work(len(scenario_records), parallel_workers)
        sim_config = self.get_sim_config()
        to_add_list = list(to_add)
        payloads = []
        start_idx = 0
        for chunk_size in chunk_sizes:
            end_idx = start_idx + chunk_size
            payloads.append({
                'db_path': db_path,
                'sim_config': sim_config,
                'ppg_pred': ppg_pred,
                'adp_samples': adp_samples,
                'scenario_records': scenario_records[start_idx:end_idx],
                'adjusted_picks': adjusted_picks,
                'shortlist': shortlist,
                'to_add': to_add_list,
            })
            start_idx = end_idx

        outputs = []
        with ProcessPoolExecutor(max_workers=len(payloads)) as executor:
            futures = [executor.submit(_best_ball_ilp_ev_worker, payload) for payload in payloads]
            for future in as_completed(futures):
                outputs.append(future.result())

        return self.merge_ev_outputs(outputs, shortlist)

    def run_sim_best_ball_ilp(
        self,
        to_add,
        to_drop,
        num_iters,
        next_year_frac=0,
        num_weeks=17,
        current_pick_ev=False,
        ev_shortlist_size=8,
        weekly_score_mode='residual',
        parallel_workers=1,
        ev_wait_candidate_size=2,
    ):
        total_start = time.perf_counter()
        timings = Counter()
        status_counts = Counter()
        failed_exception_count = 0

        self.num_iters = num_iters
        num_options = 1000
        player_selections = self.init_select_cnts()
        success_trials = 0
        scenario_records = []

        to_add_set = set(to_add)
        to_drop_set = set(to_drop)

        t0 = time.perf_counter()
        ppg_pred = self.drop_players(self.get_predictions('pred_fp_per_game', num_options=num_options), to_drop_set)
        ppg_pred_ny = None
        if next_year_frac > 0:
            ppg_pred_ny = self.drop_players(self.get_predictions('pred_fp_per_game_ny', num_options=num_options), to_drop_set)

        adp_samples = self.drop_players(self.get_adp_samples(num_options=num_options), to_drop_set)
        full_player_names = adp_samples.player.values
        full_adp_matrix = adp_samples.iloc[:, 2:].values
        timings['prediction_adp_generation'] += time.perf_counter() - t0

        adjusted_picks = self.calculate_adjusted_picks(len(to_add))
        if len(adjusted_picks) == 0:
            results = self.final_results(player_selections, 1)
            timings['total'] = time.perf_counter() - total_start
            results.attrs['timings'] = {
                'mode': 'best_ball_ilp',
                'weekly_score_mode': weekly_score_mode,
                'requested_iters': num_iters,
                'success_trials': 0,
                'failed_exception_count': 0,
                'status_counts': {},
                'model': {},
                'parallel_workers': 1,
                'sections': dict(timings),
            }
            return results

        if weekly_score_mode == 'template':
            t0 = time.perf_counter()
            num_weeks = self.load_weekly_template_profiles()
            timings['template_profile_load'] += time.perf_counter() - t0

        t0 = time.perf_counter()
        ppg_pred, ppg_pred_ny, adp_samples = self.filter_best_ball_ilp_pool(
            ppg_pred,
            ppg_pred_ny,
            adp_samples,
            to_add_set,
        )
        adp_matrix = adp_samples.iloc[:, 2:].values
        timings['pool_filter'] += time.perf_counter() - t0

        t0 = time.perf_counter()
        model = self.build_best_ball_ilp_model(ppg_pred, to_add_set, adjusted_picks, num_weeks)
        full_idx = {player: idx for idx, player in enumerate(full_player_names)}
        model_player_names = model['player_names'][model['draft_pool_indices']]
        model_full_indices = np.array([full_idx[player] for player in model_player_names], dtype=np.int64)
        selected_full_indices = np.array(
            [full_idx[player] for player in to_add_set if player in full_idx],
            dtype=np.int64,
        )
        num_full_players = len(full_player_names)
        timings['model_build'] += time.perf_counter() - t0

        db_path = self.get_db_path()
        parallel_workers = int(max(1, parallel_workers))
        use_parallel = parallel_workers > 1 and db_path is not None and self.num_iters > 1

        if use_parallel:
            t0 = time.perf_counter()
            try:
                chunk_output = self.run_parallel_best_ball_ilp_chunks(
                    db_path,
                    ppg_pred,
                    ppg_pred_ny,
                    adp_samples,
                    full_player_names,
                    full_adp_matrix,
                    adjusted_picks,
                    to_add,
                    next_year_frac,
                    weekly_score_mode,
                    num_weeks,
                    current_pick_ev,
                    parallel_workers,
                )
                parallel_wall = time.perf_counter() - t0
                timings['parallel_base_sims'] += parallel_wall
            except Exception as e:
                print(f"Parallel best-ball ILP failed, falling back to single worker: {e}")
                use_parallel = False
                timings['parallel_fallback'] += time.perf_counter() - t0

        if not use_parallel:
            chunk_output = self.run_best_ball_ilp_iteration_chunk(
                ppg_pred,
                ppg_pred_ny,
                adp_matrix,
                full_adp_matrix,
                adjusted_picks,
                model,
                model_full_indices,
                selected_full_indices,
                num_full_players,
                self.num_iters,
                to_add,
                next_year_frac,
                weekly_score_mode,
                num_weeks,
                collect_scenarios=current_pick_ev,
            )
            timings.update(chunk_output.get('timings', {}))

        player_selections = chunk_output['player_selections']
        success_trials = chunk_output['success_trials']
        failed_exception_count += chunk_output['failed_exception_count']
        status_counts.update(chunk_output.get('status_counts', {}))
        scenario_records = chunk_output.get('scenario_records', [])

        t0 = time.perf_counter()
        results = self.final_results(player_selections, max(success_trials, 1))
        timings['final_results'] += time.perf_counter() - t0

        if current_pick_ev:
            t0 = time.perf_counter()
            shortlist = self.get_current_pick_ev_shortlist(
                results,
                to_add,
                ev_shortlist_size,
                ev_wait_candidate_size=ev_wait_candidate_size,
            )
            if len(shortlist) > 0:
                if use_parallel and len(scenario_records) > 1:
                    try:
                        ev_output = self.run_parallel_current_pick_ev(
                            db_path,
                            ppg_pred,
                            adp_samples,
                            scenario_records,
                            adjusted_picks,
                            shortlist,
                            to_add,
                            parallel_workers,
                        )
                    except Exception as e:
                        print(f"Parallel current-pick EV failed, falling back to single worker: {e}")
                        ev_output = self.solve_current_pick_ev_records(
                            scenario_records,
                            model,
                            adjusted_picks,
                            shortlist,
                        )
                else:
                    ev_output = self.solve_current_pick_ev_records(
                        scenario_records,
                        model,
                        adjusted_picks,
                        shortlist,
                    )
                results = self.apply_current_pick_ev_results(results, ev_output, shortlist)
            timings['current_pick_ev'] += time.perf_counter() - t0

        timings['total'] = time.perf_counter() - total_start
        model_summary = self.best_ball_model_summary(model, num_weeks)
        results.attrs['timings'] = {
            'mode': 'best_ball_ilp',
            'weekly_score_mode': weekly_score_mode,
            'requested_iters': int(num_iters),
            'success_trials': int(success_trials),
            'failed_exception_count': int(failed_exception_count),
            'status_counts': dict(status_counts),
            'model': model_summary,
            'parallel_workers': int(chunk_output.get('worker_count', 1) if use_parallel else 1),
            'sections': dict(timings),
        }
        return results

    @staticmethod
    def select_disjoint_policy_ppg_columns(
        num_columns,
        construction_samples,
        evaluation_samples,
        construction_seed,
        evaluation_seed,
    ):
        """Select unique PPG columns for two banks with an enforced empty overlap."""
        num_columns = int(num_columns)
        construction_samples = int(construction_samples)
        evaluation_samples = int(evaluation_samples)
        if construction_samples <= 0 or evaluation_samples <= 0:
            raise ValueError("Policy bank sample counts must be positive.")
        if construction_samples + evaluation_samples > num_columns:
            raise ValueError(
                "Construction and evaluation samples exceed the available "
                "prediction columns required for disjoint banks."
            )

        construction_rng = np.random.default_rng(construction_seed)
        construction_columns = construction_rng.choice(
            num_columns,
            size=construction_samples,
            replace=False,
        ).astype(np.int64)
        evaluation_pool = np.setdiff1d(
            np.arange(num_columns, dtype=np.int64),
            construction_columns,
            assume_unique=True,
        )
        evaluation_rng = np.random.default_rng(evaluation_seed)
        evaluation_columns = evaluation_rng.choice(
            evaluation_pool,
            size=evaluation_samples,
            replace=False,
        ).astype(np.int64)
        if np.intersect1d(construction_columns, evaluation_columns).size:
            raise AssertionError("Construction and evaluation PPG columns overlap.")
        return construction_columns, evaluation_columns

    @staticmethod
    def select_additional_policy_ppg_columns(
        num_columns,
        excluded_columns,
        samples,
        seed,
        bank_name,
    ):
        """Select another policy bank disjoint from every supplied column."""
        samples = int(samples)
        if samples <= 0:
            return np.zeros(0, dtype=np.int64)

        excluded_columns = np.unique(
            np.asarray(excluded_columns, dtype=np.int64)
        )
        available_columns = np.setdiff1d(
            np.arange(int(num_columns), dtype=np.int64),
            excluded_columns,
            assume_unique=True,
        )
        if samples > len(available_columns):
            raise ValueError(
                f"{bank_name} samples exceed the prediction columns remaining "
                "after earlier policy-bank allocation."
            )
        rng = np.random.default_rng(seed)
        columns = rng.choice(
            available_columns,
            size=samples,
            replace=False,
        ).astype(np.int64)
        if np.intersect1d(excluded_columns, columns).size:
            raise AssertionError(f"{bank_name} PPG columns overlap an earlier bank.")
        return columns

    def sample_sequential_policy_score_bank(
        self,
        ppg_pred,
        ppg_pred_ny,
        num_scenarios,
        num_weeks,
        seed,
        ppg_column_indices,
        next_year_frac=0,
    ):
        """Sample an ex-ante season bank, optionally mixing next-year predictions."""
        score_bank = self.sample_template_weekly_score_bank(
            ppg_pred,
            num_scenarios,
            num_weeks=num_weeks,
            seed=seed,
            ppg_column_indices=ppg_column_indices,
        )
        if ppg_pred_ny is None or next_year_frac <= 0:
            return score_bank

        next_year_bank = self.sample_template_weekly_score_bank(
            ppg_pred_ny,
            num_scenarios,
            num_weeks=num_weeks,
            seed=seed,
            ppg_column_indices=ppg_column_indices,
        )
        rng = np.random.default_rng(seed + 1)
        use_next_year = rng.random(num_scenarios) < next_year_frac
        score_bank[use_next_year] = next_year_bank[use_next_year]
        return score_bank

    def complete_sequential_best_ball_rollout(
        self,
        initial_selected_indices,
        root_candidate_idx,
        adjusted_picks,
        base_remaining,
        draft_order,
        construction_bank,
        player_positions,
        adp_matrix,
        pos_ranges,
        survival_table=None,
        scarcity_weight=SEQUENTIAL_SCARCITY_WEIGHT,
        urgency_weight=SEQUENTIAL_URGENCY_WEIGHT,
    ):
        """Complete one candidate-consistent draft room using ex-ante EV only."""
        selected_indices = list(initial_selected_indices)
        remaining = np.asarray(base_remaining, dtype=bool).copy()
        path = []
        available_by_pick = []
        opponent_picks_by_turn = []
        order_pointer = 0

        root_candidate_idx = int(root_candidate_idx)
        if not remaining[root_candidate_idx]:
            return None, None, False
        selected_indices.append(root_candidate_idx)
        path.append(root_candidate_idx)
        remaining[root_candidate_idx] = False

        for pick_idx in range(1, len(adjusted_picks)):
            opponent_pick_count = (
                adjusted_picks[pick_idx]
                - adjusted_picks[pick_idx - 1]
                - 1
            )
            order_pointer, opponent_picks = self.advance_sequential_opponents(
                remaining,
                draft_order,
                order_pointer,
                opponent_pick_count,
            )
            opponent_picks_by_turn.append(opponent_picks)

            picks_left = len(adjusted_picks) - pick_idx
            legal_candidates = self.sequential_legal_candidate_indices(
                remaining,
                player_positions,
                selected_indices,
                picks_left,
                pos_ranges=pos_ranges,
            )
            available_by_pick.append(legal_candidates.copy())
            if len(legal_candidates) == 0:
                return None, {
                    'path': np.asarray(path, dtype=np.int64),
                    'available_by_pick': available_by_pick,
                    'opponent_picks_by_turn': opponent_picks_by_turn,
                }, False

            _, immediate_values = self.marginal_best_ball_values_bank(
                construction_bank,
                player_positions,
                selected_indices,
                legal_candidates,
            )
            next_pick = (
                adjusted_picks[pick_idx + 1]
                if pick_idx + 1 < len(adjusted_picks)
                else None
            )
            policy_scores, _, _, _ = self.sequential_policy_scores(
                legal_candidates,
                immediate_values,
                player_positions,
                adp_matrix,
                adjusted_picks[pick_idx],
                next_pick,
                scarcity_weight=scarcity_weight,
                urgency_weight=urgency_weight,
                survival_probabilities=(
                    survival_table[pick_idx]
                    if survival_table is not None
                    else None
                ),
            )
            chosen_idx = int(legal_candidates[int(np.argmax(policy_scores))])
            selected_indices.append(chosen_idx)
            path.append(chosen_idx)
            remaining[chosen_idx] = False

        selected_array = np.asarray(selected_indices, dtype=np.int64)
        final_counts = Counter(player_positions[selected_array])
        construction_is_legal = all(
            min_count <= final_counts.get(pos, 0) <= max_count
            for pos, (min_count, max_count) in pos_ranges.items()
        )
        success = (
            len(selected_indices) == self.num_rounds
            and len(set(selected_indices)) == len(selected_indices)
            and construction_is_legal
        )
        details = {
            'path': np.asarray(path, dtype=np.int64),
            'available_by_pick': available_by_pick,
            'opponent_picks_by_turn': opponent_picks_by_turn,
        }
        return selected_array if success else None, details, success

    @staticmethod
    def approximate_two_way_se(values):
        """Approximate uncertainty with draft-room and season scenario components."""
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.size == 0:
            return np.nan

        num_rooms, num_seasons = values.shape
        room_component = 0.0
        season_component = 0.0
        if num_rooms > 1:
            room_component = np.var(
                values.mean(axis=1),
                ddof=1,
            ) / num_rooms
        if num_seasons > 1:
            season_component = np.var(
                values.mean(axis=0),
                ddof=1,
            ) / num_seasons
        return float(np.sqrt(room_component + season_component))

    def run_sim_best_ball_policy(
        self,
        to_add,
        to_drop,
        num_iters=SEQUENTIAL_DRAFT_ROOMS,
        next_year_frac=0,
        num_weeks=SEQUENTIAL_POLICY_HORIZON,
        construction_samples=SEQUENTIAL_CONSTRUCTION_SAMPLES,
        evaluation_samples=SEQUENTIAL_EVALUATION_SAMPLES,
        decision_samples=SEQUENTIAL_DECISION_SAMPLES,
        decision_candidate_count=SEQUENTIAL_DECISION_CANDIDATES,
        audit_samples=0,
        candidate_pool_size=SEQUENTIAL_CANDIDATE_POOL_SIZE,
        scarcity_weight=SEQUENTIAL_SCARCITY_WEIGHT,
        urgency_weight=SEQUENTIAL_URGENCY_WEIGHT,
        seed=SEQUENTIAL_POLICY_SEED,
        evaluation_seed=None,
        decision_seed=None,
        audit_seed=None,
    ):
        """Sequential Preview with candidate-consistent opponent availability."""
        total_start = time.perf_counter()
        timings = Counter()
        num_rooms = max(1, int(num_iters))
        num_weeks = min(int(num_weeks), SEQUENTIAL_POLICY_HORIZON)
        evaluation_seed = (
            int(seed) + 202
            if evaluation_seed is None
            else int(evaluation_seed)
        )
        decision_seed = (
            int(seed) + 404
            if decision_seed is None
            else int(decision_seed)
        )
        audit_seed = (
            int(seed) + 505
            if audit_seed is None
            else int(audit_seed)
        )
        decision_samples = max(0, int(decision_samples))
        decision_candidate_count = max(
            1,
            int(decision_candidate_count),
        )
        audit_samples = max(0, int(audit_samples))
        if audit_samples > 0 and decision_samples <= 0:
            raise ValueError("Audit scoring requires an enabled decision stage.")
        to_add = list(dict.fromkeys(to_add))
        to_drop_set = set(to_drop)
        overlap = set(to_add) & to_drop_set
        if overlap:
            raise ValueError(
                "Players cannot be selected by both the user and opponents: "
                + ', '.join(sorted(overlap))
            )

        adjusted_picks = self.calculate_adjusted_picks(len(to_add))
        if len(adjusted_picks) == 0:
            results = pd.DataFrame(columns=['player', 'CurrentPickEV'])
            results.attrs['timings'] = {
                'mode': 'best_ball_policy_sequential',
                'release_stage': 'preview',
                'horizon_label': 'sequential_template_16',
                'sections': {'total': time.perf_counter() - total_start},
            }
            return results

        expected_opponent_picks = int(adjusted_picks[0] - 1 - len(to_add))
        actual_opponent_picks = len(to_drop_set)
        if actual_opponent_picks != expected_opponent_picks:
            raise ValueError(
                "Sequential Preview requires a complete physical draft state "
                "at the user's turn: "
                f"{expected_opponent_picks} opponent picks should be marked, "
                f"but {actual_opponent_picks} are marked."
            )

        t0 = time.perf_counter()
        with self.temp_seed(seed):
            ppg_pred = self.drop_players(
                self.get_predictions('pred_fp_per_game', num_options=1000),
                to_drop_set,
            )
            ppg_pred_ny = None
            if next_year_frac > 0:
                ppg_pred_ny = self.drop_players(
                    self.get_predictions(
                        'pred_fp_per_game_ny',
                        num_options=1000,
                    ),
                    to_drop_set,
                )
            adp_samples = self.drop_players(
                self.get_adp_samples(num_options=1000),
                to_drop_set,
            )
        timings['prediction_adp_generation'] += time.perf_counter() - t0

        player_names = ppg_pred.player.to_numpy()
        player_positions = ppg_pred.pos.to_numpy()
        player_idx = {player: idx for idx, player in enumerate(player_names)}
        missing_selected = [player for player in to_add if player not in player_idx]
        if missing_selected:
            raise ValueError(
                "Selected players are absent from the sequential policy pool: "
                + ', '.join(missing_selected)
            )

        if not np.array_equal(player_names, adp_samples.player.to_numpy()):
            adp_samples = (
                pd.DataFrame({'player': player_names})
                .merge(adp_samples, how='left', on='player', validate='one_to_one')
            )
            if adp_samples.iloc[:, 2:].isna().any().any():
                raise ValueError("Could not align ADP samples to prediction players.")
        adp_matrix = adp_samples.iloc[:, 2:].to_numpy(dtype=np.float32)

        selected_indices = np.asarray(
            [player_idx[player] for player in to_add],
            dtype=np.int64,
        )
        selected_mask = np.zeros(len(player_names), dtype=bool)
        selected_mask[selected_indices] = True
        pos_ranges = self.best_ball_position_ranges(
            player_positions,
            selected_mask,
        )
        base_remaining = np.ones(len(player_names), dtype=bool)
        base_remaining[selected_indices] = False
        required_undrafted_players = int(
            adjusted_picks[-1] - adjusted_picks[0] + 1
        )
        available_undrafted_players = int(base_remaining.sum())
        if available_undrafted_players < required_undrafted_players:
            raise ValueError(
                "Sequential policy player pool is too small to simulate every "
                "pick through the user's final selection: "
                f"{available_undrafted_players} modeled undrafted players for "
                f"{required_undrafted_players} required picks. "
                "Use a deeper league player pool or a smaller draft format."
            )

        t0 = time.perf_counter()
        num_ppg_columns = len(self.sample_value_columns(ppg_pred))
        construction_columns, evaluation_columns = (
            self.select_disjoint_policy_ppg_columns(
                num_ppg_columns,
                construction_samples,
                evaluation_samples,
                construction_seed=seed + 101,
                evaluation_seed=evaluation_seed,
            )
        )
        if np.intersect1d(construction_columns, evaluation_columns).size:
            raise AssertionError("Sequential policy score banks are not disjoint.")
        decision_columns = self.select_additional_policy_ppg_columns(
            num_ppg_columns,
            np.concatenate([construction_columns, evaluation_columns]),
            decision_samples,
            decision_seed,
            'Decision',
        )
        audit_columns = self.select_additional_policy_ppg_columns(
            num_ppg_columns,
            np.concatenate([
                construction_columns,
                evaluation_columns,
                decision_columns,
            ]),
            audit_samples,
            audit_seed,
            'Audit',
        )
        construction_bank = self.sample_sequential_policy_score_bank(
            ppg_pred,
            ppg_pred_ny,
            construction_samples,
            num_weeks,
            seed + 101,
            construction_columns,
            next_year_frac=next_year_frac,
        )
        evaluation_bank = self.sample_sequential_policy_score_bank(
            ppg_pred,
            ppg_pred_ny,
            evaluation_samples,
            num_weeks,
            evaluation_seed,
            evaluation_columns,
            next_year_frac=next_year_frac,
        )
        timings['score_banks'] += time.perf_counter() - t0

        t0 = time.perf_counter()
        draft_orders, adp_cols = self.build_sequential_draft_orders(
            adp_matrix,
            num_rooms,
            seed=seed + 303,
        )
        survival_table = self.build_sequential_survival_table(
            adp_matrix,
            adjusted_picks,
        )
        timings['draft_rooms'] += time.perf_counter() - t0

        t0 = time.perf_counter()
        current_candidates = self.sequential_legal_candidate_indices(
            base_remaining,
            player_positions,
            selected_indices,
            len(adjusted_picks),
            pos_ranges=pos_ranges,
        )
        if len(current_candidates) == 0:
            raise ValueError("No legal current-pick candidates remain.")
        _, immediate_values = self.marginal_best_ball_values_bank(
            construction_bank,
            player_positions,
            selected_indices,
            current_candidates,
        )
        next_pick = adjusted_picks[1] if len(adjusted_picks) > 1 else None
        screen_scores, survival, scarcity_cliffs, urgency = (
            self.sequential_policy_scores(
                current_candidates,
                immediate_values,
                player_positions,
                adp_matrix,
                adjusted_picks[0],
                next_pick,
                scarcity_weight=scarcity_weight,
                urgency_weight=urgency_weight,
                survival_probabilities=survival_table[0],
            )
        )
        root_candidates, inclusion_reasons = (
            self.select_sequential_root_candidates(
                current_candidates,
                immediate_values,
                screen_scores,
                survival,
                player_positions,
                candidate_pool_size,
            )
        )
        timings['root_screen'] += time.perf_counter() - t0

        current_loc = {
            int(candidate_idx): loc
            for loc, candidate_idx in enumerate(current_candidates)
        }
        records = []
        policy_paths = {}
        candidate_value_matrices = {}
        candidate_rosters = {}
        t0 = time.perf_counter()
        for root_candidate_idx in root_candidates:
            room_values = []
            completed_room_indices = []
            completed_rosters = []
            candidate_paths = []
            for room_idx in range(num_rooms):
                roster, details, success = (
                    self.complete_sequential_best_ball_rollout(
                        selected_indices,
                        root_candidate_idx,
                        adjusted_picks,
                        base_remaining,
                        draft_orders[room_idx],
                        construction_bank,
                        player_positions,
                        adp_matrix,
                        pos_ranges,
                        survival_table=survival_table,
                        scarcity_weight=scarcity_weight,
                        urgency_weight=urgency_weight,
                    )
                )
                if not success:
                    continue
                room_values.append(
                    self.best_ball_roster_scores_bank(
                        evaluation_bank,
                        player_positions,
                        roster,
                    )
                )
                completed_room_indices.append(room_idx)
                completed_rosters.append(roster.copy())
                candidate_paths.append({
                    'room_idx': room_idx,
                    'path': player_names[details['path']].tolist(),
                    'opponent_picks_by_turn': [
                        player_names[picks].tolist()
                        for picks in details['opponent_picks_by_turn']
                    ],
                })

            candidate_idx = int(root_candidate_idx)
            loc = current_loc[candidate_idx]
            value_matrix = (
                np.stack(room_values)
                if room_values
                else np.empty((0, evaluation_samples), dtype=np.float32)
            )
            records.append({
                'player': player_names[candidate_idx],
                'pos': player_positions[candidate_idx],
                'CurrentPickEV': (
                    float(value_matrix.mean()) if value_matrix.size else np.nan
                ),
                'DraftRoomSamples': num_rooms,
                'EvaluationSamples': int(evaluation_samples),
                'PolicyCompletedRooms': int(value_matrix.shape[0]),
                'CurrentPickEVSamples': int(value_matrix.size),
                'ApproxSE': self.approximate_two_way_se(value_matrix),
                'ImmediateValue': float(immediate_values[loc]),
                'ScreenScore': float(screen_scores[loc]),
                'SurviveNext': float(survival[loc]),
                'ScarcityCliff': float(scarcity_cliffs[loc]),
                'UrgencyValue': float(urgency[loc]),
                'InclusionReasons': ','.join(inclusion_reasons[candidate_idx]),
            })
            policy_paths[player_names[candidate_idx]] = candidate_paths
            candidate_value_matrices[player_names[candidate_idx]] = {
                'rooms': np.asarray(completed_room_indices, dtype=np.int64),
                'values': value_matrix,
            }
            candidate_rosters[player_names[candidate_idx]] = {
                'rooms': np.asarray(completed_room_indices, dtype=np.int64),
                'rosters': (
                    np.stack(completed_rosters)
                    if completed_rosters
                    else np.empty((0, self.num_rounds), dtype=np.int64)
                ),
            }
        timings['candidate_rollouts'] += time.perf_counter() - t0

        results = pd.DataFrame(records)
        if len(results) > 0:
            best_ev = results.CurrentPickEV.max()
            results['CurrentPickEVVsBest'] = results.CurrentPickEV - best_ev
            best_player = results.loc[
                results.CurrentPickEV.idxmax(),
                'player',
            ]
            best_matrix = candidate_value_matrices[best_player]
            paired_se = []
            for player in results.player:
                candidate_matrix = candidate_value_matrices[player]
                common_rooms = np.intersect1d(
                    candidate_matrix['rooms'],
                    best_matrix['rooms'],
                )
                if len(common_rooms) == 0:
                    paired_se.append(np.nan)
                    continue
                candidate_locs = np.searchsorted(
                    candidate_matrix['rooms'],
                    common_rooms,
                )
                best_locs = np.searchsorted(
                    best_matrix['rooms'],
                    common_rooms,
                )
                paired_differences = (
                    candidate_matrix['values'][candidate_locs]
                    - best_matrix['values'][best_locs]
                )
                paired_se.append(
                    self.approximate_two_way_se(paired_differences)
                )
            results['PairedSEVsBest'] = paired_se
            results = results.sort_values(
                ['CurrentPickEV', 'ScreenScore'],
                ascending=False,
            ).reset_index(drop=True)

        def score_candidate_rosters(candidate_players, score_bank):
            value_matrices = {}
            for player in candidate_players:
                roster_data = candidate_rosters[player]
                if len(roster_data['rosters']) == 0:
                    continue
                values = np.stack([
                    self.best_ball_roster_scores_bank(
                        score_bank,
                        player_positions,
                        roster,
                    )
                    for roster in roster_data['rosters']
                ])
                value_matrices[player] = {
                    'rooms': roster_data['rooms'],
                    'values': values,
                }
            return value_matrices

        def attach_bank_metrics(prefix, value_matrices):
            ev_col = f'{prefix}EV'
            gap_col = f'{prefix}EVVsBest'
            se_col = f'{prefix}PairedSEVsBest'
            sample_col = f'{prefix}Samples'
            results[ev_col] = np.nan
            results[gap_col] = np.nan
            results[se_col] = np.nan
            results[sample_col] = 0
            for player, matrix in value_matrices.items():
                player_mask = results.player == player
                results.loc[player_mask, ev_col] = float(
                    matrix['values'].mean()
                )
                results.loc[player_mask, sample_col] = int(
                    matrix['values'].size
                )
            scored = results[results[ev_col].notna()]
            if len(scored) == 0:
                return None
            top_player = str(
                scored.loc[scored[ev_col].idxmax(), 'player']
            )
            best_ev = float(scored[ev_col].max())
            results.loc[results[ev_col].notna(), gap_col] = (
                results.loc[results[ev_col].notna(), ev_col] - best_ev
            )
            best_matrix = value_matrices[top_player]
            for player, matrix in value_matrices.items():
                common_rooms = np.intersect1d(
                    matrix['rooms'],
                    best_matrix['rooms'],
                )
                matrix_locs = np.searchsorted(matrix['rooms'], common_rooms)
                best_locs = np.searchsorted(
                    best_matrix['rooms'],
                    common_rooms,
                )
                paired_differences = (
                    matrix['values'][matrix_locs]
                    - best_matrix['values'][best_locs]
                )
                results.loc[
                    results.player == player,
                    se_col,
                ] = self.approximate_two_way_se(paired_differences)
            return top_player

        decision_candidates = []
        decision_value_matrices = {}
        decision_top_player = None
        audit_value_matrices = {}
        audit_top_player = None
        if len(results) > 0:
            results['PilotRank'] = np.arange(1, len(results) + 1)

        if decision_samples > 0 and len(results) > 0:
            t0 = time.perf_counter()
            decision_bank = self.sample_sequential_policy_score_bank(
                ppg_pred,
                ppg_pred_ny,
                decision_samples,
                num_weeks,
                decision_seed,
                decision_columns,
                next_year_frac=next_year_frac,
            )
            timings['decision_bank'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            decision_candidates = results.head(
                min(decision_candidate_count, len(results))
            ).player.tolist()
            decision_value_matrices = score_candidate_rosters(
                decision_candidates,
                decision_bank,
            )
            decision_top_player = attach_bank_metrics(
                'Decision',
                decision_value_matrices,
            )
            results['DecisionCandidate'] = results.player.isin(
                decision_candidates
            )
            results = results.sort_values(
                [
                    'DecisionCandidate',
                    'DecisionEV',
                    'CurrentPickEV',
                    'ScreenScore',
                ],
                ascending=False,
                na_position='last',
            ).reset_index(drop=True)
            timings['decision_scoring'] += time.perf_counter() - t0
        elif len(results) > 0:
            decision_top_player = str(results.iloc[0].player)
            results['DecisionCandidate'] = False

        if audit_samples > 0 and decision_candidates:
            t0 = time.perf_counter()
            audit_bank = self.sample_sequential_policy_score_bank(
                ppg_pred,
                ppg_pred_ny,
                audit_samples,
                num_weeks,
                audit_seed,
                audit_columns,
                next_year_frac=next_year_frac,
            )
            timings['audit_bank'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            audit_value_matrices = score_candidate_rosters(
                decision_candidates,
                audit_bank,
            )
            audit_top_player = attach_bank_metrics(
                'Audit',
                audit_value_matrices,
            )
            timings['audit_scoring'] += time.perf_counter() - t0

        if len(results) > 0:
            results['RecommendationRank'] = np.arange(1, len(results) + 1)

        timings['total'] = time.perf_counter() - total_start
        results.attrs['timings'] = {
            'mode': 'best_ball_policy_sequential',
            'release_stage': 'preview',
            'horizon_label': 'sequential_template_16',
            'requested_iters': int(num_rooms),
            'success_trials': int(
                results.PolicyCompletedRooms.min() if len(results) else 0
            ),
            'failed_exception_count': 0,
            'num_weeks': int(num_weeks),
            'construction_samples': int(construction_samples),
            'evaluation_samples': int(evaluation_samples),
            'decision_samples': int(decision_samples),
            'decision_candidate_count': int(decision_candidate_count),
            'audit_samples': int(audit_samples),
            'draft_room_samples': int(num_rooms),
            'candidate_pool_size': int(candidate_pool_size),
            'scarcity_weight': float(scarcity_weight),
            'urgency_weight': float(urgency_weight),
            'seed': int(seed),
            'evaluation_seed': int(evaluation_seed),
            'decision_seed': int(decision_seed),
            'audit_seed': int(audit_seed),
            'deterministic_seed': True,
            'sections': dict(timings),
        }
        results.attrs['policy_paths'] = policy_paths
        results.attrs['candidate_value_matrices'] = candidate_value_matrices
        results.attrs['decision_candidates'] = decision_candidates
        results.attrs['decision_value_matrices'] = decision_value_matrices
        results.attrs['decision_top_player'] = decision_top_player
        results.attrs['audit_value_matrices'] = audit_value_matrices
        results.attrs['audit_top_player'] = audit_top_player
        results.attrs['scenario_banks'] = {
            'construction_ppg_columns': construction_columns.tolist(),
            'evaluation_ppg_columns': evaluation_columns.tolist(),
            'decision_ppg_columns': decision_columns.tolist(),
            'audit_ppg_columns': audit_columns.tolist(),
            'disjoint': True,
        }
        results.attrs['draft_room_adp_columns'] = adp_cols.tolist()
        return results



    def run_sim_best_ball_marginal(self, to_add, to_drop, num_iters, next_year_frac=0, num_weeks=17, candidate_pool_size=24):
        self.num_iters = num_iters
        num_options = 1000
        player_selections = self.init_select_cnts()
        success_trials = 0

        to_drop_set = set(to_drop)

        ppg_pred = self.drop_players(self.get_predictions('pred_fp_per_game', num_options=num_options), to_drop_set)
        ppg_pred_ny = None
        if next_year_frac > 0:
            ppg_pred_ny = self.drop_players(self.get_predictions('pred_fp_per_game_ny', num_options=num_options), to_drop_set)

        adp_samples = self.drop_players(self.get_adp_samples(num_options=num_options), to_drop_set)
        adp_matrix = adp_samples.iloc[:, 2:].values

        adjusted_picks = self.calculate_adjusted_picks(len(to_add))
        if len(adjusted_picks) == 0:
            return self.final_results(player_selections, 1)

        player_names = ppg_pred.player.values
        player_positions = ppg_pred.pos.values
        player_idx = {player: idx for idx, player in enumerate(player_names)}

        for _ in range(self.num_iters):
            predictions = ppg_pred_ny if (next_year_frac > 0 and np.random.random() < next_year_frac) else ppg_pred
            score_matrix = predictions[self.sample_value_columns(predictions)].values.astype(np.float32)
            week_cols = np.random.randint(0, score_matrix.shape[1], size=num_weeks)
            weekly_scores = score_matrix[:, week_cols]

            adp_col = np.random.randint(0, adp_matrix.shape[1])
            adp_sample = adp_matrix[:, adp_col]

            selected_indices = [player_idx[p] for p in to_add if p in player_idx]
            selected_set = set(selected_indices)

            pos_require_adjusted = copy.deepcopy(self.pos_require_start)
            for idx in selected_indices:
                pos = player_positions[idx]
                if pos in pos_require_adjusted and pos_require_adjusted[pos] > 0:
                    pos_require_adjusted[pos] -= 1

            current_candidates = self.get_candidate_indices(
                selected_set,
                player_positions,
                pos_require_adjusted,
                adp_sample,
                adjusted_picks[0],
                use_adp=False
            )
            if len(current_candidates) == 0:
                continue

            selected_scores = (
                weekly_scores[selected_indices]
                if selected_indices else np.empty((0, num_weeks), dtype=np.float32)
            )
            selected_positions = (
                player_positions[selected_indices]
                if selected_indices else np.array([], dtype=object)
            )
            immediate_values = self.marginal_best_ball_values(
                selected_scores,
                selected_positions,
                weekly_scores[current_candidates],
                player_positions[current_candidates]
            )

            eval_candidates = self.select_rollout_candidates(
                current_candidates,
                immediate_values,
                adp_sample,
                player_positions,
                candidate_pool_size,
            )

            best_score = -np.inf
            best_picks = None
            best_available = None

            for current_pick_idx in eval_candidates:
                rollout_selected = selected_indices + [current_pick_idx]
                rollout_require = copy.deepcopy(pos_require_adjusted)
                pick_pos = player_positions[current_pick_idx]
                if pick_pos in rollout_require and rollout_require[pick_pos] > 0:
                    rollout_require[pick_pos] -= 1

                future_picks, future_available, success = self.complete_greedy_best_ball_path(
                    rollout_selected,
                    rollout_require,
                    1,
                    adjusted_picks,
                    weekly_scores,
                    player_positions,
                    adp_sample,
                )
                if not success:
                    continue

                full_picks = [current_pick_idx] + future_picks
                full_selected = selected_indices + full_picks
                roster_score = self.best_ball_total_score(full_selected, weekly_scores, player_positions)

                if roster_score > best_score:
                    best_score = roster_score
                    best_picks = full_picks
                    best_available = [current_candidates] + future_available

            if best_picks is None:
                continue

            original_best_picks = best_picks
            original_best_available = best_available
            ordered_picks, ordered_available, ordered_success = self.order_path_by_availability(
                best_picks,
                selected_indices,
                pos_require_adjusted,
                adjusted_picks,
                player_positions,
                adp_sample,
            )
            if ordered_success:
                best_picks = ordered_picks
                best_available = ordered_available
            else:
                best_picks = original_best_picks
                best_available = original_best_available

            for round_idx, (pick_idx, available_indices) in enumerate(zip(best_picks, best_available)):
                round_num = round_idx + len(to_add) + 1
                for idx in available_indices:
                    player_name = player_names[idx]
                    player_selections[player_name][f'round_{round_num}_available'] += 1
                    player_selections[player_name]['total_available_count'] += 1

                player_name = player_names[pick_idx]
                player_selections[player_name][f'round_{round_num}_count'] += 1
                player_selections[player_name]['total_counts'] += 1

            success_trials += 1

        return self.final_results(player_selections, max(success_trials, 1))

    def run_sim(
        self,
        to_add,
        to_drop,
        num_iters,
        num_avg_pts=5,
        next_year_frac=0,
        scoring_mode='total_points',
        current_pick_ev=False,
        ev_shortlist_size=8,
        weekly_score_mode='residual',
        parallel_workers=1,
    ):
        if scoring_mode == 'best_ball_ilp':
            return self.run_sim_best_ball_ilp(
                to_add,
                to_drop,
                num_iters,
                next_year_frac=next_year_frac,
                current_pick_ev=current_pick_ev,
                ev_shortlist_size=ev_shortlist_size,
                weekly_score_mode=weekly_score_mode,
                parallel_workers=parallel_workers,
            )

        if scoring_mode == 'best_ball_policy':
            return self.run_sim_best_ball_policy(
                to_add,
                to_drop,
                num_iters=num_iters,
                next_year_frac=next_year_frac,
            )

        if scoring_mode in ('best_ball_marginal', 'best_ball_lookahead'):
            return self.run_sim_best_ball_marginal(to_add, to_drop, num_iters, next_year_frac=next_year_frac)

        
        # Initialize simulation parameters
        self.num_iters = num_iters
        num_options = 1000
        player_selections = self.init_select_cnts()
        success_trials = 0
        
        # Pre-convert to sets for faster lookups
        to_add_set = set(to_add)
        to_drop_set = set(to_drop)
        
        # Pre-generate prediction batches to avoid repeated generation
        batch_size = max(1, int(num_options/2))
        num_batches = max(1, (num_iters + batch_size - 1) // batch_size)  # Ceiling division
                
        # Pre-generate all prediction batches
        ppg_pred_batches = []
        ppg_pred_ny_batches = []
        adp_sample_batches = []
        
        for batch_idx in range(num_batches):
            # Generate predictions for this batch
            ppg_pred = self.get_predictions('pred_fp_per_game', num_options=num_options)
            ppg_pred = self.drop_players(ppg_pred, to_drop_set)
            ppg_pred_batches.append(ppg_pred)

            if next_year_frac > 0:
                ppg_pred_ny = self.get_predictions('pred_fp_per_game_ny', num_options=num_options)
                ppg_pred_ny = self.drop_players(ppg_pred_ny, to_drop_set)
                ppg_pred_ny_batches.append(ppg_pred_ny)

            adp_samples = self.get_adp_samples(num_options=num_options)
            adp_samples = self.drop_players(adp_samples, to_drop_set)
            adp_sample_batches.append(adp_samples)

        for i in range(self.num_iters):
            
            # Use pre-generated batches instead of regenerating
            batch_idx = i // batch_size
            ppg_pred = ppg_pred_batches[batch_idx]
            
            if next_year_frac > 0:
                ppg_pred_ny = ppg_pred_ny_batches[batch_idx]
            
            adp_samples = adp_sample_batches[batch_idx]

            # Select prediction type
            use_next_year = next_year_frac > 0 and np.random.random() < next_year_frac

            if use_next_year: 
                predictions = ppg_pred_ny.copy()
            else: 
                predictions = ppg_pred.copy()


            if i == 0:
                # Build constraint matrices for all picks at once
                num_players = len(predictions)
                num_picks = len(self.my_picks)
                
                # Calculate adjusted picks - remove first N picks if N players already selected
                adjusted_picks = self.calculate_adjusted_picks(len(to_add))
                
                # Adjust position requirements based on already owned players (no FLEX for now)
                pos_require_adjusted = copy.deepcopy(self.pos_require_start)
            
                for player in to_add:
                    if player in predictions.player.values:
                        player_pos = predictions[predictions.player == player].pos.iloc[0]
                        if player_pos in pos_require_adjusted and pos_require_adjusted[player_pos] > 0:
                            pos_require_adjusted[player_pos] -= 1
            
                # Remove already owned players from consideration
                available_predictions = predictions[~predictions.player.isin(to_add_set)].reset_index(drop=True)
                available_adp_samples = adp_samples[~adp_samples.player.isin(to_add_set)].reset_index(drop=True)
                
                # Pre-convert ADP data columns to numpy array for faster access
                adp_data_matrix = available_adp_samples.iloc[:, 2:].values  # Skip player and pos columns
                
                if len(available_predictions) == 0:
                    continue
            
                # Recalculate number of picks needed based on adjusted picks
                remaining_picks = len(adjusted_picks)
                if remaining_picks <= 0:
                    success_trials += 1
                    continue

                # Build constraint matrices for remaining picks
                num_players = len(available_predictions)
                num_rounds = len(adjusted_picks)

                if num_rounds == 0:
                    success_trials += 1
                    continue
                
                # Cache position mapping for faster constraint creation
                position_cache = {}
                player_positions = available_predictions['pos'].values
                for pos_idx, position in enumerate(['QB', 'RB', 'WR', 'TE']):  # Common positions
                    if position in pos_require_adjusted:
                        position_cache[position] = np.where(player_positions == position)[0]
            
                # 1. Position Requirements (exact constraints - use A matrix for == constraints)
                A_position, b_position = self.create_position_constraint_matrix(available_predictions, pos_require_adjusted, num_rounds)
                
                # 2. Player Uniqueness (each player selected at most once across all rounds)
                G_players, h_players = self.create_player_uniqueness_constraints(num_players, num_rounds)

                # 4. Round Selection Constraints (exactly 1 player per round - use A matrix)
                A_rounds, b_rounds = self.create_round_selection_constraints(num_players, num_rounds)
                
                # Pre-allocate combined matrices once (much faster than repeated vstack)
                # Create identity matrix for availability constraints (reused every iteration)
                total_constraints = num_players * num_rounds
                G_availability_identity = np.eye(total_constraints, dtype=np.float64)
                
                # Pre-allocate G matrices
                G_rows = G_players.shape[0] + total_constraints
                G_cols = G_players.shape[1]
                G_combined = np.empty((G_rows, G_cols), dtype=np.float64)
                G_combined[:G_players.shape[0]] = G_players  # Static part
                G_combined[G_players.shape[0]:] = G_availability_identity  # Static identity part
                
                # Pre-allocate h vector
                h_rows = h_players.shape[0] + total_constraints
                h_combined = np.empty((h_rows, 1), dtype=np.float64)
                h_combined[:h_players.shape[0]] = h_players  # Static part
                
                # Pre-allocate A matrices
                A_rows = A_position.shape[0] + A_rounds.shape[0]
                A_cols = A_position.shape[1]
                A_combined = np.empty((A_rows, A_cols), dtype=np.float64)
                A_combined[:A_position.shape[0]] = A_position  # Static part
                A_combined[A_position.shape[0]:] = A_rounds    # Static part
                
                # Pre-allocate b vector
                b_rows = b_position.shape[0] + b_rounds.shape[0]
                b_combined = np.empty((b_rows, 1), dtype=np.float64)
                b_combined[:b_position.shape[0]] = b_position  # Static part
                b_combined[b_position.shape[0]:] = b_rounds    # Static part

                # Pre-convert static matrices to cvxopt format (only once)
                A = matrix(A_combined, tc='d')
                b = matrix(b_combined, tc='d')
                G_static = matrix(G_combined, tc='d')  # Pre-converted G matrix
                
            

            # 3. Availability Constraints (players only available when ADP >= pick number)
            # Use pre-converted matrix for faster access
            col_idx = np.random.randint(0, adp_data_matrix.shape[1])
            adp_sample = adp_data_matrix[:, col_idx]
            _, h_availability = self.create_availability_constraints(adp_sample, available_predictions, adjusted_picks)
            
            # Update only the changing parts (availability constraints)
            h_combined[h_players.shape[0]:] = h_availability

            # Objective function (maximize points)
            _, c_points = self.sample_c_points(available_predictions, num_options, num_avg_pts, num_rounds)
            
            # Convert only the changing parts to cvxopt matrices
            h = matrix(h_combined, tc='d')            
            c = matrix(c_points, tc='d')
            # Solve ILP

            try:
                
                status, x = self.solve_ilp(c, G_static, h, A, b)
                if status == 'optimal':
                    # Track player availability for this iteration (vectorized approach)
                    h_avail_array = h_availability.flatten()
                    
                    # Vectorized availability tracking
                    # Reshape availability array to (num_players, num_rounds)
                    availability_matrix = h_avail_array.reshape(num_players, num_rounds)
                    
                    # Get player names as array for faster indexing
                    player_names = available_predictions['player'].values
                    
                    # Find all available player-round combinations at once
                    available_positions = np.where(availability_matrix == 1)
                    player_indices = available_positions[0]
                    round_indices = available_positions[1]
                    
                    # Batch update availability counts
                    for i in range(len(player_indices)):
                        player_idx = player_indices[i]
                        round_idx = round_indices[i]
                        player_name = player_names[player_idx]
                        adjusted_round_num = round_idx + len(to_add) + 1
                        player_selections[player_name][f'round_{adjusted_round_num}_available'] += 1
                        player_selections[player_name]['total_available_count'] += 1
                    
                    # Track selections and availability (vectorized approach)
                    x_solution = np.array(x)[:, 0]
                    
                    # Vectorized solution parsing: find selected players by round
                    x_reshaped = x_solution.reshape(num_players, num_rounds)
                    selected_indices = np.where(x_reshaped == 1)
                    
                    selected_players_by_round = {}
                    for i in range(len(selected_indices[0])):
                        player_idx = selected_indices[0][i]
                        round_idx = selected_indices[1][i]
                        player_name = available_predictions.iloc[player_idx].player
                        adjusted_round_idx = round_idx + len(to_add)  # Adjust for already selected players
                        selected_players_by_round[adjusted_round_idx + 1] = player_name
                    
                    for round_num, player in selected_players_by_round.items():
                        player_selections[player][f'round_{round_num}_count'] += 1
                        player_selections[player]['total_counts'] += 1
                    
                    success_trials += 1
                    
            except Exception as e:
                # If optimization fails, continue to next iteration
                print(f"Optimization failed in iteration {i}: {e}")
                pass

        results = self.final_results(player_selections, success_trials)

        return results

    def calculate_adjusted_picks(self, num_already_selected):
        """Calculate adjusted picks by removing the first N picks if N players are already selected"""
        if num_already_selected >= len(self.my_picks):
            return []  # All picks have been used
        return self.my_picks[num_already_selected:]

    def calculate_snake_picks(self):
        """Calculate the pick numbers for snake draft based on position and rounds"""
        picks = []
        for round_num in range(1, self.num_rounds + 1):
            if round_num % 2 == 1:  # Odd rounds: normal order
                pick = (round_num - 1) * self.num_teams + self.my_pick_position
            else:  # Even rounds: reverse order
                pick = round_num * self.num_teams - self.my_pick_position + 1
            picks.append(pick)
        return picks


def _best_ball_ilp_base_worker(payload):
    conn = sqlite3.connect(payload['db_path'])
    try:
        sim = FootballSimulation(conn=conn, **payload['sim_config'])
        np.random.seed(payload['seed'])

        ppg_pred = payload['ppg_pred']
        ppg_pred_ny = payload['ppg_pred_ny']
        adp_samples = payload['adp_samples']
        full_player_names = payload['full_player_names']
        full_adp_matrix = payload['full_adp_matrix']
        adjusted_picks = payload['adjusted_picks']
        to_add = payload['to_add']
        to_add_set = set(to_add)
        num_weeks = payload['num_weeks']
        if payload['weekly_score_mode'] == 'template':
            sim.load_weekly_template_profiles()

        model = sim.build_best_ball_ilp_model(ppg_pred, to_add_set, adjusted_picks, num_weeks)
        full_idx = {player: idx for idx, player in enumerate(full_player_names)}
        model_player_names = model['player_names'][model['draft_pool_indices']]
        model_full_indices = np.array([full_idx[player] for player in model_player_names], dtype=np.int64)
        selected_full_indices = np.array(
            [full_idx[player] for player in to_add_set if player in full_idx],
            dtype=np.int64,
        )

        return sim.run_best_ball_ilp_iteration_chunk(
            ppg_pred,
            ppg_pred_ny,
            adp_samples.iloc[:, 2:].values,
            full_adp_matrix,
            adjusted_picks,
            model,
            model_full_indices,
            selected_full_indices,
            len(full_player_names),
            payload['num_iters'],
            to_add,
            payload['next_year_frac'],
            payload['weekly_score_mode'],
            num_weeks,
            collect_scenarios=payload['collect_scenarios'],
            seed=payload['seed'],
        )
    finally:
        conn.close()


def _best_ball_ilp_ev_worker(payload):
    conn = sqlite3.connect(payload['db_path'])
    try:
        sim = FootballSimulation(conn=conn, **payload['sim_config'])
        scenario_records = payload['scenario_records']
        shortlist = payload['shortlist']
        if len(scenario_records) == 0:
            return {
                'ev_values_by_player': {player_name: [] for player_name in shortlist},
                'complete_scenarios': 0,
                'failed_scenarios': 0,
            }

        ppg_pred = payload['ppg_pred']
        to_add = payload['to_add']
        model = sim.build_best_ball_ilp_model(
            ppg_pred,
            set(to_add),
            payload['adjusted_picks'],
            scenario_records[0]['weekly_scores'].shape[1],
        )

        return sim.solve_current_pick_ev_records(
            scenario_records,
            model,
            payload['adjusted_picks'],
            shortlist,
        )
    finally:
        conn.close()

#%%
    
# conn = sqlite3.connect("C:/Users/borys/OneDrive/Documents/Github/Fantasy_Football_Snake/app/Simulation.sqlite3")
# year = 2026
# league = 'dk'
# num_teams = 12
# num_rounds = 20
# my_pick_position = 7
# num_iters = 300
# pos_require_start = {'QB': 3, 'RB': 6, 'WR': 8, 'TE': 3}  # No FLEX for now


# sim = FootballSimulation(conn, year, pos_require_start, num_teams, num_rounds, my_pick_position,
#                             pred_vers='final_ensemble', league=league, use_ownership=0)

# print(f"Snake picks: {sim.my_picks}")
# print(f"Player data shape: {sim.player_data.shape}")

# # Test run
# to_add = []  # No pre-selected players
# to_drop = ["Ja'Marr Chase", 'Saquon Barkley', 'Puka Nacua',
#             'Bijan Robinson', 'Christian Mccaffrey',
#             'Justin Jefferson', 'Jahmyr Gibbs', 
#             ]

# results = sim.run_sim(to_add, to_drop, num_iters, num_avg_pts=3, next_year_frac=0)
# results.sort_values(by='Round2Count', ascending=False).iloc[:10]

# %%
