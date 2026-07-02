#%%

# core packages
import pandas as pd
import numpy as np
import copy
from collections import Counter
import contextlib
import sqlite3
import time

# scipy imports
import scipy.stats as stats

# linear optimization
from cvxopt import matrix, spmatrix
from cvxopt.glpk import ilp

import cvxopt
cvxopt.glpk.options['msg_lev'] = 'GLP_MSG_OFF'

class FootballSimulation:

    def __init__(self, conn, set_year, pos_require_start, num_teams, num_rounds, my_pick_position,
                 pred_vers='final_ensemble', league='beta', use_ownership=0, position_ranges=None):

        self.set_year = set_year
        self.pos_require_start = pos_require_start
        self.pred_vers = pred_vers
        self.league = league
        self.use_ownership = use_ownership
        self.position_ranges = position_ranges
        self.conn = conn
        self.num_teams = num_teams
        self.num_rounds = num_rounds
        self.my_pick_position = my_pick_position
        
        # Calculate my picks for snake draft
        self.my_picks = self.calculate_snake_picks()
        self.weekly_template_profiles = None
        self.weekly_template_week_cols = None

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

        resid_cols = [c for c in df.columns if c.startswith('pred_resid_')]
        df[resid_cols] = df[resid_cols].fillna(0)
        df['pred_fp_per_game_ny'] = df.pred_fp_per_game_ny.fillna(df.pred_fp_per_game)
        df['pred_p10'] = np.maximum(0, df.pred_fp_per_game + df.pred_resid_10)
        df['pred_p90'] = np.maximum(0, df.pred_fp_per_game + df.pred_resid_90)

        return df
    

    def join_adp(self, df):

        # add ADP data to the dataframe 
        adp_data = pd.read_sql_query(f'''SELECT player, 
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

        df = pd.merge(df, adp_data, how='left', left_on='player', right_on='player')
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

        labels = self.player_data[['player', 'pos']]
        predictions = self.trunc_normal_dist(pred_label, num_options)
        predictions = pd.concat([labels, predictions], axis=1)

        return predictions
    
    def get_adp_samples(self, num_options=500):
        labels = self.player_data[['player', 'pos']]
        adp = self.trunc_normal_dist('adp', num_options).astype('int64')
        adp = pd.concat([labels, adp], axis=1)
        return adp

    def load_weekly_template_profiles(self):
        if self.weekly_template_profiles is not None:
            return len(self.weekly_template_week_cols)

        template_cols = pd.read_sql_query(
            "SELECT * FROM Best_Ball_Weekly_Templates LIMIT 0",
            self.conn,
        ).columns
        week_cols = sorted(
            [c for c in template_cols if c.startswith('week_')],
            key=lambda c: int(c.split('_')[1]),
        )
        if len(week_cols) == 0:
            raise ValueError("Best_Ball_Weekly_Templates has no week_* columns.")

        week_select = ', '.join([f't.{c}' for c in week_cols])
        profiles = pd.read_sql_query(
            f'''
            SELECT m.player,
                   p.template_id,
                   {week_select}
            FROM Best_Ball_Weekly_Player_Map m
            INNER JOIN Best_Ball_Weekly_Template_Pools p
                    ON m.template_pool_key = p.template_pool_key
            INNER JOIN Best_Ball_Weekly_Templates t
                    ON p.template_id = t.template_id
            WHERE m.year = {self.set_year}
                  AND m.version = '{self.league}'
                  AND m.dataset = '{self.pred_vers}'
            ORDER BY m.player, p.template_id
            ''',
            self.conn,
        )

        if len(profiles) == 0:
            raise ValueError(
                f"No weekly template profiles found for "
                f"year={self.set_year}, version={self.league}, dataset={self.pred_vers}."
            )

        self.weekly_template_week_cols = week_cols
        self.weekly_template_profiles = {
            player: group[week_cols].to_numpy(dtype=np.float32)
            for player, group in profiles.groupby('player', sort=False)
        }
        return len(week_cols)

    def sample_template_weekly_scores(self, predictions, num_weeks):
        template_weeks = self.load_weekly_template_profiles()
        num_weeks = min(num_weeks, template_weeks)

        players = predictions.player.values
        score_matrix = predictions.iloc[:, 2:].values.astype(np.float32)
        ppg_col = np.random.randint(0, score_matrix.shape[1])
        sampled_ppg = score_matrix[:, ppg_col]

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
            template_idx = np.random.randint(0, profiles.shape[0])
            weekly_scores[idx] = sampled_ppg[idx] * profiles[template_idx, :num_weeks]

        return weekly_scores

    def sample_ilp_weekly_scores(self, predictions, num_weeks, weekly_score_mode='residual'):
        if weekly_score_mode == 'template':
            return self.sample_template_weekly_scores(predictions, num_weeks)

        score_matrix = predictions.iloc[:, 2:].values.astype(np.float32)
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
        num_players = len(predictions)
        num_rounds = len(adjusted_picks)

        selected_mask = np.array([p in to_add_set for p in player_names])
        pos_ranges = self.best_ball_position_ranges(player_positions, selected_mask)
        draft_pool_indices = np.where(~selected_mask)[0]
        num_draftable = len(draft_pool_indices)

        x_count = num_draftable * num_rounds
        y_offset = x_count
        start_offset = y_offset + num_players
        next_var = start_offset

        slot_specs = self.best_ball_slot_specs()
        start_vars_by_slot_week = {
            (slot, week): []
            for slot, _, _ in slot_specs
            for week in range(num_weeks)
        }
        start_vars_by_player_week = {
            (player_idx, week): []
            for player_idx in range(num_players)
            for week in range(num_weeks)
        }
        objective_start_entries = []

        for week in range(num_weeks):
            for slot, eligible_positions, _ in slot_specs:
                eligible_players = np.where(np.isin(player_positions, eligible_positions))[0]
                for player_idx in eligible_players:
                    var_idx = next_var
                    next_var += 1
                    start_vars_by_slot_week[(slot, week)].append(var_idx)
                    start_vars_by_player_week[(player_idx, week)].append(var_idx)
                    objective_start_entries.append((var_idx, player_idx, week))

        num_vars = next_var
        start_var_indices = np.arange(start_offset, num_vars)
        x_var_indices = np.arange(x_count).reshape(num_draftable, num_rounds) if x_count else np.empty((0, num_rounds), dtype=int)
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
            add_a_row(
                [(x_var_indices[draft_idx, round_idx], 1.0) for draft_idx in range(num_draftable)],
                1.0
            )

        # Link draft assignment variables to final roster variables.
        for draft_idx, player_idx in enumerate(draft_pool_indices):
            coeffs = [(y_var_indices[player_idx], 1.0)]
            coeffs.extend((x_var_indices[draft_idx, round_idx], -1.0) for round_idx in range(num_rounds))
            add_a_row(coeffs, 0.0)

        # Already drafted players are fixed on the final roster.
        for player_idx in np.where(selected_mask)[0]:
            add_a_row([(y_var_indices[player_idx], 1.0)], 1.0)

        # Final roster size is fixed, while position counts are constrained by ranges below.
        add_a_row([(y_var_indices[player_idx], 1.0) for player_idx in range(num_players)], self.num_rounds)

        # Weekly best-ball lineup slots.
        for week in range(num_weeks):
            for slot, _, starter_count in slot_specs:
                add_a_row(
                    [(var_idx, 1.0) for var_idx in start_vars_by_slot_week[(slot, week)]],
                    starter_count
                )

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

        # A player can only occupy one scoring lineup slot per week, and only if rostered.
        for player_idx in range(num_players):
            y_idx = y_var_indices[player_idx]
            for week in range(num_weeks):
                start_vars = start_vars_by_player_week[(player_idx, week)]
                if len(start_vars) == 0:
                    continue
                coeffs = [(var_idx, 1.0) for var_idx in start_vars]
                coeffs.append((y_idx, -1.0))
                add_g_row(coeffs, 0.0)

        # Position construction ranges for the final roster.
        for pos, (min_count, max_count) in pos_ranges.items():
            pos_players = np.where(player_positions == pos)[0]
            pos_coeffs = [(y_var_indices[player_idx], 1.0) for player_idx in pos_players]
            add_g_row(pos_coeffs, max_count)
            add_g_row([(col_idx, -1.0) for col_idx, _ in pos_coeffs], -min_count)

        availability_row_start = len(h_static)
        for draft_idx in range(num_draftable):
            for round_idx in range(num_rounds):
                add_g_row([(x_var_indices[draft_idx, round_idx], 1.0)], 1.0)

        # Weekly start variables are relaxed continuous variables, so they need explicit lower bounds.
        for var_idx in start_var_indices:
            add_g_row([(var_idx, -1.0)], 0.0)

        A = spmatrix(a_v, a_i, a_j, (len(b_vals), num_vars), tc='d')
        b = matrix(b_vals, tc='d')
        G = spmatrix(g_v, g_i, g_j, (len(h_static), num_vars), tc='d')
        h_template = np.array(h_static, dtype=np.float64)
        binary_var_count = start_offset
        objective_start_entries = np.array(objective_start_entries, dtype=np.int64)
        objective_var_indices = objective_start_entries[:, 0]
        objective_player_indices = objective_start_entries[:, 1]
        objective_week_indices = objective_start_entries[:, 2]
        availability_slice_end = availability_row_start + (num_draftable * num_rounds)

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
            'start_offset': start_offset,
            'start_var_indices': start_var_indices,
            'pos_ranges': pos_ranges,
            'draft_pool_indices': draft_pool_indices,
            'x_var_indices': x_var_indices,
            'future_pick_nums': np.array(adjusted_picks[1:], dtype=np.float64),
            'player_names': player_names,
            'player_positions': player_positions,
            'objective_var_indices': objective_var_indices,
            'objective_player_indices': objective_player_indices,
            'objective_week_indices': objective_week_indices,
            'c_template': np.zeros(num_vars, dtype=np.float64),
            'binary_vars': set(range(binary_var_count)),
        }

    @staticmethod
    def best_ball_ilp_objective(model, weekly_scores):
        c_vals = model['c_template'].copy()
        c_vals[model['objective_var_indices']] = -weekly_scores[
            model['objective_player_indices'],
            model['objective_week_indices']
        ]
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
        h_vals[model['availability_row_start']:model['availability_slice_end']] = availability.reshape(-1)
        return availability, matrix(h_vals, tc='d')

    def simulate_opponent_draft_availability(
        self,
        model,
        num_full_players,
        model_full_indices,
        selected_full_indices,
        full_adp_sample,
        adjusted_picks,
        adp_temperature=10.0,
        reach_temperature=8.0,
    ):
        """Simulate opponent picks without replacement, then project availability onto ILP players."""
        num_draftable = model['num_draftable']
        num_rounds = model['num_rounds']
        availability = np.ones((num_draftable, num_rounds), dtype=np.float64)

        full_adp_sample = np.asarray(full_adp_sample, dtype=np.float64)

        remaining = np.ones(num_full_players, dtype=bool)
        if len(selected_full_indices) > 0:
            remaining[selected_full_indices] = False

        for round_idx in range(1, num_rounds):
            prev_pick = adjusted_picks[round_idx - 1]
            cur_pick = adjusted_picks[round_idx]

            for pick_num in range(prev_pick + 1, cur_pick):
                remaining_indices = np.where(remaining)[0]
                if len(remaining_indices) == 0:
                    break

                adp_vals = np.maximum(full_adp_sample[remaining_indices], 1.0)
                reach_penalty = np.maximum(adp_vals - pick_num, 0) / reach_temperature
                logits = (-adp_vals / adp_temperature) - reach_penalty
                logits -= logits.max()
                probs = np.exp(logits)
                prob_sum = probs.sum()
                if prob_sum <= 0 or not np.isfinite(prob_sum):
                    chosen_idx = remaining_indices[np.argmin(adp_vals)]
                else:
                    chosen_idx = np.random.choice(remaining_indices, p=probs / prob_sum)
                remaining[chosen_idx] = False

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
        pool['proj_mean'] = ppg_pred.iloc[:, 2:].mean(axis=1).values
        pool['adp_mean'] = adp_samples.iloc[:, 2:].mean(axis=1).values
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

    def add_current_pick_ev(self, results, scenario_records, model, adjusted_picks, to_add, ev_shortlist_size):
        if len(scenario_records) == 0:
            return results

        current_round = len(to_add) + 1
        count_col = f'Round{current_round}Count'
        available_col = f'Round{current_round}Available'
        if count_col not in results.columns or available_col not in results.columns:
            return results

        shortlist = (
            results[(results[available_col] > 0) & (results[count_col] > 0)]
            .nlargest(ev_shortlist_size, count_col)
            .player
            .tolist()
        )
        if len(shortlist) == 0:
            return results

        ev_records = {}
        for player_name in shortlist:
            ev_values = []
            for scenario in scenario_records:
                ev = self.solve_forced_current_pick_best_ball_ilp(
                    model,
                    scenario['weekly_scores'],
                    scenario['adp_sample'],
                    adjusted_picks,
                    scenario['availability'],
                    player_name,
                )
                if ev is not None:
                    ev_values.append(ev)

            if len(ev_values) > 0:
                ev_records[player_name] = {
                    'CurrentPickEV': float(np.mean(ev_values)),
                    'CurrentPickEVSamples': len(ev_values),
                }

        if len(ev_records) == 0:
            return results

        best_ev = max(v['CurrentPickEV'] for v in ev_records.values())
        for player_name, ev_data in ev_records.items():
            ev_data['CurrentPickEVVsBest'] = ev_data['CurrentPickEV'] - best_ev

        ev_df = pd.DataFrame.from_dict(ev_records, orient='index').reset_index().rename(columns={'index': 'player'})
        results = pd.merge(results, ev_df, how='left', on='player')
        return results

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
    ):
        self.num_iters = num_iters
        num_options = 1000
        player_selections = self.init_select_cnts()
        success_trials = 0
        scenario_records = []

        to_add_set = set(to_add)
        to_drop_set = set(to_drop)

        ppg_pred = self.drop_players(self.get_predictions('pred_fp_per_game', num_options=num_options), to_drop_set)
        ppg_pred_ny = None
        if next_year_frac > 0:
            ppg_pred_ny = self.drop_players(self.get_predictions('pred_fp_per_game_ny', num_options=num_options), to_drop_set)

        adp_samples = self.drop_players(self.get_adp_samples(num_options=num_options), to_drop_set)
        full_player_names = adp_samples.player.values
        full_adp_matrix = adp_samples.iloc[:, 2:].values

        adjusted_picks = self.calculate_adjusted_picks(len(to_add))
        if len(adjusted_picks) == 0:
            return self.final_results(player_selections, 1)

        if weekly_score_mode == 'template':
            num_weeks = self.load_weekly_template_profiles()

        ppg_pred, ppg_pred_ny, adp_samples = self.filter_best_ball_ilp_pool(
            ppg_pred,
            ppg_pred_ny,
            adp_samples,
            to_add_set,
        )
        adp_matrix = adp_samples.iloc[:, 2:].values

        model = self.build_best_ball_ilp_model(ppg_pred, to_add_set, adjusted_picks, num_weeks)
        full_idx = {player: idx for idx, player in enumerate(full_player_names)}
        model_player_names = model['player_names'][model['draft_pool_indices']]
        model_full_indices = np.array([full_idx[player] for player in model_player_names], dtype=np.int64)
        selected_full_indices = np.array(
            [full_idx[player] for player in to_add_set if player in full_idx],
            dtype=np.int64,
        )
        num_full_players = len(full_player_names)

        for iter_idx in range(self.num_iters):
            predictions = ppg_pred_ny if (next_year_frac > 0 and np.random.random() < next_year_frac) else ppg_pred
            weekly_scores = self.sample_ilp_weekly_scores(
                predictions,
                num_weeks,
                weekly_score_mode=weekly_score_mode,
            )

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

            try:
                status, x, availability = self.solve_best_ball_ilp(
                    model,
                    weekly_scores,
                    adp_sample,
                    adjusted_picks,
                    availability,
                )
            except Exception as e:
                print(f"Best-ball ILP failed in iteration {iter_idx}: {e}")
                continue

            if status != 'optimal':
                continue

            player_names = model['player_names']
            draft_pool_indices = model['draft_pool_indices']
            num_rounds = model['num_rounds']
            x_solution = np.array(x)[:, 0]
            draft_solution = x_solution[model['x_var_indices'].reshape(-1)].reshape(model['num_draftable'], num_rounds)

            available_positions = np.where(availability == 1)
            for avail_idx in range(len(available_positions[0])):
                draft_idx = available_positions[0][avail_idx]
                round_idx = available_positions[1][avail_idx]
                player_name = player_names[draft_pool_indices[draft_idx]]
                round_num = round_idx + len(to_add) + 1
                player_selections[player_name][f'round_{round_num}_available'] += 1
                player_selections[player_name]['total_available_count'] += 1

            selected_positions = np.where(draft_solution > 0.5)
            for selected_idx in range(len(selected_positions[0])):
                draft_idx = selected_positions[0][selected_idx]
                round_idx = selected_positions[1][selected_idx]
                player_name = player_names[draft_pool_indices[draft_idx]]
                round_num = round_idx + len(to_add) + 1
                player_selections[player_name][f'round_{round_num}_count'] += 1
                player_selections[player_name]['total_counts'] += 1

            success_trials += 1
            if current_pick_ev:
                scenario_records.append({
                    'weekly_scores': weekly_scores.copy(),
                    'adp_sample': adp_sample.copy(),
                    'availability': availability.copy(),
                })

        results = self.final_results(player_selections, max(success_trials, 1))
        if current_pick_ev:
            results = self.add_current_pick_ev(
                results,
                scenario_records,
                model,
                adjusted_picks,
                to_add,
                ev_shortlist_size,
            )

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
        sample_start_col = 2

        for _ in range(self.num_iters):
            predictions = ppg_pred_ny if (next_year_frac > 0 and np.random.random() < next_year_frac) else ppg_pred
            score_matrix = predictions.iloc[:, sample_start_col:].values.astype(np.float32)
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
