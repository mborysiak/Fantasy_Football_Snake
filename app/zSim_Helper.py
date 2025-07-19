#%%

# core packages
import pandas as pd
import numpy as np
import copy
from collections import Counter
import contextlib
import sqlite3
import time

# linear optimization
from cvxopt import matrix
from cvxopt.glpk import ilp

import cvxopt
cvxopt.glpk.options['msg_lev'] = 'GLP_MSG_OFF'

class FootballSimulation:

    def __init__(self, conn, set_year, pos_require_start, num_teams, num_rounds, my_pick_position,
                 pred_vers='final_ensemble', league='beta', use_ownership=0):

        self.set_year = set_year
        self.pos_require_start = pos_require_start
        self.pred_vers = pred_vers
        self.league = league
        self.use_ownership = use_ownership
        self.conn = conn
        self.num_teams = num_teams
        self.num_rounds = num_rounds
        self.my_pick_position = my_pick_position
        
        # Calculate my picks for snake draft
        self.my_picks = self.calculate_snake_picks()

        player_data = self.get_model_predictions()

        # join in ADP data to player data
        self.player_data = self.join_adp(player_data)

    def get_model_predictions(self):
        df = pd.read_sql_query(f'''SELECT player, 
                                          pos, 
                                          pred_fp_per_game, 
                                          pred_prob_upside prob_upside,
                                          pred_prob_top prob_top,
                                          pred_fp_per_game_ny,
                                          std_dev_ny,
                                          std_dev, 
                                          min_score, 
                                          max_score,
                                          min_score_ny,
                                          max_score_ny,
                                          0 as prob_min,
                                          1 as prob_max
                                   FROM Final_Predictions
                                   WHERE year={self.set_year}
                                         AND dataset='{self.pred_vers}'
                                         AND version='{self.league}'
                                        
                                ''', self.conn)
        
        df.loc[df.std_dev < 0.1, 'std_dev'] = 0.2 * df.loc[df.std_dev < 0.1, 'pred_fp_per_game']  # Avoid division by zero in calculations
        df.loc[df.std_dev_ny < 0.1, 'std_dev_ny'] = 0.2 * df.loc[df.std_dev_ny < 0.1, 'pred_fp_per_game_ny']


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
        df.loc[df.std_dev<0.1, 'std_dev'] = 0.2 * df.loc[df.std_dev<0.1, 'avg_pick'] # Avoid division by zero in calculations
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
    def trunc_normal(mean_val, sdev, min_sc, max_sc, num_samples=50):

        import scipy.stats as stats

        # create truncated distribution
        lower_bound = (min_sc - mean_val) / sdev, 
        upper_bound = (max_sc - mean_val) / sdev
        trunc_dist = stats.truncnorm(lower_bound, upper_bound, loc=mean_val, scale=sdev)
        
        estimates = trunc_dist.rvs(num_samples)

        return estimates


    def trunc_normal_dist(self, col, num_options=50):
        
        if col=='pred_fp_per_game':
            cols = ['pred_fp_per_game', 'std_dev', 'min_score', 'max_score']
        elif col == 'prob_top':
            cols = ['prob_top', 'prob_upside', 'prob_min', 'prob_max']
        elif col == 'pred_fp_per_game_ny':
            cols = ['pred_fp_per_game_ny', 'std_dev_ny', 'min_score_ny', 'max_score_ny']
        elif col=='adp':
            cols = ['avg_pick', 'adp_std_dev', 'adp_min_pick', 'adp_max_pick']

        # Vectorized approach: process all players at once when possible
        data_values = self.player_data[cols].values
        pred_list = [self.trunc_normal(mean_val, sdev, min_sc, max_sc, num_options) 
                     for mean_val, sdev, min_sc, max_sc in data_values]

        return pd.DataFrame(pred_list)
    

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



    def run_sim(self, to_add, to_drop, num_iters, num_avg_pts=5, upside_frac=0, next_year_frac=0):
        
        # Initialize simulation parameters
        self.num_iters = num_iters
        num_options = 1000
        player_selections = self.init_select_cnts()
        success_trials = 0
        
        # Pre-convert to sets for faster lookups
        to_add_set = set(to_add)
        to_drop_set = set(to_drop)

        for i in range(self.num_iters):

            start1 = time.time()
            
            if i % int(num_options/10) == 0:
                
                # get predictions and remove already drafted players
                ppg_pred = self.get_predictions('pred_fp_per_game', num_options=num_options)
                ppg_pred = self.drop_players(ppg_pred, to_drop_set)

                if next_year_frac > 0:
                    # Get next year predictions if applicable
                    ppg_pred_ny = self.get_predictions('pred_fp_per_game_ny', num_options=num_options)
                    ppg_pred_ny = self.drop_players(ppg_pred_ny, to_drop_set)

                if upside_frac > 0:
                    # Get upside predictions if applicable
                    prob_upside = self.get_predictions('prob_upside', num_options=num_options)
                    prob_upside = self.drop_players(prob_upside, to_drop_set)   

                adp_samples = self.get_adp_samples(num_options=num_options)
                adp_samples = self.drop_players(adp_samples, to_drop_set)

            # Select prediction type
            use_upside = np.random.choice([True, False], p=[upside_frac, 1-upside_frac])
            use_next_year = np.random.choice([True, False], p=[next_year_frac, 1-next_year_frac])

            if use_upside: 
                predictions = prob_upside.copy()
            elif use_next_year: 
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
            time1 = time.time() - start1
            # Solve ILP
            try:
                start2 = time.time()
                status, x = self.solve_ilp(c, G_static, h, A, b)
                print(status)
                time2 = time.time() - start2
                if status == 'optimal':
                    start3 = time.time()
                    
                    # Track player availability for this iteration (before tracking selections)
                    # Use the h_availability vector to determine which players were available
                    h_avail_array = h_availability.flatten()
                    
                    for player_idx in range(num_players):
                        player_name = available_predictions.iloc[player_idx].player
                        for round_idx in range(num_rounds):
                            # Calculate the constraint index for this player-round combination
                            constraint_idx = player_idx * num_rounds + round_idx
                            if h_avail_array[constraint_idx] == 1:  # Player was available
                                adjusted_round_num = round_idx + len(to_add) + 1  # Adjust for already selected players
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
                    time3 = time.time() - start3
                    
            except Exception as e:
                # If optimization fails, continue to next iteration
                print(f"Optimization failed in iteration {i}: {e}")
                pass

            print(f'Time1: {time1:.2f}s, Time2: {time2:.2f}s, Time3: {time3:.2f}s, Success Trials: {success_trials}', end='\r')

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
# year = 2025
# league = 'dk'
# num_teams = 12
# num_rounds = 20
# my_pick_position = 7
# num_iters = 100
# pos_require_start = {'QB': 3, 'RB': 6, 'WR': 8, 'TE': 3}  # No FLEX for now

# try:
#     sim = FootballSimulation(conn, year, pos_require_start, num_teams, num_rounds, my_pick_position,
#                              pred_vers='final_ensemble', league=league, use_ownership=0)
    
#     print(f"Snake picks: {sim.my_picks}")
#     print(f"Player data shape: {sim.player_data.shape}")
    
#     # Test run
#     to_add = []  # No pre-selected players
#     to_drop = ["Ja'Marr Chase", 'Saquon Barkley', 'Puka Nacua',
#                 'Bijan Robinson', 'Christian Mccaffrey',
#                 'Justin Jefferson', 'Jahmyr Gibbs', 
#                 ]
    
#     results = sim.run_sim(to_add, to_drop, num_iters, num_avg_pts=3, upside_frac=0, next_year_frac=0)
#     print("Top 10 results:")
#     print(results.head(10))
    
#     # Show round-by-round breakdown for top players
#     print("\nRound-by-round breakdown for top 3 players:")
#     for i in range(min(3, len(results))):
#         player = results.iloc[i]
#         print(f"\n{player['player']}:")
#         print(f"  Total: {player['TotalSelectionCounts']}/{player['TotalAvailableCount']} ({player['PctSelectedWhenAvailable']:.1f}%)")
        
#         # Show round data
#         for round_num in range(1, num_rounds + 1):
#             count_col = f'Round{round_num}Count'
#             avail_col = f'Round{round_num}Available'
#             pct_col = f'Round{round_num}Pct'
            
#             if count_col in player.index and avail_col in player.index:
#                 count = player[count_col]
#                 available = player[avail_col]
#                 pct = player[pct_col] if pct_col in player.index else 0
#                 print(f"  Round {round_num}: {count}/{available} ({pct:.1f}%)")
    
# except Exception as e:
#     print(f"Error: {e}")
#     import traceback
#     traceback.print_exc()
# # %%

# results.sort_values(by='Round2Count', ascending=False).iloc[:10]
# # %%
# results.sum().iloc[:20]
# %%
