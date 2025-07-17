#%%

# core packages
import pandas as pd
import numpy as np
import copy
from collections import Counter
import contextlib
import sqlite3


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
        df = df.fillna({'avg_pick': 200, 'adp_std_dev': 20, 'adp_min_pick': 180, 'adp_max_pick': 250})
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

        pred_list = []
        for mean_val, sdev, min_sc, max_sc in self.player_data[cols].values:
            pred_list.append(self.trunc_normal(mean_val, sdev, min_sc, max_sc, num_options))

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
            player_selections[p] = {'counts': 0, 'available_count': 0}
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
        return df[~df.player.isin(to_drop)].reset_index(drop=True)


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


    @staticmethod
    def position_matrix_mapping(pos_require):
        position_map = {}
        i = 0
        for k, _ in pos_require.items():
            position_map[k] = i
            i+=1

        return position_map


    @staticmethod
    def create_A_position(position_map, player_map):

        num_positions = len(position_map)
        num_players = len(player_map)
        A_positions = np.zeros(shape=(num_positions, num_players))

        for i in range(num_players):
            cur_pos = player_map[i]['pos']
            row_idx = position_map[cur_pos]
            A_positions[row_idx, i] = 1
            
            # Add FLEX eligibility for RB, WR, TE
            if cur_pos in ['RB', 'WR', 'TE'] and 'FLEX' in position_map:
                flex_row_idx = position_map['FLEX']
                A_positions[flex_row_idx, i] = 1

        return A_positions

    @staticmethod
    def create_b_matrix(pos_require):
        # Convert to minimum constraints (>= instead of ==)
        return np.array(list(pos_require.values())).reshape(-1,1)

    def create_position_constraint_matrix(self, predictions, pos_require):
        """Create position constraint matrix where each position has minimum requirements"""
        num_players = len(predictions)
        
        # Remove FLEX from position requirements for now
        positions = [pos for pos in pos_require.keys() if pos != 'FLEX']
        num_positions = len(positions)
        
        if num_positions == 0:
            # If no position constraints, return empty matrix
            return np.zeros((0, num_players))
        
        A_position = np.zeros((num_positions, num_players))
        
        for i, (pred_idx, player_row) in enumerate(predictions.iterrows()):
            player_pos = player_row.pos
            
            # Add to specific position constraint only
            if player_pos in positions:
                pos_idx = positions.index(player_pos)
                A_position[pos_idx, i] = 1
        
        return A_position
    
    def create_availability_vector(self, adp_sample, predictions, adp_samples, adjusted_picks):
        """Create availability constraint vector based on ADP vs picks"""
        num_players = len(predictions)
        h_availability = np.zeros((num_players, 1))
        
        for i, (pred_idx, player_row) in enumerate(predictions.iterrows()):
            player_name = player_row.player
            player_adp = adp_sample[i]  # ADP sample is aligned with predictions after filtering
            
            # For the next pick (first pick in adjusted_picks), all players are available
            # For future picks, use ADP constraints
            if len(adjusted_picks) > 0:
                next_pick = adjusted_picks[0]  # Our very next pick
                future_picks = adjusted_picks[1:]  # All subsequent picks
                
                # Always available for the next pick, or available based on ADP for future picks
                available_for_next = True  # Always available for immediate next pick
                available_for_future = len(future_picks) == 0 or any(player_adp >= pick_num for pick_num in future_picks)
                
                # Player is available if they can be selected in next pick OR future picks
                available = available_for_next or available_for_future
            else:
                # No picks left, no one is available
                available = False
            
            h_availability[i, 0] = 1 if available else 0
        
        return h_availability

    # Removed old constraint methods - using new matrix-based approach

    @staticmethod
    def sample_c_points(data, max_entries, num_avg_pts):

        labels = data[['player', 'pos']]
        current_points = -1 * data.iloc[:, np.random.choice(range(2, max_entries+2), size=num_avg_pts)].mean(axis=1)

        return labels, current_points

    @staticmethod
    def solve_ilp(c, G, h, A, b):
    
        (status, x) = ilp(c, G, h, A=A, b=b, B=set(range(0, len(c))))

        return status, x


    # Removed old tally method - using inline tracking in run_sim

    
    def final_results(self, player_selections, success_trials):
        for k, _ in player_selections.items():
            available_count = player_selections[k]['available_count']
            if available_count > 0:
                player_selections[k]['pct_selected_when_available'] = 100 * player_selections[k]['counts'] / available_count
            else:
                player_selections[k]['pct_selected_when_available'] = 0
                
        results = pd.DataFrame(player_selections).T
        results.columns = ['SelectionCounts', 'AvailableCount', 'PctSelectedWhenAvailable']

        results = results.sort_values(by='SelectionCounts', ascending=False).iloc[:59]
        results = results.reset_index().rename(columns={'index': 'player'})
        results['PctSelected'] = 100*np.round(results.SelectionCounts / success_trials, 3)
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



    def run_sim(self, to_add, to_drop, num_iters, num_avg_pts=3, upside_frac=0, next_year_frac=0):
        
        # Initialize simulation parameters
        self.num_iters = num_iters
        num_options = 500
        player_selections = self.init_select_cnts()
        success_trials = 0

        for i in range(self.num_iters):
            
            if i % int(num_options/5) == 0:
                
                # get predictions and remove already drafted players
                ppg_pred = self.get_predictions('pred_fp_per_game', num_options=num_options)
                ppg_pred = self.drop_players(ppg_pred, to_drop)

                ppg_pred_ny = self.get_predictions('pred_fp_per_game_ny', num_options=num_options)
                ppg_pred_ny = self.drop_players(ppg_pred_ny, to_drop)
                
                prob_top = self.get_predictions('prob_top', num_options=num_options)
                prob_top = self.drop_players(prob_top, to_drop)

                adp_samples = self.get_adp_samples(num_options=num_options)
                adp_samples = self.drop_players(adp_samples, to_drop)

            # Select prediction type
            use_upside = np.random.choice([True, False], p=[upside_frac, 1-upside_frac])
            use_next_year = np.random.choice([True, False], p=[next_year_frac, 1-next_year_frac])

            if use_upside: 
                predictions = prob_top.copy()
            elif use_next_year: 
                predictions = ppg_pred_ny.copy()
            else: 
                predictions = ppg_pred.copy()

            # Build constraint matrices for all picks at once
            num_players = len(predictions)
            num_picks = len(self.my_picks)
            
            # Account for already owned players
            already_selected = set(to_add)
            
            # Calculate adjusted picks - remove first N picks if N players already selected
            adjusted_picks = self.calculate_adjusted_picks(len(to_add))
            
            # Adjust position requirements based on already owned players (no FLEX for now)
            pos_require_adjusted = copy.deepcopy(self.pos_require_start)
            
            # Remove FLEX from requirements temporarily
            if 'FLEX' in pos_require_adjusted:
                del pos_require_adjusted['FLEX']
            
            for player in to_add:
                if player in predictions.player.values:
                    player_pos = predictions[predictions.player == player].pos.iloc[0]
                    if player_pos in pos_require_adjusted and pos_require_adjusted[player_pos] > 0:
                        pos_require_adjusted[player_pos] -= 1
            
            # Remove already owned players from consideration
            available_predictions = predictions[~predictions.player.isin(to_add)].reset_index(drop=True)
            available_adp_samples = adp_samples[~adp_samples.player.isin(to_add)].reset_index(drop=True)
            
            if len(available_predictions) == 0:
                continue
            
            # Recalculate number of picks needed based on adjusted picks
            remaining_picks = len(adjusted_picks)
            if remaining_picks <= 0:
                success_trials += 1
                continue
            
            # Sample ADP for availability constraints
            adp_sample = available_adp_samples.iloc[:, np.random.choice(range(2, available_adp_samples.shape[1]))].values
            
            # Build constraint matrices for remaining picks
            num_players = len(available_predictions)
            
            # Create master constraint matrix
            # Rows: position constraints + player uniqueness + availability + total selection
            # Columns: players
            
            # 1. Position Requirements (minimum constraints - use G matrix for >= constraints)
            A_position = self.create_position_constraint_matrix(available_predictions, pos_require_adjusted)
            
            # Convert position constraints to G matrix format (negate for >= to become <=)
            if A_position.shape[0] > 0:
                G_position = -A_position  # Negate for >= constraints
                h_position = -np.array([pos_require_adjusted[pos] for pos in pos_require_adjusted.keys() if pos != 'FLEX']).reshape(-1, 1)
            else:
                G_position = np.zeros((0, num_players))
                h_position = np.zeros((0, 1))
            
            # 2. Player Selection (each player at most once - use G matrix)
            G_players = np.eye(num_players)  # Each player <= 1
            h_players = np.ones((num_players, 1))
            
            # 3. Availability Constraints (players unavailable based on ADP - use G matrix)
            G_availability = np.eye(num_players)  # Player selection <= availability
            h_availability = self.create_availability_vector(adp_sample, available_predictions, available_adp_samples, adjusted_picks)
            
            # 4. Total Selection Constraint (select exactly remaining_picks players - use A matrix)
            A_total = np.ones((1, num_players))
            b_total = np.array([[remaining_picks]])
            
            # Combine ALL G constraints (inequality <=)
            G_combined = np.vstack([G_position, G_players, G_availability])
            h_combined = np.vstack([h_position, h_players, h_availability])
            
            # Only A constraint is total selection (equality ==)
            A_combined = A_total
            b_combined = b_total

            # Objective function (maximize points)
            _, c_points = self.sample_c_points(available_predictions, num_options, num_avg_pts)
            
            # Convert to cvxopt matrices
            G = matrix(G_combined, tc='d')
            h = matrix(h_combined, tc='d')
            A = matrix(A_combined, tc='d')
            b = matrix(b_combined, tc='d')
            c = matrix(c_points, tc='d')
            
            # Solve ILP
            try:
                status, x = self.solve_ilp(c, G, h, A, b)
                
                if status == 'optimal':
                    # Track selections and availability
                    x_selected = np.array(x)[:, 0] == 1
                    selected_players = available_predictions.player.values[x_selected]
                    
                    # Track availability for all players across adjusted picks
                    for j, player in enumerate(available_predictions.player):
                        player_available = False
                        player_adp = adp_sample[j]
                        
                        # Same logic as create_availability_vector
                        if len(adjusted_picks) > 0:
                            next_pick = adjusted_picks[0]  # Our very next pick
                            future_picks = adjusted_picks[1:]  # All subsequent picks
                            
                            # Always available for the next pick, or available based on ADP for future picks
                            available_for_next = True  # Always available for immediate next pick
                            available_for_future = len(future_picks) == 0 or any(player_adp >= pick_num for pick_num in future_picks)
                            
                            # Player is available if they can be selected in next pick OR future picks
                            player_available = available_for_next or available_for_future
                        
                        if player_available:
                            player_selections[player]['available_count'] += 1
                    
                    # Track selections
                    for player in selected_players:
                        player_selections[player]['counts'] += 1
                    
                    success_trials += 1
                    
            except Exception as e:
                # If optimization fails, continue to next iteration
                pass

        results = self.final_results(player_selections, success_trials)
        results = results.iloc[:30]

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
# league = 'nffc'
# num_teams = 12
# num_rounds = 20  # Small test - just 3 rounds
# my_pick_position = 1
# num_iters = 25  # Small number for testing
# pos_require_start = {'QB': 3, 'RB': 6, 'WR': 8, 'TE': 3}  # No FLEX for now

# try:
#     sim = FootballSimulation(conn, year, pos_require_start, num_teams, num_rounds, my_pick_position,
#                              pred_vers='final_ensemble', league=league, use_ownership=0)
    
#     print(f"Snake picks: {sim.my_picks}")
#     print(f"Player data shape: {sim.player_data.shape}")
    
#     # Test run
#     to_add = ['Ceedee Lamb', 'Brock Bowers', 'Jaxon Smith Njigba']  # No pre-selected players
#     to_drop = ["Ja'Marr Chase", 'Ashton Jeanty']  # No excluded players
    
#     results = sim.run_sim(to_add, to_drop, num_iters, num_avg_pts=3, upside_frac=0, next_year_frac=0)
#     print("Top 10 results:")
#     print(results.head(10))
    
# except Exception as e:
#     print(f"Error: {e}")
#     import traceback
#     traceback.print_exc()
# %%
