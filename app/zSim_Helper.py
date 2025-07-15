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

    def __init__(self, conn, set_year, pos_require_start, salary_cap,
                 pred_vers='final_ensemble', league='beta', use_ownership=0,
                 sal_pred_actual='pred'):

        self.set_year = set_year
        self.pos_require_start = pos_require_start
        self.pred_vers = pred_vers
        self.league = league
        self.use_ownership = use_ownership
        self.conn = conn
        self.salary_cap = salary_cap
        self.sal_pred_actual = sal_pred_actual

        player_data = self.get_model_predictions()

        # join in salary data to player data
        self.player_data = self.join_salary(player_data)

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

        return df
    

    def join_salary(self, df):

        # add salaries to the dataframe and set index to player
        salaries = pd.read_sql_query(f'''SELECT player, 
                                                salary,
                                                std_dev salary_std_dev,
                                                min_score salary_min_score,
                                                max_score salary_max_score
                                         FROM Salaries_Pred
                                         WHERE year={self.set_year}
                                               AND league='{self.league}{self.sal_pred_actual}' ''', 
                                        self.conn)

        df = pd.merge(df, salaries, how='left', left_on='player', right_on='player')
        df = df.fillna({'salary': 2, 'salary_std_dev': 0.5, 'salary_min_score': 1, 'salary_max_score': 5})

        return df
    
    def get_flex_pct(self, to_drop):
        drafted = (
            self.player_data[self.player_data.player.isin(to_drop)]
            .pos
            .value_counts()
            .reset_index()
            .rename(columns={'index': 'pos', 'pos': 'drafted'})
        )

        starters = (
            pd.DataFrame({'RB': [24], 'WR': [24], 'TE': [12]})
            .T
            .reset_index()
            .rename(columns={'index': 'pos', 0: 'starters'})
        )

        top_players = (
            self.player_data[(self.player_data.pos.isin(['RB', 'WR', 'TE'])
                            & ~self.player_data.player.isin(to_drop))]
            .sort_values(by='pred_fp_per_game', ascending=False)
            .copy()
        )

        top_players['pos_rank'] = (
            top_players
            .groupby('pos')
            .cumcount()
        )

        if len(drafted)>0: starters = pd.merge(starters, drafted, on='pos', how='left').fillna(0)
        else: starters['drafted'] = 0
        starters['starters'] = starters.starters - starters.drafted

        top_players = pd.merge(top_players, starters, on='pos')
        top_players = top_players.sort_values(by='pred_fp_per_game', ascending=False)
        top_players = (
            top_players[top_players.pos_rank >= top_players.starters]
            .iloc[:84]
            .pos
            .value_counts()
        )

        top_players = top_players / top_players.sum()
        wr_frac = top_players['WR']
        
        try: rb_frac = top_players['RB']
        except: rb_frac = 0

        try: te_frac = top_players['TE']
        except: te_frac = 0

        return wr_frac, te_frac, rb_frac
    

    def add_flex(self, open_pos_require, wr_frac, te_frac, rb_frac):

        cur_pos_require = copy.deepcopy(self.pos_require_start)

        for p, v in open_pos_require.items():
            if v < 0:
                cur_pos_require[p] += -v
                cur_pos_require['FLEX'] += v

        while cur_pos_require['FLEX'] > 0:
            flex_pos = np.random.choice(['RB', 'WR', 'TE'], p=[rb_frac, wr_frac, te_frac])
            cur_pos_require[flex_pos] += 1
            cur_pos_require['FLEX'] -= 1

        return cur_pos_require
    

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
        elif col=='salary':
            cols = ['salary', 'salary_std_dev', 'salary_min_score', 'salary_max_score']

        pred_list = []
        for mean_val, sdev, min_sc, max_sc in self.player_data[cols].values:
            pred_list.append(self.trunc_normal(mean_val, sdev, min_sc, max_sc, num_options))

        return pd.DataFrame(pred_list)
    

    def get_predictions(self, pred_label, num_options=500):

        labels = self.player_data[['player', 'pos']]
        labels['salary'] = 1
        predictions = self.trunc_normal_dist(pred_label, num_options)
        predictions = pd.concat([labels, predictions], axis=1)

        return predictions
    
    def get_salaries(self, num_options=500):
        labels = self.player_data[['player', 'pos']]
        salary = self.trunc_normal_dist('salary', num_options).astype('int64')
        salary = pd.concat([labels, salary], axis=1)
        return salary
    
    def init_select_cnts(self):
        
        player_selections = {}
        for p in self.player_data.player:
            player_selections[p] = {'counts': 0, 'salary': []}
        return player_selections


    def add_players(self, to_add):
        
        h_player_add = {}
        open_pos_require = copy.deepcopy(self.pos_require_start)
        df_add = self.player_data[self.player_data.player.isin(to_add['players'])]
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
                'pos': row.pos,
                'salary': row.salary
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

        return A_positions

    @staticmethod
    def create_b_matrix(pos_require):
        return np.array(list(pos_require.values())).reshape(-1,1)

    @staticmethod
    def create_G_salaries(df, to_add):
        cur_salaries = df[['player', 'salary']].copy()
        cur_salaries.loc[cur_salaries.player.isin(to_add['players']), 'salary'] = to_add['salaries']
        return cur_salaries.salary.values.reshape(1, len(df))

    def create_h_salaries(self):
        return np.array(self.salary_cap).reshape(1, 1)

    @staticmethod
    def create_G_players(player_map):

        num_players = len(player_map)
        G_players = np.zeros(shape=(num_players, num_players))
        np.fill_diagonal(G_players, -1)

        return G_players

    @staticmethod
    def create_h_players(player_map, h_player_add):
        num_players = len(player_map)
        h_players = np.zeros(shape=(num_players, 1))

        for player, val in h_player_add.items():
            h_players[player_map[player]] = val

        return h_players

    @staticmethod
    def sample_c_points(data, max_entries, num_avg_pts):

        labels = data[['player', 'pos', 'salary']]
        current_points = -1 * data.iloc[:, np.random.choice(range(3, max_entries+3), size=num_avg_pts)].mean(axis=1)

        return labels, current_points

    @staticmethod
    def solve_ilp(c, G, h, A, b):
    
        (status, x) = ilp(c, G, h, A=A, b=b, B=set(range(0, len(c))))

        return status, x


    @staticmethod
    def tally_player_selections(predictions, player_selections, x):
            
        # find all LP results chosen and equal to 1
        x = np.array(x)[:, 0]==1
        names = predictions.player.values[x]

        # add up the player selections
        if len(names) != len(np.unique(names)):
            pass
        else:
            for n in names:
                player_selections[n]['counts'] += 1
                player_selections[n]['salary'].append(predictions[predictions.player==n].salary.values[0])

        return player_selections

    
    def final_results(self, player_selections, success_trials):
        for k, _ in player_selections.items():
            player_selections[k]['mean_salary'] = np.mean(player_selections[k]['salary'])
            if len(player_selections[k]['salary']) > 0: 
                player_selections[k]['perc_salary'] = np.percentile(player_selections[k]['salary'], 80)
            else:
                player_selections[k]['perc_salary'] = 0
            del player_selections[k]['salary']
                
        results = pd.DataFrame(player_selections).T
        results.columns = ['SelectionCounts', 'MeanSalary', 'PercSalary']

        results = results.sort_values(by='SelectionCounts', ascending=False).iloc[:59]
        results = results.reset_index().rename(columns={'index': 'player'})
        results.SelectionCounts = 100*np.round(results.SelectionCounts / success_trials, 3)
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
        
        # can set as argument, but static set for now
        self.num_iters = num_iters
        num_options=500
        player_selections = self.init_select_cnts()
        success_trials = 0

        wr_frac, te_frac, rb_frac = self.get_flex_pct(to_drop)

        for i in range(self.num_iters):
            
            if i ==0:
                # pull out current add players and added teams
                h_player_add, open_pos_require = self.add_players(to_add)

            if i % int(num_options/5) == 0:
                
                # get predictions and remove to drop players
                ppg_pred = self.get_predictions('pred_fp_per_game', num_options=num_options)
                ppg_pred = self.drop_players(ppg_pred, to_drop)

                # get predictions and remove to drop players
                ppg_pred_ny = self.get_predictions('pred_fp_per_game_ny', num_options=num_options)
                ppg_pred_ny = self.drop_players(ppg_pred_ny, to_drop)
                
                prob_top = self.get_predictions('prob_top', num_options=num_options)
                prob_top = self.drop_players(prob_top, to_drop)

                salaries = self.get_salaries(num_options=num_options)
                salaries = self.drop_players(salaries, to_drop)

            use_upside = np.random.choice([True, False], p=[upside_frac, 1-upside_frac])
            use_next_year = np.random.choice([True, False], p=[next_year_frac, 1-next_year_frac])

            if use_upside: predictions = prob_top.copy()
            elif use_next_year: predictions = ppg_pred_ny.copy()
            else: predictions = ppg_pred.copy()

            predictions['salary'] = salaries.iloc[:, np.random.choice(range(2, num_options+2))].values
            G_salaries = self.create_G_salaries(predictions, to_add)

             # append the flex position to position requirements
            cur_pos_require = self.add_flex(open_pos_require, wr_frac, te_frac, rb_frac)
            b_position = self.create_b_matrix(cur_pos_require)
            
            if i == 0:

                position_map = self.position_matrix_mapping(cur_pos_require)
                idx_player_map, player_idx_map = self.player_matrix_mapping(predictions)

                A_position = self.create_A_position(position_map, idx_player_map)
                h_salaries = self.create_h_salaries()
                
                G_players = self.create_G_players(player_idx_map)
                h_players = self.create_h_players(player_idx_map, h_player_add)
        
            # generate the c matrix with the point values to be optimized
            self.labels, self.c_points = self.sample_c_points(predictions, num_options, num_avg_pts)

            G = np.concatenate([G_salaries, G_players])
            h = np.concatenate([h_salaries, h_players])
            G = matrix(G, tc='d')
            h = matrix(h, tc='d')
            b = matrix(b_position, tc='d')
            A = matrix(A_position, tc='d')    
            c = matrix(self.c_points, tc='d')

            if len(to_add) < np.sum(list(self.pos_require_start.values())):
                status, x = self.solve_ilp(c, G, h, A, b)
                if status=='optimal':
                    player_selections = self.tally_player_selections(predictions, player_selections, x)
                    success_trials += 1

        results = self.final_results(player_selections, success_trials)

        results = results.iloc[:30]

        return results

#%%
    
# conn = sqlite3.connect("C:/Users/borys/OneDrive/Documents/Github/Fantasy_Football_App/app/Simulation.sqlite3")
# year = 2024
# league = 'beta'
# salary_cap = 300
# num_iters = 200
# pos_require_start = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 4}

# sim = FootballSimulation(conn, year, pos_require_start, salary_cap,
#                          pred_vers='final_ensemble', league=league, use_ownership=0, sal_pred_actual='_actual')

# to_add = {
#           'players': [
#                       ],
#           'salaries': [
#                        ]
#         }

# print(salary_cap - np.sum(to_add['salaries']))
# to_drop = []


# import time

# start = time.time()
# results = sim.run_sim(to_add, to_drop, num_iters, num_avg_pts=3, upside_frac=0, next_year_frac=1)
# print(time.time()-start)

# results

# %%