#%%
import streamlit as st
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import copy
import sqlite3
from zSim_Helper import FootballSimulation

db_name = 'Simulation.sqlite3'
year = 2025
league='dk'
pred_vers = 'final_ensemble'
sal_vers = 'pred'

keepers = {
    # 'Bucky Irving': [12],
    # # 'Malik Nabers': [50],

    # 'Brock Bowers': [19],
    # 'Kyren Williams': [26],

    # # 'James Cook': [54],
    # # 'Jayden Daniels': [13],

    # 'Nico Collins': [27],
    # 'Puka Nacua': [67],

    # 'Ladd Mcconkey': [19],

    # 'Josh Jacobs': [57],

    # 'Jalen Hurts': [37],

    # 'Jaxon Smith-Njigba': [27],
    # 'Rashee Rice': [11],

    # 'Brian Thomas': [13],

    # 'Jerry Jeudy': [11],   
}



# keepers = {
#     'Breece Hall': [35],
#     'Devon Achane': [12],

#     'Garrett Wilson': [31],
#     'Kyler Murray': [12],

#     'Tyreek Hill': [79],
#     "D'Andre Swift": [14],

#     'Josh Allen': [76],
#     'Jordan Love': [20],

#     'Ceedee Lamb': [79],
#     'Anthony Richardson': [40],

#     'Kyren Williams': [28],
#     'Joe Burrow': [14],

#     'Patrick Mahomes': [83],
#     'Christian Mccaffrey': [80],

#     'Aj Brown': [67],
#     "Ja'Marr Chase": [51],

#     'Isiah Pacheco': [31],
#     'Jonathan Taylor': [38],

#     'Terry Mclaurin': [19],
#     'Trevor Lawrence': [34],

# }

#-----------------
# Pull Data In
#-----------------

if 'spending_rate' not in st.session_state:
    st.session_state['spending_rate'] = 0
if 'spending_diff' not in st.session_state:
    st.session_state['spending_diff'] = 0

def get_conn(filename):
    from pathlib import Path
    filepath = Path(__file__).parents[0] / filename
    conn = sqlite3.connect(filepath)
    
    return conn

def pull_sim_requirements(league):

    # set league information, included position requirements, number of teams, and salary cap
    league_info = {}

    if league == 'beta': league_info['pos_require'] = {'QB': 1, 'RB': 3, 'WR': 3, 'TE': 1, 'FLEX': 1}
    elif league == 'nv': league_info['pos_require'] = {'QB': 2, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1}
    elif league == 'dk': league_info['pos_require'] = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'FLEX': 1}

    league_info['num_teams'] = 12
    league_info['initial_cap'] = 298
    league_info['salary_cap'] = 298

    total_pos = np.sum(list(league_info['pos_require'].values()))
    
    return league_info, total_pos


def init_sim(filename, pred_vers, year, league, league_info):

    conn = get_conn(filename)
    sim = FootballSimulation(conn, year,league_info['pos_require'], league_info['salary_cap'], pred_vers, league, sal_pred_actual=sal_vers)

    return sim

#------------------
# App Components
#------------------

def create_interactive_grid(data):

    selected = st.data_editor(
            data,
            column_config={
                "MyTeam": st.column_config.CheckboxColumn(
                    "MyTeam",
                    help="Choose players to add to your team",
                    default=False,
                ),
                "UpdateSal": st.column_config.NumberColumn(
                        "UpdateSal",
                        help="Update the price of the player",
                        min_value=0,
                        max_value=300,
                        default=None
                ),
                "ActSal": st.column_config.NumberColumn(
                        "ActSal",
                        help="Input the price of the player",
                        min_value=0,
                        max_value=300,
                        default=None
                )
            },
            use_container_width=True,
            disabled=["widgets"],
            hide_index=True,
            height=500
        )
    return selected


def run_sim(sim, df, num_iters, upside_frac, next_frac, spending_rate):

    to_add_df = df[df.MyTeam==True]
    to_drop_df = df[(df.MyTeam==False) & ~(df.ActSal.isnull())]

    to_add = {'players': to_add_df.Player.tolist(), 'salaries': to_add_df.ActSal.tolist()}
    to_drop = {'players': to_drop_df.Player.tolist(), 'salaries': to_drop_df.ActSal.tolist()}

    # update the salary based on manual inputs
    update_sal = df[['Player', 'PredSal', 'UpdateSal']].copy().rename(columns={'Player': 'player'})
    update_sal.loc[update_sal.PredSal == update_sal.UpdateSal, 'UpdateSal'] = np.nan
    update_sal['PredSal'] = (update_sal.PredSal * (1 - spending_rate/100)).round(1)
    
    # update salary
    sim.player_data = sim.player_data.merge(update_sal, on='player', how='left')
    sim.player_data = sim.player_data.fillna({'UpdateSal': sim.player_data.PredSal})
    sim.player_data['salary_diff'] = sim.player_data.salary - sim.player_data.UpdateSal
    for c in ['salary', 'salary_min_score', 'salary_max_score']:
        sim.player_data[c] = sim.player_data[c] - sim.player_data['salary_diff']
    
    sim.player_data = sim.player_data.drop(['UpdateSal', 'PredSal', 'salary_diff'], axis=1)
    
    results = sim.run_sim(to_add, to_drop['players'], num_iters, num_avg_pts=5, upside_frac=upside_frac, next_year_frac=next_frac)
    results = results.iloc[len(to_add['players']):]
    results.MeanSalary = results.MeanSalary.round(1)
    results = results.rename(columns={'SelectionCounts': 'Selections'})

    results = pd.merge(results, update_sal[['player', 'PredSal']], on='player', how='left').rename(columns={'player': 'Player'})
    results = results[['Player',  'Selections','PredSal', 'MeanSalary', 'PercSalary']]
    return results


def init_my_team_df(pos_require):

    my_team_list = []
    for k, v in pos_require.items():
        for _ in range(v):
            my_team_list.append([k, None, 0])

    my_team_df = pd.DataFrame(my_team_list, columns=['Position', 'Player', 'Salary'])
    
    return my_team_df


def team_fill(df, df2):
    '''
    INPUT: df: blank team template to be filled in with chosen players
           df2: chosen players dataframe

    OUTPUT: Dataframe filled in with selected player information
    '''
    # loop through chosen players dataframe
    for _, row in df2.iterrows():

        # pull out the current position and find min position index not filled (if any)
        cur_pos = row.Pos
        min_idx = df.loc[(df.Position==cur_pos) & (df.Player.isnull())].index.min()

        # if position still needs to be filled, fill it
        if min_idx is not np.nan:
            df.loc[min_idx, ['Player', 'Salary']] = [row.Player, row.ActSal]

        # if normal positions filled, fill in the FLEX if applicable
        elif cur_pos in ('RB', 'WR', 'TE'):
            cur_pos = 'FLEX'
            min_idx = df.loc[(df.Position==cur_pos) & (df.Player.isnull())].index.min()
            if min_idx is not np.nan:
                df.loc[min_idx, ['Player', 'Salary']] = [row.Player, row.ActSal]

    return df[['Position', 'Player', 'Salary']]


##########################################

def get_player_data(sim, keepers):

    player_data = sim.player_data.reset_index().sort_values(by='salary', ascending=False)
    player_data = player_data.loc[player_data.pos!='FLEX', ['player', 'pos', 'pred_fp_per_game', 'pred_fp_per_game_ny', 'prob_top', 'prob_upside', 'salary']]
    player_data.prob_upside = player_data.prob_top + player_data.prob_upside
    player_data.pred_fp_per_game_ny = player_data.pred_fp_per_game_ny - player_data.pred_fp_per_game
    player_data = player_data.drop('prob_top', axis=1)
    player_data = player_data.rename(columns={'player': 'Player',
                                              'pos': 'Pos', 
                                              'salary': 'PredSal', 
                                              'pred_fp_per_game': 'PredPPG',
                                              'prob_upside': 'Upside',
                                              'pred_fp_per_game_ny': 'NextYearDiff'})
    player_data.PredPPG = player_data.PredPPG.round(1)
    player_data.Upside = player_data.Upside.round(3)
    player_data.NextYearDiff = player_data.NextYearDiff.round(1)

    player_data['UpdateSal'] = player_data['PredSal']
    player_data['ActSal'] = None
    player_data['MyTeam'] = False
    player_data['MyTeam'] = player_data['MyTeam'].astype(bool)

    for p,s in keepers.items():
        player_data.loc[player_data.Player==p, 'ActSal'] = s
        player_data.loc[player_data.Player==p, 'MyTeam'] = False

    player_data = player_data.sort_values(by=['PredSal', 'PredPPG'], ascending=False)
    player_data = player_data[['Player', 'Pos', 'ActSal', 'MyTeam', 'UpdateSal', 'PredSal', 'PredPPG', 'Upside', 'NextYearDiff']]

    return player_data

def get_my_team( selected):
    if len(selected.loc[(selected.MyTeam==True) & (selected.ActSal.isnull())]) > 0:
        st.error('Please fill in the salary for all players on your team')
        return None
    
    my_team = selected.loc[selected.MyTeam==True]
    return my_team

def update_pos_require(league_info):
    for k, v in league_info['pos_require'].items():
        if k == 'FLEX': max_value = 7
        else: max_value = v + 5
        league_info['pos_require'][k] = st.number_input(k, min_value=1, max_value=max_value, value=v, step=1)

    return league_info

def update_salary_cap(league_info):
    league_info['salary_cap'] = league_info['initial_cap'] - st.number_input('Bench Salary', min_value=0, 
                                                                             max_value=100, value=10, step=1)
    return league_info

def update_upside_frac():
    upside_frac = st.number_input('Upside Fraction', min_value=0.0, max_value=0.5, value=0.0, step=0.05)
    return upside_frac

def update_next_frac():
    next_frac = st.number_input('Next Year Fraction', min_value=0.0, max_value=0.5, value=0.0, step=0.05)
    return next_frac

def update_num_iters():
    num_iters = st.number_input('Number of Simulations', min_value=10, max_value=400, value=200, step=10)
    return num_iters

def get_spending_rate(selected):
    spending_diff = (selected.loc[~selected.ActSal.isnull(), 'ActSal'] - selected.loc[~selected.ActSal.isnull(), 'PredSal']).sum() 
    spending_rate = 100*(spending_diff / (3576-selected.loc[~selected.ActSal.isnull(), 'PredSal'].sum())).round(3)

    st.session_state['spending_diff'] = spending_diff
    st.session_state['spending_rate'] = spending_rate

#======================
# Run the App
#======================

def main():
    
    st.set_page_config(layout="wide")
    # col1, col2, col3 = st.columns([4, 3, 2])

    league_info, total_pos = pull_sim_requirements(league)

    with st.sidebar:
        st.write('Position Requirements')
        league_info = update_pos_require(league_info)

        st.write('Bench Salary')
        league_info = update_salary_cap(league_info)

        st.write('Upside Fraction')
        upside_frac = update_upside_frac()

        st.write('Next Year Fraction')
        next_frac = update_next_frac()

        st.write('Number of Simulations')
        num_iters = update_num_iters()

    team_display = init_my_team_df(league_info['pos_require']) 
    sim = init_sim(db_name, pred_vers, year, league, league_info, )

    display_data = get_player_data(sim, keepers)

    # Custom CSS to reduce padding
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 0rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
  
    with st.container():
        
        st.header('1. Choose Players')
        st.write('*Check **Myteam** box to select a player and fill in drafted_salary* ✅')
        
        selected = create_interactive_grid(display_data)
        get_spending_rate(selected)
        my_team = selected.loc[selected.MyTeam==True]

    results = run_sim(sim, selected, num_iters, upside_frac, next_frac, st.session_state['spending_rate'])
    
    with st.container():
        col1, col2 = st.columns([2, 1])
        with col1:
            st.header('2. Review Top Choices')
            st.write('*These are the optimal players to choose from* ⬇️')

            st.data_editor(
                        results,
                        column_config={
                            "Selections": st.column_config.ProgressColumn(
                                "Selections",
                                help="Percent of selections in best lineup",
                                format="%.1f",
                                min_value=0,
                                max_value=100,
                            ),
                        },
                        hide_index=True,
                        height=500,
                        use_container_width=True
                    )

        with col2:      
            st.header("⚡:green[Your Team]⚡")  
            st.write('*Players selected so far 🏈*')
            
            try: st.table(team_fill(team_display, my_team))
            except: st.table(team_display)

            subcol1, subcol2, subcol3 = st.columns(3)
            remaining_salary = league_info['salary_cap'] - my_team.ActSal.sum()
            subcol1.metric('Remaining Salary', remaining_salary)
            
            if total_pos-len(my_team) > 0: subcol2.metric('Per Player', int(remaining_salary / (total_pos-len(my_team))))
            else: subcol2.metric('Per Player', 'N/A')

            subcol3.metric('Spending Rate', f"${st.session_state['spending_diff']}", delta=f"{st.session_state['spending_rate']}%")    

if __name__ == '__main__':
    main()
# %%

