#%%
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from zSim_Helper import FootballSimulation

# Configuration
db_name = 'Simulation.sqlite3'
year = 2025
league = 'nffc'
pred_vers = 'final_ensemble'

#-----------------
# Helper Functions
#-----------------

def get_conn(filename):
    from pathlib import Path
    filepath = Path(__file__).parents[0] / filename
    conn = sqlite3.connect(filepath)
    return conn

def init_sim(filename, pred_vers, year, league, num_teams, num_rounds, my_pick_position, pos_require):
    conn = get_conn(filename)
    sim = FootballSimulation(
        conn=conn, 
        set_year=year,
        pos_require_start=pos_require, 
        num_teams=num_teams,
        num_rounds=num_rounds,
        my_pick_position=my_pick_position,
        pred_vers=pred_vers, 
        league=league, 
        use_ownership=0
    )
    return sim

def get_player_data(sim):
    """Get player data formatted for the app"""
    player_data = sim.player_data.reset_index()
    
    # Select and rename relevant columns
    player_data = player_data[['player', 'pos', 'years_of_experience', 'pred_fp_per_game', 'pred_fp_per_game_ny', 'prob_top', 'prob_upside', 'avg_pick']]
    
    # Rename columns for display
    player_data = player_data.rename(columns={
        'player': 'Player',
        'pos': 'Pos', 
        'avg_pick': 'ADP',
        'pred_fp_per_game': 'PredPPG',
        'years_of_experience': 'Years_Exp',
    })
    
    # Round values for display
    player_data['PredPPG'] = player_data['PredPPG'].round(1)
    player_data['ADP'] = player_data['ADP'].round(1)
    
    # Add selection columns
    player_data['MyTeam'] = False
    player_data['OtherTeam'] = False
    
    # Sort by ADP (best players first)
    player_data = player_data.sort_values(by='ADP')
    
    # Reorder columns
    player_data = player_data[['Player', 'Pos', 'Years_Exp', 'MyTeam', 'OtherTeam', 'ADP', 'PredPPG']]
    
    return player_data

def create_interactive_grid(data):
    """Create the interactive data editor for player selection"""
    selected = st.data_editor(
        data,
        column_config={
            "MyTeam": st.column_config.CheckboxColumn(
                "My Team",
                help="Check to add player to your team",
                default=False,
            ),
            "OtherTeam": st.column_config.CheckboxColumn(
                "Other Team",
                help="Check to exclude player (drafted by others)",
                default=False,
            ),
            "ADP": st.column_config.NumberColumn(
                "ADP",
                help="Average Draft Position",
                format="%.1f"
            ),
            "PredPPG": st.column_config.NumberColumn(
                "Pred PPG",
                help="Predicted Points Per Game",
                format="%.1f"
            ),

        },
        use_container_width=True,
        disabled=["Player", "Pos", "ADP", "PredPPG"],
        hide_index=True,
        height=500
    )
    return selected

def run_simulation(sim, selected_data, num_iters):
    """Run the snake draft simulation"""
    
    # Get selected players
    my_team = selected_data[selected_data['MyTeam'] == True]['Player'].tolist()
    other_teams = selected_data[selected_data['OtherTeam'] == True]['Player'].tolist()
    
    # Run simulation
    results = sim.run_sim(
        to_add=my_team, 
        to_drop=other_teams, 
        num_iters=num_iters, 
        num_avg_pts=3, 

    )
    
    # Format results
    results = results.rename(columns={
        'player': 'Player',
        'SelectionCounts': 'Selections',
        'AvailableCount': 'Available',
        'PctSelectedWhenAvailable': 'Pct When Available',
        'PctSelected': 'Pct Selected'
    })
    
    return results

def create_my_team_display(selected_data, pos_require):
    """Create display of current team with position requirements"""
    my_team = selected_data[selected_data['MyTeam'] == True]
    
    # Count positions
    pos_counts = my_team['Pos'].value_counts() if len(my_team) > 0 else {}
    
    team_display = []
    for pos, required in pos_require.items():
        if pos == 'FLEX':
            continue  # Skip FLEX for now
        current_count = pos_counts.get(pos, 0)
        
        # Add header row for each position
        team_display.append([f"{pos} ({current_count}/{required})", ""])
        
        # Add filled positions
        pos_players = my_team[my_team['Pos'] == pos]['Player'].tolist()
        for i, player in enumerate(pos_players):
            team_display.append(["", player])
        
        # Add empty slots
        for i in range(max(0, required - current_count)):
            team_display.append(["", "-- Empty --"])
    
    return pd.DataFrame(team_display, columns=['Position', 'Player'])

#------------------
# App Components
#------------------

def sidebar_controls():
    """Create sidebar controls"""
    st.sidebar.header("Draft Settings")
    
    # League settings
    num_teams = st.sidebar.number_input(
        'Number of Teams', 
        min_value=8, 
        max_value=16, 
        value=12, 
        step=1
    )
    
    my_pick_position = st.sidebar.number_input(
        'My Pick Position', 
        min_value=1, 
        max_value=num_teams, 
        value=7, 
        step=1
    )
    
    st.sidebar.header("Position Requirements")
    
    # Position requirements
    pos_require = {}
    pos_require['QB'] = st.sidebar.number_input('QB', min_value=1, max_value=5, value=3, step=1)
    pos_require['RB'] = st.sidebar.number_input('RB', min_value=2, max_value=15, value=6, step=1)
    pos_require['WR'] = st.sidebar.number_input('WR', min_value=3, max_value=15, value=9, step=1)
    pos_require['TE'] = st.sidebar.number_input('TE', min_value=1, max_value=5, value=3, step=1)
    # pos_require['FLEX'] = st.sidebar.number_input('FLEX', min_value=0, max_value=3, value=1, step=1)
    
    # Calculate total rounds from position requirements
    num_rounds = sum(pos_require.values())
    st.sidebar.write(f"**Total Rounds:** {num_rounds}")
    
    st.sidebar.header("Simulation Settings")
    
    num_iters = st.sidebar.number_input(
        'Number of Simulations', 
        min_value=10, 
        max_value=500, 
        value=100, 
        step=10
    )
    
    return {
        'num_teams': num_teams,
        'num_rounds': num_rounds,
        'my_pick_position': my_pick_position,
        'pos_require': pos_require,
        'num_iters': num_iters,
    }

#======================
# Main App
#======================

def main():
    st.set_page_config(page_title="Snake Draft Optimizer", layout="wide")
    
    # Custom CSS to reduce padding
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("🐍 Snake Draft Optimizer")
    
    # Get settings from sidebar
    settings = sidebar_controls()
    
    try:
        # Initialize simulation
        sim = init_sim(
            db_name, 
            pred_vers, 
            year, 
            league,
            settings['num_teams'],
            settings['num_rounds'],
            settings['my_pick_position'],
            settings['pos_require']
        )
        
        # Display draft information
        st.sidebar.header("Draft Info")
        st.sidebar.write(f"**My Picks:** {sim.my_picks[:5]}..." if len(sim.my_picks) > 5 else f"**My Picks:** {sim.my_picks}")
        
        # Get player data
        player_data = get_player_data(sim)
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("1. Select Players")
            
            # Player selection grid
            selected_data = create_interactive_grid(player_data)
            
            # Get current selections for validation
            current_my_team = selected_data[selected_data['MyTeam'] == True]
            current_other_team = selected_data[selected_data['OtherTeam'] == True]
            
            # Check for conflicts
            conflicts = set(current_my_team['Player']) & set(current_other_team['Player'])
            if conflicts:
                st.error(f"Players cannot be on both your team and other teams: {', '.join(conflicts)}")
                return
            
            # Calculate draft status for simulation button
            my_team_players = selected_data[selected_data['MyTeam'] == True]
            num_selected = len(my_team_players)
            adjusted_picks = sim.calculate_adjusted_picks(num_selected)
            
            # Run simulation section
            st.header("2. Optimal Recommendations")
            
            if st.button("🚀 Run Simulation", type="primary"):
                if len(adjusted_picks) == 0:
                    st.info("All picks completed! No more recommendations needed.")
                else:
                    with st.spinner("Running simulation..."):
                        results = run_simulation(
                            sim, 
                            selected_data, 
                            settings['num_iters'], 
                        )
                    
                    st.write("Players ranked by selection frequency in optimal lineups")
                    
                    # Display simplified results - just Player and Pct Selected
                    display_results = results[['Player', 'Pct Selected']].copy()
                    
                    st.dataframe(
                        display_results,
                        column_config={
                            "Pct Selected": st.column_config.ProgressColumn(
                                "Pct Selected",
                                help="Percentage of optimal lineups",
                                format="%.1f%%",
                                min_value=0,
                                max_value=100,
                            ),
                        },
                        use_container_width=True,
                        hide_index=True,
                        height=400
                    )
        
        with col2:
            st.header("📋 My Team")
            
            # Display current team
            team_display = create_my_team_display(selected_data, settings['pos_require'])
            st.dataframe(team_display, use_container_width=True, hide_index=True)
            
            # Calculate draft status based on current selections
            my_team_players = selected_data[selected_data['MyTeam'] == True]
            num_selected = len(my_team_players)
            adjusted_picks = sim.calculate_adjusted_picks(num_selected)
            
            st.subheader("📊 Draft Status")
            st.metric("Players Selected", num_selected)
            st.metric("Remaining Picks", len(adjusted_picks))
            
            if len(adjusted_picks) > 0:
                st.write(f"**Next Pick:** {adjusted_picks[0]}")
                if len(adjusted_picks) > 1:
                    st.write(f"**Future Picks:** {adjusted_picks[1:3]}..." if len(adjusted_picks) > 3 else f"**Future Picks:** {adjusted_picks[1:]}")
                
                # Show pick timing info
                total_picks = settings['num_teams'] * settings['num_rounds']
                pick_progress = adjusted_picks[0] / total_picks * 100
                st.progress(pick_progress / 100, text=f"Draft Progress: {pick_progress:.1f}%")
                
            else:
                st.success("All picks completed!")
                
            # Position requirements status
            if len(my_team_players) > 0:
                st.subheader("📋 Position Status")
                pos_status = {}
                for pos, required in settings['pos_require'].items():
                    if pos != 'FLEX':
                        current = len(my_team_players[my_team_players['Pos'] == pos])
                        pos_status[pos] = f"{current}/{required}"
                
                for pos, status in pos_status.items():
                    current, required = map(int, status.split('/'))
                    if current >= required:
                        st.success(f"{pos}: {status} ✓")
                    elif current > 0:
                        st.warning(f"{pos}: {status}")
                    else:
                        st.error(f"{pos}: {status}")
                
    except Exception as e:
        st.error(f"Error initializing simulation: {str(e)}")
        st.write("Please check your database connection and settings.")

if __name__ == '__main__':
    main()
# %%
