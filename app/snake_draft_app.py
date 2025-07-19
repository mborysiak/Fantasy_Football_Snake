#%%
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from zSim_Helper import FootballSimulation
import io
from datetime import datetime

# Configuration
db_name = 'Simulation.sqlite3'
year = 2025
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
    player_data = player_data[['player', 'pos', 'years_of_experience', 'pred_fp_per_game', 'avg_pick', 'adp_std_dev', 'adp_min_pick', 'adp_max_pick']]
    
    # Rename columns for display
    player_data = player_data.rename(columns={
        'player': 'Player',
        'pos': 'Pos', 
        'avg_pick': 'ADP',
        'pred_fp_per_game': 'PredPPG',
        'years_of_experience': 'Years_Exp',
        'adp_std_dev': 'ADP_StdDev',
        'adp_min_pick': 'ADP_Min',
        'adp_max_pick': 'ADP_Max',
    })
    
    # Round values for display
    player_data['PredPPG'] = player_data['PredPPG'].round(1)
    player_data['ADP'] = player_data['ADP'].round(1)
    player_data['ADP_StdDev'] = player_data['ADP_StdDev'].round(1)
    player_data['ADP_Min'] = player_data['ADP_Min'].round(1)
    player_data['ADP_Max'] = player_data['ADP_Max'].round(1)

    # Add selection columns
    player_data['MyTeam'] = False
    player_data['OtherTeam'] = False
    
    # Sort by ADP (best players first)
    player_data = player_data.sort_values(by='ADP')
    
    # Reorder columns
    player_data = player_data[['Player', 'Pos', 'Years_Exp', 'MyTeam', 'OtherTeam', 'ADP', 'PredPPG', 'ADP_StdDev', 'ADP_Min', 'ADP_Max']]
    
    return player_data

def create_interactive_grid(data, key_suffix=""):
    """Create the interactive data editor for player selection"""
    selected = st.data_editor(
        data,
        key=f"player_grid_{key_suffix}",
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
    
    # Keep all the round-specific columns
    return results

def get_round_recommendations(results, sim, my_team_count, max_rounds=3):
    """Get top recommendations for the current and next few rounds"""
    
    # Calculate which rounds to show
    adjusted_picks = sim.calculate_adjusted_picks(my_team_count)
    
    if len(adjusted_picks) == 0:
        return None, []
    
    # Determine the rounds we care about
    current_round_num = my_team_count + 1
    rounds_to_show = []
    
    for i in range(min(max_rounds, len(adjusted_picks))):
        round_num = current_round_num + i
        rounds_to_show.append(round_num)
    
    # Create round-specific recommendations
    round_data = {}
    
    for round_num in rounds_to_show:
        count_col = f'Round{round_num}Count'
        available_col = f'Round{round_num}Available'
        
        # Check if the required columns exist
        if all(col in results.columns for col in [count_col, available_col]) and 'player' in results.columns:
            # Filter for players who were available for this round
            round_results = results[results[available_col] > 0].copy()
            
            if len(round_results) > 0:
                # Calculate percentage selected when available for this round
                round_results[f'Round{round_num}PctAvailable'] = (
                    round_results[count_col] / round_results[available_col] * 100
                ).round(1)
                
                # Sort by actual selection count for this round (not percentage) and take top 10
                top_picks = round_results.nlargest(10, count_col)[
                    ['player', count_col, available_col, f'Round{round_num}PctAvailable']
                ].copy()
                
                # Rename columns for display
                top_picks = top_picks.rename(columns={
                    'player': 'Player',
                    count_col: 'Selected',
                    available_col: 'Available', 
                    f'Round{round_num}PctAvailable': 'Pct Selected'
                })
                
                round_data[round_num] = top_picks
    
    return adjusted_picks, round_data

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

def save_draft_state(selected_data, settings):
    """Save current draft state to CSV format"""
    # Get only the players that have selections
    my_team = selected_data[selected_data['MyTeam'] == True][['Player', 'Pos', 'ADP', 'PredPPG']].copy()
    other_team = selected_data[selected_data['OtherTeam'] == True][['Player', 'Pos', 'ADP', 'PredPPG']].copy()
    
    # Add team designation
    my_team['Team'] = 'MyTeam'
    other_team['Team'] = 'OtherTeam'
    
    # Combine the data
    draft_data = pd.concat([my_team, other_team], ignore_index=True)
    
    # Add settings as metadata (we'll store this in the CSV as well)
    settings_data = pd.DataFrame([{
        'Type': 'Settings',
        'League': settings['league'],
        'NumTeams': settings['num_teams'],
        'MyPickPosition': settings['my_pick_position'],
        'NumRounds': settings['num_rounds'],
        'QB': settings['pos_require']['QB'],
        'RB': settings['pos_require']['RB'],
        'WR': settings['pos_require']['WR'],
        'TE': settings['pos_require']['TE'],
        'NumIters': settings['num_iters'],
        'SavedDate': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }])
    
    return draft_data, settings_data

def load_draft_state(uploaded_file):
    """Load draft state from uploaded CSV file"""
    try:
        # Read the CSV
        df = pd.read_csv(uploaded_file)
        
        # Check if this is a valid draft state file
        if 'Team' not in df.columns:
            return None, None, "Invalid file format. Please upload a valid draft state CSV."
        
        # Extract settings if they exist (for backward compatibility)
        settings_data = None
        if 'Type' in df.columns:
            settings_rows = df[df['Type'] == 'Settings']
            if len(settings_rows) > 0:
                settings_data = settings_rows.iloc[0]
                # Remove settings rows from main data
                df = df[df['Type'] != 'Settings']
        
        return df, settings_data, None
        
    except Exception as e:
        return None, None, f"Error loading file: {str(e)}"

def apply_loaded_state(player_data, loaded_data):
    """Apply loaded draft state to current player data"""
    if loaded_data is None or len(loaded_data) == 0:
        return player_data
    
    # Reset all selections
    player_data['MyTeam'] = False
    player_data['OtherTeam'] = False
    
    # Apply loaded selections
    for _, row in loaded_data.iterrows():
        player_name = row['Player']
        team = row['Team']
        
        # Find the player in current data
        player_idx = player_data[player_data['Player'] == player_name].index
        
        if len(player_idx) > 0:
            if team == 'MyTeam':
                player_data.loc[player_idx, 'MyTeam'] = True
            elif team == 'OtherTeam':
                player_data.loc[player_idx, 'OtherTeam'] = True
    
    return player_data

#------------------
# App Components
#------------------

def render_save_button(selected_data, settings, placeholder):
    """Render the Save & Download button in the sidebar placeholder"""
    
    # Count current selections to determine if button should be enabled
    my_team_count = len(selected_data[selected_data['MyTeam'] == True])
    other_team_count = len(selected_data[selected_data['OtherTeam'] == True])
    has_selections = my_team_count > 0 or other_team_count > 0
    
    if has_selections:
        # Generate the CSV data
        draft_data, settings_data = save_draft_state(selected_data, settings)
        
        if len(draft_data) > 0:
            # Combine draft data and settings
            combined_data = pd.concat([draft_data, settings_data], ignore_index=True)
            
            # Create CSV data
            csv_buffer = io.StringIO()
            combined_data.to_csv(csv_buffer, index=False)
            csv_string = csv_buffer.getvalue()
            
            # Generate filename with custom name and timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = st.session_state.save_file_name
            if file_name.strip():
                # Clean the filename (remove invalid characters)
                clean_name = "".join(c for c in file_name.strip() if c.isalnum() or c in (' ', '-', '_')).strip()
                clean_name = clean_name.replace(' ', '_')
                filename = f"{clean_name}_{timestamp}.csv"
            else:
                filename = f"draft_state_{timestamp}.csv"
            
            # Direct download button
            placeholder.download_button(
                label="💾 Save & Download Draft",
                data=csv_string,
                file_name=filename,
                mime="text/csv",
                help="Save and download current draft selections as CSV",
                key="save_draft_button"
            )
        else:
            placeholder.button("💾 Save & Download Draft", disabled=True, help="No players selected to save")
    else:
        placeholder.button("💾 Save & Download Draft", disabled=True, help="No players selected to save")

def sidebar_controls():
    """Create sidebar controls"""
    st.sidebar.header("Draft Settings")
    
    # League selection
    league = st.sidebar.selectbox(
        'League Type',
        options=['nffc', 'dk'],
        index=0,  # Default to 'nffc'
        help="Select the league type for predictions and ADP data"
    )
    
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
    
    st.sidebar.header("Save/Load Draft")
    
    # File upload for loading draft state
    uploaded_file = st.sidebar.file_uploader(
        "Load Draft State",
        type=['csv'],
        help="Upload a previously saved draft state CSV file",
        key=f"file_uploader_{st.session_state.file_uploader_key}"
    )
    
    if uploaded_file is not None:
        loaded_data, loaded_settings, error = load_draft_state(uploaded_file)
        if error:
            st.sidebar.error(error)
        else:
            st.session_state.loaded_draft_data = loaded_data
            st.session_state.loaded_settings_data = loaded_settings
            st.session_state.data_loaded_applied = False  # Reset flag to allow reapplication
            st.sidebar.success("Draft state loaded successfully!")
            
            # Show loaded info
            if loaded_settings is not None:
                st.sidebar.write(f"**Loaded:** {loaded_settings.get('SavedDate', 'Unknown date')}")
                st.sidebar.write(f"**League:** {loaded_settings.get('League', 'Unknown').upper()}")
            
            st.sidebar.write(f"**Players loaded:** {len(loaded_data) if loaded_data is not None else 0}")
    
    # Save draft state section
    st.sidebar.subheader("Save Current Draft")
    
    # File name input
    file_name = st.sidebar.text_input(
        "Draft Name (optional)",
        placeholder="e.g., my_draft, league_name",
        help="Enter a custom name for your draft file"
    )
    
    # Update session state with file name
    st.session_state.save_file_name = file_name
    
    # Create a placeholder for the save button that will be populated later
    save_button_placeholder = st.sidebar.empty()
    
    # Clear selections section  
    st.sidebar.subheader("Clear Draft")
    
    # Clear all selections button with confirmation
    if st.sidebar.button("🗑️ Clear All Selections", help="Remove all player selections"):
        st.session_state.confirm_clear = True
    
    # Handle confirmation dialog
    if 'confirm_clear' in st.session_state and st.session_state.confirm_clear:
        with st.sidebar:
            st.warning("⚠️ Are you sure you want to clear all selections?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, Clear All", key="confirm_clear_yes"):
                    st.session_state.clear_selections = True
                    st.session_state.confirm_clear = False
                    # Clear any loaded data to ensure it doesn't interfere
                    st.session_state.loaded_draft_data = None
                    st.session_state.loaded_settings_data = None
                    st.session_state.data_loaded_applied = False
                    # Reset file uploader by changing its key
                    st.session_state.file_uploader_key += 1
            with col2:
                if st.button("❌ Cancel", key="confirm_clear_no"):
                    st.session_state.confirm_clear = False
    
    return {
        'league': league,
        'num_teams': num_teams,
        'num_rounds': num_rounds,
        'my_pick_position': my_pick_position,
        'pos_require': pos_require,
        'num_iters': num_iters,
        'save_button_placeholder': save_button_placeholder
    }

#======================
# Main App
#======================

def main():
    st.set_page_config(page_title="Snake Draft Optimizer", layout="wide")
    
    # Initialize all session state variables at the start
    if 'loaded_draft_data' not in st.session_state:
        st.session_state.loaded_draft_data = None
    if 'loaded_settings_data' not in st.session_state:
        st.session_state.loaded_settings_data = None
    if 'confirm_clear' not in st.session_state:
        st.session_state.confirm_clear = False
    if 'file_uploader_key' not in st.session_state:
        st.session_state.file_uploader_key = 0
    if 'save_file_name' not in st.session_state:
        st.session_state.save_file_name = ""
    if 'clear_selections' not in st.session_state:
        st.session_state.clear_selections = False
    if 'grid_key_counter' not in st.session_state:
        st.session_state.grid_key_counter = 0
    if 'league_changed' not in st.session_state:
        st.session_state.league_changed = False
    if 'data_loaded_applied' not in st.session_state:
        st.session_state.data_loaded_applied = False
    
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
        # Initialize simulation (will be fresh if refresh button was clicked)
        sim = init_sim(
            db_name, 
            pred_vers, 
            year, 
            settings['league'],
            settings['num_teams'],
            settings['num_rounds'],
            settings['my_pick_position'],
            settings['pos_require']
        )
        
        # Reset the flag after initialization
        if st.session_state.league_changed:
            st.session_state.league_changed = False
        
        # Get player data
        player_data = get_player_data(sim)
        
        # Apply loaded draft state if available and not yet applied
        if (st.session_state.loaded_draft_data is not None and 
            not st.session_state.data_loaded_applied):
            player_data = apply_loaded_state(player_data, st.session_state.loaded_draft_data)
            # Mark as applied but don't clear the data yet
            st.session_state.data_loaded_applied = True
        
        # Handle clear selections if requested (before creating the grid)
        if st.session_state.clear_selections:
            # Reset player data by clearing selections
            player_data['MyTeam'] = False
            player_data['OtherTeam'] = False
            st.session_state.clear_selections = False
            # Force grid to reset by changing the key
            st.session_state.grid_key_counter += 1
            st.success("All selections cleared!")
        
        # Get grid key for forcing reset when needed
        grid_key = f"main_{st.session_state.grid_key_counter}"
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("1. Select Players")
            
            # Player selection grid
            selected_data = create_interactive_grid(player_data, key_suffix=grid_key)
            
            # Render the Save & Download button in the sidebar placeholder
            # This ensures we capture the latest grid state
            render_save_button(selected_data, settings, settings['save_button_placeholder'])
            
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
                    
                    # Get round-specific recommendations
                    picks_remaining, round_recommendations = get_round_recommendations(
                        results, sim, num_selected, max_rounds=3
                    )
                    
                    if round_recommendations:
                        st.write("**Top recommendations by round** (based on optimal draft simulations)")
                        
                        # Add explanation
                        st.info(
                            "🎯 **Current Round**: All available players can be selected on your turn\n\n"
                            "📊 **Future Rounds**: Based on typical ADP patterns and availability"
                        )
                        
                        # Create tabs for each round
                        if len(round_recommendations) == 1:
                            # Single round - no tabs needed
                            round_num = list(round_recommendations.keys())[0]
                            pick_num = picks_remaining[0] if picks_remaining else "Unknown"
                            st.subheader(f"Round {round_num} (Pick #{pick_num})")
                            
                            round_data = round_recommendations[round_num]
                            st.dataframe(
                                round_data,
                                column_config={
                                    "Pct Selected": st.column_config.ProgressColumn(
                                        "Pct Selected",
                                        help="Percentage selected when available in this round",
                                        format="%.1f%%",
                                        min_value=0,
                                        max_value=100,
                                    ),
                                },
                                use_container_width=True,
                                hide_index=True,
                                height=400
                            )
                        else:
                            # Multiple rounds - use tabs
                            tab_labels = []
                            for i, round_num in enumerate(round_recommendations.keys()):
                                pick_num = picks_remaining[i] if i < len(picks_remaining) else "TBD"
                                if i == 0:
                                    tab_labels.append(f"🎯 Round {round_num} (Pick #{pick_num})")
                                else:
                                    tab_labels.append(f"Round {round_num} (Pick #{pick_num})")
                            
                            tabs = st.tabs(tab_labels)
                            
                            for i, (round_num, round_data) in enumerate(round_recommendations.items()):
                                with tabs[i]:
                                    if i == 0:
                                        st.write("**Current round - all available players can be selected**")
                                    else:
                                        st.write("**Future round - based on typical ADP availability**")
                                    
                                    st.dataframe(
                                        round_data,
                                        column_config={
                                            "Pct Selected": st.column_config.ProgressColumn(
                                                "Pct Selected",
                                                help="Percentage selected when available in this round",
                                                format="%.1f%%",
                                                min_value=0,
                                                max_value=100,
                                            ),
                                        },
                                        use_container_width=True,
                                        hide_index=True,
                                        height=350
                                    )
                    else:
                        st.warning("No recommendations available for upcoming rounds.")
        
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
