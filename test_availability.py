import sys
sys.path.append('app')
from zSim_Helper import FootballSimulation
import sqlite3

# Test the updated availability logic
conn = sqlite3.connect("app/Simulation.sqlite3")
year = 2025
league = 'dk'
num_teams = 12
num_rounds = 5  # Just test first 5 rounds
my_pick_position = 7
num_iters = 10  
pos_require_start = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1}  

try:
    sim = FootballSimulation(conn, year, pos_require_start, num_teams, num_rounds, my_pick_position,
                             pred_vers='final_ensemble', league=league, use_ownership=0)
    
    print(f"Snake picks: {sim.my_picks}")
    
    # Test scenario: NO pre-selected players first
    to_add = []  # No pre-selected players
    to_drop = []  
    
    print(f"Adjusted picks with no selections: {sim.calculate_adjusted_picks(len(to_add))}")
    
    results = sim.run_sim(to_add, to_drop, num_iters, num_avg_pts=3, upside_frac=0, next_year_frac=0)
    
    print(f"\nTotal results: {len(results)}")
    print(f"Round 1 available count: {(results['Round1Available'] > 0).sum()}")
    print(f"Round 2 available count: {(results['Round2Available'] > 0).sum()}")
    print(f"Round 3 available count: {(results['Round3Available'] > 0).sum()}")
    
    # Show max availability for each round
    print(f"\nMax Round 1 availability: {results['Round1Available'].max()}")
    print(f"Max Round 2 availability: {results['Round2Available'].max()}")
    print(f"Max Round 3 availability: {results['Round3Available'].max()}")
    
    print("\nFor the current round logic to work:")
    print("- Round 1 should have all players available (since it's the current round)")
    print("- Round 2+ should have ADP-based availability")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
