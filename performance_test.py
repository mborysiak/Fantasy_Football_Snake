"""
Test script to verify the performance improvements in matrix creation
"""
import numpy as np
import pandas as pd
import time

# Mock the availability constraints optimization
def old_create_availability_constraints(num_players, num_rounds):
    """Old method - creates full identity matrix"""
    total_constraints = num_players * num_rounds
    G_availability = np.eye(total_constraints)  # Expensive!
    h_availability = np.ones((total_constraints, 1))
    return G_availability, h_availability

def new_create_availability_constraints(num_players, num_rounds):
    """New method - returns None for G, only creates h vector"""
    total_constraints = num_players * num_rounds
    h_availability = np.ones((total_constraints, 1))
    return None, h_availability

# Test matrix creation performance
def test_matrix_performance():
    print("Testing matrix creation performance...")
    
    test_sizes = [(100, 10), (200, 15), (300, 20)]
    
    for num_players, num_rounds in test_sizes:
        print(f"\nTesting with {num_players} players, {num_rounds} rounds:")
        
        # Test old method
        start_time = time.time()
        for _ in range(100):  # 100 iterations
            G_old, h_old = old_create_availability_constraints(num_players, num_rounds)
        old_time = time.time() - start_time
        
        # Test new method
        start_time = time.time()
        for _ in range(100):  # 100 iterations
            G_new, h_new = new_create_availability_constraints(num_players, num_rounds)
        new_time = time.time() - start_time
        
        speedup = old_time / new_time if new_time > 0 else float('inf')
        print(f"  Old method: {old_time:.4f}s")
        print(f"  New method: {new_time:.4f}s")
        print(f"  Speedup: {speedup:.1f}x")

# Test ADP sampling optimization
def test_adp_sampling():
    print("\n\nTesting ADP sampling performance...")
    
    # Create mock ADP data
    num_players = 300
    num_columns = 1000
    
    # Create DataFrame (old method)
    df = pd.DataFrame(np.random.rand(num_players, num_columns + 2))
    df.columns = ['player', 'pos'] + [f'col_{i}' for i in range(num_columns)]
    
    # Create numpy array (new method)
    array_data = np.random.rand(num_players, num_columns)
    
    # Test old method
    start_time = time.time()
    for _ in range(1000):
        col_idx = np.random.choice(range(2, num_columns + 2))
        sample = df.iloc[:, col_idx].values
    old_time = time.time() - start_time
    
    # Test new method
    start_time = time.time()
    for _ in range(1000):
        col_idx = np.random.randint(0, num_columns)
        sample = array_data[:, col_idx]
    new_time = time.time() - start_time
    
    speedup = old_time / new_time if new_time > 0 else float('inf')
    print(f"  Old method (DataFrame.iloc): {old_time:.4f}s")
    print(f"  New method (numpy array): {new_time:.4f}s")
    print(f"  Speedup: {speedup:.1f}x")

if __name__ == "__main__":
    test_matrix_performance()
    test_adp_sampling()
    print("\n\nOptimizations Summary:")
    print("1. Eliminated expensive np.eye() identity matrix creation")
    print("2. Pre-converted static matrices to cvxopt format")
    print("3. Replaced DataFrame.iloc with direct numpy array access")
    print("4. Replaced np.random.choice with np.random.randint")
    print("5. Vectorized position constraint creation")
    print("6. Added position caching for constraint building")
