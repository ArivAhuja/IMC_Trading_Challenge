import numpy as np
import pandas as pd
from itertools import permutations

# define commodities
commodities = ['Snowballs', 'Pizza\'s', 'Silicon Nuggets', 'SeaShells']

# define conversion table as a DataFrame as shown on the site
conversion_rates = pd.DataFrame([
    [1, 1.45, 0.52, 0.72],      # Snowballs to Others
    [0.7, 1, 0.31, 0.48],       # Pizza's to Others
    [1.95, 3.1, 1, 1.49],       # Silicon Nuggets to Others
    [1.34, 1.98, 0.64, 1]       # SeaShells to Others
], columns=commodities, index=commodities)

# track profitable cycles
profitable_cycles = []

# function to calculate profit for a given cycle
def calculate_cycle_profit(cycle):
    # start with 1 unit of SeaShells
    profit = 1  
    # for each step in the cycle, multiply by the conversion rate
    for i in range(len(cycle) - 1):
        from_commodity = cycle[i]
        to_commodity = cycle[i + 1]
        profit *= conversion_rates.loc[from_commodity, to_commodity]
    return profit

# Generate all permutations of commodities of length 1 to 5
for length in range(1, 6):
    for cycle in permutations(commodities[:-1], length):
        # Ensure the cycle starts and ends with 'SeaShells'
        full_cycle = ['SeaShells'] + list(cycle) + ['SeaShells']
        profit = calculate_cycle_profit(full_cycle)
        if profit > 1:
            profitable_cycles.append((full_cycle, profit))


# print the profitable cycles
for cycle, profit in profitable_cycles:
    print(f"Cycle: {' -> '.join(cycle)}, Profit: {profit:.2f}")

