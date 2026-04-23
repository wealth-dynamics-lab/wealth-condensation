"""
Panel Threshold Regression (Hansen 1999)
Author: Wealth Dynamics Lab
Date: 2026
Purpose: Estimate thresholds for epsilon and CRJ on Top10% wealth share
         using within-transformed fixed-effects panel regression.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

def within_demean(df, group_col, vars_to_demean):
    """Apply within transformation (fixed effects) by demeaning within groups."""
    df = df.copy()
    for var in vars_to_demean:
        group_mean = df.groupby(group_col)[var].transform('mean')
        df[var + '_dm'] = df[var] - group_mean
    return df

def estimate_threshold(y, q, x_dm, grid):
    """
    Grid search to find threshold that minimizes SSR.
    """
    ssr_values = []
    for gamma in grid:
        d = (q <= gamma).astype(int)
        x1 = x_dm * d
        x2 = x_dm * (1 - d)
        X = np.column_stack([x1, x2])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        ssr = np.sum(resid**2)
        ssr_values.append(ssr)
    best_idx = np.argmin(ssr_values)
    return grid[best_idx], ssr_values[best_idx]

def run_panel_threshold_regression(df, threshold_var, dependent_var, n_grid=100):
    """
    Main function for Hansen panel threshold regression.
    """
    # Within demeaning
    df_dm = within_demean(df, 'country', [dependent_var, threshold_var])
    
    y = df_dm[dependent_var + '_dm'].values
    q = df_dm[threshold_var].values
    x_dm = df_dm[threshold_var + '_dm'].values
    
    # Grid for threshold search
    grid = np.percentile(q, np.linspace(10, 90, n_grid))
    
    # Estimate threshold
    gamma_hat, min_ssr = estimate_threshold(y, q, x_dm, grid)
    
    print(f"Estimated threshold for {threshold_var}: {gamma_hat:.4f}")
    print(f"Minimum SSR: {min_ssr:.4f}")
    
    return gamma_hat, min_ssr

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/WID_Data_21042026-163112.csv', 
                     delimiter=';', 
                     skiprows=1)
    
    # Rename and prepare (adjust column names based on your exact dataset)
    df = df.rename(columns={df.columns[0]: 'Percentile', df.columns[1]: 'Year'})
    # Assume CRJ and epsilon have already been computed and added
    # For demonstration, we use epsilon as threshold variable and Top10 as dependent
    
    # Example: run for epsilon
    gamma_epsilon, _ = run_panel_threshold_regression(df, threshold_var='epsilon', dependent_var='p90p100')
    
    # Example: run for CRJ
    gamma_crj, _ = run_panel_threshold_regression(df, threshold_var='CRJ', dependent_var='p90p100')
    
    print("\nThreshold regression completed successfully!")
