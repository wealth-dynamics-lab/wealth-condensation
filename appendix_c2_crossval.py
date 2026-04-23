"""
Appendix C-2: Subsample Cross-Validation for Threshold Stability
Author: Wealth Dynamics Lab
Date: 2026
Purpose: Perform 500 random subsamples (30 out of 40 countries) and estimate
         threshold values for epsilon and CRJ to validate stability of main results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def within_demean(df, group_col, vars_to_demean):
    """Within transformation for fixed effects."""
    df = df.copy()
    for var in vars_to_demean:
        group_mean = df.groupby(group_col)[var].transform('mean')
        df[var + '_dm'] = df[var] - group_mean
    return df

def estimate_threshold(y, q, x_dm, grid):
    """Grid search to minimize SSR for threshold estimation."""
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
    return grid[best_idx]

def run_subsample_cross_validation(df, threshold_var, dependent_var, n_subsamples=500, subsample_size=30):
    """
    Main cross-validation: randomly draw 30 countries, estimate threshold, repeat 500 times.
    """
    countries = df['country'].unique()
    thresholds = []
    
    for _ in range(n_subsamples):
        # Random subsample
        sample_countries = np.random.choice(countries, subsample_size, replace=False)
        sample_df = df[df['country'].isin(sample_countries)].copy()
        
        # Within demeaning
        sample_dm = within_demean(sample_df, 'country', [dependent_var, threshold_var])
        
        y = sample_dm[dependent_var + '_dm'].values
        q = sample_dm[threshold_var].values
        x_dm = sample_dm[threshold_var + '_dm'].values
        
        # Grid search
        grid = np.percentile(q, np.linspace(10, 90, 100))
        gamma_hat = estimate_threshold(y, q, x_dm, grid)
        thresholds.append(gamma_hat)
    
    return np.array(thresholds)

def plot_cross_validation(thresholds, threshold_var):
    """
    Generate histogram for Appendix C-2.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(thresholds, bins=40, density=True, alpha=0.75, color='steelblue', edgecolor='black')
    
    mean_thresh = np.mean(thresholds)
    plt.axvline(mean_thresh, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_thresh:.4f}')
    plt.axvline(np.percentile(thresholds, 2.5), color='gray', linestyle=':', linewidth=1.5)
    plt.axvline(np.percentile(thresholds, 97.5), color='gray', linestyle=':', linewidth=1.5)
    
    plt.xlabel(f'{threshold_var} Threshold')
    plt.ylabel('Frequency')
    plt.title(f'Appendix C-2: Subsample Cross-Validation for {threshold_var} (500 draws of 30 countries)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f'appendix_c2_crossval_{threshold_var.lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Appendix C-2 cross-validation for {threshold_var} completed!")
    print(f"Saved as: {filename}")
    print(f"Mean threshold: {mean_thresh:.4f}")
    print(f"95% interval: [{np.percentile(thresholds, 2.5):.4f}, {np.percentile(thresholds, 97.5):.4f}]")

if __name__ == "__main__":
    # Load panel data (must contain country, year, epsilon, CRJ, p90p100)
    df = pd.read_csv('data/WID_Data_21042026-163112.csv', delimiter=';', skiprows=1)
    # Rename columns as needed (adjust to your exact dataset)
    df = df.rename(columns={df.columns[0]: 'Percentile', df.columns[1]: 'Year'})
    
    # Run for epsilon
    thresholds_epsilon = run_subsample_cross_validation(df, threshold_var='epsilon', dependent_var='p90p100')
    plot_cross_validation(thresholds_epsilon, 'Epsilon')
    
    # Run for CRJ
    thresholds_crj = run_subsample_cross_validation(df, threshold_var='CRJ', dependent_var='p90p100')
    plot_cross_validation(thresholds_crj, 'CRJ')
    
    print("\nAppendix C-2 cross-validation completed successfully for both variables!")
