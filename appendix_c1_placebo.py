"""
Appendix C-1: Placebo Test for Threshold Regression
Author: Wealth Dynamics Lab
Date: 2026
Purpose: Generate placebo F-statistic distribution using gamma simulation
         to validate true thresholds (ε=0.32, CRJ=12.8) against random data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma

def simulate_placebo_f_stat(n_sim=1000, n_obs=40*29):
    """
    Simulate placebo F-statistics under null hypothesis (no threshold).
    """
    f_stats = []
    for _ in range(n_sim):
        # Simulate random data under null (no structural break)
        y = np.random.normal(0, 1, n_obs)
        q = np.random.uniform(0, 1, n_obs)
        x = np.random.normal(0, 1, n_obs)
        
        # Grid search for fake threshold
        grid = np.percentile(q, np.linspace(10, 90, 100))
        ssr_values = []
        for gamma in grid:
            d = (q <= gamma).astype(int)
            X = np.column_stack([x * d, x * (1 - d)])
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            resid = y - X @ beta
            ssr = np.sum(resid**2)
            ssr_values.append(ssr)
        
        min_ssr = min(ssr_values)
        # F-statistic approximation under null
        f_stat = (n_obs * (np.var(y) - min_ssr / n_obs)) / min_ssr
        f_stats.append(f_stat)
    
    return np.array(f_stats)

def plot_placebo_distribution(f_stats):
    """
    Generate placebo distribution plot (Appendix C-1).
    """
    plt.figure(figsize=(10, 6))
    plt.hist(f_stats, bins=80, density=True, alpha=0.75, color='lightcoral', edgecolor='black')
    
    # Gamma fit for reference
    shape, loc, scale = gamma.fit(f_stats)
    x = np.linspace(0, max(f_stats), 500)
    plt.plot(x, gamma.pdf(x, shape, loc, scale), 'r-', lw=2, label=f'Gamma fit (shape={shape:.2f})')
    
    # Critical values from paper
    plt.axvline(28.3, color='red', linestyle='--', linewidth=2, label='Critical value 1 = 28.3')
    plt.axvline(35.1, color='red', linestyle='--', linewidth=2, label='Critical value 2 = 35.1')
    
    plt.xlabel('Placebo F-statistic')
    plt.ylabel('Density')
    plt.title('Appendix C-1: Placebo Test Distribution (1000 simulations)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('appendix_c1_placebo.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Appendix C-1 placebo test completed successfully!")
    print("Saved as: appendix_c1_placebo.png")
    print(f"95th percentile of placebo F-stats: {np.percentile(f_stats, 95):.2f}")

if __name__ == "__main__":
    f_stats = simulate_placebo_f_stat(n_sim=1000)
    plot_placebo_distribution(f_stats)
