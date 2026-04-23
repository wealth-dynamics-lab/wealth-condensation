"""
Pareto Index Inversion and Absolute Abundance Calculation
Author: Wealth Dynamics Lab
Date: 2026
Purpose: Compute Pareto shape parameter α and absolute abundance ε
         from WID top wealth shares (p99p100 / p90p100)
"""

import pandas as pd
import numpy as np

def calculate_pareto_inversion(df):
    """
    Calculate Pareto index α and absolute abundance ε.
    
    Formula:
        α = 1 / (1 - log(S_top1 / S_top10) / log(0.1))
        ε = 1 - exp(-(α - 1))
    
    Parameters
    ----------
    df : DataFrame
        Must contain columns: 'Percentile', 'Year', 'p99p100', 'p90p100'
    
    Returns
    -------
    DataFrame with added columns 'alpha' and 'epsilon'
    """
    # Ensure required columns exist
    required_cols = ['Percentile', 'Year', 'p99p100', 'p90p100']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing column: {col}")
    
    # Compute Pareto index α
    df = df.copy()
    df['alpha'] = 1 / (1 - (np.log(df['p99p100'] / df['p90p100']) / np.log(0.1)))
    
    # Compute absolute abundance ε
    df['epsilon'] = 1 - np.exp(-(df['alpha'] - 1))
    
    # Round for readability
    df['alpha'] = df['alpha'].round(4)
    df['epsilon'] = df['epsilon'].round(4)
    
    return df

if __name__ == "__main__":
    # Load raw WID data (adjust path if needed)
    df = pd.read_csv('data/WID_Data_21042026-163112.csv', 
                     delimiter=';', 
                     skiprows=1)
    
    # Rename columns if needed (based on your dataset structure)
    df = df.rename(columns={
        df.columns[0]: 'Percentile',
        df.columns[1]: 'Year'
    })
    
    # Compute
    result = calculate_pareto_inversion(df)
    
    # Save results
    result.to_csv('data/pareto_inversion_results.csv', index=False)
    
    # Summary
    print("Pareto inversion completed successfully!")
    print(result[['Year', 'Percentile', 'alpha', 'epsilon']].head(10))
    print(f"\nResults saved to: data/pareto_inversion_results.csv")
