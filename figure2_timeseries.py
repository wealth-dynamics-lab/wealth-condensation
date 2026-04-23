"""
Figure 2: 6-Country Time Series of ε and CRJ (1995–2024)
Author: Wealth Dynamics Lab
Date: 2026
Purpose: Generate dual-axis time series plot for USA, UK, Japan, Germany, Brazil, India
         matching paper Figure 2 (ε linear + CRJ log scale)
"""

import pandas as pd
import matplotlib.pyplot as plt

def generate_figure2():
    """
    Load pre-computed 6-country data and create Figure 2.
    """
    # Load corrected 6-country data
    df = pd.read_csv('data/fig2_data_final_corrected.csv')
    
    # Ensure required columns
    required = ['country', 'year', 'epsilon', 'CRJ']
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing column: {col}")
    
    countries = ['USA', 'UK', 'Japan', 'Germany', 'Brazil', 'India']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6', '#e67e22']
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # ε on left axis (linear)
    for i, country in enumerate(countries):
        data = df[df['country'] == country].sort_values('year')
        ax1.plot(data['year'], data['epsilon'], 
                 color=colors[i], linewidth=2.5, label=f'{country} (ε)')
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Absolute Abundance ε', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    
    # CRJ on right axis (log scale)
    ax2 = ax1.twinx()
    for i, country in enumerate(countries):
        data = df[df['country'] == country].sort_values('year')
        ax2.plot(data['year'], data['CRJ'], 
                 color=colors[i], linestyle='--', linewidth=2.5, label=f'{country} (CRJ)')
    
    ax2.set_ylabel('Capital Re-concentration Junction (CRJ)', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_yscale('log')
    
    # Title and legend
    plt.title('Figure 2: 6-Country ε and CRJ Time Series Evolution (1995–2024)')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', ncol=2)
    
    plt.tight_layout()
    
    # Save
    plt.savefig('figure2_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Figure 2 generated successfully!")
    print("Saved as: figure2_timeseries.png")

if __name__ == "__main__":
    generate_figure2()
