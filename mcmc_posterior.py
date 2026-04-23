"""
Bayesian MCMC Posterior Sampling for Edge Extraction Coefficient β
Author: Wealth Dynamics Lab
Date: 2026
Purpose: Generate posterior draws for β (edge extraction coefficient)
         and produce Figure 3 diagnostics (trace plot, ACF, ESS)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma

def log_posterior(beta, data, prior_shape=2.0, prior_rate=1.0):
    """
    Log posterior for beta under Gamma prior + Gaussian likelihood.
    """
    # Simulated likelihood (replace with real panel data likelihood in production)
    likelihood = -0.5 * np.sum((data - beta)**2)
    prior = (prior_shape - 1) * np.log(beta) - prior_rate * beta
    return likelihood + prior

def metropolis_hastings(n_samples=10000, burn_in=2000, proposal_sd=0.05):
    """
    Simple Metropolis-Hastings sampler for beta.
    """
    # Initialize
    beta_current = 0.32
    samples = np.zeros(n_samples)
    
    for i in range(n_samples):
        beta_proposal = np.random.normal(beta_current, proposal_sd)
        if beta_proposal <= 0:
            samples[i] = beta_current
            continue
        
        # Data simulation for demonstration (in real run, load from panel)
        simulated_data = np.random.normal(0.32, 0.05, 500)
        
        log_post_current = log_posterior(beta_current, simulated_data)
        log_post_proposal = log_posterior(beta_proposal, simulated_data)
        
        if np.log(np.random.rand()) < (log_post_proposal - log_post_current):
            beta_current = beta_proposal
        
        samples[i] = beta_current
    
    # Discard burn-in
    posterior = samples[burn_in:]
    return posterior

def plot_diagnostics(posterior):
    """
    Generate Figure 3 diagnostics: trace, ACF, ESS.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Trace plot
    axes[0].plot(posterior[:2000], alpha=0.7)
    axes[0].set_title("Trace Plot of β")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("β")
    
    # ACF
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(posterior, lags=50, ax=axes[1])
    axes[1].set_title("Autocorrelation Function (ACF)")
    
    # ESS
    ess = len(posterior) / (1 + 2 * np.sum(np.correlate(posterior - np.mean(posterior), 
                                                         posterior - np.mean(posterior), 
                                                         mode='full')[len(posterior):] / np.var(posterior)))
    axes[2].hist(posterior, bins=50, density=True, alpha=0.7, color='steelblue')
    axes[2].axvline(np.mean(posterior), color='red', linestyle='--', label=f'Mean = {np.mean(posterior):.4f}')
    axes[2].set_title(f"Posterior Distribution of β (ESS = {ess:.0f})")
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('figure3_mcmc_diagnostics.png', dpi=300)
    plt.close()
    
    print(f"MCMC completed. Posterior mean of β: {np.mean(posterior):.4f}")
    print(f"Effective Sample Size (ESS): {ess:.0f}")
    print("Figure 3 saved as: figure3_mcmc_diagnostics.png")

if __name__ == "__main__":
    posterior_samples = metropolis_hastings(n_samples=12000, burn_in=2000)
    
    # Save posterior draws
    pd.DataFrame({'beta_draw': posterior_samples}).to_csv('data/mcmc_posterior_draws.csv', index=False)
    
    # Generate diagnostics
    plot_diagnostics(posterior_samples)
    
    print("\nBayesian MCMC posterior sampling completed successfully!")
