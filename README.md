# Wealth Condensation Analysis Code  
**财富凝聚态诊断完整分析代码**

**Repository for the paper "Wealth Thermalization Phase Transition Equation and CRJ Intensity Model"**

This repository contains all reproducible code for:
- Pareto index inversion and absolute abundance ε calculation  
- Hansen panel threshold regression (ε and CRJ thresholds)  
- Bayesian MCMC posterior sampling (Figure 3)  
- All figures (Figure 1–3, Appendix C-1, C-2)

**本文涉及的帕累托指数反推计算、面板门槛回归、贝叶斯MCMC估计及全部图表生成的完整可复现代码**

## Repository Contents / 仓库内容

- `pareto_inversion.py` → Pareto index α and ε calculation / 帕累托指数反推计算  
- `threshold_regression.py` → Panel threshold regression (Figure C-2) / 面板门槛回归  
- `mcmc_posterior.py` → Bayesian MCMC posterior (Figure 3) / 贝叶斯MCMC估计  
- `figure1_scatter.py` → Figure 1: 40-country ε-CRJ scatter  
- `figure2_timeseries.py` → Figure 2: 6-country time series (1995–2024)  
- `figure3_mcmc_diagnostics.py` → Figure 3 + trace, ACF, ESS diagnostics  
- `appendix_c1_placebo.py` → Appendix C-1 placebo test  
- `appendix_c2_crossval.py` → Appendix C-2 subsample cross-validation  
- `data/` → WID raw data files

## How to Run / 如何运行 (JupyterLab / Python 3.10+)

```bash
pip install -r requirements.txt
jupyter lab
