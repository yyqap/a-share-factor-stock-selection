# Quant Research Strategy Prototypes

This repository contains two Python prototypes for systematic investing research. It is organized as a concise portfolio project for quantitative research internship applications: source code and project summaries are included, while raw datasets, internship reports, internal reference PDFs, and other sensitive materials are excluded.

## Task 1: Factor-Based A-Share Stock Selection

Task 1 develops a cross-sectional A-share stock selection model using a monthly factor panel. The model labels the top and bottom 30% of next-month excess returns, preprocesses 70 equity factors with MAD-based winsorization and standardization, and trains a logistic regression classifier to estimate stock-level positive-return probabilities.

The prototype then uses out-of-sample predictions to construct a monthly top-N portfolio weighted by predicted probability. The workflow covers factor preprocessing, supervised classification, out-of-sample forecasting, and basic portfolio backtesting.

Code: `src/task1_factor_stock_selection.py`

## Task 2: GMAT-Style Multi-Asset Trend Strategy

Task 2 implements a GMAT-style global multi-asset allocation prototype. The strategy covers equity indices, government bonds, and commodities, and combines trend-following signals with volatility-aware position sizing, asset-level caps, and periodic rebalancing.

The code includes GMAT2-style and GMAT3-style configurations. It uses reproducible synthetic price paths to validate the implementation structure, rather than publishing or relying on proprietary market data.

Code: `src/task2_gmat2_style_allocation.py` and `src/task2_gmat3_style_allocation.py`

## Repository Structure

```text
.
├── src/
│   ├── task1_factor_stock_selection.py
│   └── task2_gmat2_style_allocation.py
│   └── task2_gmat3_style_allocation.py
├── requirements.txt
└── README.md
```


## Notes

- Task 1 requires local factor CSV files, which are not included in this public repository.
- Task 2 uses simulated asset prices for implementation validation.
- This repository is for research demonstration only and does not provide investment advice.
