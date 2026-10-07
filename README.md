# Quantitative Trading Research Pipeline

A quantitative finance research and backtesting pipeline built during the CSOT Quant PS'26 at IIT Delhi. This repository contains end-to-end explorations of statistical signal generation, machine learning return prediction, and cross-sectional portfolio allocation.

The project models real-world institutional constraints, including strict market neutrality, maximum exposure limits, and transaction cost modeling.

## Project Architecture

### 1. Signal Analysis & Benchmarking (`quantweek1.ipynb`)
An analysis of price momentum and moving average (MA) crossover signals on AAPL. 
* **The Strategy:** Implemented a 5-day / 20-day MA crossover signal to capture short-term momentum.
* **Risk Metrics:** Calculated annualized volatility, maximum drawdown, and Sharpe ratio from daily log returns.
* **Results:** Improved the annualized Sharpe ratio from 0.50 (baseline buy-and-hold) to 0.90 by actively managing momentum exposure.

### 2. ML Return Prediction (`week2quant.ipynb`)
A machine learning pipeline to predict forward stock returns over 1,989 trading days.
* **Feature Engineering:** Constructed custom market microstructure and volatility features, including 10-day rolling volatility, 20-day SMA ratios, and high-low spreads.
* **Data Processing:** Handled extreme market anomalies by clipping target variables at the 5th and 95th percentiles.
* **Modeling:** Utilized `TimeSeriesSplit` to prevent lookahead bias while training and evaluating XGBoost and Linear Regression models. 

### 3. Long-Short Portfolio Backtester (`quantweek3.ipynb`)
A highly optimized, vectorized cross-sectional backtester running a market-neutral strategy across a universe of 2,167 stocks over a 10-year period.
* **Vectorized Execution:** Replaced standard row-by-row iteration with Pandas vectorization, reducing backtest latency for 5,000+ days from ~50 minutes to under 3 seconds.
* **Portfolio Construction:** Demeaned daily momentum signals to enforce market neutrality (Zero Net Exposure). Scaled allocations to a gross book value of 1.
* **Risk Management:** Enforced a strict 10% maximum position limit per asset (`clip(-0.1, 0.1)`) to prevent over-concentration.
* **Friction & PnL:** Simulated real-world execution by injecting a 1 bps (0.01%) transaction cost penalty on daily portfolio turnover to calculate and compare **Gross Sharpe** vs. **Net Sharpe**.

## Technology Stack
* **Languages:** Python
* **Data Manipulation:** Pandas, NumPy, Parquet
* **Machine Learning:** Scikit-learn, XGBoost
* **Financial Data:** yfinance
* **Visualization:** Matplotlib

## Key Takeaways
This pipeline highlights the transition from theoretical ML models to realistic quantitative research. By explicitly measuring trading friction (turnover penalties) and optimizing the backtest engine via vectorization, the project accurately simulates the structural and performance constraints of a mid-frequency statistical arbitrage strategy.
