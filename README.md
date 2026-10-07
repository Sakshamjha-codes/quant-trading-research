# Quantitative Equity Research Pipeline

A quantitative finance research and backtesting pipeline developed during the CSOT Quant PS'26 at IIT Delhi. This repository documents a three-phase exploration of statistical signal generation, machine learning return prediction, and cross-sectional portfolio allocation. 

Rather than presenting overfitted, unrealistic returns, this project focuses on rigorous evaluation, strict out-of-sample cross-validation, and the severe impact of transaction costs and turnover on theoretical alpha.

## 📊 Phase 1: Signal Analysis & Benchmarking
**Objective:** Evaluate price momentum via moving-average (MA) crossovers on a single-asset baseline (AAPL).
* **Methodology:** Implemented 5-day / 20-day MA crossover signals.
* **Findings:** The strategy improved the in-sample Sharpe ratio from 0.50 (buy-and-hold) to 0.90. However, further analysis revealed the signal possessed minimal directional predictive power (5-day forward return of 0.27% when ON vs 0.29% when OFF). The Sharpe improvement primarily resulted from risk reduction (being out of the market during volatile periods) rather than alpha generation.

## 🤖 Phase 2: Machine Learning Return Prediction
**Objective:** Forecast forward stock returns over 1,989 trading days using tree-based and linear models.
* **Methodology:** Engineered custom features (10-day rolling volatility, 20-day SMA ratios, high-low spreads). To handle market anomalies, target variables were clipped at the 5th and 95th percentiles. Models were evaluated using strict `TimeSeriesSplit` cross-validation to prevent lookahead bias.
* **Findings:** Both XGBoost and Linear Regression models yielded negative $R^2$ scores across validation folds (Linear Regression Mean $R^2$: -0.12). This accurately reflects the inherently low signal-to-noise ratio in raw daily equity returns and demonstrates the difficulty of extracting linear predictive alpha without alternative datasets.

## 📈 Phase 3: Long-Short Market-Neutral Portfolio
**Objective:** Construct a highly optimized, vectorized cross-sectional backtester running a statistical strategy across a point-in-time universe of ~1,000 tradeable stocks.
* **Methodology:** Engineered a Pandas-vectorized backtesting engine (reducing 5,000-day execution latency to under 3 seconds). Demeaned short-term reversal signals to enforce strict market neutrality, capped maximum weights at 10%, and modeled a 1 bps transaction cost penalty on daily turnover.
* **Findings:** 

| Metric | Theoretical (Gross) | Realized (Net of 1 bps Cost) |
| :--- | :--- | :--- |
| **Sharpe Ratio** | 0.42 | -0.47 |
| **Interpretation** | Signal holds edge | Erased by turnover friction |

* **Diagnosis:** The short-term reversal signal requires aggressive daily rebalancing. The resulting portfolio turnover generated transaction costs that completely consumed the theoretical Gross PnL, highlighting the necessity of turnover-smoothing algorithms or longer-horizon signals in production environments.

## ⚠️ Limitations & Biases Handled
* **Survivorship Bias Mitigation:** Utilized a dynamic, point-in-time universe mask rather than a static list of currently active tickers.
* **Lookahead Bias:** Enforced `shift(1)` on all signal generations and strictly utilized `TimeSeriesSplit` for ML model validation.
* **Transaction Cost Friction:** Explicitly modeled the PnL drag caused by turnover, preventing the illusion of profitability on high-frequency signals.
* **Overfitting Risk:** Maintained full transparency on negative out-of-sample $R^2$ results rather than hyper-tuning models to fit the noise.

## 🛠 Technology Stack
**Python** | **Pandas** | **NumPy** | **Scikit-learn** | **XGBoost** | **Matplotlib**
