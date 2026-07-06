# Apple Stock Price Prediction Using Machine Learning

## Overview

This project investigates whether Apple's (AAPL) next trading day closing price can be predicted from historical daily price and trading data. It follows a complete data science workflow, from exploratory analysis and feature engineering through model training, hyperparameter tuning, and evaluation against a naive baseline.

The project seeks to answer a single question: can historical stock prices and trading information be used to accurately predict the next trading day's closing price? The analysis is deliberately structured to reach an honest answer rather than an impressive-looking one, with explicit safeguards against look-ahead bias and the persistence illusion that inflates performance metrics on trending price data.

## Dataset

The data is drawn from the S&P 500 daily price dataset (Kaggle: `camnugent/sandp500`), from which a single instrument, Apple (AAPL), was isolated to develop the pipeline on one coherent time series.

| Property | Detail |
|---|---|
| Instrument | Apple (AAPL), filtered from the S&P 500 dataset |
| Frequency | Daily (trading days) |
| Period | Approximately 2013 to 2018 |
| Fields | Open, High, Low, Close, Volume (OHLCV) |
| Split adjustment | Confirmed clean across the June 2014 7-for-1 split |
| Dividend adjustment | Not dividend-adjusted (minor, acceptable effect) |
| Calendar integrity | No unexplained missing trading days |

## Methodology

The workflow proceeds through data integrity checks, exploratory analysis, feature engineering, and a modeling phase with rigorous validation.

Data integrity checks confirmed the series was split-adjusted, free of duplicate dates, free of zero or negative volumes, free of phantom price spikes, and consistent in its OHLC relationships.

The following features were engineered entirely from past values to avoid look-ahead bias:

| Feature | Description |
|---|---|
| `close_lag_1`, `close_lag_2`, `close_lag_3`, `close_lag_5` | Closing price 1, 2, 3, and 5 trading days prior |
| `ma_5`, `ma_10`, `ma_20` | Rolling means of close over 5, 10, and 20 days |
| `std_5` | 5-day rolling standard deviation of close (local volatility) |
| `daily_return` | Day-over-day percentage change in close |
| `volume_change` | Day-over-day percentage change in volume |
| `high_low_range` | Daily high minus daily low |
| `open_close_range` | Daily open minus daily close |
| `next_close` | Target: next trading day's closing price |

Exploratory analysis established three findings that shaped the modeling strategy. An Augmented Dickey-Fuller test confirmed that the raw price is non-stationary (p = 0.83) while daily returns are stationary (p near zero). Autocorrelation analysis showed price is almost perfectly autocorrelated while returns are near-random from their own past. A correlation review revealed severe multicollinearity among the price-level features, with the only non-trivial non-price signal concentrated in the volatility features.

The modeling phase applied the following safeguards. The data was split chronologically, never shuffled. Feature scaling statistics were learned from the training set only. Hyperparameter tuning used an expanding-window `TimeSeriesSplit` with a buffer gap between train and validation folds. The held-out test set was evaluated only once, after all tuning was complete. Every model was compared against a naive persistence baseline that predicts tomorrow's close equals today's close.

## Results

The naive persistence baseline produced the reference performance below. Its R-squared of 0.9752 is notable: a rule requiring no learning achieves an apparently excellent R-squared purely through the autocorrelation of price, which demonstrates that R-squared is not a meaningful measure of skill for this target.

| Metric | Naive Baseline |
|---|---|
| MSE | 3.5802 |
| RMSE | 1.8921 |
| MAE | 1.3072 |
| MAPE | 0.83% |
| R-squared | 0.9752 |

Five model configurations were evaluated on the price-level target. None beat the baseline's MSE of 3.5802, while all reported a near-identical R-squared around 0.97.

| Model | MSE | MAE | R-squared | Beats baseline? |
|---|---|---|---|---|
| Naive baseline | 3.5802 | 1.3072 | 0.9752 | — |
| Linear Regression | 3.6702 | — | 0.9746 | No |
| Ridge (untuned) | 4.1237 | 1.4179 | 0.9715 | No |
| Lasso | 4.3832 | 1.4690 | 0.9697 | No |
| Ridge (tuned, alpha = 0.001) | 3.6908 | 1.3405 | 0.9745 | No |

Directional accuracy was measured on the log-return target to test whether models could at least predict the sign of the next move. The results converged tightly across structurally different model families, all landing near 47%, slightly below the 50% expected by chance.

| Model (log-return target) | Directional Accuracy |
|---|---|
| Ridge Regression | 47.65% |
| XGBoost | 47.48% |
| Gradient Boosting | 47.24% |
| Random guessing (reference) | ~50.00% |

## Key Findings

No model beat the naive persistence baseline on error metrics, and directional accuracy across all model families remained below chance. Next-day AAPL price movement was not reliably predictable from the engineered features over the period studied, a result consistent with the weak-form efficient market hypothesis.

Hyperparameter tuning drove Ridge toward a very small regularization strength (alpha = 0.001), effectively concluding that regularization does not help and that the model should behave as closely as possible to ordinary least squares. Stronger regularization made performance distinctly worse, confirming that the features carry little genuine signal.

The convergence of results across linear, bagged-tree, and boosted-tree approaches indicates that the limiting factor is the data itself rather than the choice or tuning of any particular model. The rigor of the pipeline, including clean data, chronologically honest splits, no leakage, and an explicit baseline, is what allows this conclusion to be trusted. Exploratory analysis suggests volatility forecasting as a more tractable reformulation, since volatility, unlike direction, clusters and persists over time.

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Statistical tests | statsmodels |
| Machine learning | scikit-learn |
| Gradient boosting | xgboost |
| Data access | kagglehub |
