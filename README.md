
This repository contains two quantitative trading pipelines built on the **Precog Quant Task 2026 dataset** (100 anonymized S&P500 assets, 10 years daily OHLCV).

The goal of this project is to:

* Design alpha-generating strategies
* Avoid overfitting
* Evaluate models on unseen data
* Select the most robust trading model

Two pipelines are provided:

### 1. Original Baseline Model (Provided / Converted)

Feature-engineered cross-sectional ML strategy with adaptive retraining.

### 2. Improved Robust Model

Enhanced pipeline designed for:

* better generalization
* overfitting detection
* clean backtesting
* reproducible evaluation

This repo allows a third person to **run both models and objectively decide which one is better**.

---

# Dataset Information

Dataset used:
[https://www.kaggle.com/datasets/iamspace/precog-quant-task-2026](https://www.kaggle.com/datasets/iamspace/precog-quant-task-2026)

Contents:

* 100 CSV files (Asset_001 … Asset_100)
* 10 years daily OHLCV
* anonymized S&P500 subset

Each file contains:
Date, Open, High, Low, Close, Volume

Goal:
Predict next-day return direction and generate trading alpha.

---

# Project Structure

```
repo/
│
├── original_model.py          # Original pipeline (baseline)
├── new_model.py              # Improved robust model
├── compare_models.py          # Runs both and compares performance
├── requirements.txt
├── README.md
└── data/
```

---

# Installation

Create virtual environment:

```
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
```

---

# How to Run Everything

## Step 1 — Download dataset automatically

Dataset downloads automatically via KaggleHub when running code.

OR manually download and place CSVs inside `/data`.

---

## Step 2 — Run baseline model

```
python original_model.py
```

This will:

* load dataset
* generate features
* train adaptive ML model
* run backtest
* print Sharpe ratio
* show equity curve

---

## Step 3 — Run improved model

```
python new_model.py
```

This runs the enhanced strategy with:

* better feature stability
* cleaner training loop
* robust backtest engine
* proper signal generation

Outputs:

* equity curve
* drawdown plot
* Sharpe ratio
* final return

---

## Step 4 — Compare both models (IMPORTANT)

```
python compare_models.py
```

This prints:

* Sharpe ratio comparison
* final returns
* max drawdown
* stability across time

This determines which model is better.

---

# How to Decide Which Model is Better

Use these metrics:

## 1. Sharpe Ratio (MOST IMPORTANT)

Measures risk-adjusted return.

| Sharpe         | Meaning |
| -------------- | ------- |
| <0.5 = weak    |         |
| 0.5–1 = decent |         |
| 1–2 = strong   |         |

> 2 = excellent

Higher Sharpe = better model.

---

## 2. Max Drawdown

Worst capital fall.

Lower drawdown = safer model.

If one model:

* has slightly lower return
* but MUCH lower drawdown

→ it is usually better.

---

## 3. Equity Curve Stability

Check graph:

Good model:
smooth upward curve

Bad model:
spiky or sudden crashes

---

## 4. Out-of-Sample Performance (Overfitting Check)

If:
train performance >> test performance
→ model is overfitting

If:
train ≈ validation ≈ test
→ robust model

---

# How We Prevent Overfitting

This repo uses:

### Rolling training window

Model only sees past data.

### No future leakage

Targets shifted correctly.

### Transaction costs included

Realistic trading simulation.

### Cross-sectional ranking

Prevents scale bias.

### Validation through time

Simulates real trading.

---

# Alpha Logic Used

The model combines:

### Momentum

Recent price movement.

### Volatility

Market uncertainty.

### Volume anomaly

Unusual trading activity.

### Hurst exponent

Trend vs mean reversion detection.

### Cross-sectional ranking

Relative strength across stocks.

These features are fed into a Random Forest classifier to predict next-day direction.

Predictions converted into:

* long positions
* short positions
* neutral positions

Then backtested.

---

# What is Alpha Here?

Alpha = strategy return − benchmark return.

If strategy grows:
$1 → $2
and benchmark grows:
$1 → $1.5

Alpha = +50%

This repo prints final alpha value.

---

# How Backtest Works

Each day:

1. Model predicts probability of up/down
2. Convert to signals:

   * buy if probability > threshold
   * short if < threshold
3. Apply transaction cost
4. Calculate daily portfolio return
5. Build cumulative equity curve

---

# Important Evaluation Questions

When comparing models, ask:

### Does it beat buy-and-hold?

If not → useless.

### Is Sharpe stable?

If Sharpe only high in one period → overfit.

### Is drawdown acceptable?

If -70% drawdown → unusable.

### Does performance collapse on test data?

If yes → overfit.

---

# Expected Output

When you run compare_models.py you should see:

```
Baseline Sharpe: 0.62
Improved Sharpe: 1.18

Baseline Return: 38%
Improved Return: 91%

Max Drawdown Baseline: -35%
Max Drawdown Improved: -18%

Better model: Improved strategy
```

(Example output)

---

# How to Extend This Project

Possible improvements:

* add XGBoost / LightGBM
* sector neutral portfolio
* volatility scaling
* ensemble models
* regime switching models

---

# Final Goal of This Repo

Allow any reviewer to:

* run both models
* verify performance
* detect overfitting
* choose best strategy objectively

This simulates real quant research workflow.

---

# If You Are a Reviewer

Run:

```
python compare_models.py
```

and select model with:

highest Sharpe
lowest drawdown
most stable equity curve

That is the best strategy.

---


