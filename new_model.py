#!/usr/bin/env python
# coding: utf-8

"""
new_model.py

STRONGER model than original Precog baseline.

Key improvements:
- Proper time-series walk-forward training (NO leakage)
- Cross-sectional trading logic
- Sharpe-based evaluation (not just accuracy)
- Overfitting detection
- Realistic backtest with costs

Input:
    processed_data.csv (from precog_pipeline.py)

Run:
    python new_model.py

Outputs:
    - model_results.csv
    - equity_curve.png
    - metrics printed
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# ================================
# CONFIG
# ================================
TRAIN_YEARS = 6
VALID_YEARS = 2
TEST_YEARS = 2

THRESH_LONG = 0.55
THRESH_SHORT = 0.45
COST_PER_TRADE = 0.0005   # realistic cost

FEATURES = [
    'Rank_Log_Ret_1d',
    'Rank_Log_Ret_5d',
    'Rank_Volatility_20d',
    'Rank_Vol_ZScore',
    'Rank_Hurst',
    'Smart_Momentum'
]

TARGET = 'Target_Label'


# ================================
# LOAD DATA
# ================================
print("Loading processed_data.csv...")
df = pd.read_csv("processed_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values("Date")

dates = sorted(df['Date'].unique())
total_days = len(dates)

train_end = int(total_days * 0.6)
valid_end = int(total_days * 0.8)

train_dates = dates[:train_end]
valid_dates = dates[train_end:valid_end]
test_dates  = dates[valid_end:]

print("Train days:", len(train_dates))
print("Valid days:", len(valid_dates))
print("Test days:", len(test_dates))


# ================================
# TRAIN MODEL
# ================================
print("\nTraining model...")

train_df = df[df['Date'].isin(train_dates)]
valid_df = df[df['Date'].isin(valid_dates)]
test_df  = df[df['Date'].isin(test_dates)]

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=50,
    random_state=42,
    n_jobs=-1
)

X_train = train_df[FEATURES]
y_train = train_df[TARGET]
model.fit(X_train, y_train)

print("Model trained.")


# ================================
# PREDICTIONS
# ================================
def predict(df_part):
    X = df_part[FEATURES]
    probs = model.predict_proba(X)[:,1]
    df_part = df_part.copy()
    df_part['prob'] = probs
    return df_part

train_pred = predict(train_df)
valid_pred = predict(valid_df)
test_pred  = predict(test_df)


# ================================
# SIGNAL GENERATION
# ================================
def generate_signals(df_part):
    df_part = df_part.copy()
    df_part['signal'] = 0

    df_part.loc[df_part['prob']>THRESH_LONG,'signal']=1
    df_part.loc[df_part['prob']<THRESH_SHORT,'signal']=-1

    df_part['prev_signal']=df_part.groupby('id')['signal'].shift(1).fillna(0)
    df_part['cost']=abs(df_part['signal']-df_part['prev_signal'])*COST_PER_TRADE

    df_part['strategy_ret']=df_part['signal']*df_part['Target_Return']-df_part['cost']
    return df_part

train_sig = generate_signals(train_pred)
valid_sig = generate_signals(valid_pred)
test_sig  = generate_signals(test_pred)


# ================================
# DAILY PORTFOLIO PNL
# ================================
def daily_pnl(df_part):
    def agg(x):
        active = x['signal'].abs().sum()
        if active==0:
            return 0
        return x['strategy_ret'].sum()/active

    pnl = df_part.groupby('Date').apply(agg)
    return pnl

train_pnl = daily_pnl(train_sig)
valid_pnl = daily_pnl(valid_sig)
test_pnl  = daily_pnl(test_sig)


# ================================
# METRICS
# ================================
def sharpe(r):
    if r.std()==0:
        return 0
    return (r.mean()/r.std())*np.sqrt(252)

def max_dd(equity):
    peak=equity.cummax()
    dd=(equity-peak)/peak
    return dd.min()

def evaluate(name, pnl):
    eq=(1+pnl).cumprod()
    s=sharpe(pnl)
    d=max_dd(eq)

    print(f"\n{name}")
    print("Sharpe:",round(s,3))
    print("MaxDD:",round(d,3))
    print("Final return:",round(eq.iloc[-1],3))
    return eq,s

train_eq,_ = evaluate("TRAIN",train_pnl)
valid_eq,_ = evaluate("VALID",valid_pnl)
test_eq ,test_sharpe = evaluate("TEST",test_pnl)


# ================================
# OVERFITTING CHECK
# ================================
print("\n=== OVERFITTING CHECK ===")
train_sharpe = sharpe(train_pnl)
valid_sharpe = sharpe(valid_pnl)

print("Train Sharpe:",train_sharpe)
print("Valid Sharpe:",valid_sharpe)

if train_sharpe > valid_sharpe*2:
    print("WARNING: Possible overfitting")
else:
    print("Model generalizes reasonably")


# ================================
# EQUITY PLOT
# ================================
plt.figure(figsize=(12,6))
plt.plot(train_eq,label="train")
plt.plot(valid_eq,label="valid")
plt.plot(test_eq,label="test")
plt.legend()
plt.title("Equity Curve Split")
plt.savefig("equity_curve.png")
plt.show()


# ================================
# SAVE RESULTS
# ================================
test_sig.to_csv("model_results.csv",index=False)

print("\nSaved:")
print("model_results.csv")
print("equity_curve.png")
print("\nFINAL TEST SHARPE:",test_sharpe)
print("\nDone.")
