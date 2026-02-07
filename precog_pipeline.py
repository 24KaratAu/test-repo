#!/usr/bin/env python
# coding: utf-8
"""
precog_pipeline.py

Refactored data pipeline from the notebook 'PrecogPipeline.ipynb'.

Usage:
    # download via kagglehub (if available) and process
    python precog_pipeline.py --download

    # or process an existing local folder containing Asset_*.csv
    python precog_pipeline.py --data-dir path/to/anonymized_data

Output:
    - processed_data.csv (ready for modeling)
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd

# Optional: kagglehub download; keep import local to avoid hard dependency when not used
try:
    import kagglehub
    _HAS_KAGGLEHUB = True
except Exception:
    _HAS_KAGGLEHUB = False

from typing import Optional
pd.options.mode.chained_assignment = None  # silence SettingWithCopy warnings


# -------------------------
# Math / feature helpers
# -------------------------
def get_hurst_exponent(time_series: np.ndarray, max_lag: int = 20) -> float:
    """
    Estimate Hurst exponent using the rescaled range / slope method.
    Returns ~0.5 for random walk, >0.5 trending, <0.5 mean-reverting.
    """
    if len(time_series) < max_lag + 2:
        return 0.5
    lags = range(2, max_lag)
    tau = []
    for lag in lags:
        diff = np.subtract(time_series[lag:], time_series[:-lag])
        tau.append(np.sqrt(np.std(diff)))
    try:
        model = np.polyfit(np.log(lags), np.log(tau), 1)
        return float(model[0] * 2.0)
    except Exception:
        return 0.5


def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a single-asset dataframe with columns ['Date','Open','High','Low','Close','Volume'],
    compute technical features and target labels.
    """
    df = df.copy().reset_index(drop=True)
    df['Close'] = df['Close'].astype(float)
    # Log returns
    df['Log_Ret_1d'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Log_Ret_5d'] = np.log(df['Close'] / df['Close'].shift(5))

    # 20-day volatility of 1d log returns
    df['Volatility_20d'] = df['Log_Ret_1d'].rolling(window=20, min_periods=5).std()

    # Volume z-score (20-day)
    if 'Volume' in df.columns:
        vol_mean = df['Volume'].rolling(20, min_periods=5).mean()
        vol_std = df['Volume'].rolling(20, min_periods=5).std()
        df['Vol_ZScore'] = (df['Volume'] - vol_mean) / (vol_std + 1e-8)
    else:
        df['Vol_ZScore'] = np.nan

    # Hurst exponent computed on rolling windows of Close
    # use .rolling(...).apply with lambda to call get_hurst_exponent
    df['Hurst'] = df['Close'].rolling(window=100, min_periods=50).apply(
        lambda arr: get_hurst_exponent(arr), raw=True
    )

    # Targets: forward 1-day return and binary label
    df['Target_Return'] = df['Log_Ret_1d'].shift(-1)
    df['Target_Label'] = (df['Target_Return'] > 0).astype(int)

    return df


# -------------------------
# Pipeline
# -------------------------
def load_and_process_data(kaggle_download: bool = False, local_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load all Asset_*.csv files and compute features + cross-sectional ranks.
    Returns a processed dataframe with columns:
      ['Date','id','Open','High','Low','Close','Volume',..., 'Rank_*', 'Smart_Momentum', 'Target_*']
    """
    # 1) Acquire files
    if kaggle_download:
        if not _HAS_KAGGLEHUB:
            raise RuntimeError("kagglehub not available. Either install kagglehub or provide --data-dir")
        print("Downloading dataset via kagglehub ...")
        path = kagglehub.dataset_download("iamspace/precog-quant-task-2026")
        target_folder = os.path.join(path, "anonymized_data")
        if not os.path.exists(target_folder):
            target_folder = path
    else:
        if local_dir is None:
            raise ValueError("local_dir must be provided if kaggle_download is False")
        target_folder = local_dir

    all_files = sorted(glob.glob(os.path.join(target_folder, "Asset_*.csv")))
    if len(all_files) == 0:
        raise FileNotFoundError(f"No Asset_*.csv files found in {target_folder}")

    print(f"Found {len(all_files)} asset files. Loading...")

    # 2) Load and concat
    frames = []
    for f in all_files:
        df = pd.read_csv(f)
        # ensure Date column exists
        if 'Date' not in df.columns:
            raise ValueError(f"{f} missing 'Date' column")
        df['id'] = os.path.splitext(os.path.basename(f))[0]  # Asset_001 etc
        frames.append(df)
    full_df = pd.concat(frames, axis=0, ignore_index=True)
    full_df['Date'] = pd.to_datetime(full_df['Date'])
    full_df = full_df.sort_values(['Date', 'id']).reset_index(drop=True)

    print("Calculating features per asset (this may take a few minutes)...")
    # 3) Group-by asset and compute features
    df_processed = full_df.groupby('id', group_keys=False).apply(calculate_technical_features)
    df_processed = df_processed.reset_index(drop=True)

    # 4) Cross-sectional ranking per date
    features_to_rank = ['Log_Ret_1d', 'Log_Ret_5d', 'Volatility_20d', 'Vol_ZScore', 'Hurst']
    for feat in features_to_rank:
        rank_col = f"Rank_{feat}"
        # rank pct per Date; preserve NA handling
        df_processed[rank_col] = df_processed.groupby('Date')[feat].rank(pct=True, method='first')

    # 5) Example composite feature: Smart_Momentum
    df_processed['Smart_Momentum'] = df_processed['Rank_Log_Ret_5d'] * df_processed['Rank_Hurst']

    # 6) Drop rows with NaNs in target/features (optional)
    df_processed = df_processed.dropna(subset=['Target_Return', 'Rank_Log_Ret_1d', 'Rank_Log_Ret_5d'])

    print("Processing complete. Returning processed DataFrame.")
    return df_processed


# -------------------------
# Visualization / Sanity checks
# -------------------------
def visualize_data_physics(df: pd.DataFrame, sample_id: Optional[str] = None):
    """Quick sanity visualizations for one sample asset and rank distribution."""
    import matplotlib.pyplot as plt

    if sample_id is None:
        sample_id = df['id'].unique()[0]

    sample_df = df[df['id'] == sample_id].sort_values('Date').tail(300)
    if sample_df.empty:
        print(f"No data for sample id {sample_id}")
        return

    # Price vs Hurst
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(sample_df['Date'], sample_df['Close'], label='Close', color='C0')
    ax1.set_ylabel('Price', color='C0')
    ax2 = ax1.twinx()
    ax2.plot(sample_df['Date'], sample_df['Hurst'], label='Hurst', color='C1', linestyle='--')
    ax2.axhline(0.5, color='gray', linestyle=':')
    ax2.set_ylabel('Hurst', color='C1')
    plt.title(f'Price vs Hurst — {sample_id}')
    plt.show()

    # Rank histogram
    plt.figure(figsize=(8, 3))
    plt.hist(df['Rank_Log_Ret_1d'].dropna(), bins=40)
    plt.title('Cross-Sectional Rank Distribution — Log_Ret_1d')
    plt.show()


# -------------------------
# CLI
# -------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="Precog data processing pipeline")
    p.add_argument("--download", action="store_true", help="download dataset via kagglehub (requires kagglehub installed & authenticated)")
    p.add_argument("--data-dir", type=str, default=None, help="path to local folder containing Asset_*.csv files")
    p.add_argument("--out", type=str, default="processed_data.csv", help="output CSV filename")
    p.add_argument("--visualize", action="store_true", help="show sample visualizations after processing")
    return p.parse_args()


def main():
    args = _parse_args()
    if args.download:
        if not _HAS_KAGGLEHUB:
            raise RuntimeError("kagglehub not available. Install kagglehub or run with --data-dir")
        df = load_and_process_data(kaggle_download=True)
    else:
        if args.data_dir is None:
            raise ValueError("Please provide --data-dir when not using --download")
        df = load_and_process_data(kaggle_download=False, local_dir=args.data_dir)

    df.to_csv(args.out, index=False)
    print(f"Saved processed data to {args.out}")

    if args.visualize:
        visualize_data_physics(df)


if __name__ == "__main__":
    main()
