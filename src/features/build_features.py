"""Feature engineering — turning a raw demand series into model inputs.

An LSTM can't magically know "it's Saturday" or "demand was high last
Christmas". We hand it that context as FEATURES. Three families:

1. LAG features — demand N days ago (N = 1, 7, 14, 28, 364).
   These align with real cycles: yesterday, last week, last month, last year.

2. TRAILING rolling statistics — mean and std of the last 7 / 28 days.
   "Trailing" is critical: the window ends at *yesterday*, never includes
   today or the future. A centered window (pandas default in some examples)
   would leak future values into the feature -> look-ahead bias.

3. CALENDAR features — day-of-week and month encoded as sin/cos pairs.
   Why sin/cos instead of the raw number 0..6? Because the raw number says
   Sunday(6) and Monday(0) are far apart, when they're actually adjacent.
   sin/cos places the days on a circle so the model sees the cyclicality.

THE GOLDEN RULE (look-ahead bias / temporal leakage):
   Every feature for day t may only use information available BEFORE day t
   is over. That's why:
     - all lags/rollings are shifted by at least 1 day,
     - the scaler (normalization) is fit on the TRAINING period only and
       merely applied to validation/test data (see fit_scaler below).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TARGET_COL = "demand"


def build_features(df: pd.DataFrame, lags=(1, 7, 14, 28, 364), rolling_windows=(7, 28)) -> pd.DataFrame:
    """Add lag, rolling and calendar features. Returns a new DataFrame."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    # --- 1. Lag features -------------------------------------------------
    for lag in lags:
        out[f"lag_{lag}"] = out[TARGET_COL].shift(lag)

    # --- 2. Trailing rolling statistics ----------------------------------
    # .shift(1) FIRST, so the window covers [t-window, t-1] — never day t.
    shifted = out[TARGET_COL].shift(1)
    for w in rolling_windows:
        out[f"roll_mean_{w}"] = shifted.rolling(w).mean()
        out[f"roll_std_{w}"] = shifted.rolling(w).std()

    # --- 3. Calendar features (cyclical encodings) -----------------------
    dow = out["date"].dt.dayofweek
    month = out["date"].dt.month
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    # The first max(lags) rows have missing lag values (there's no "364 days
    # ago" for day 10). Drop them — with 7,504 rows we can afford it.
    out = out.dropna().reset_index(drop=True)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Everything except date and the target is a model input."""
    return [c for c in df.columns if c not in ("date", TARGET_COL)]


def fit_scaler(train_df: pd.DataFrame, cols: list[str]) -> StandardScaler:
    """Fit normalization statistics on TRAINING data only.

    Neural networks train best when inputs are roughly mean 0 / std 1.
    But if we computed the mean/std over ALL data (including the test
    period), information about the future would leak into training —
    the single most common leakage bug in published time-series code.
    """
    scaler = StandardScaler()
    scaler.fit(train_df[cols].to_numpy())
    return scaler


def apply_scaler(df: pd.DataFrame, cols: list[str], scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(df[cols].to_numpy())
