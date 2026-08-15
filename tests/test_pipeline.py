"""Tests for the properties that make the metrics trustworthy.

These aren't token tests — each one guards a claim from the resume:
  - "temporal-split validation"   -> test_temporal_split_order
  - "preventing look-ahead bias"  -> test_features_use_only_past
  - "KS drift detection"          -> test_drift_detects_injected_shift
  - the model itself works        -> test_model_forward_shape

Run:  python -m pytest tests/ -v
"""

import numpy as np
import pandas as pd
import torch

from src.data.generate_data import generate_demand
from src.features.build_features import build_features, feature_columns
from src.models.lstm_model import LSTMForecaster, make_windows
from src.monitoring.drift import detect_drift
from src.training.train import temporal_split


def _small_df(n_days=900, seed=0):
    return generate_demand(n_days, seed)


def test_features_use_only_past():
    """THE leakage test: changing FUTURE demand must not change any feature
    computed for an earlier day. If it does, information travels backwards
    in time — look-ahead bias."""
    df = _small_df()
    feats_before = build_features(df)

    df_mutated = df.copy()
    df_mutated.loc[df_mutated.index[-100:], "demand"] += 9999.0   # rewrite the future
    feats_after = build_features(df_mutated)

    cols = feature_columns(feats_before)
    cutoff = len(feats_before) - 101   # rows strictly before the mutation
    pd.testing.assert_frame_equal(
        feats_before.iloc[:cutoff][cols],
        feats_after.iloc[:cutoff][cols],
    )


def test_temporal_split_order():
    """Every training day must be older than every validation day, which
    must be older than every test day."""
    df = build_features(_small_df())
    train, val, test = temporal_split(df, val_days=100, test_days=100)
    assert train["date"].max() < val["date"].min()
    assert val["date"].max() < test["date"].min()
    assert len(train) + len(val) + len(test) == len(df)


def test_drift_detects_injected_shift():
    """The generator injects a +18% level shift in the last 60 days —
    the year-over-year KS test must catch it, and must stay quiet on
    unshifted (but still seasonal!) data.

    Full-length series on purpose: the generator scales its growth trend to
    the series length, so a short series grows unrealistically fast and
    year-over-year growth itself would (correctly!) register as drift."""
    df = _small_df(n_days=7504)                      # includes the shift
    report = detect_drift(df, detection_days=30, alpha=0.05)
    assert report["any_drift"] is True

    calm = df.iloc[:-200]                            # well before the shift
    report_calm = detect_drift(calm, detection_days=30, alpha=0.05)
    assert report_calm["any_drift"] is False


def test_model_forward_shape():
    """A batch of 8 windows in -> 8 predictions out."""
    n_features, lookback = 12, 60
    model = LSTMForecaster(n_features=n_features)
    x = torch.randn(8, lookback, n_features)
    assert model(x).shape == (8,)


def test_make_windows_alignment():
    """Window t must end at row t and predict target[t]."""
    feats = np.arange(20, dtype=np.float32).reshape(-1, 1)
    target = np.arange(20, dtype=np.float32)
    X, y = make_windows(feats, target, lookback=5)
    assert X.shape == (16, 5, 1)
    assert y[0] == 4.0                 # first prediction is day index 4
    assert X[0, -1, 0] == 4.0          # ...and its window ends at that day
