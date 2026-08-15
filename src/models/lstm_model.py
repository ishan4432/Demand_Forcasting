"""The LSTM forecasting model and the windowing logic that feeds it.

What is an LSTM (in plain words)?
    A normal neural network looks at one input and produces one output — it
    has no memory. An LSTM (Long Short-Term Memory network) reads a SEQUENCE
    step by step and carries a memory state along, deciding at every step
    what to remember and what to forget. That makes it a natural fit for
    time series: "given the last 60 days, what happens tomorrow?"

How the data is shaped:
    For each day t we build one training example:
        input  X = the feature rows for days [t-59 ... t]   (60 x n_features)
        target y = demand on day t
    Note the features for day t only contain information from BEFORE day t
    (lags are shifted — see build_features.py), so this is leak-free.
"""

import numpy as np
import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """LSTM -> take the last time step's hidden state -> linear head -> 1 number."""

    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,          # tensors are (batch, sequence, features)
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)       # output: (batch, lookback, hidden)
        last_step = output[:, -1, :]   # the hidden state after reading all 60 days
        return self.head(last_step).squeeze(-1)


def make_windows(features: np.ndarray, target: np.ndarray, lookback: int):
    """Slice a long series into (windows, next-day targets) training pairs.

    features: (n_days, n_features) — already scaled
    target:   (n_days,)            — already scaled
    returns X: (n_samples, lookback, n_features), y: (n_samples,)
    """
    X, y = [], []
    for t in range(lookback - 1, len(target)):
        X.append(features[t - lookback + 1 : t + 1])
        y.append(target[t])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
