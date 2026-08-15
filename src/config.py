"""Central configuration for the whole pipeline.

Every "magic number" in the project lives here, in one place, with an
explanation of why it has the value it has. Change a value here and every
stage of the pipeline (data -> features -> training -> evaluation -> API)
picks it up.
"""

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — everything the pipeline reads/writes lives under these folders.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"   # trained models, scalers, reports
RAW_DATA_PATH = DATA_DIR / "demand.csv"


@dataclass
class DataConfig:
    """Settings for the synthetic demand dataset.

    n_days = 7504 ≈ 20.5 years of daily history. Why so long? An LSTM learns
    seasonal patterns by seeing them repeat. 20 years of data = 20 examples
    of the yearly cycle and ~1,070 examples of the weekly cycle — enough to
    learn the pattern instead of memorizing one year's noise.
    """
    n_days: int = 7504
    seed: int = 42                # fixed seed -> the dataset is reproducible


@dataclass
class FeatureConfig:
    """Settings for feature engineering.

    Lags are chosen to line up with real demand cycles:
      1   = yesterday          (short-term momentum)
      7   = same day last week (weekly seasonality)
      14  = two weeks ago
      28  = ~same day last month
      364 = same day last year (364, not 365, so the WEEKDAY matches too:
            364 = 52 exact weeks. Demand cares more about "first Monday of
            July" than "July 3rd".)
    """
    lags: tuple = (1, 7, 14, 28, 364)
    rolling_windows: tuple = (7, 28)   # trailing means/stds over these windows


@dataclass
class ModelConfig:
    """LSTM architecture + training settings.

    lookback=60: the model sees the last 60 days of features when predicting
    the next day. 60 days spans ~8 weekly cycles and any monthly effects.
    Yearly effects come in through the lag-364 FEATURE instead of a 365-day
    window (which would make training far slower for little gain).
    """
    lookback: int = 60
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 64
    learning_rate: float = 1e-3
    max_epochs: int = 30
    early_stopping_patience: int = 5   # stop if val loss hasn't improved in 5 epochs
    seed: int = 42


@dataclass
class EvalConfig:
    """Walk-forward (rolling-origin) evaluation settings.

    n_folds=5: we get 5 independent-ish MAPE measurements, enough to compute
    a mean and a 95% confidence interval instead of trusting one lucky split.
    test_days=365: each fold is tested on one full year, so every fold's
    score covers all seasons.
    """
    n_folds: int = 5
    test_days: int = 365


@dataclass
class DriftConfig:
    """Kolmogorov–Smirnov drift detection settings.

    alpha=0.05: we accept a 5% false-alarm rate per test. This is exactly why
    the pipeline has a MANUAL promotion gate — with several features tested,
    occasional false alarms are a statistical certainty.

    The comparison is YEAR-OVER-YEAR: the most recent `detection_days` are
    compared against the same window 364 days earlier (52 exact weeks, so
    weekdays align). Comparing adjacent windows instead would flag ordinary
    seasonality as drift — see src/monitoring/drift.py for the full story.
    """
    alpha: float = 0.05
    detection_days: int = 30
    year_offset: int = 364


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)


CONFIG = Config()
