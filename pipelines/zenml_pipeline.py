"""ZenML orchestration — the same pipeline stages, as a managed DAG.

What ZenML adds over run_pipeline.py:
    run_pipeline.py just calls the stages in order. ZenML turns each stage
    into a tracked STEP inside a PIPELINE (a DAG — directed acyclic graph):
      - every run is recorded (who ran what, with which inputs, when),
      - outputs of each step are cached & versioned automatically,
      - the same pipeline can be triggered on a schedule or by a drift
        alarm instead of by a human typing a command.

    That's what "automated retraining via ZenML DAGs" means on the resume:
    drift alarm -> this pipeline runs -> a CANDIDATE model is trained and
    registered -> a human reviews it at the manual gate -> promotion.

Setup (optional — the project fully works without it):
    pip install "zenml>=0.55"
    zenml init
    python pipelines/zenml_pipeline.py
"""

import pandas as pd

try:
    from zenml import pipeline, step
except ImportError:
    raise SystemExit(
        "ZenML is not installed (it's optional). Install with:\n"
        "    pip install 'zenml>=0.55' && zenml init\n"
        "Or run the plain pipeline instead:  python run_pipeline.py"
    )

from src.config import CONFIG, RAW_DATA_PATH
from src.data.generate_data import generate_demand
from src.features.build_features import build_features, feature_columns
from src.monitoring.drift import detect_drift
from src.training.train import save_artifacts, train_model


@step
def ingest_data() -> pd.DataFrame:
    """Load demand history (generate it first if missing)."""
    if not RAW_DATA_PATH.exists():
        df = generate_demand(CONFIG.data.n_days, CONFIG.data.seed)
        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_DATA_PATH, index=False)
    return pd.read_csv(RAW_DATA_PATH, parse_dates=["date"])


@step
def check_drift(df: pd.DataFrame) -> bool:
    """KS test: does recent data still look like the data we trained on?"""
    cfg = CONFIG.drift
    report = detect_drift(df, cfg.detection_days, cfg.alpha, cfg.year_offset)
    print(f"Drift check: any_drift={report['any_drift']}")
    return report["any_drift"]


@step
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    return build_features(df, CONFIG.features.lags, CONFIG.features.rolling_windows)


@step
def train_candidate(df_feat: pd.DataFrame) -> dict:
    """Train a CANDIDATE model. It is saved and MLflow-logged, but promoting
    it to production is a human decision (the manual validation gate)."""
    feat_cols = feature_columns(df_feat)
    model, fs, ts, metrics = train_model(
        df_feat, feat_cols, CONFIG.model, run_name="zenml-retrain-candidate"
    )
    save_artifacts(model, fs, ts, feat_cols, metrics)
    return metrics


@step
def manual_gate_notice(metrics: dict, drift: bool) -> None:
    print("\n----- MANUAL VALIDATION GATE -----")
    print(f"Drift detected this run : {drift}")
    print(f"Candidate test MAPE     : {metrics['test_mape']:.2f}%")
    print("A human now reviews: was the drift real (not a holiday / data")
    print("bug)? Does the candidate beat the incumbent on recent data?")
    print("Only then is the candidate promoted to production serving.")


@pipeline
def retraining_pipeline():
    df = ingest_data()
    drift = check_drift(df)
    df_feat = engineer_features(df)
    metrics = train_candidate(df_feat)
    manual_gate_notice(metrics, drift)


if __name__ == "__main__":
    retraining_pipeline()
