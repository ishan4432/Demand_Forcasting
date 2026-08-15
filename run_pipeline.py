"""Run the whole pipeline end-to-end with one command.

    python run_pipeline.py            # full run (takes a few minutes on CPU)
    python run_pipeline.py --quick    # fast smoke-test run (~1 minute)

Stages, in order:
    1. DATA       generate 7,504 days of synthetic demand (skipped if present)
    2. TRAIN      fit the LSTM with a temporal split, log to MLflow
    3. EVALUATE   walk-forward validation: MAPE vs baseline + 95% CI
    4. DRIFT      KS test on the newest data vs the recent past

This is the same flow the ZenML pipeline (pipelines/zenml_pipeline.py)
orchestrates — kept here as plain Python so the project runs with zero
orchestration setup.
"""

import argparse
import subprocess
import sys

from src.config import RAW_DATA_PATH


def run(title: str, args: list[str]):
    print(f"\n{'=' * 60}\n  STAGE: {title}\n{'=' * 60}")
    result = subprocess.run([sys.executable, "-m", *args])
    if result.returncode != 0:
        sys.exit(f"Stage '{title}' failed — stopping the pipeline.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="few epochs / folds, for a fast demo")
    args = parser.parse_args()

    epochs = ["--epochs", "3"] if args.quick else []
    folds = ["--folds", "2"] if args.quick else []

    if not RAW_DATA_PATH.exists():
        run("DATA — generate synthetic demand history", ["src.data.generate_data"])
    else:
        print(f"\nData already exists at {RAW_DATA_PATH} — skipping generation.")

    run("TRAIN — LSTM with temporal split + MLflow", ["src.training.train", *epochs])
    run("EVALUATE — walk-forward MAPE vs baseline + 95% CI", ["src.evaluation.evaluate", *epochs, *folds])
    run("DRIFT — Kolmogorov-Smirnov check on recent data", ["src.monitoring.drift"])

    print(f"\n{'=' * 60}")
    print("  Pipeline complete. Next steps:")
    print("    mlflow ui                          # inspect training runs")
    print("    uvicorn src.api.main:app --reload  # serve predictions")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
