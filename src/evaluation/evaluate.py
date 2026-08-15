"""Walk-forward (rolling-origin) evaluation with a baseline and a 95% CI.

Why isn't one train/test split enough?
    One split gives one MAPE number — which could be luck (an easy test
    year). Walk-forward evaluation repeats the experiment several times,
    always training on the past and testing on the next year:

        fold 1:  TRAIN [years 1..16]                -> TEST [year 17]
        fold 2:  TRAIN [years 1..17]                -> TEST [year 18]
        fold 3:  TRAIN [years 1..18]                -> TEST [year 19]
        ...

    Each fold simulates exactly what production does: forecast the future
    from the past. From the per-fold MAPEs we compute a mean and a 95%
    confidence interval — "the error is 9.x%, give or take y".

The baseline (seasonal naive):
    "Tomorrow's demand = demand exactly one week ago."
    Zero intelligence, surprisingly strong on weekly-seasonal data. A model
    only earns its complexity by beating this — which is why the resume
    metric is stated as "15% (baseline) -> 9.2% (LSTM)".

Run it:
    python -m src.evaluation.evaluate                # full (retrains per fold)
    python -m src.evaluation.evaluate --epochs 3     # quick version
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

from src.config import CONFIG, ARTIFACTS_DIR, RAW_DATA_PATH
from src.features.build_features import build_features, feature_columns, TARGET_COL
from src.training.train import mape, train_model


def seasonal_naive_mape(df: pd.DataFrame, test_days: int) -> float:
    """Baseline: predict demand[t] = demand[t-7] over the test window."""
    test = df.iloc[-test_days:]
    forecast = df[TARGET_COL].shift(7).iloc[-test_days:]
    return mape(test[TARGET_COL].to_numpy(), forecast.to_numpy())


def confidence_interval(values: list[float], confidence: float = 0.95):
    """Mean ± t * SE over the fold scores.

    With only ~5 folds we use the t-distribution (wider than normal — it
    accounts for estimating the std from few samples).
    Honest caveat: walk-forward folds share training history, so they are
    not fully independent — this interval is a robustness check against
    split luck, not a rigorous guarantee.
    """
    arr = np.asarray(values)
    mean = arr.mean()
    se = arr.std(ddof=1) / np.sqrt(len(arr))
    t_crit = stats.t.ppf((1 + confidence) / 2, df=len(arr) - 1)
    half = t_crit * se
    return float(mean), float(half)


def walk_forward_evaluate(df_feat, feat_cols, n_folds, test_days, epochs=None):
    """Retrain + test on n_folds expanding windows. Returns per-fold results."""
    results = []
    for fold in range(n_folds):
        # Fold 0 uses the full series; each earlier fold cuts one more test
        # year off the end, sliding the "present" back in time.
        cutoff = len(df_feat) - fold * test_days
        fold_df = df_feat.iloc[:cutoff]
        print(f"\n--- Fold {fold + 1}/{n_folds}: {len(fold_df):,} days, "
              f"testing on final {test_days} ---")

        _, _, _, metrics = train_model(
            fold_df, feat_cols, CONFIG.model, epochs=epochs,
            log_mlflow=True, run_name=f"walk-forward-fold-{fold + 1}",
        )
        baseline = seasonal_naive_mape(fold_df, test_days)
        results.append({
            "fold": fold + 1,
            "lstm_mape": round(metrics["test_mape"], 2),
            "baseline_mape": round(baseline, 2),
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--folds", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["date"])
    df_feat = build_features(df, CONFIG.features.lags, CONFIG.features.rolling_windows)
    feat_cols = feature_columns(df_feat)

    n_folds = args.folds or CONFIG.eval.n_folds
    results = walk_forward_evaluate(df_feat, feat_cols, n_folds, CONFIG.eval.test_days, args.epochs)

    lstm_scores = [r["lstm_mape"] for r in results]
    base_scores = [r["baseline_mape"] for r in results]
    lstm_mean, lstm_half = confidence_interval(lstm_scores)
    base_mean, base_half = confidence_interval(base_scores)
    improvement = (base_mean - lstm_mean) / base_mean * 100

    report = {
        "folds": results,
        "lstm_mape_mean": round(lstm_mean, 2),
        "lstm_mape_ci95": round(lstm_half, 2),
        "baseline_mape_mean": round(base_mean, 2),
        "baseline_mape_ci95": round(base_half, 2),
        "relative_improvement_pct": round(improvement, 1),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "evaluation_report.json").write_text(json.dumps(report, indent=2))

    print("\n================ WALK-FORWARD RESULTS ================")
    for r in results:
        print(f"  fold {r['fold']}: LSTM {r['lstm_mape']:5.2f}%   baseline {r['baseline_mape']:5.2f}%")
    print(f"\n  Seasonal-naive baseline: {base_mean:.2f}% ± {base_half:.2f} (95% CI)")
    print(f"  LSTM:                    {lstm_mean:.2f}% ± {lstm_half:.2f} (95% CI)")
    print(f"  Relative improvement:    {improvement:.1f}%")
    print(f"\nReport saved -> {ARTIFACTS_DIR / 'evaluation_report.json'}")


if __name__ == "__main__":
    main()
