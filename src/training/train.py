"""Train the LSTM with a strict temporal split and MLflow tracking.

The temporal split (THE most important idea in this project):
    ┌────────────────── time ──────────────────────────────►
    │  TRAIN (oldest ~18.5 years) │ VALIDATION (1 yr) │ TEST (newest 1 yr) │

    We NEVER split randomly. With a random split, a model tested on
    Monday could have trained on the very next Tuesday — and because
    consecutive days are correlated, it has effectively seen the answer.
    That is "look-ahead bias": test scores look great, production doesn't.

What each piece of data is for:
    TRAIN      — the model learns its weights here.
    VALIDATION — used for early stopping + hyperparameter choices.
    TEST       — touched exactly once, at the very end, to report honest error.

MLflow:
    Every run logs its parameters (hidden size, learning rate, ...) and
    metrics (val/test MAPE per epoch) so runs are comparable later in a UI:
        mlflow ui   ->  http://localhost:5000

Run it:
    python -m src.training.train                 # full training
    python -m src.training.train --epochs 3      # quick smoke test
"""

import argparse
import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import CONFIG, ARTIFACTS_DIR, RAW_DATA_PATH
from src.features.build_features import (
    TARGET_COL,
    apply_scaler,
    build_features,
    feature_columns,
    fit_scaler,
)
from src.models.lstm_model import LSTMForecaster, make_windows

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:              # the pipeline still works without MLflow
    MLFLOW_AVAILABLE = False


def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean Absolute Percentage Error — "on average, how many % are we off?"

    MAPE = mean( |actual - forecast| / |actual| ) * 100
    """
    mask = np.abs(actual) > 1e-8          # avoid dividing by zero-demand days
    return float(np.mean(np.abs(actual[mask] - forecast[mask]) / np.abs(actual[mask])) * 100)


def temporal_split(df: pd.DataFrame, val_days: int = 365, test_days: int = 365):
    """Chronological split: oldest -> train, then validation, newest -> test."""
    train = df.iloc[: -(val_days + test_days)]
    val = df.iloc[-(val_days + test_days) : -test_days]
    test = df.iloc[-test_days:]
    return train, val, test


def prepare_tensors(df_part, feat_cols, feat_scaler, target_scaler, lookback, context_df=None):
    """Scale a data slice and cut it into LSTM windows.

    context_df: the `lookback-1` rows immediately BEFORE this slice. The first
    day of the validation set still needs the previous 59 days as input —
    those days come from the past, which is legal (using history is not
    leakage; using the future is).
    """
    if context_df is not None:
        df_part = pd.concat([context_df.tail(lookback - 1), df_part])
    feats = apply_scaler(df_part, feat_cols, feat_scaler)
    target = target_scaler.transform(df_part[[TARGET_COL]].to_numpy()).ravel()
    # make_windows yields its first target at index lookback-1, which after
    # prepending lookback-1 context rows is exactly the first row of df_part —
    # so every returned window predicts a day inside this slice.
    X, y = make_windows(feats, target, lookback)
    return torch.from_numpy(X), torch.from_numpy(y)


def train_model(df_feat, feat_cols, model_cfg, epochs=None, log_mlflow=True, run_name="lstm-train"):
    """Full training loop. Returns (model, scalers, metrics)."""
    torch.manual_seed(model_cfg.seed)
    epochs = epochs or model_cfg.max_epochs

    train_df, val_df, test_df = temporal_split(df_feat)

    # Scalers are fit on TRAIN ONLY — see build_features.py for why.
    feat_scaler = fit_scaler(train_df, feat_cols)
    from sklearn.preprocessing import StandardScaler
    target_scaler = StandardScaler().fit(train_df[[TARGET_COL]].to_numpy())

    lookback = model_cfg.lookback
    X_train, y_train = prepare_tensors(train_df, feat_cols, feat_scaler, target_scaler, lookback)
    X_val, y_val = prepare_tensors(val_df, feat_cols, feat_scaler, target_scaler, lookback, context_df=train_df)
    X_test, y_test = prepare_tensors(test_df, feat_cols, feat_scaler, target_scaler, lookback, context_df=val_df)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=model_cfg.batch_size, shuffle=True)

    model = LSTMForecaster(len(feat_cols), model_cfg.hidden_size, model_cfg.num_layers, model_cfg.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg.learning_rate)
    # We TRAIN with Huber loss (stable gradients) but REPORT MAPE (business
    # metric). MAPE is a bad training loss — its gradient explodes near zero.
    loss_fn = nn.HuberLoss()

    def evaluate(X, y):
        model.eval()
        with torch.no_grad():
            pred_scaled = model(X).numpy()
        pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        actual = target_scaler.inverse_transform(y.numpy().reshape(-1, 1)).ravel()
        return mape(actual, pred)

    best_val, best_state, patience_left = float("inf"), None, model_cfg.early_stopping_patience
    history = []

    mlflow_active = MLFLOW_AVAILABLE and log_mlflow
    if mlflow_active:
        mlflow.set_experiment("demand-forecasting")
        mlflow.start_run(run_name=run_name)
        mlflow.log_params({**asdict(model_cfg), "n_features": len(feat_cols)})

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(X_train)

        val_mape = evaluate(X_val, y_val)
        history.append({"epoch": epoch, "train_loss": epoch_loss, "val_mape": val_mape})
        print(f"epoch {epoch:2d}  train_loss={epoch_loss:.4f}  val_MAPE={val_mape:.2f}%")
        if mlflow_active:
            mlflow.log_metrics({"train_loss": epoch_loss, "val_mape": val_mape}, step=epoch)

        # Early stopping: keep the best-on-validation weights, stop when the
        # model stops improving (prevents overfitting to the training years).
        if val_mape < best_val - 1e-4:
            best_val = val_mape
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = model_cfg.early_stopping_patience
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"Early stopping at epoch {epoch} (best val MAPE {best_val:.2f}%)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_mape = evaluate(X_test, y_test)
    metrics = {"val_mape": best_val, "test_mape": test_mape, "epochs_run": len(history)}
    print(f"\nFinal: val MAPE {best_val:.2f}%  |  test MAPE {test_mape:.2f}%")

    if mlflow_active:
        mlflow.log_metrics({"best_val_mape": best_val, "test_mape": test_mape})
        mlflow.end_run()

    return model, feat_scaler, target_scaler, metrics


def save_artifacts(model, feat_scaler, target_scaler, feat_cols, metrics):
    """Persist everything the API needs to serve predictions."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "feat_scaler": feat_scaler,
            "target_scaler": target_scaler,
            "feat_cols": feat_cols,
            "model_cfg": asdict(CONFIG.model),
            "metrics": metrics,
        },
        ARTIFACTS_DIR / "model.pt",
    )
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model + scalers -> {ARTIFACTS_DIR / 'model.pt'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None, help="override max epochs (for quick runs)")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["date"])
    df_feat = build_features(df, CONFIG.features.lags, CONFIG.features.rolling_windows)
    feat_cols = feature_columns(df_feat)
    print(f"{len(df_feat):,} usable days, {len(feat_cols)} features: {feat_cols}")

    model, fs, ts, metrics = train_model(
        df_feat, feat_cols, CONFIG.model, epochs=args.epochs, log_mlflow=not args.no_mlflow
    )
    save_artifacts(model, fs, ts, feat_cols, metrics)


if __name__ == "__main__":
    main()
