"""Data drift detection with the two-sample Kolmogorov–Smirnov (KS) test.

What is drift, and why care?
    A model learns patterns from the world as it looked during training.
    If the world changes — demand jumps because a new sales channel opened,
    a competitor closes, prices change — the model's assumptions go stale
    and its accuracy quietly degrades. Drift detection is the tripwire that
    says "the incoming data no longer looks like the training data" BEFORE
    the errors show up in business numbers.

What the KS test does (plain words):
    Take two samples — e.g. demand over the last 90 "known" days vs. the
    most recent 30 days. Draw each sample's cumulative distribution (the
    curve "what fraction of days had demand below x?"). The KS statistic D
    is simply the BIGGEST vertical gap between the two curves:

        D = max |F_reference(x) − F_recent(x)|

    If the two samples come from the same distribution, that gap stays
    small. The test converts the gap into a p-value: "if nothing changed,
    how likely is a gap this big by pure chance?" A small p-value means
    "very unlikely to be chance" -> we call it drift.

Why alpha = 0.05?
    We reject "no drift" when p < 0.05, accepting a 5% false-alarm rate per
    test. Testing several features multiplies alarm chances — which is
    exactly why a detected drift triggers a CANDIDATE retrain reviewed by a
    human (the manual gate), never an automatic production deployment.

Why KS and not, say, a t-test?
    - Non-parametric: assumes nothing about demand's distribution (demand is
      skewed and spiky, not a neat bell curve).
    - Catches ANY change in shape — mean, spread, or skew — while a t-test
      only sees mean shifts.

The seasonality trap (a real lesson this project ran into):
    Naively comparing "last 30 days" against "the 90 days before that"
    fires CONSTANTLY on seasonal data — demand in June genuinely has a
    different distribution than demand in March. That's seasonality, not
    drift. The fix used here: compare the recent window against the SAME
    window one year earlier (364 days back, so weekdays align too).
    Seasonality then cancels out, and what remains is genuine change.

Run it:
    python -m src.monitoring.drift
"""

import json

import pandas as pd
from scipy.stats import ks_2samp

from src.config import CONFIG, ARTIFACTS_DIR, RAW_DATA_PATH


# Columns we monitor. Demand itself is the most important; is_promo would
# catch a change in marketing behavior.
MONITORED_COLUMNS = ["demand"]


def detect_drift(df: pd.DataFrame, detection_days: int, alpha: float, year_offset: int = 364) -> dict:
    """Compare the most recent window against the same window last year.

    year_offset=364 (not 365): 364 = 52 exact weeks, so Mondays line up
    with Mondays — otherwise the weekly pattern itself would look like drift.
    """
    recent = df.iloc[-detection_days:]
    reference = df.iloc[-(detection_days + year_offset) : -year_offset]

    checks = []
    for col in MONITORED_COLUMNS:
        stat, p_value = ks_2samp(reference[col], recent[col])
        checks.append({
            "column": col,
            "ks_statistic": round(float(stat), 4),
            "p_value": float(p_value),
            "drift_detected": bool(p_value < alpha),
        })

    return {
        "alpha": alpha,
        "reference_window": f"same {detection_days} days, {year_offset} days earlier",
        "detection_window_days": detection_days,
        "reference_mean": round(float(reference["demand"].mean()), 2),
        "recent_mean": round(float(recent["demand"].mean()), 2),
        "checks": checks,
        "any_drift": any(c["drift_detected"] for c in checks),
    }


def main():
    cfg = CONFIG.drift
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["date"])
    report = detect_drift(df, cfg.detection_days, cfg.alpha)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "drift_report.json").write_text(json.dumps(report, indent=2))

    print("================ DRIFT REPORT ================")
    print(f"reference window : same {cfg.detection_days} days one year earlier "
          f"(mean demand {report['reference_mean']})")
    print(f"detection window : most recent {cfg.detection_days}d "
          f"(mean demand {report['recent_mean']})")
    for c in report["checks"]:
        flag = "DRIFT DETECTED" if c["drift_detected"] else "ok"
        print(f"  {c['column']:>10}: KS={c['ks_statistic']:.4f}  p={c['p_value']:.2e}  -> {flag}")
    if report["any_drift"]:
        print("\n=> Drift detected. Next step in the pipeline: retrain a CANDIDATE")
        print("   model and hold it at the manual validation gate — a human")
        print("   confirms the drift is real (not a holiday/data bug) before")
        print("   the candidate is promoted to production.")
    else:
        print("\n=> No drift. Production model stays as is.")


if __name__ == "__main__":
    main()
