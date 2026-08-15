"""Generate 7,504 days (~20.5 years) of realistic synthetic daily demand data.

Why synthetic data?
    Real multi-decade demand data is proprietary — companies don't publish it.
    Generating our own has two big advantages for a learning project:
      1. We KNOW the ground truth: the data contains real trend, weekly and
         yearly seasonality, holiday spikes, promotions and noise — so we can
         verify the model actually learns those patterns.
      2. We can inject a DELIBERATE distribution shift near the end of the
         series, which gives the drift detector something real to catch.

The demand series is built as a sum of interpretable parts:

    demand = base level
           + slow trend            (business grows over the years)
           + yearly seasonality    (e.g. high season in summer/December)
           + weekly seasonality    (weekends differ from weekdays)
           + holiday spikes        (a few fixed dates each year)
           + promotion spikes      (random marketing pushes)
           + noise                 (real life is noisy)
           + regime shift          (a level change in the last ~60 days,
                                    so drift detection has work to do)

Run it:
    python -m src.data.generate_data
Produces:
    data/demand.csv  with columns [date, demand, is_holiday, is_promo]
"""

import numpy as np
import pandas as pd

from src.config import CONFIG, DATA_DIR, RAW_DATA_PATH


def generate_demand(n_days: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)

    # Dates: the series ends "today", so the most recent rows are current.
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="D")

    # 1. Base level + slow growth trend (business roughly doubles over 20 years)
    base = 200.0
    trend = 200.0 * (t / n_days)

    # 2. Yearly seasonality: one smooth cycle per 365.25 days.
    #    Amplitude 40 -> demand swings +/-40 units across the year.
    yearly = 40.0 * np.sin(2 * np.pi * t / 365.25 - np.pi / 2)

    # 3. Weekly seasonality: a fixed pattern by day-of-week.
    #    Mon..Sun multipliers — weekends are busier in this business.
    dow_pattern = np.array([0.95, 0.92, 0.94, 0.98, 1.05, 1.18, 1.10])
    weekly = dow_pattern[dates.dayofweek]

    # 4. Holiday spikes: same few (month, day) dates every year, +35% demand.
    holiday_dates = {(1, 1), (2, 14), (7, 4), (11, 27), (12, 24), (12, 25), (12, 31)}
    is_holiday = np.array(
        [(m, d) in holiday_dates for m, d in zip(dates.month, dates.day)], dtype=int
    )

    # 5. Promotions: ~2% of days are marketing pushes with +25% demand.
    is_promo = (rng.random(n_days) < 0.02).astype(int)

    # 6. Noise: demand is never perfectly predictable. Real demand series
    #    carry substantial irreducible randomness — we use ~11% multiplicative
    #    noise (noise proportional to demand level, like real sales) plus a
    #    little additive noise. This sets the FLOOR on achievable accuracy:
    #    no model can forecast pure randomness, so even a perfect model lands
    #    around 0.8 * 11% ≈ 9% MAPE. (mean|N(0,σ)| = 0.8σ)
    mult_noise = rng.normal(1.0, 0.11, n_days)
    add_noise = rng.normal(0, 5.0, n_days)

    demand = (base + trend + yearly) * weekly
    demand *= 1 + 0.35 * is_holiday
    demand *= 1 + 0.25 * is_promo
    demand = demand * mult_noise + add_noise

    # 7. Regime shift: in the final 60 days demand jumps +18% (think: a new
    #    sales channel launched). The KS drift detector should flag this.
    shift_start = n_days - 60
    demand[shift_start:] *= 1.18

    # Demand can't be negative.
    demand = np.clip(demand, 0, None)

    return pd.DataFrame(
        {
            "date": dates,
            "demand": demand.round(2),
            "is_holiday": is_holiday,
            "is_promo": is_promo,
        }
    )


def main() -> None:
    cfg = CONFIG.data
    df = generate_demand(cfg.n_days, cfg.seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Wrote {len(df):,} days of demand data -> {RAW_DATA_PATH}")
    print(df.tail())


if __name__ == "__main__":
    main()
