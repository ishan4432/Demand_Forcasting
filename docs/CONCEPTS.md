# Every Concept in This Project, From Zero

No prior ML knowledge assumed. Read top to bottom — each concept builds on the previous one. Every section ends with **where it lives in the code**, so you can read the real implementation right after the idea.

---

## 1. What is a time series, and what is forecasting?

A **time series** is just a sequence of measurements taken over time, in order: daily temperature, monthly sales, hourly website visits. Our series is **daily product demand** — one number per day, 7,504 days (~20.5 years) of it.

**Forecasting** = predicting the next values of the series. "Given everything up to today, how much demand tomorrow?"

The crucial property that makes time series special: **order matters**. Yesterday influences today. You can shuffle rows in a spreadsheet of house prices; you cannot shuffle days of a demand history without destroying the information. Almost every rule in this project ("never split randomly", "features may only look backwards") follows from this one property.

📍 *In the code:* [`src/data/generate_data.py`](../src/data/generate_data.py) builds the series from interpretable parts — trend + yearly season + weekly season + holidays + promotions + noise. Read it to see what a demand series is "made of".

---

## 2. What patterns live inside a demand series?

- **Trend** — the slow long-term direction. Our business grows over 20 years.
- **Seasonality** — repeating cycles. Two of them here:
  - *Weekly:* weekends differ from weekdays, every week.
  - *Yearly:* summer differs from winter, every year.
- **Events** — holidays and promotions cause spikes on known dates.
- **Noise** — pure randomness. A customer walks in or doesn't. **No model can predict noise**, which is why forecast error can never reach 0%. Our data has ~11% noise, so ~9% error is close to the theoretical best. Knowing the noise floor tells you when to *stop* tuning.

A forecasting model's whole job is: learn the trend, the seasons, and the event effects — and don't chase the noise (chasing noise is called **overfitting**).

---

## 3. What is a feature? (And the features we use)

A model doesn't magically know "it's Saturday" or "demand was high last Christmas." A **feature** is a piece of context we compute and hand to the model as an input column. Ours, three families:

| Family | Example | What it tells the model |
|---|---|---|
| **Lags** | `lag_7` = demand 7 days ago | "Same day last week looked like this" |
| **Rolling stats** | `roll_mean_28` = average of the last 28 days | "The current overall level is this" |
| **Calendar** | `dow_sin`, `dow_cos` | "It's a Saturday" / "It's December" |

Two details that show real care:

- **Lag 364, not 365.** 364 = 52 exact weeks, so "one year ago" lands on the same *weekday*. Demand cares more about "first Saturday of July" than "July 3rd".
- **Sin/cos encoding for cycles.** If we encoded Monday=0 … Sunday=6, the model would think Sunday and Monday are far apart (6 vs 0) when they're adjacent. Sin/cos places days on a circle, so distances reflect reality.

📍 *In the code:* [`src/features/build_features.py`](../src/features/build_features.py)

---

## 4. Look-ahead bias — the most important idea in this whole project

Imagine grading a stock-picking strategy while letting it peek at tomorrow's newspaper. Its test results would look amazing and mean nothing. In ML this mistake happens by accident, constantly, and it's called **look-ahead bias** (or **temporal leakage**): information from the future sneaking into training or features.

The three places it sneaks in, and how this project blocks each:

1. **Random train/test splits.** Standard ML practice shuffles data and splits randomly. For time series that's fatal: the model trains on Tuesday and gets tested on the Monday *right before it* — and since adjacent days are correlated, it has effectively seen the answer.
   → *Block:* we split **chronologically**: oldest data trains, newest data tests. The test set is always strictly in the future. (`temporal_split` in [`src/training/train.py`](../src/training/train.py))

2. **Features that peek forward.** A "7-day rolling average" computed with a centered window includes 3 days of *future* demand.
   → *Block:* all rolling features are **trailing** — shifted so the window ends yesterday. And we have a test that mutates future data and asserts no past feature changes ([`tests/test_pipeline.py`](../tests/test_pipeline.py) → `test_features_use_only_past`).

3. **Normalization fitted on all data.** Neural nets need inputs scaled to roughly mean 0 / std 1. If you compute that mean over the *whole* dataset, the test period's statistics leak into training. This is the single most common leak in published time-series code.
   → *Block:* the scaler is fitted on the **training window only**, then merely applied to validation/test. (`fit_scaler` in [`src/features/build_features.py`](../src/features/build_features.py))

Why care so much? Because leakage produces models that ace evaluation and fail in production — the most expensive kind of failure, discovered last.

---

## 5. What is an LSTM?

A plain neural network maps one input to one output — no memory. An **LSTM (Long Short-Term Memory network)** reads a *sequence* step by step, carrying a memory along, learning what to keep and what to forget at each step. Perfect fit for "read the last 60 days, predict tomorrow."

How our data becomes LSTM food: for each day `t` we build one example —

```
input  X = feature rows for days [t-59 ... t]   (a 60 × 15 matrix)
target y = demand on day t
```

Slide that window along 20 years of history → ~7,000 training examples.

Architecture (deliberately modest): 2 LSTM layers of 64 hidden units with dropout, then one linear layer producing a single number. **This is a small model** — a few hundred thousand parameters, not billions. Small is a feature: it trains in minutes on a laptop CPU and predicts in milliseconds, which is what makes the <100 ms API possible.

Honest context worth knowing: for tabular demand data, gradient-boosted trees (XGBoost/LightGBM) on lag features are a famously strong competitor — they dominated the M5 forecasting competition. An LSTM is a defensible choice with 20 years of data and a learning goal; claiming it always wins is not.

📍 *In the code:* [`src/models/lstm_model.py`](../src/models/lstm_model.py)

---

## 6. How training works (loss, epochs, early stopping)

Training = show the model examples, measure how wrong it is (the **loss**), nudge every weight slightly to reduce the loss, repeat. One pass through all examples = one **epoch**.

Data is split three ways, each part with one job:

| Split | Age | Used for |
|---|---|---|
| **Train** (~18.5 yr) | oldest | learning the weights |
| **Validation** (1 yr) | middle | deciding when to stop; comparing settings |
| **Test** (1 yr) | newest | touched **once**, at the end — the honest score |

**Early stopping:** after each epoch we check the validation error. When it stops improving for 5 consecutive epochs, we stop and keep the best weights. This prevents the model from memorizing training noise (overfitting).

**A subtlety interviewers love:** we *train* with Huber loss but *report* MAPE. MAPE is a terrible training signal (its gradient explodes when demand is near zero) but a great business metric. Using different metrics for optimization and reporting is standard, deliberate practice.

📍 *In the code:* [`src/training/train.py`](../src/training/train.py)

---

## 7. MAPE — the error metric, and how to talk about it

**MAPE = Mean Absolute Percentage Error**: "on average, how many percent is the forecast off?"

```
MAPE = mean( |actual − forecast| / actual ) × 100
```

If we forecast 95 and actual demand is 100 → that day contributes 5%. Average over all test days → the MAPE.

Why it's the industry default for demand: it's **scale-free** (a 9% error means the same for a product selling 100/day or 10,000/day) and **business-legible** (no translation needed for a planning team).

Its known weaknesses — volunteer these before being asked: it **explodes near zero demand** (dividing by tiny actuals), and it's **asymmetric** (over-forecasting is penalized differently than under-forecasting). For zero-heavy "intermittent" demand you'd switch to WAPE or SMAPE.

**The headline claim, decoded:** *"improved 15% → 9.2% MAPE (38% gain)"* means the naive baseline was off by ~15% on average, the LSTM by ~9%. The "38%" is the **relative error reduction**: (15 − 9.2) / 15 = 38.7%. Careful: that's NOT "38% more accurate" — accuracy moved ~5.8 *percentage points* (from ~85% to ~91%).

---

## 8. The baseline — why 15% is the number to beat

A **baseline** is the simplest possible forecast, and every model must be measured against one. Ours is the **seasonal naive**: *"tomorrow's demand = demand exactly 7 days ago."* Zero intelligence — and surprisingly hard to beat on weekly-seasonal data, because it gets the weekly pattern right for free.

On our data the seasonal naive scores ~14.5% MAPE. The LSTM's job is to beat it by also learning the trend, yearly season, holidays, and promotions. Without a baseline, "9% MAPE" is meaningless — maybe the naive approach also gets 9%, and the neural network adds nothing but complexity.

📍 *In the code:* `seasonal_naive_mape` in [`src/evaluation/evaluate.py`](../src/evaluation/evaluate.py)

---

## 9. Walk-forward validation and the 95% confidence interval

One train/test split gives one number — which might be luck (an easy test year). **Walk-forward (rolling-origin) validation** repeats the experiment:

```
fold 1: train on years 1–16  →  test on year 17
fold 2: train on years 1–17  →  test on year 18
fold 3: train on years 1–18  →  test on year 19
...
```

Each fold simulates exactly what production does — forecast the future from the past. Five folds give five MAPE scores, and from them:

```
mean = average of the 5 scores
SE   = std of scores / √5
95% CI = mean ± t × SE        (t ≈ 2.78 for 5 samples)
```

Result stated properly: **"9.5% MAPE, 95% CI ± 0.3"** — meaning: if the folds' spread is representative, the true error plausibly lies in that band. The baseline's interval (~14.5 ± 0.4) doesn't overlap it, so the improvement isn't a lucky split.

**Honest caveat (worth saying out loud):** the folds share training history, so they're not fully independent — the interval is a robustness check, not a rigorous statistical guarantee. Conceding this precisely is a *senior* move.

📍 *In the code:* [`src/evaluation/evaluate.py`](../src/evaluation/evaluate.py)

---

## 10. Data drift and the Kolmogorov–Smirnov test

A model learns the world as it looked during training. When the world changes — new sales channel, competitor closes, prices move — the model's assumptions go stale and accuracy quietly rots. **Drift detection** is the tripwire that catches the change *before* the business notices bad forecasts.

**The KS test in plain words:** take two samples of demand (recent days vs. a reference period). For each, draw the curve "what fraction of days had demand below x?" (the cumulative distribution). The KS statistic **D** is the biggest vertical gap between the two curves. Same world → small gap. Changed world → big gap. The test converts the gap into a **p-value**: "if nothing had changed, how likely is a gap this big by chance?" If p < **α = 0.05**, we call it drift.

**Why KS and not a t-test?** KS is *non-parametric* — it assumes nothing about demand's distribution (which is spiky and skewed, not a bell curve) and catches changes in spread or shape, not just the average.

**What α = 0.05 buys and costs:** a 5% false-alarm rate *per test, by design*. Monitor weekly for a year and false alarms are near-certain. This single fact justifies the manual gate (next section).

**The seasonality trap (this project hit it — great story):** the first version compared "last 30 days" vs "the 90 days before." It flagged drift *constantly* — because on seasonal data, adjacent windows genuinely differ. That's seasonality, not drift. The fix: compare against the **same 30 days one year earlier** (364 days back, so weekdays align). Seasonality cancels; only genuine change remains. Our generator injects a deliberate +18% shift in the final 60 days, and a test proves the detector catches it — and stays quiet on calm data.

📍 *In the code:* [`src/monitoring/drift.py`](../src/monitoring/drift.py) · test in [`tests/test_pipeline.py`](../tests/test_pipeline.py)

---

## 11. Automated retraining and the manual validation gate

When drift fires, the pipeline retrains automatically — but the fresh model does **not** go straight to production. It becomes a **candidate** that waits at a **manual validation gate**: a human checks (a) was the drift real, or a holiday spike / broken data feed? (b) does the candidate actually beat the current model on recent data? Only then is it promoted.

The tradeoff being balanced:

| | Fully automatic | Gated (chosen) |
|---|---|---|
| Freshness | immediate | slightly delayed |
| Risk | one corrupted data feed silently poisons production | minutes of human review per alarm |
| Stability | forecasts change without warning | every promotion deliberate & reversible |

Given guaranteed false alarms (α = 0.05!) and low alarm volume, the gate is the rational choice. At high volume you'd automate the *evaluation* (champion vs challenger) and keep the human only for final sign-off.

📍 *In the code:* [`pipelines/zenml_pipeline.py`](../pipelines/zenml_pipeline.py)

---

## 12. MLflow — experiment tracking and the model registry

Train 30 model variants over a month and you *will* forget which settings produced which score. **MLflow** fixes this: every training run automatically logs its **parameters** (hidden size, learning rate…), **metrics** (val/test MAPE per epoch), and **artifacts** (the model file). Run `mlflow ui` and compare every run ever made in a browser table.

The **model registry** part assigns models **versions** with lifecycle stages (candidate → production). That's what makes two things possible: *lineage* (any prediction traces back to the exact run, data window, and parameters that produced its model) and *rollback* (bad deploy? switch back to the previous version — a tag change, not a scramble).

📍 *In the code:* the `mlflow.log_*` calls in [`src/training/train.py`](../src/training/train.py)

---

## 13. ZenML — pipeline orchestration (what "DAG" means)

`run_pipeline.py` calls the stages in order — fine for a human at a laptop. **ZenML** turns each stage into a tracked **step** in a **pipeline**, connected as a **DAG** (Directed Acyclic Graph — steps and arrows, no loops: data → features → train → evaluate). What that buys:

- every run is recorded (what ran, with which inputs, when, producing what),
- step outputs are cached and versioned,
- the pipeline can be **triggered** — by a schedule or a drift alarm — instead of typed by hand. That's the "automated" in *automated retraining*.

It's optional here (heavy install); the plain-Python `run_pipeline.py` runs the identical flow.

📍 *In the code:* [`pipelines/zenml_pipeline.py`](../pipelines/zenml_pipeline.py)

---

## 14. FastAPI, Docker, and serving under 100 ms

**FastAPI** wraps the model in a web API so any system can ask for forecasts with an HTTP request. Endpoints: `/predict`, `/health` (for load balancers), `/model/info` (which version is serving), `/drift`. It also auto-generates an interactive playground at `/docs`.

**The one trick behind <100 ms latency:** the model is loaded into memory **once, at server startup** — never per request. A request is then just: parse JSON → build features from recent history → one LSTM forward pass (a few ms on CPU) → respond. When quoting latency, say the **percentile**: "p95 < 100 ms" (95% of requests are faster) — averages hide slow outliers.

**Docker** freezes the exact environment — Python version, every library, the code — into an image. The model that passed evaluation on the laptop is byte-for-byte the model serving on the cloud. "Works on my machine" stops being a sentence anyone says.

**AWS deployment (the resume's last bullet):** the image runs on 1–2 small EC2 machines behind a load balancer with an Auto Scaling Group. At ~1,000 predictions/day (≈1 request every 86 seconds on average!) the second node is about **availability and bursts** — traffic clusters in morning planning windows, and one node can die or be redeployed without downtime — not average throughput. Scaling to one node off-peak keeps costs near single-machine level.

📍 *In the code:* [`src/api/main.py`](../src/api/main.py) · [`Dockerfile`](../Dockerfile)

---

## The one-paragraph summary of everything

We generate 20 years of realistic daily demand, engineer lag/rolling/calendar features that only ever look backwards, and train a small LSTM under a strict "past predicts future" regime. We prove the model's worth against a naive baseline using walk-forward validation with a confidence interval — numbers that can't lie. In production, a KS test watches for the world changing (comparing year-over-year so seasonality doesn't cry wolf), retraining is automated but promotion is human-gated because false alarms are a statistical certainty, every model is versioned and traceable in MLflow, and a Dockerized FastAPI serves memory-resident predictions in milliseconds. **The model is the least interesting part — the system around it is the project.**
