# Interview Guide — Presenting This Project

How to pitch the project, justify every number, and answer the questions interviewers actually ask. Read [CONCEPTS.md](CONCEPTS.md) first if any term here is unfamiliar.

---

## The 60-second pitch

> "I built an end-to-end demand forecasting system — not just a model, but the full ML lifecycle. The core is an LSTM trained on about 20 years of daily demand history. What I'm most proud of isn't the architecture, it's the rigor around it: strict temporal splits and walk-forward validation so the model is never evaluated on data it could have 'seen' from the future — the classic silent failure in time-series work. That discipline took error from a ~14.3% MAPE seasonal-naive baseline down to ~9.7% — about a 32% relative reduction — with a 95% confidence interval from rolling-origin cross-validation rather than a single lucky split.
>
> Then I treated it as a living system: a Kolmogorov–Smirnov test monitors incoming data for distribution drift at α = 0.05, and when drift fires, the pipeline retrains a candidate model with everything tracked in MLflow. I deliberately put a manual validation gate before promotion — drift tests have a designed-in false-positive rate, and auto-deploying on every alarm trades stability for freshness. Serving is a Dockerized FastAPI with the model memory-resident, answering in well under 100 ms."

This framing pre-answers the three hardest questions (baseline? why believe the number? why a manual gate?) and makes the **MLOps judgment** the star, not the LSTM.

---

## Every metric, justified

### "7,504 days of data"
≈ 20.5 years daily. The justification: an LSTM learns seasonality by seeing cycles *repeat* — 20 examples of the yearly cycle, ~1,070 weekly cycles. With only 2–3 years, simpler baselines usually win.
**If asked "what dataset?"**: it's a synthetic generator ([src/data/generate_data.py](../src/data/generate_data.py)) built from realistic components — trend, dual seasonality, holidays, promotions, ~11% noise, and a deliberate regime shift for the drift detector. Say this plainly; for a personal project it's a *strength*: you know the ground truth, so you can verify the model learns real patterns, and the pipeline is dataset-agnostic (any `date,demand` CSV drops in).

### "~14.3% → ~9.7% MAPE (≈32% gain)"
- MAPE = average percent error: `mean(|actual − forecast| / actual) × 100`.
- 14.3% is the **seasonal-naive baseline** ("tomorrow = same day last week") measured under the same walk-forward folds — never quote a model's error without the baseline it beat.
- (14.27 − 9.71)/14.27 = **32% relative error reduction**. Never say "32% more accurate" — accuracy moved ~4.6 *percentage points*.
- Trained with Huber loss, *evaluated* with MAPE (MAPE is unstable as a training loss near zero demand).
- Run `python -m src.evaluation.evaluate` and read `artifacts/evaluation_report.json` for the exact current numbers — quote *those* in interviews, they're reproducible on demand.

### "95% confidence interval via cross-validation"
Five walk-forward folds → five MAPE scores → `mean ± t·SE` (t ≈ 2.78 at n=5). States that the improvement survives across different test years, not just one lucky split.
**Known limitation to volunteer:** folds share training history so they're not independent; the interval is a robustness check, not a rigorous guarantee. A blocked bootstrap would be the upgrade.

### "KS drift detection (α = 0.05)"
- Two-sample KS: largest gap between the cumulative distributions of recent vs reference demand; p < 0.05 ⇒ drift.
- **Why KS:** non-parametric (demand isn't Gaussian), catches shape/spread changes, not just mean shifts.
- **Why α = 0.05:** the accepted 5%-per-test false-alarm rate — the standard balance between missing drift and crying wolf.
- **Best story in the project:** the first implementation compared adjacent windows and flagged *seasonality* as drift, constantly. The fix compares **year-over-year** (364 days back, so weekdays align). Tell this story — hitting a real monitoring pitfall and solving it is worth more than a clean run.

### "Manual validation gate"
α = 0.05 across ongoing monitoring ⇒ false alarms are a statistical certainty. Auto-deploying on every alarm means one broken data feed retrains production on garbage. The gate: drift → **candidate** retrain (tracked in MLflow) → human confirms drift is real and candidate beats incumbent → promote. Frame it as a deliberate freshness-vs-stability tradeoff; at higher alarm volume you'd automate champion/challenger evaluation and keep the human only for sign-off.

### "<100 ms latency, 1,000+ predictions/day, 1–2 node autoscaling"
- Latency: model loaded **once at startup**; a request = JSON parse → feature build on a trimmed history window → one LSTM forward pass (few ms on CPU). Quote the **percentile**: "p95 under 100 ms."
- **Do the math before they do:** 1,000/day ≈ one request every 86 seconds on average. So the honest answer to "why autoscale?": traffic is bursty around planning windows, and min-1/max-2 behind a load balancer is an **availability** pattern (a node can die or redeploy with zero downtime) plus a cost pattern (one node off-peak). Admitting the average load is small and the design is for bursts + failover + learning reads as engineering maturity.

---

## The questions they'll actually ask

**Why an LSTM and not ARIMA / Prophet / XGBoost?**
Three-part honest answer: (1) technical fit — nonlinear, handles multiple seasonalities and exogenous features (holiday/promo flags) natively, and 20 years of data can feed it; (2) awareness — gradient-boosted trees on lag features are a brutally strong baseline (they dominated the M5 competition) and would be the first benchmark in a business setting; (3) candor — productionizing a deep learning model was part of the learning goal.

**Walk me through what happens when drift is detected.**
KS fires → ZenML DAG runs: ingest → drift check → features → retrain → evaluate → register candidate in MLflow → manual gate → promote → API serves the new version. Rollback = revert to previous registry version.

**How do you know there's no leakage?**
It's *tested*: `test_features_use_only_past` mutates future demand and asserts every past feature is unchanged; `test_temporal_split_order` asserts strict chronological ordering. Scalers are fit on train only. Point at the tests — "I enforce it in CI" beats "I was careful."

**What loss did you train with?** Huber (stable gradients), evaluated with MAPE (business metric). Train-loss ≠ report-metric is deliberate.

**Multi-day forecasts?** Recursive strategy — predict tomorrow, append, predict the next day. Known tradeoff: errors compound with horizon; the direct (multi-output) strategy is the alternative.

**Why EC2, not Lambda or SageMaker?** Lambda: cold-starting PyTorch + weights breaks the latency budget; the model must stay memory-resident. SageMaker: managed endpoints trade cost/control for convenience — hand-building the serving stack was the point. Knowing what you gave up is the senior signal.

**What would you improve?** Ranked: (1) probabilistic forecasts via quantile loss so planning gets uncertainty bands; (2) champion/challenger automation at the gate; (3) gradient-boosting benchmark; (4) blocked-bootstrap CIs; (5) CI/CD for the pipeline itself; (6) monitor live forecast error (rolling MAPE on arriving actuals) as the ultimate drift backstop.

---

## The mental model to carry into the room

**The model is the least interesting part of this project — and that's the pitch.** The LSTM is a few dozen lines of PyTorch. What the project demonstrates is the discipline around it: evaluation that can't lie (temporal splits, walk-forward CV, confidence intervals), monitoring that knows its own false-alarm rate (KS at α = 0.05 → manual gate), and serving with full lineage from any prediction back to the exact run that produced it. That's the difference between "I trained a model" and "I operated an ML system."
