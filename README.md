# Marketing Mix Modeling, Experimentation & Budget Optimization

An end-to-end project on **simulated** weekly marketing data with a known ground truth, so every estimate can be checked against the real answer. It covers predictive modeling, time series, causal inference, experimental design and constrained optimization.

```bash
pip install -r requirements.txt
python mmm_budget_optimizer.py      # ~1 min, deterministic (seed 42); writes results.json + 2 figures
```
## What it does

| Part | Method | Result (seed 42) |
|---|---|---|
| **Media mix model** | Adstock carry-over + exponential saturation per channel, constrained regression (channel effects >= 0), nonlinear parameters tuned with differential evolution | 13-week holdout MAPE **1.4%** vs 3.7% Holt-Winters and 4.9% seasonal-naive; train R^2 0.96 |
| **Causal analysis** | Geo-lift experiment (40 geos, 12 pre / 6 post weeks) analysed with difference-in-differences, cluster-robust SEs | Estimated lift **5.2%** (95% CI covers the true 6% lift) |
| **Experimental design** | Power analysis, two-proportion z-test, 2,000-run simulation check, CUPED | **25,531 users/arm**; empirical power 79.8% (target 80%), false-positive rate 5.2% (target 5%); CUPED cuts variance **79%** and CI width 54% |
| **Budget optimization** | SLSQP, fixed total budget, each channel bounded to 0.25x-3x its historical spend | Reallocation lifts **ground-truth** marketing revenue **1.2%** at equal spend |

## Honest limitations (read before quoting numbers)

- **Data is simulated.** That is the point: the true response curves are known, so results can be verified. It is not evidence about real marketing performance.
- **Forecast accuracy is not attribution accuracy.** Across 10 re-simulated seeds, holdout MAPE stays around 2%, but channel-contribution error averages about 10 percentage points (median 7.8, worst case 33.6). Weekly marketing data is collinear and noisy, so a model can forecast well while splitting credit between channels poorly. An early version of this simulation, with smoother spend, attributed 87% of revenue to display when the truth was 24%. Adding realistic campaign flighting (on/off bursts) fixed most of that, but the heavy tail remains. In practice you would calibrate with lift experiments like the geo test in this repo.
- **The optimizer overstated its gain.** The model predicted +5.9% marketing revenue; the ground truth says +1.2%. The model is misspecified on purpose (it fits an exponential saturation curve to Hill-shaped truth), and optimizers tend to exploit model errors. Treat model-predicted lift as an upper bound.
- Ridge shrinkage on the channel coefficients was tried and did not improve attribution, so it is not used.

## Files

- `mmm_budget_optimizer.py` - the whole pipeline
- `results.json` - every number quoted above
- `forecast_holdout.png`, `budget_allocation.png` - figures
