"""
Marketing Mix Modeling, Experimentation & Budget Optimization
-------------------------------------------------------------
End-to-end project on SIMULATED weekly marketing data with a known ground truth,
so every estimate can be checked against the real answer.

Parts
  1. Simulate 156 weeks of revenue driven by 4 channels (adstock carry-over +
     Hill saturation), trend, seasonality and noise.
  2. Media mix model: adstock + exponential saturation, constrained regression,
     parameters tuned with differential evolution. Evaluated on a 13-week holdout
     against time-series baselines (seasonal naive, Holt-Winters).
  3. Causal analysis: geo-lift experiment analysed with difference-in-differences
     (cluster-robust SEs), compared with the true effect.
  4. Experimental design: power analysis + A/B test, empirical power / false-positive
     check by simulation, and CUPED variance reduction.
  5. Constrained budget optimization (SLSQP), verified against the ground truth.

Run:  python mmm_budget_optimizer.py
"""
import json
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, differential_evolution, minimize
from scipy.signal import lfilter
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SEED = 42
CHANNELS = ["search", "social", "display", "email"]
T, HOLDOUT = 156, 13
N_TRAIN = T - HOLDOUT

# Ground-truth response curves (only used to simulate and to verify results)
TRUE = {
    "search":  dict(theta=0.30, K=60, slope=1.4, beta=140, mean_spend=60),
    "social":  dict(theta=0.50, K=45, slope=1.3, beta=90,  mean_spend=40),
    "display": dict(theta=0.65, K=50, slope=1.2, beta=70,  mean_spend=45),
    "email":   dict(theta=0.20, K=12, slope=1.5, beta=40,  mean_spend=10),
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def adstock(x, theta):
    """Geometric carry-over: a_t = x_t + theta * a_{t-1}."""
    return lfilter([1.0], [1.0, -theta], x)


def hill(a, K, slope):
    return a ** slope / (a ** slope + K ** slope)


def mape(y, yhat):
    return float(np.mean(np.abs((y - yhat) / y)) * 100)


def fourier(t, period=52, harmonics=2):
    cols = []
    for k in range(1, harmonics + 1):
        cols += [np.sin(2 * np.pi * k * t / period), np.cos(2 * np.pi * k * t / period)]
    return np.column_stack(cols)


# ----------------------------------------------------------------------------
# 1. Simulation
# ----------------------------------------------------------------------------
def simulate(rng):
    t = np.arange(T)
    season_driver = np.sin(2 * np.pi * t / 52)
    spend = {}
    for i, c in enumerate(CHANNELS):
        p = TRUE[c]
        wobble = 1 + 0.25 * np.sin(2 * np.pi * t / 52 + i)           # planned seasonality
        noise = rng.lognormal(0, 0.50, T)
        # Campaign flighting: 4-week on/off bursts. Real budgets are bursty, and this
        # independent variation is what lets a mix model separate channels.
        on = np.repeat(rng.random(T // 4 + 1) < 0.7, 4)[:T]
        spend[c] = p["mean_spend"] * wobble * noise * np.where(on, 1.3, 0.4)

    base = 500 + 0.8 * t
    season = 55 * season_driver + 25 * np.cos(4 * np.pi * t / 52)
    media = {}
    for c in CHANNELS:
        p = TRUE[c]
        media[c] = p["beta"] * hill(adstock(spend[c], p["theta"]), p["K"], p["slope"])
    noise = rng.normal(0, 15, T)
    revenue = base + season + sum(media.values()) + noise

    df = pd.DataFrame({"week": t, "revenue": revenue, **{f"spend_{c}": spend[c] for c in CHANNELS}})
    truth = {"media": pd.DataFrame(media)}
    return df, truth


# ----------------------------------------------------------------------------
# 2. Media mix model
# ----------------------------------------------------------------------------
def design(df, params, means):
    """Control columns + one saturated-adstock column per channel."""
    t = df["week"].to_numpy()
    controls = np.column_stack([np.ones(len(df)), t / T, fourier(t)])
    feats = []
    for i, c in enumerate(CHANNELS):
        theta, K = params[2 * i], params[2 * i + 1]
        a = adstock(df[f"spend_{c}"].to_numpy() / means[c], theta)
        feats.append(1 - np.exp(-a / K))
    return controls, np.column_stack(feats)


def solve_linear(controls, feats, y):
    A = np.hstack([controls, feats])
    lb = np.r_[np.full(controls.shape[1], -np.inf), np.zeros(feats.shape[1])]   # channel effects >= 0
    ub = np.full(A.shape[1], np.inf)
    res = lsq_linear(A, y, bounds=(lb, ub))
    return res.x, A @ res.x


def fit_mmm(df_train, means):
    y = df_train["revenue"].to_numpy()

    def loss(params):
        controls, feats = design(df_train, params, means)
        _, pred = solve_linear(controls, feats, y)
        return float(np.mean((y - pred) ** 2))

    bounds = [(0.0, 0.9), (0.2, 6.0)] * len(CHANNELS)
    res = differential_evolution(loss, bounds, seed=SEED, maxiter=60, popsize=12, tol=1e-6, polish=True)
    controls, feats = design(df_train, res.x, means)
    coef, pred = solve_linear(controls, feats, y)
    return res.x, coef, pred


# ----------------------------------------------------------------------------
# 3. Geo-lift experiment + difference-in-differences
# ----------------------------------------------------------------------------
def geo_lift(rng, n_geo=40, pre=12, post=6, true_lift=0.06):
    base = rng.lognormal(np.log(100), 0.4, n_geo)
    treated = np.zeros(n_geo, dtype=int)
    treated[rng.choice(n_geo, n_geo // 2, replace=False)] = 1
    weeks = pre + post
    week_shock = rng.normal(0, 4, weeks)
    rows = []
    for g in range(n_geo):
        for w in range(weeks):
            is_post = int(w >= pre)
            rev = base[g] * (1 + true_lift * treated[g] * is_post) + week_shock[w] + rng.normal(0, 6)
            rows.append((g, w, treated[g], is_post, rev))
    d = pd.DataFrame(rows, columns=["geo", "week", "treated", "post", "rev"])
    model = smf.ols("rev ~ treated + post + treated:post", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["geo"]})
    est = model.params["treated:post"]
    lo, hi = model.conf_int().loc["treated:post"]
    true_abs = float(np.mean(base[treated == 1]) * true_lift)
    pre_treated_mean = float(d[(d.treated == 1) & (d.post == 0)]["rev"].mean())
    return dict(
        true_effect=true_abs, estimate=float(est), ci_low=float(lo), ci_high=float(hi),
        ci_covers_truth=bool(lo <= true_abs <= hi), p_value=float(model.pvalues["treated:post"]),
        relative_lift_pct=float(est / pre_treated_mean * 100),
    )


# ----------------------------------------------------------------------------
# 4. A/B test: power, empirical power / false positives, CUPED
# ----------------------------------------------------------------------------
def ab_test(rng):
    p0, p1, alpha, power = 0.040, 0.045, 0.05, 0.80
    es = proportion_effectsize(p1, p0)
    n = int(np.ceil(NormalIndPower().solve_power(es, alpha=alpha, power=power, alternative="two-sided")))

    # One concrete experiment
    c, t = rng.binomial(n, p0), rng.binomial(n, p1)
    z, pval = proportions_ztest([t, c], [n, n])

    # Empirical power (true effect) and false-positive rate (A/A) over 2000 simulated tests
    def rejection_rate(pa, pb, reps=2000):
        a = rng.binomial(n, pa, reps)
        b = rng.binomial(n, pb, reps)
        pool = (a + b) / (2 * n)
        se = np.sqrt(2 * pool * (1 - pool) / n)
        zs = (b / n - a / n) / se
        return float(np.mean(np.abs(zs) > 1.96))

    emp_power = rejection_rate(p0, p1)
    type1 = rejection_rate(p0, p0)

    # CUPED on a continuous metric (revenue per user) using pre-period revenue
    m = 5000
    pre_c, pre_t = rng.lognormal(2, 0.8, m), rng.lognormal(2, 0.8, m)
    y_c = 0.6 * pre_c + rng.normal(0, 3, m)
    y_t = 0.6 * pre_t + rng.normal(0, 3, m) + 0.4
    y = np.r_[y_c, y_t]
    x = np.r_[pre_c, pre_t]
    g = np.r_[np.zeros(m), np.ones(m)]
    theta = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
    y_adj = y - theta * (x - x.mean())

    def lift_ci(v):
        d = v[g == 1].mean() - v[g == 0].mean()
        se = np.sqrt(v[g == 1].var(ddof=1) / m + v[g == 0].var(ddof=1) / m)
        return d, d - 1.96 * se, d + 1.96 * se, se

    raw, adj = lift_ci(y), lift_ci(y_adj)
    return dict(
        sample_size_per_arm=n, baseline_rate=p0, target_rate=p1,
        observed_control=float(c / n), observed_treatment=float(t / n), p_value=float(pval),
        empirical_power=emp_power, false_positive_rate=type1,
        cuped=dict(
            variance_reduction_pct=float((1 - y_adj.var(ddof=1) / y.var(ddof=1)) * 100),
            ci_width_raw=float(raw[2] - raw[1]), ci_width_cuped=float(adj[2] - adj[1]),
            ci_width_reduction_pct=float((1 - (adj[2] - adj[1]) / (raw[2] - raw[1])) * 100),
            true_effect=0.4, estimate_raw=float(raw[0]), estimate_cuped=float(adj[0]),
        ),
    )


# ----------------------------------------------------------------------------
# 5. Budget optimization
# ----------------------------------------------------------------------------
def optimize_budget(params, coef, means, hist):
    ncontrol = 6
    betas = coef[ncontrol:]
    th = np.array([params[2 * i] for i in range(4)])
    K = np.array([params[2 * i + 1] for i in range(4)])
    mean_vec = np.array([means[c] for c in CHANNELS])

    def model_rev(s):
        a = (s / mean_vec) / (1 - th)
        return float(np.sum(betas * (1 - np.exp(-a / K))))

    def true_rev(s):
        return float(sum(TRUE[c]["beta"] * hill(s[i] / (1 - TRUE[c]["theta"]), TRUE[c]["K"], TRUE[c]["slope"])
                         for i, c in enumerate(CHANNELS)))

    h = np.array([hist[c] for c in CHANNELS])
    budget = h.sum()
    res = minimize(lambda s: -model_rev(s), x0=h, method="SLSQP",
                   bounds=[(0.25 * v, 3.0 * v) for v in h],
                   constraints=[{"type": "eq", "fun": lambda s: s.sum() - budget}])
    opt = res.x
    out = dict(
        weekly_budget=float(budget),
        historical_allocation={c: float(h[i]) for i, c in enumerate(CHANNELS)},
        optimized_allocation={c: float(opt[i]) for i, c in enumerate(CHANNELS)},
        model_predicted_lift_pct=float((model_rev(opt) / model_rev(h) - 1) * 100),
        true_marketing_revenue_historical=true_rev(h),
        true_marketing_revenue_optimized=true_rev(opt),
        true_lift_pct=float((true_rev(opt) / true_rev(h) - 1) * 100),
        true_incremental_revenue_per_week=float(true_rev(opt) - true_rev(h)),
    )
    return out, h, opt


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def robustness(seeds):
    """Re-simulate and re-fit across seeds to check the headline results are not a lucky draw."""
    mapes, maes = [], []
    for sd in seeds:
        df, truth = simulate(np.random.default_rng(sd))
        train = df.iloc[:N_TRAIN]
        means = {c: float(train[f"spend_{c}"].mean()) for c in CHANNELS}
        params, coef, _ = fit_mmm(train, means)
        controls, feats = design(df, params, means)
        pred = np.hstack([controls, feats]) @ coef
        mapes.append(mape(df["revenue"].to_numpy()[N_TRAIN:], pred[N_TRAIN:]))
        est = coef[6:] * feats[:N_TRAIN].mean(axis=0)
        tru = truth["media"].iloc[:N_TRAIN].mean().to_numpy()
        maes.append(float(np.mean(np.abs(est / est.sum() * 100 - tru / tru.sum() * 100))))
    return dict(seeds=len(seeds), holdout_mape_mean=float(np.mean(mapes)),
                contribution_share_mae_pp_mean=float(np.mean(maes)),
                contribution_share_mae_pp_median=float(np.median(maes)),
                contribution_share_mae_pp_max=float(np.max(maes)))


def main():
    rng = np.random.default_rng(SEED)
    df, truth = simulate(rng)
    train, test = df.iloc[:N_TRAIN], df.iloc[N_TRAIN:]
    means = {c: float(train[f"spend_{c}"].mean()) for c in CHANNELS}

    # --- MMM ---
    params, coef, pred_train = fit_mmm(train, means)
    controls_all, feats_all = design(df, params, means)
    full_pred = np.hstack([controls_all, feats_all]) @ coef
    pred_test = full_pred[N_TRAIN:]
    y_test = test["revenue"].to_numpy()
    ss_res = np.sum((train["revenue"] - pred_train) ** 2)
    ss_tot = np.sum((train["revenue"] - train["revenue"].mean()) ** 2)

    # --- Time-series baselines ---
    snaive = df["revenue"].to_numpy()[N_TRAIN - 52:N_TRAIN - 52 + HOLDOUT]
    hw = ExponentialSmoothing(train["revenue"].to_numpy(), trend="add", seasonal="add",
                              seasonal_periods=52).fit()
    hw_pred = hw.forecast(HOLDOUT)

    # --- Contribution recovery (train window) ---
    est_contrib = coef[6:] * feats_all[:N_TRAIN].mean(axis=0)
    true_contrib = truth["media"].iloc[:N_TRAIN].mean().to_numpy()
    est_share = est_contrib / est_contrib.sum() * 100
    true_share = true_contrib / true_contrib.sum() * 100

    mmm = dict(
        weeks=T, train_weeks=N_TRAIN, holdout_weeks=HOLDOUT, channels=len(CHANNELS),
        train_r2=float(1 - ss_res / ss_tot),
        holdout_mape_mmm=mape(y_test, pred_test),
        holdout_mape_holt_winters=mape(y_test, hw_pred),
        holdout_mape_seasonal_naive=mape(y_test, snaive),
        contribution_share_est_pct={c: float(v) for c, v in zip(CHANNELS, est_share)},
        contribution_share_true_pct={c: float(v) for c, v in zip(CHANNELS, true_share)},
        contribution_share_mae_pp=float(np.mean(np.abs(est_share - true_share))),
        fitted_carryover={c: float(params[2 * i]) for i, c in enumerate(CHANNELS)},
        true_carryover={c: TRUE[c]["theta"] for c in CHANNELS},
    )

    geo = geo_lift(rng)
    ab = ab_test(rng)
    opt, h, o = optimize_budget(params, coef, means, means)

    rob = robustness(range(1, 11))
    results = dict(mmm=mmm, robustness_10_seeds=rob, geo_lift_did=geo, ab_test=ab, budget_optimization=opt)
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- Figures ---
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(df["week"], df["revenue"], color="#444", lw=1, label="Actual")
    ax.plot(df["week"][N_TRAIN:], pred_test, color="#cc0000", lw=2, label=f"MMM (MAPE {mmm['holdout_mape_mmm']:.1f}%)")
    ax.plot(df["week"][N_TRAIN:], hw_pred, color="#1f77b4", lw=2, ls="--",
            label=f"Holt-Winters (MAPE {mmm['holdout_mape_holt_winters']:.1f}%)")
    ax.axvline(N_TRAIN, color="grey", ls=":")
    ax.set_xlabel("Week"); ax.set_ylabel("Revenue ($K)"); ax.set_title("13-week holdout forecast")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig("forecast_holdout.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(CHANNELS)); w = 0.38
    ax.bar(x - w / 2, h, w, label="Historical", color="#999")
    ax.bar(x + w / 2, o, w, label="Optimized", color="#cc0000")
    ax.set_xticks(x); ax.set_xticklabels(CHANNELS); ax.set_ylabel("Weekly spend ($K)")
    ax.set_title(f"Budget allocation (true lift {opt['true_lift_pct']:.1f}%)")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig("budget_allocation.png", dpi=150); plt.close(fig)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
