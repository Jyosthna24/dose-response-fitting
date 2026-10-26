"""Four-parameter Hill (log-logistic) fitting, uncertainty and model comparison.

Model, on log10 concentration x:

    y = bottom + (top - bottom) / (1 + 10 ** ((logEC50 - x) * hill))

hill > 0 for activators / potentiators, hill < 0 for inhibitors.
EC50 is fitted on the log scale because that is how its error is distributed;
its confidence interval is therefore asymmetric in linear units.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

PARAMS = ["bottom", "top", "logEC50", "hill"]


def hill4(x, bottom, top, log_ec50, hill):
    return bottom + (top - bottom) / (1.0 + 10.0 ** ((log_ec50 - x) * hill))


@dataclass
class FitResult:
    params: dict
    se: dict
    ci95: dict
    n: int
    dof: int
    ssr: float
    r2: float
    fixed: dict = field(default_factory=dict)

    @property
    def ec50(self) -> float:
        return 10 ** self.params["logEC50"]

    @property
    def ec50_ci95(self) -> tuple[float, float]:
        lo, hi = self.ci95["logEC50"]
        return 10 ** lo, 10 ** hi

    def predict(self, conc) -> np.ndarray:
        return hill4(np.log10(np.asarray(conc, float)), *(self.params[p] for p in PARAMS))

    def as_row(self) -> dict:
        lo, hi = self.ec50_ci95
        return {"EC50": self.ec50, "EC50_ci_low": lo, "EC50_ci_high": hi,
                "logEC50": self.params["logEC50"], "logEC50_se": self.se["logEC50"],
                "hill": self.params["hill"], "top": self.params["top"],
                "bottom": self.params["bottom"], "r2": self.r2, "n": self.n}


def _prepare(conc, resp):
    conc, resp = np.asarray(conc, float), np.asarray(resp, float)
    ok = np.isfinite(conc) & np.isfinite(resp) & (conc > 0)  # vehicle (0) cannot go on a log axis
    if len(np.unique(conc[ok])) < 4:
        raise ValueError("need at least 4 distinct non-zero concentrations to fit 4 parameters")
    return np.log10(conc[ok]), resp[ok]


def _initial_guess(x, y):
    order = np.argsort(x)
    x, y = x[order], y[order]
    slope_sign = 1.0 if np.corrcoef(x, y)[0, 1] >= 0 else -1.0
    mid = (y.min() + y.max()) / 2
    log_ec50 = x[np.argmin(np.abs(y - mid))]
    return {"bottom": y.min(), "top": y.max(), "logEC50": log_ec50, "hill": slope_sign}


def fit_hill(conc, resp, fix_bottom: float | None = None, fix_top: float | None = None) -> FitResult:
    """Least-squares 4PL fit. Optionally fix bottom/top (e.g. bottom = vehicle mean)."""
    x, y = _prepare(conc, resp)
    fixed = {k: v for k, v in (("bottom", fix_bottom), ("top", fix_top)) if v is not None}
    free = [p for p in PARAMS if p not in fixed]
    guess = _initial_guess(x, y)

    def model(xx, *theta):
        p = dict(fixed, **dict(zip(free, theta)))
        return hill4(xx, *(p[k] for k in PARAMS))

    span = x.max() - x.min()
    lower = {"bottom": -np.inf, "top": -np.inf, "logEC50": x.min() - span, "hill": -10}
    upper = {"bottom": np.inf, "top": np.inf, "logEC50": x.max() + span, "hill": 10}
    theta, cov = curve_fit(model, x, y, p0=[guess[p] for p in free],
                           bounds=([lower[p] for p in free], [upper[p] for p in free]),
                           maxfev=20000)

    resid = y - model(x, *theta)
    ssr = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    dof = len(y) - len(free)
    se_free = np.sqrt(np.diag(cov)) if np.all(np.isfinite(cov)) else np.full(len(free), np.nan)
    tcrit = stats.t.ppf(0.975, dof) if dof > 0 else np.nan

    params = dict(fixed, **dict(zip(free, map(float, theta))))
    se = {p: 0.0 for p in fixed} | dict(zip(free, map(float, se_free)))
    ci = {p: (params[p] - tcrit * se[p], params[p] + tcrit * se[p]) for p in PARAMS}
    return FitResult(params, se, ci, len(y), dof, ssr, 1 - ssr / sst if sst else np.nan, fixed)


def bootstrap_ec50(conc, resp, n_boot: int = 1000, seed: int = 0, **fit_kwargs) -> dict:
    """Case-resampling bootstrap of EC50: a check on the asymptotic CI that
    does not assume normally distributed parameter errors."""
    rng = np.random.default_rng(seed)
    conc, resp = np.asarray(conc, float), np.asarray(resp, float)
    logs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(conc), len(conc))
        try:
            logs.append(fit_hill(conc[idx], resp[idx], **fit_kwargs).params["logEC50"])
        except (RuntimeError, ValueError):
            continue  # resample with too few concentrations or non-convergence
    logs = np.array(logs)
    lo, hi = np.percentile(logs, [2.5, 97.5])
    return {"EC50_boot_low": 10 ** lo, "EC50_boot_high": 10 ** hi,
            "n_boot_ok": len(logs), "n_boot": n_boot}


def compare_ec50(df: pd.DataFrame, conc: str, resp: str, group: str) -> dict:
    """Extra-sum-of-squares F-test: do the groups share one EC50?

    Full model    - every group has its own bottom, top, logEC50, hill.
    Reduced model - bottom, top, hill per group, but ONE shared logEC50.
    A small p-value means the EC50 shift between groups is real.
    """
    d = df[(df[conc] > 0) & df[[conc, resp]].notna().all(axis=1)]
    groups = list(pd.unique(d[group]))
    k = len(groups)
    if k < 2:
        raise ValueError("need at least two groups to compare")

    fits = {g: fit_hill(s[conc], s[resp]) for g, s in d.groupby(group, sort=False)}
    ssr_full = sum(f.ssr for f in fits.values())
    n = len(d)
    df_full = n - 4 * k

    x = np.log10(d[conc].to_numpy(float))
    y = d[resp].to_numpy(float)
    gi = d[group].map({g: i for i, g in enumerate(groups)}).to_numpy()

    def shared(X, log_ec50, *rest):
        xx, g = X
        b, t, h = (np.asarray(rest[i * k:(i + 1) * k]) for i in range(3))
        return hill4(xx, b[g.astype(int)], t[g.astype(int)], log_ec50, h[g.astype(int)])

    p0 = ([np.mean([f.params["logEC50"] for f in fits.values()])]
          + [fits[g].params["bottom"] for g in groups]
          + [fits[g].params["top"] for g in groups]
          + [fits[g].params["hill"] for g in groups])
    theta, _ = curve_fit(shared, (x, gi), y, p0=p0, maxfev=50000)
    resid = y - shared((x, gi), *theta)
    ssr_red = float(resid @ resid)
    df_red = n - (3 * k + 1)

    F = ((ssr_red - ssr_full) / (df_red - df_full)) / (ssr_full / df_full)
    p = float(stats.f.sf(F, df_red - df_full, df_full))
    return {"F": float(F), "df_num": df_red - df_full, "df_den": df_full, "p_value": p,
            "shared_EC50": float(10 ** theta[0]),
            "EC50_by_group": {g: fits[g].ec50 for g in groups}}


def fit_groups(df: pd.DataFrame, conc: str, resp: str, group: str,
               n_boot: int = 0, seed: int = 0) -> pd.DataFrame:
    """One summary row per group (EC50 with CI, Hill slope, top, bottom, R²)."""
    rows = []
    for g, s in df.groupby(group, sort=False):
        row = {group: g}
        try:
            row |= fit_hill(s[conc], s[resp]).as_row()
            if n_boot:
                row |= bootstrap_ec50(s[conc], s[resp], n_boot=n_boot, seed=seed)
        except (RuntimeError, ValueError) as e:
            row["error"] = str(e)
        rows.append(row)
    return pd.DataFrame(rows)
