"""Publication-style dose-response plot: mean ± SEM per concentration plus fitted curve."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .fit import fit_hill


def plot_curves(df: pd.DataFrame, conc: str, resp: str, group: str,
                units: str = "µM", ylabel: str | None = None, ax=None):
    ax = ax or plt.subplots(figsize=(5.5, 4))[1]
    d = df[df[conc] > 0]
    for i, (g, s) in enumerate(d.groupby(group, sort=False)):
        color = f"C{i}"
        agg = s.groupby(conc)[resp].agg(["mean", "sem"]).reset_index()
        ax.errorbar(agg[conc], agg["mean"], yerr=agg["sem"], fmt="o", color=color,
                    capsize=3, ms=5)
        try:
            f = fit_hill(s[conc], s[resp])
        except (RuntimeError, ValueError):
            continue
        xs = np.logspace(np.log10(s[conc].min()), np.log10(s[conc].max()), 200)
        ax.plot(xs, f.predict(xs), color=color,
                label=f"{g}  EC50 = {f.ec50:.3g} {units}, h = {f.params['hill']:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel(f"{conc} ({units})")
    ax.set_ylabel(ylabel or resp)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    return ax
