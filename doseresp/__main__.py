"""Command-line interface: python -m doseresp DATA.csv [options]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .fit import compare_ec50, fit_groups


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="doseresp",
                                description="Fit 4-parameter Hill curves and compare EC50s.")
    p.add_argument("data", type=Path, help="CSV/Excel in long format: one row per well/replicate")
    p.add_argument("--conc", default="concentration")
    p.add_argument("--resp", default="response")
    p.add_argument("--group", default="condition")
    p.add_argument("--units", default="µM")
    p.add_argument("--bootstrap", type=int, default=0, metavar="N",
                   help="also compute bootstrap EC50 CIs with N resamples")
    p.add_argument("--compare", action="store_true",
                   help="F-test: do all groups share one EC50?")
    p.add_argument("-o", "--outdir", type=Path, default=Path("results"))
    a = p.parse_args(argv)

    df = (pd.read_excel(a.data) if a.data.suffix.lower() in (".xlsx", ".xls")
          else pd.read_csv(a.data))
    a.outdir.mkdir(parents=True, exist_ok=True)

    summary = fit_groups(df, a.conc, a.resp, a.group, n_boot=a.bootstrap)
    summary.to_csv(a.outdir / "fit_summary.csv", index=False)
    with pd.option_context("display.float_format", "{:.3g}".format):
        print(summary.to_string(index=False))

    if a.compare:
        res = compare_ec50(df, a.conc, a.resp, a.group)
        (a.outdir / "ec50_comparison.json").write_text(json.dumps(res, indent=2))
        print(f"\nShared-EC50 F-test: F({res['df_num']}, {res['df_den']}) = {res['F']:.2f}, "
              f"p = {res['p_value']:.3g}")

    from .plot import plot_curves
    ax = plot_curves(df, a.conc, a.resp, a.group, units=a.units)
    ax.figure.tight_layout()
    ax.figure.savefig(a.outdir / "dose_response.png", dpi=200)
    print(f"\nResults in {a.outdir.resolve()}")


if __name__ == "__main__":
    main()
