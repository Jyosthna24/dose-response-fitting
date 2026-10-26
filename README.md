# dose-response-fitting

![tests](https://github.com/Jyosthna24/dose-response-fitting/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Fits four-parameter Hill models to dose-response data, quantifies the uncertainty in each EC50, and tests whether two conditions differ in potency. It's the analysis GraphPad Prism does, made scriptable, reproducible and testable.**

During my PhD (UCL) I characterised CFTR modulators such as potentiators and correctors across CFTR variants with dose-response experiments. This package is a clean, open re-implementation of that analysis.

## Why it matters
"Is compound X more potent on variant A than on variant B?" is a **statistical** question, not a visual one. Answering it properly requires:
- a sensible non-linear model
- honest confidence intervals on EC50 (they're asymmetric!)
- a formal test of the difference

Point-and-click tools hide these choices. Here every step is explicit and covered by tests.

## Model

Four-parameter log-logistic (Hill) equation on $x = \log_{10}[\text{concentration}]$:

$$
y = \text{Bottom} + \frac{\text{Top} - \text{Bottom}}{1 + 10^{(\log EC_{50} - x)\cdot h}}
$$

| Parameter | Meaning |
|---|---|
| Bottom / Top | baseline and maximal (efficacy) response |
| $\log EC_{50}$ | potency: concentration giving a half-maximal effect |
| $h$ (Hill slope) | steepness; $h > 0$ activator, $h < 0$ inhibitor |

## Methods

| Step | Method | Why |
|---|---|---|
| Fit | Bounded non-linear least squares (`scipy.optimize.curve_fit`) with data-driven starting values | Robust convergence without manual tuning |
| EC50 uncertainty | Fitted on the **log scale**, with a t-based 95 % CI back-transformed to concentration | EC50 error is log-normal, so its CI is asymmetric in µM |
| Robustness check | **Case-resampling bootstrap** CI (`--bootstrap N`) | Doesn't rely on asymptotic normality |
| Constraints | Optionally fix Bottom (e.g. to the vehicle mean) or Top | Common when the curve doesn't plateau |
| Comparing EC50s | **Extra-sum-of-squares F-test**: separate EC50s vs one shared EC50 (other parameters free per group) | A formal test of a potency shift, the same approach as Prism's "Is logEC50 different?" |
| Vehicle wells | Excluded from the log-axis fit automatically | log(0) is undefined |

## Validation
The test suite fits **simulated experiments with known parameters** and checks that:
- noise-free data recover the true parameters exactly
- the 95 % EC50 confidence interval contains the true value in about 95 % of repeated experiments (**coverage**)
- the F-test detects a real 7-fold EC50 shift (p < 0.001)
- the F-test keeps its **false-positive rate close to 5 %** when the curves are identical
- inhibitors (negative Hill slope), fixed parameters and too-sparse data are handled correctly

## Usage

```bash
pip install -r requirements.txt
python scripts/make_synthetic_data.py                     # writes data/synthetic_dose_response.csv
python -m doseresp data/synthetic_dose_response.csv --compare --bootstrap 1000
```

Input is long format, with one row per well or replicate: `condition, concentration, response` (column names configurable with `--conc`, `--resp` and `--group`).

Outputs in `results/`:
- `fit_summary.csv`: EC50 with its 95 % CI (asymptotic and bootstrap), Hill slope, Top, Bottom, R² and n, for each condition
- `ec50_comparison.json`: F statistic, degrees of freedom, p-value and the shared EC50
- `dose_response.png`: mean ± SEM per concentration, with the fitted curves

As a library:

```python
from doseresp import fit_hill, compare_ec50
f = fit_hill(conc, resp)
f.ec50, f.ec50_ci95, f.params["hill"], f.r2
compare_ec50(df, "concentration", "response", "condition")["p_value"]
```

## Skills demonstrated
Non-linear regression · parameter uncertainty (asymptotic and bootstrap) · nested-model hypothesis testing · simulation-based validation of statistical methods (CI coverage, type I error) · reproducible CLI and library design · testing and CI

---
*Part of my PhD work on CFTR modulator pharmacology. See my [GitHub profile](https://github.com/Jyosthna24).*
