# RBDO: Regularised Boundary-Degree Over-Sampling

Code for the paper *"Regularised Boundary-Degree Over-Sampling with Closed-Form Aided
samples and Minority-Anchored Covariance Shrinkage for Imbalanced Data"*.

> **Note.** This repository was reconstructed from the algorithms, theorems and worked
> examples in the paper, because the original code was lost. It reproduces the paper's
> Examples 1-3 exactly and passes numerical checks of Theorems 1-3 and Propositions 1-2,
> but it is **not** the original code. Numbers from `experiments/` use a simplified task
> builder and a minimal SMOTE, so they will not match the paper's tables. Before
> submission, re-run the full protocol if you need the paper's tables to be exactly
> reproducible.

## Install

```bash
pip install -r requirements.txt
```

## Use

```python
from rbdo import RBDO, BDO, RBDOIso

X_res, y_res = RBDO(random_state=0).fit_resample(X_train, y_train)  # X_train standardised
```

`fit_resample` expects a binary problem (minority = rarer class) and **features already
standardised on the training fold**. Always resample inside the training fold only.

| Class | Aided sample | Local covariance | Mixture |
|---|---|---|---|
| `BDO` | iterative (Eq. 5) | maximum likelihood | sub-sampling |
| `RBDOIso` | closed form (Eq. 8) | shrunk to `nu I` | multinomial, weights `IE^gamma` |
| `RBDO` | closed form (Eq. 8) | shrunk to minority covariance `T_rho` (Eq. 10) | multinomial, weights `IE^gamma` |

`BoundaryDegreeOversampler` exposes every option (`aided`, `shrinkage`, `allocation`,
`entropy`, `kappa`, `gamma`, `lam0`, `rho0`, `lam`, `rho`, `fallback`) for the ablations
and sensitivity studies. Diagnostics (`M`, fallback use, per-group `lambda`, `rho`) are in
`sampler.diagnostics_`.

Defaults follow the paper: `kappa=1, gamma=1, lam0=0.05, rho0=0.01, alpha=1e-3, T=1000`,
Shannon entropy, `k = n_plus`.

## Layout

```
rbdo/core.py                 boundary degree, sigma-rule + fallback, aided point,
                             minority-anchored shrinkage, factored sampler, BDO/RBDO
tests/test_theory.py         Examples 1-3, Theorems 1-3, Propositions 1-2
experiments/run_experiment.py  in-fold resampling, G-mean, four classifiers
```

## Test

```bash
pip install pytest
python -m pytest -q tests
```

## Experiments

```bash
python experiments/run_experiment.py --tasks wine-3 cancer-IR10 digits8-IR30 --reps 2
```

Add your own license and citation before publishing the repository.
