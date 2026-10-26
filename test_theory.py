"""Numerical checks of the paper's results (Sections 3.2-3.4 and Examples 1-3)."""
import numpy as np
import pytest

from rbdo import (BDO, RBDO, MinorityTarget, aided_point, explicit_covariance,
                  intensity, positive_boundary_set, renyi, sample_local,
                  select_boundary, shannon, tsallis)


# ---------------------------------------------------------------- Theorem 1
def test_rank_invariance_and_example1():
    k = 10
    c = np.array([0] * 55 + [1] * 9 + [2] * 9 + [3] * 6 + [4] * 2 + [5] * 5
                 + [6] * 4 + [7] * 2 + [8] * 5 + [9] * 3 + [10] * 0)
    assert len(c) == 100
    p = c / k
    sets = {}
    for name, h in {"shannon": shannon, "renyi2": renyi(2), "tsallis2": tsallis(2),
                    "renyi.5": renyi(0.5)}.items():
        IE = h(p)
        # same ranking as |2c-k| (Theorem 1(i)); round away 1e-16 symmetry noise
        u = np.abs(2 * c - k)
        IEr = np.round(IE, 12)
        assert np.all((IEr[:, None] >= IEr[None]) == (u[:, None] <= u[None]))
        sel, _ = select_boundary(IE, kappa=1.0)
        sets[name] = set(sel)
    assert sets["shannon"] == sets["renyi2"] == sets["tsallis2"]
    assert len(sets["shannon"]) == 19                      # Example 1
    assert sets["shannon"] < sets["renyi.5"] and len(sets["renyi.5"]) == 33


# ---------------------------------------------------------------- Prop. 1
def test_empty_selection_and_fallback():
    IE = np.zeros(50)
    assert select_boundary(IE, 1.0, fallback=False)[0].size == 0
    sel, used = select_boundary(IE, 1.0, dist_to_P=np.arange(50.0))
    assert used and len(sel) == int(np.ceil(np.sqrt(50)))
    # two-valued case: empty iff q >= 1/(1+kappa^2)
    for q, empty in [(0.49, False), (0.5, True), (0.6, True)]:
        n = 1000
        IE = np.zeros(n)
        IE[: int(q * n)] = 1.0
        assert (select_boundary(IE, 1.0, fallback=False)[0].size == 0) == empty


# ---------------------------------------------------------------- Theorem 2
def test_example2():
    xnb = np.zeros(2)
    P = np.array([[2.0, 1.0], [3.0, -0.5]])
    Pn, d = positive_boundary_set(xnb, P)
    assert len(Pn) == 2
    zcf, s = aided_point(xnb, Pn, d, 1e-3, 1000, "closed")
    zit, _ = aided_point(xnb, Pn, d, 1e-3, 1000, "iterative")
    assert s == pytest.approx(0.6946, abs=1e-3)
    assert zcf == pytest.approx([1.6038, 0.3728], abs=1e-3)
    assert zit == pytest.approx([1.6047, 0.3730], abs=1e-3)


def test_closed_form_error_bound_and_half_bound():
    rng = np.random.default_rng(0)
    for _ in range(200):
        dim = int(rng.integers(2, 30))
        P = rng.standard_normal((int(rng.integers(3, 20)), dim))
        xnb = rng.standard_normal(dim) * 1.5
        Pn, d = positive_boundary_set(xnb, P)
        for alpha in (1e-1, 1e-2, 1e-3):
            T = int(round(1 / alpha))
            zi, s = aided_point(xnb, Pn, d, alpha, T, "iterative")
            zc, _ = aided_point(xnb, Pn, d, alpha, T, "closed")
            w = np.exp(-(d - d.min())); w /= w.sum()
            g = w @ (Pn - xnb)
            assert np.linalg.norm(zi - zc) <= alpha * np.linalg.norm(g) * (1 + 1e-9)
            assert s >= 0.5 - 1e-9
            # Proposition 2: the point lies in the hypersphere
            xpb = P[np.argmin(np.linalg.norm(P - xnb, axis=1))]
            assert np.linalg.norm(zc - xpb) <= np.linalg.norm(xnb - xpb) * (1 + 1e-9)


# ---------------------------------------------------------------- Theorem 3
def test_example3():
    P = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], float)
    Z = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0]], float)
    tgt = MinorityTarget(P, rho0=0.01, rho=0.3)
    Sig = explicit_covariance(Z, tgt, 0.5)
    assert np.diag(Sig) == pytest.approx([0.5583, 0.3361, 0.05], abs=1e-3)
    assert np.trace(Sig) == pytest.approx(0.9444, abs=1e-3)
    ev = np.linalg.eigvalsh(Sig)
    assert ev.max() / ev.min() == pytest.approx(11.2, abs=0.1)


@pytest.mark.parametrize("seed", range(5))
def test_theorem3_properties(seed):
    rng = np.random.default_rng(seed)
    d, n_pos, m = 25, 12, 5
    P = rng.standard_normal((n_pos, d)) * rng.uniform(0.2, 2, d)
    Z = rng.standard_normal((m, d))
    tgt = MinorityTarget(P)
    mu = Z.mean(0); Zc = Z - mu
    lam = intensity(Zc, tgt, "minority", 0.05)
    Sig = explicit_covariance(Z, tgt, lam)
    S = Zc.T @ Zc / m
    SP = tgt.Pc.T @ tgt.Pc / n_pos
    # (i) positive definite, lambda_min >= lam*rho*nu_P, condition number bound
    ev = np.linalg.eigvalsh(Sig)
    assert ev.min() >= lam * tgt.rho * tgt.nu * (1 - 1e-9)
    q = np.trace(S) / np.trace(SP)
    bound = 1 + d / tgt.rho * ((1 - lam) * q / lam + 1 - tgt.rho)
    assert ev.max() / ev.min() <= bound
    # (ii) trace
    assert np.trace(Sig) == pytest.approx((1 - lam) * np.trace(S) + lam * np.trace(SP))
    # (iii) variance outside L
    L = np.linalg.matrix_rank(np.vstack([Zc, tgt.Pc]), tol=1e-9)
    U, s, Vt = np.linalg.svd(np.vstack([Zc, tgt.Pc]))
    Vperp = Vt[L:]
    off = np.trace(Vperp @ Sig @ Vperp.T)
    assert off == pytest.approx(lam * tgt.rho * tgt.nu * (d - L), rel=1e-8)
    # (v) Gram-matrix intensity equals explicit-matrix intensity
    delta2 = np.sum((S - tgt.explicit()) ** 2)
    beta2 = sum(np.sum((np.outer(v, v) - S) ** 2) for v in Zc) / m**2
    assert lam == pytest.approx(min(1.0, max(beta2 / delta2, 0.05)), rel=1e-9)


def test_factored_sampler_reproduces_covariance():
    rng = np.random.default_rng(1)
    d = 8
    P = rng.standard_normal((20, d))
    Z = rng.standard_normal((4, d))
    tgt = MinorityTarget(P)
    mu = Z.mean(0); Zc = Z - mu
    lam = 0.3
    X = sample_local(mu, Zc, 400_000, rng, "minority", lam, tgt)
    emp = np.cov(X.T, bias=True)
    ref = explicit_covariance(Z, tgt, lam)
    assert np.linalg.norm(emp - ref) / np.linalg.norm(ref) < 0.02


# ---------------------------------------------------------------- end to end
def _toy(seed=0, d=10, n_pos=25, n_neg=500):
    rng = np.random.default_rng(seed)
    Xn = rng.standard_normal((n_neg, d))
    Xp = rng.standard_normal((n_pos, d)) + 1.2
    return np.vstack([Xn, Xp]), np.r_[np.zeros(n_neg, int), np.ones(n_pos, int)]


@pytest.mark.parametrize("cls", [RBDO, BDO])
def test_balances_classes(cls):
    X, y = _toy()
    Xr, yr = cls(random_state=0).fit_resample(X, y)
    assert (yr == 1).sum() == (yr == 0).sum()
    assert np.isfinite(Xr).all()


def test_rbdo_is_deterministic_given_seed():
    X, y = _toy()
    a = RBDO(random_state=3).fit_resample(X, y)[0]
    b = RBDO(random_state=3).fit_resample(X, y)[0]
    assert np.array_equal(a, b)
