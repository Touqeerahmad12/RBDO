"""Regularised boundary-degree over-sampling (RBDO) and the original BDO.

Reconstructed from the paper "Regularised Boundary-Degree Over-Sampling with
Closed-Form Aided samples and Minority-Anchored Covariance Shrinkage for
Imbalanced Data".  Equation / theorem numbers in comments refer to that paper.

Binary problems only.  Features should be standardised on the training fold
BEFORE calling ``fit_resample`` (the experiment driver does this).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.covariance import ledoit_wolf_shrinkage

_TOL = 1e-12


# --------------------------------------------------------------------------
# Step 1: boundary degree (Eq. 1, Definition 1)
# --------------------------------------------------------------------------
def shannon(p):
    p = np.asarray(p, dtype=float)
    out = np.zeros_like(p)
    m = (p > 0) & (p < 1)
    q = p[m]
    out[m] = -q * np.log2(q) - (1 - q) * np.log2(1 - q)
    return out


def renyi(a):
    if a <= 0 or a == 1:
        raise ValueError("Renyi order must be > 0 and != 1")

    def h(p):
        p = np.asarray(p, dtype=float)
        return np.log2(p**a + (1 - p) ** a) / (1 - a)

    return h


def tsallis(a):
    if a <= 0 or a == 1:
        raise ValueError("Tsallis order must be > 0 and != 1")

    def h(p):
        p = np.asarray(p, dtype=float)
        return (1 - p**a - (1 - p) ** a) / (a - 1)

    return h


ENTROPIES = {"shannon": shannon}


def get_entropy(h):
    if callable(h):
        return h
    if h in ENTROPIES:
        return ENTROPIES[h]
    raise ValueError(f"unknown entropy {h!r}")


def neighbour_counts(X, is_pos, k, chunk=512):
    """c_i = number of positives among the k nearest neighbours of every
    negative sample (neighbours taken in D minus the sample itself)."""
    neg = np.where(~is_pos)[0]
    c = np.empty(len(neg), dtype=int)
    for s in range(0, len(neg), chunk):
        idx = neg[s : s + chunk]
        D = cdist(X[idx], X)
        D[np.arange(len(idx)), idx] = np.inf  # exclude self
        nn = np.argpartition(D, k - 1, axis=1)[:, :k]
        c[s : s + chunk] = is_pos[nn].sum(axis=1)
    return neg, c


# --------------------------------------------------------------------------
# Step 2: sigma-rule with fallback (Eq. 2, Proposition 1)
# --------------------------------------------------------------------------
def select_boundary(IE, kappa, dist_to_P=None, fallback=True):
    """Return indices (into the negative array) of the boundary set."""
    mu, sigma = IE.mean(), IE.std()
    sel = np.where(IE > mu + kappa * sigma + _TOL)[0]
    if sel.size or not fallback:
        return sel, False
    if IE.max() > 0:
        return np.array([int(np.argmax(IE))]), True
    n_sel = int(np.ceil(np.sqrt(len(IE))))
    return np.argsort(dist_to_P)[:n_sel], True


# --------------------------------------------------------------------------
# Step 3: aided boundary point (Eqs. 3-5, 7-8, Theorem 2)
# --------------------------------------------------------------------------
def positive_boundary_set(xnb, P):
    """P(x^nb): positives inside the hypersphere around the nearest positive
    that passes through x^nb (Eq. 3)."""
    d_nb = np.linalg.norm(P - xnb, axis=1)
    j0 = int(np.argmin(d_nb))
    r = d_nb[j0]
    inside = np.linalg.norm(P - P[j0], axis=1) <= r * (1 + _TOL)
    return P[inside], d_nb[inside]


def aided_point(xnb, Pn, d, alpha=1e-3, T=1000, method="closed"):
    """Aided sample for one boundary negative.

    method='closed'    -> z^cf = x^nb + min(s*, alpha*T) g          (Eq. 8)
    method='iterative' -> constant-direction walk of Eq. (5)
    Returns (z, s_star).
    """
    diff = Pn - xnb
    w = np.exp(-(d - d.min()))  # stable softmax of -d
    w /= w.sum()
    g = w @ diff  # Eq. (4)
    a = d**2
    b = 2.0 * diff @ g  # Eq. (7)
    if np.linalg.norm(g) == 0:
        return xnb.copy(), np.inf
    s_star = np.max(a / b) if np.all(b > 0) else np.inf
    if method == "closed":
        return xnb + min(s_star, alpha * T) * g, s_star
    if method == "iterative":
        for t in range(T):
            z = xnb + t * alpha * g
            if np.all(np.linalg.norm(z - Pn, axis=1) < np.linalg.norm(z - xnb)):
                return z, s_star
        return xnb + T * alpha * g, s_star
    raise ValueError(method)


# --------------------------------------------------------------------------
# Step 4: local Gaussians and minority-anchored shrinkage (Section 3.4)
# --------------------------------------------------------------------------
class MinorityTarget:
    """Quantities of T_rho = (1-rho) S_P + rho nu_P I (Eq. 10), computed once."""

    def __init__(self, P, rho0=0.01, rho=None):
        self.P = P
        self.n, self.d = P.shape
        self.pbar = P.mean(axis=0)
        self.Pc = P - self.pbar
        self.C = self.Pc @ self.Pc.T
        self.tr_SP = np.trace(self.C) / self.n
        self.nu = self.tr_SP / self.d
        self.fro2_SP = np.sum(self.C**2) / self.n**2
        if rho is None:
            rho = max(float(ledoit_wolf_shrinkage(P)), rho0)
        self.rho = float(min(max(rho, rho0), 1.0)) if rho is not None else rho0
        r = self.rho
        self.fro2_T = (
            (1 - r) ** 2 * self.fro2_SP
            + 2 * (1 - r) * r * self.nu * self.tr_SP
            + r**2 * self.nu**2 * self.d
        )

    def explicit(self):
        SP = self.Pc.T @ self.Pc / self.n
        return (1 - self.rho) * SP + self.rho * self.nu * np.eye(self.d)


def intensity(Zc, target, kind, lam0):
    """Ledoit-Wolf-type intensity lam = beta^2 / delta^2 truncated to [lam0, 1],
    computed from Gram matrices only (Theorem 3(v))."""
    m, d = Zc.shape
    A = Zc @ Zc.T
    fro2_S = np.sum(A**2) / m**2
    tr_S = np.trace(A) / m
    beta2 = (np.sum(np.diag(A) ** 2) - m * fro2_S) / m**2
    if kind == "minority":
        B = Zc @ target.Pc.T
        inner_SSP = np.sum(B**2) / (m * target.n)
        inner_ST = (1 - target.rho) * inner_SSP + target.rho * target.nu * tr_S
        delta2 = fro2_S - 2 * inner_ST + target.fro2_T
    else:  # isotropic target nu I, nu = tr S / d
        delta2 = fro2_S - tr_S**2 / d
    if delta2 <= _TOL:
        return 1.0
    return float(min(1.0, max(beta2 / delta2, lam0)))


def sample_local(mu, Zc, K, rng, kind, lam, target=None):
    """Draw K points from N(mu, Sigma_lam) in factored form (Eq. 12); no d x d
    matrix is ever formed.  kind in {'none', 'iso', 'minority'}."""
    m, d = Zc.shape
    if kind == "none":  # maximum-likelihood covariance S
        return mu + (rng.standard_normal((K, m)) @ Zc) / np.sqrt(m)
    x = mu + np.sqrt((1 - lam) / m) * (rng.standard_normal((K, m)) @ Zc)
    if kind == "iso":
        nu = np.sum(Zc**2) / m / d
        return x + np.sqrt(lam * nu) * rng.standard_normal((K, d))
    x += np.sqrt(lam * (1 - target.rho) / target.n) * (
        rng.standard_normal((K, target.n)) @ target.Pc
    )
    return x + np.sqrt(lam * target.rho * target.nu) * rng.standard_normal((K, d))


def explicit_covariance(Z, target, lam):
    """Explicit Sigma_lam = (1-lam) S + lam T_rho (for tests / diagnostics)."""
    mu = Z.mean(axis=0)
    S = (Z - mu).T @ (Z - mu) / len(Z)
    return (1 - lam) * S + lam * target.explicit()


# --------------------------------------------------------------------------
# The sampler
# --------------------------------------------------------------------------
class BoundaryDegreeOversampler:
    """General sampler; ``BDO`` and ``RBDO`` below fix the options.

    aided      : 'closed' | 'iterative' | 'none'
    shrinkage  : 'minority' | 'iso' | 'none'
    allocation : 'weighted' (multinomial, pi ~ IE^gamma) | 'subsample' (BDO)
    """

    def __init__(self, kappa=1.0, gamma=1.0, lam0=0.05, rho0=0.01, rho=None,
                 lam=None, alpha=1e-3, T=1000, entropy="shannon",
                 aided="closed", shrinkage="minority", allocation="weighted",
                 fallback=True, random_state=None):
        self.kappa, self.gamma, self.lam0, self.rho0 = kappa, gamma, lam0, rho0
        self.rho, self.lam, self.alpha, self.T = rho, lam, alpha, T
        self.entropy, self.aided, self.shrinkage = entropy, aided, shrinkage
        self.allocation, self.fallback = allocation, fallback
        self.random_state = random_state

    # ------------------------------------------------------------------
    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        labels, counts = np.unique(y, return_counts=True)
        if len(labels) != 2:
            raise ValueError("binary problems only")
        pos_label = labels[np.argmin(counts)]
        is_pos = y == pos_label
        P = X[is_pos]
        n_pos, n_neg = len(P), int((~is_pos).sum())
        K = n_neg - n_pos
        self.diagnostics_ = {"K": K}
        if K <= 0 or n_pos < 2:
            return X, y

        # Step 1
        h = get_entropy(self.entropy)
        neg, c = neighbour_counts(X, is_pos, k=n_pos)
        IE = h(c / n_pos)
        # Step 2
        dist_to_P = cdist(X[neg], P).min(axis=1)
        sel, used_fb = select_boundary(IE, self.kappa, dist_to_P, self.fallback)
        self.diagnostics_.update(M=len(sel), fallback_used=used_fb)
        if len(sel) == 0:  # BDO is undefined here: return the data unchanged
            return X, y

        target = None
        if self.shrinkage == "minority":
            target = MinorityTarget(P, self.rho0, self.rho)
            self.diagnostics_["rho"] = target.rho

        # Steps 3-4: local groups
        comps = []
        for i in sel:
            xnb = X[neg[i]]
            Pn, d = positive_boundary_set(xnb, P)
            if self.aided == "none":
                Z = Pn if len(Pn) >= 2 else np.vstack([xnb, Pn])
            else:
                z, _ = aided_point(xnb, Pn, d, self.alpha, self.T, self.aided)
                Z = np.vstack([z, Pn])
            mu = Z.mean(axis=0)
            Zc = Z - mu
            if self.shrinkage == "none":
                lam = 0.0
            elif self.lam is not None:
                lam = self.lam
            else:
                lam = intensity(Zc, target, self.shrinkage, self.lam0)
            comps.append((mu, Zc, lam))
        self.diagnostics_["lambdas"] = [cp[2] for cp in comps]

        # Mixture
        M = len(comps)
        kind = self.shrinkage
        if self.allocation == "weighted":
            w = IE[sel] ** self.gamma
            pi = w / w.sum() if w.sum() > 0 else np.full(M, 1.0 / M)
            Ks = rng.multinomial(K, pi)
            parts = [sample_local(mu, Zc, Ki, rng, kind, lam, target)
                     for (mu, Zc, lam), Ki in zip(comps, Ks) if Ki > 0]
            Xsyn = np.vstack(parts)
        else:  # BDO: K draws from every component, keep K of the MK pool
            pool = np.vstack([sample_local(mu, Zc, K, rng, kind, lam, target)
                              for (mu, Zc, lam) in comps])
            Xsyn = pool[rng.choice(len(pool), size=K, replace=False)]

        X_res = np.vstack([X, Xsyn])
        y_res = np.concatenate([y, np.full(len(Xsyn), pos_label)])
        return X_res, y_res


class BDO(BoundaryDegreeOversampler):
    """Original BDO (Chen et al. 2023): iterative aided sample, maximum-likelihood
    local covariance, sub-sampling, no fallback."""

    def __init__(self, kappa=1.0, alpha=1e-3, T=1000, entropy="shannon",
                 random_state=None):
        super().__init__(kappa=kappa, alpha=alpha, T=T, entropy=entropy,
                         aided="iterative", shrinkage="none",
                         allocation="subsample", fallback=False,
                         random_state=random_state)


class RBDO(BoundaryDegreeOversampler):
    """Proposed method: closed-form aided sample, minority-anchored shrinkage,
    multinomial allocation, fallback selection."""

    def __init__(self, kappa=1.0, gamma=1.0, lam0=0.05, rho0=0.01, alpha=1e-3,
                 T=1000, entropy="shannon", random_state=None):
        super().__init__(kappa=kappa, gamma=gamma, lam0=lam0, rho0=rho0,
                         alpha=alpha, T=T, entropy=entropy, aided="closed",
                         shrinkage="minority", allocation="weighted",
                         fallback=True, random_state=random_state)


class RBDOIso(BoundaryDegreeOversampler):
    """First version of the method: isotropic target nu I (RBDO-iso)."""

    def __init__(self, kappa=1.0, gamma=1.0, lam0=0.05, alpha=1e-3, T=1000,
                 random_state=None):
        super().__init__(kappa=kappa, gamma=gamma, lam0=lam0, alpha=alpha, T=T,
                         aided="closed", shrinkage="iso", allocation="weighted",
                         fallback=True, random_state=random_state)
