"""Small cross-validation driver: resampling happens INSIDE each training fold,
features are standardised on the training fold only, metric is the G-mean.

Usage:
    python experiments/run_experiment.py --reps 2 --tasks wine-3 cancer-IR10 digits8-IR30
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_wine, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rbdo import BDO, RBDO, RBDOIso  # noqa: E402


# ------------------------------------------------------------------ baselines
class SMOTE:
    def __init__(self, k=5, random_state=None):
        self.k, self.random_state = k, random_state

    def fit_resample(self, X, y):
        rng = np.random.default_rng(self.random_state)
        labels, counts = np.unique(y, return_counts=True)
        pos = labels[np.argmin(counts)]
        P = X[y == pos]
        K = counts.max() - counts.min()
        k = min(self.k, len(P) - 1)
        D = np.linalg.norm(P[:, None] - P[None], axis=2)
        nn = np.argsort(D, axis=1)[:, 1 : k + 1]
        i = rng.integers(0, len(P), K)
        j = nn[i, rng.integers(0, k, K)]
        Xs = P[i] + rng.random((K, 1)) * (P[j] - P[i])
        return np.vstack([X, Xs]), np.concatenate([y, np.full(K, pos)])


class NoResampling:
    def fit_resample(self, X, y):
        return X, y


# ------------------------------------------------------------------ tasks
def _subsample(X, y, pos_label, n_pos, rng):
    pos = np.where(y == pos_label)[0]
    neg = np.where(y != pos_label)[0]
    pos = rng.choice(pos, size=min(n_pos, len(pos)), replace=False)
    idx = np.concatenate([pos, neg])
    return X[idx], (y[idx] == pos_label).astype(int)


def make_task(name, seed=0):
    rng = np.random.default_rng(seed)
    if name == "wine-3":
        X, y = load_wine(return_X_y=True)
        return _subsample(X, y, 2, 16, rng)
    if name.startswith("cancer-IR"):
        ir = int(name.split("IR")[1])
        X, y = load_breast_cancer(return_X_y=True)  # 0 = malignant
        return _subsample(X, y, 0, round(357 / ir), rng)
    if name.startswith("digits"):
        digit, ir = name[6:].split("-IR")
        X, y = load_digits(return_X_y=True)
        return _subsample(X, y, int(digit), round(1617 / int(ir)), rng)
    if name.startswith("clf-d"):
        d, ir = name[5:].split("-IR")
        n_neg = 600
        X, y = make_classification(n_samples=n_neg + n_neg // int(ir) * 1,
                                   n_features=int(d), n_informative=max(2, int(d) // 3),
                                   n_redundant=0, weights=[1 - 1 / int(ir), 1 / int(ir)],
                                   random_state=seed)
        return X, y
    raise ValueError(f"unknown task {name}")


CLASSIFIERS = {
    "rf": lambda s: RandomForestClassifier(100, random_state=s, n_jobs=1),
    "svm": lambda s: SVC(kernel="rbf", C=1.0, gamma="scale", random_state=s),
    "knn": lambda s: KNeighborsClassifier(5),
    "gnb": lambda s: GaussianNB(),
}
METHODS = {
    "None": lambda s: NoResampling(),
    "SMOTE": lambda s: SMOTE(random_state=s),
    "BDO": lambda s: BDO(random_state=s),
    "RBDO-iso": lambda s: RBDOIso(random_state=s),
    "RBDO": lambda s: RBDO(random_state=s),
}


def gmean(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    return float(np.sqrt(tpr * tnr))


def evaluate(task, reps, folds=5):
    X, y = make_task(task)
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=reps, random_state=0)
    scores = {(m, c): [] for m in METHODS for c in CLASSIFIERS}
    for f, (tr, te) in enumerate(cv.split(X, y)):
        sc = StandardScaler().fit(X[tr])           # training fold only
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        for m, make in METHODS.items():
            Xr, yr = make(f).fit_resample(Xtr, y[tr])  # training fold only
            for c, mk in CLASSIFIERS.items():
                clf = mk(f).fit(Xr, yr)
                scores[(m, c)].append(gmean(y[te], clf.predict(Xte)))
    return {k: float(np.mean(v)) for k, v in scores.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["wine-3", "cancer-IR10", "clf-d20-IR30"])
    ap.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()
    print(f"{'task':16s} {'method':10s} " + " ".join(f"{c:>6s}" for c in CLASSIFIERS) + "   mean")
    for t in a.tasks:
        t0 = time.time()
        res = evaluate(t, a.reps)
        for m in METHODS:
            row = [res[(m, c)] for c in CLASSIFIERS]
            print(f"{t:16s} {m:10s} " + " ".join(f"{v:6.3f}" for v in row) + f"   {np.mean(row):.3f}")
        print(f"  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
