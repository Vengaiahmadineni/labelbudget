"""Logistic regression written by hand: L2-regularised Newton/IRLS.

No numpy/sklearn is available (and none is needed). The learner is deliberately
simple and deterministic so that every difference between two runs is
attributable to the acquisition function, not to the optimiser.

Class weighting is on by default. Without it a labelled set of 118 negatives and
2 positives collapses to the majority prediction and balanced accuracy pins at
0.5 for every strategy, which would hide the very effect the imbalanced regime
is meant to expose.
"""

import math
from operator import mul

# Optimiser constants. Newton/IRLS converges quadratically, so a handful of
# iterations reaches the optimum; tests/test_model.py checks the fitted model on
# the full pool reaches the planted Bayes ceiling, which is the only
# correctness requirement the benchmark places on the learner.
DEFAULT_EPOCHS = 8
CEILING_EPOCHS = 25
DEFAULT_L2 = 1.0   # ridge strength in the (weighted) log-likelihood; keeps
                   # tiny / separable labelled sets from diverging.


def _sigmoid(z):
    # Split the branches to avoid math.exp overflow on large |z|.
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _solve(a, rhs):
    """Gaussian elimination with partial pivoting (a is modified)."""
    n = len(rhs)
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(a[r][c]))
        a[c], a[piv] = a[piv], a[c]
        rhs[c], rhs[piv] = rhs[piv], rhs[c]
        d = a[c][c]
        if abs(d) < 1e-12:
            continue
        for r in range(c + 1, n):
            f = a[r][c] / d
            if f:
                ar, ac = a[r], a[c]
                for j in range(c, n):
                    ar[j] -= f * ac[j]
                rhs[r] -= f * rhs[c]
    out = [0.0] * n
    for c in range(n - 1, -1, -1):
        s = rhs[c] - sum(a[c][j] * out[j] for j in range(c + 1, n))
        out[c] = s / a[c][c] if abs(a[c][c]) > 1e-12 else 0.0
    return out


class LogisticRegression:
    def __init__(self, dim, l2=DEFAULT_L2, class_weight=True):
        self.w = [0.0] * dim
        self.b = 0.0
        self.l2 = l2
        self.class_weight = class_weight

    def _weights(self, ys):
        if not self.class_weight:
            return {0: 1.0, 1: 1.0}
        n1 = sum(ys)
        n0 = len(ys) - n1
        # Inverse-frequency weights normalised to average 1.
        w1 = len(ys) / (2.0 * n1) if n1 else 0.0
        w0 = len(ys) / (2.0 * n0) if n0 else 0.0
        return {0: w0, 1: w1}

    def fit(self, xs, ys, epochs=DEFAULT_EPOCHS):
        """Regularised Newton-Raphson (IRLS), warm-started from current weights.

        The bias is not penalised. Warm starting is intentional: the
        active-learning loop refits after every batch."""
        if not xs:
            return self
        dim = len(xs[0])
        if len(self.w) != dim:
            self.w = [0.0] * dim
        cw = self._weights(ys)
        m = dim + 1
        rows = [list(x) + [1.0] for x in xs]
        for _ in range(epochs):
            theta = self.w + [self.b]
            grad = [0.0] * m
            hess = [[0.0] * m for _ in range(m)]
            for r, y in zip(rows, ys):
                c = cw[y]
                if c == 0.0:
                    continue
                p = _sigmoid(sum(map(mul, theta, r)))
                g = c * (p - y)
                h = c * p * (1.0 - p)
                for j in range(m):
                    rj = r[j]
                    grad[j] += g * rj
                    hj = h * rj
                    row = hess[j]
                    for k in range(j + 1):
                        row[k] += hj * r[k]
            for j in range(m):
                for k in range(j):
                    hess[k][j] = hess[j][k]
            for j in range(dim):
                grad[j] += self.l2 * theta[j]
                hess[j][j] += self.l2
            hess[dim][dim] += 1e-6
            step = _solve(hess, grad)
            theta = [t - s for t, s in zip(theta, step)]
            self.w, self.b = theta[:dim], theta[dim]
            if max(abs(s) for s in step) < 1e-6:
                break
        return self

    def decision(self, x):
        return self.b + sum(map(mul, self.w, x))

    def predict_proba(self, x):
        return _sigmoid(self.decision(x))

    def predict(self, x):
        return 1 if self.decision(x) > 0.0 else 0


def balanced_accuracy(model, xs, ys):
    """Mean of recall and specificity.

    Plain accuracy is the wrong headline in the imbalanced regime: predicting
    the majority class everywhere scores 0.96 there while being useless.
    """
    tp = fn = tn = fp = 0
    for x, y in zip(xs, ys):
        pred = model.predict(x)
        if y == 1:
            if pred == 1:
                tp += 1
            else:
                fn += 1
        else:
            if pred == 0:
                tn += 1
            else:
                fp += 1
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return 0.5 * (rec + spec)


def plain_accuracy(model, xs, ys):
    return sum(1 for x, y in zip(xs, ys) if model.predict(x) == y) / len(ys)
