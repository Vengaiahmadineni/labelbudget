"""Synthetic pool-based classification data with PLANTED ground truth.

Each dataset is generated from a known logistic model: features are drawn from
two class-conditional Gaussians plus pure-noise dimensions, so the Bayes-optimal
decision boundary is known in closed form and the achievable accuracy ceiling is
a property of the generator rather than of the learner.

Two regimes exist because the honest answer to "does active learning help?" is
"it depends on the regime":

  balanced   - 50/50 classes, overlapping clusters. Random sampling already
               draws an informative labelled set, so there is little for a
               clever acquisition function to add.
  imbalanced - ~4% positives. Random sampling spends most of its budget on the
               majority class, which is the textbook situation where active
               learning is supposed to pay for itself.
"""

import math
import random
from dataclasses import dataclass, field


@dataclass
class Dataset:
    """A pool of unlabelled points, their hidden labels, and a held-out test set."""

    pool_x: list
    pool_y: list
    test_x: list
    test_y: list
    true_w: list
    true_b: float
    regime: str
    # Average similarity of each pool point to a random reference subsample.
    # Cached here because the density-weighted strategy needs it every round and
    # recomputing an O(n^2) similarity matrix per round dominates runtime.
    density: list = field(default_factory=list)

    @property
    def n_pool(self):
        return len(self.pool_x)

    @property
    def dim(self):
        return len(self.pool_x[0])

    def positive_rate(self):
        return sum(self.pool_y) / len(self.pool_y)


def _sample(rng, n, dim, n_informative, sep, pos_rate, w, b):
    xs, ys = [], []
    for _ in range(n):
        y = 1 if rng.random() < pos_rate else 0
        # Class-conditional means are +-sep/2 along the informative directions,
        # oriented by the planted weight vector so that the Bayes boundary is
        # exactly the hyperplane (w, b).
        row = []
        for j in range(dim):
            if j < n_informative:
                mu = (sep / 2.0) * w[j] * (1.0 if y == 1 else -1.0)
            else:
                mu = 0.0
            row.append(rng.gauss(mu, 1.0))
        xs.append(row)
        ys.append(y)
    return xs, ys


def make_dataset(seed, regime="balanced", n_pool=300, n_test=400, dim=12,
                 n_informative=5, density_refs=60):
    """Build one dataset. `seed` fully determines it (reproducibility test)."""
    rng = random.Random(seed * 7919 + 13)

    if regime == "balanced":
        # sep 2.0 between class means (w is unit-norm, so the Bayes accuracy is
        # Phi(sep/2) = 0.841) with 7 pure-noise dimensions. Chosen by a sweep so
        # the budget actually binds: with sep 3.0 / 8 dims the 8-label seed set
        # already hit the target on most seeds and there was nothing to compare.
        pos_rate, sep = 0.5, 2.0
    elif regime == "imbalanced":
        # 4% positives: a labelling budget of ~120 drawn at random contains only
        # ~5 positives, which is the regime active learning is sold for.
        pos_rate, sep = 0.04, 2.2
    else:
        raise ValueError("unknown regime: %s" % regime)

    w = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in w[:n_informative])) or 1.0
    w = [v / norm if j < n_informative else 0.0 for j, v in enumerate(w)]
    b = 0.0
    ds_sep = sep

    pool_x, pool_y = _sample(rng, n_pool, dim, n_informative, sep, pos_rate, w, b)
    test_x, test_y = _sample(rng, n_test, dim, n_informative, sep, pos_rate, w, b)

    ds = Dataset(pool_x, pool_y, test_x, test_y, w, b, regime)
    ds.sep = ds_sep
    ds.density = _density(pool_x, rng, density_refs)
    return ds


def _density(xs, rng, n_refs):
    """Mean RBF similarity of each point to a random reference subsample.

    A subsample is used rather than the full O(n^2) matrix: the density term only
    has to rank points, and 70 references give a stable enough ranking while
    keeping the whole benchmark inside a few seconds per seed.
    """
    n = len(xs)
    refs = xs if n <= n_refs else rng.sample(xs, n_refs)
    dim = len(xs[0])
    gamma = 1.0 / dim
    out = []
    for x in xs:
        s = 0.0
        for r in refs:
            d2 = 0.0
            for j in range(dim):
                t = x[j] - r[j]
                d2 += t * t
            s += math.exp(-gamma * d2)
        out.append(s / len(refs))
    return out


def bayes_balanced_accuracy(ds):
    """Balanced accuracy of the planted (Bayes) boundary on the test set."""
    tp = fn = tn = fp = 0
    for x, y in zip(ds.test_x, ds.test_y):
        z = sum(wi * xi for wi, xi in zip(ds.true_w, x)) + ds.true_b
        pred = 1 if z > 0 else 0
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
