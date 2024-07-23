"""Acquisition strategies.

All strategies share the signature

    select(k, candidates, labeled, ds, model, rng) -> list[int]

and return exactly k pool indices drawn from `candidates` (never from
`labeled`). Keeping random in the same interface as the rest is deliberate:
the baseline has to go through identical machinery, otherwise a difference in
the harness gets misread as a difference in the strategy.
"""

import math


def _shuffled(candidates, rng):
    c = list(candidates)
    rng.shuffle(c)
    return c


def select_random(k, candidates, labeled, ds, model, rng):
    return _shuffled(candidates, rng)[:k]


def least_confidence_score(p):
    """Uncertainty of a binary prediction: 1 - max(p, 1-p)."""
    return 1.0 - max(p, 1.0 - p)


def margin_score(p):
    """Negative margin between the top two class probabilities.

    For BINARY problems this is an affine function of least confidence
    (margin = |2p-1| = 1 - 2*least_confidence), so margin and least-confidence
    sampling select identical points. tests/test_strategies.py asserts that
    equivalence rather than letting the benchmark pretend they are two
    independent strategies.
    """
    return -abs(2.0 * p - 1.0)


def entropy_score(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def _top_k(scored, k, rng):
    # Shuffle before the sort so that ties (very common in the cold-start rounds
    # when the model is still all-zeros and every score is identical) are broken
    # uniformly at random instead of by pool order. Without this, "uncertainty"
    # silently degenerates into "lowest index first", which is worse than random
    # for the wrong reason and would corrupt the cold-start finding.
    rng.shuffle(scored)
    scored.sort(key=lambda t: t[0], reverse=True)
    return [i for _, i in scored[:k]]


def select_uncertainty(k, candidates, labeled, ds, model, rng):
    scored = [(least_confidence_score(model.predict_proba(ds.pool_x[i])), i)
              for i in candidates]
    return _top_k(scored, k, rng)


def select_margin(k, candidates, labeled, ds, model, rng):
    scored = [(margin_score(model.predict_proba(ds.pool_x[i])), i)
              for i in candidates]
    return _top_k(scored, k, rng)


def select_density(k, candidates, labeled, ds, model, rng):
    """Information density: uncertainty weighted by how typical a point is.

    Settles & Craven's beta exponent is fixed at 1.0 - the benchmark is about
    the seed variance of the comparison, and a swept exponent would let us pick
    the winner after seeing the answer.
    """
    scored = [(least_confidence_score(model.predict_proba(ds.pool_x[i])) * ds.density[i], i)
              for i in candidates]
    return _top_k(scored, k, rng)


def select_coreset(k, candidates, labeled, ds, model, rng):
    """Greedy k-centre (core-set) selection: pure diversity, no model input.

    Repeatedly takes the candidate furthest from the nearest already-selected or
    already-labelled point. Model-free, so it is immune to the cold-start
    failure that afflicts uncertainty sampling.
    """
    cand = list(candidates)
    if not cand:
        return []
    # Pairwise squared distances are cached on the dataset: recomputing them
    # every round made core-set ~90% of the demo's runtime (138s -> see README).
    D = getattr(ds, "_d2", None)
    if D is None:
        X = ds.pool_x
        D = [[sum((p - q) * (p - q) for p, q in zip(a, b)) for b in X] for a in X]
        ds._d2 = D

    def d2(i, j):
        return D[i][j]

    if labeled:
        nearest = [min(D[i][j] for j in labeled) for i in cand]
    else:
        # No labels yet: seed the greedy sweep from a random point so the result
        # is not an artefact of pool ordering.
        start = cand[rng.randrange(len(cand))]
        nearest = [D[i][start] for i in cand]

    picked = []
    taken = [False] * len(cand)
    for _ in range(min(k, len(cand))):
        best, best_d = -1, -1.0
        for i in range(len(cand)):
            if not taken[i] and nearest[i] > best_d:
                best, best_d = i, nearest[i]
        taken[best] = True
        picked.append(cand[best])
        bp = cand[best]
        for i in range(len(cand)):
            if not taken[i]:
                d = d2(cand[i], bp)
                if d < nearest[i]:
                    nearest[i] = d
    return picked


def select_most_confident(k, candidates, labeled, ds, model, rng):
    """Deliberately BAD strategy: labels the points the model is surest about.

    Not a recommendation. It exists so the benchmark has a PLANTED real effect
    to detect: a comparison guard that cannot tell this apart from random
    across seeds is broken. Excluded from DEFAULT_STRATEGIES.
    """
    scored = [(-least_confidence_score(model.predict_proba(ds.pool_x[i])), i)
              for i in candidates]
    return _top_k(scored, k, rng)


STRATEGIES = {
    "random": select_random,
    "uncertainty": select_uncertainty,
    "margin": select_margin,
    "density": select_density,
    "coreset": select_coreset,
    "most_confident": select_most_confident,
}

DEFAULT_STRATEGIES = ["random", "uncertainty", "margin", "density", "coreset"]

BASELINE = "random"
