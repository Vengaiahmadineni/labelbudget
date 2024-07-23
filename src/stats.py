"""Distribution-free statistics for the comparison, in pure stdlib.

Everything here operates on PAIRED differences: strategy A and strategy B are
run on the same seed, which means the same dataset draw and the same initial
labelled set. Pairing removes the dataset-to-dataset variance, which is the
largest term; an unpaired comparison would need several times as many seeds to
resolve the same effect.
"""

import math
import random
import statistics


def median(xs):
    return statistics.median(xs)


def quantile(xs, q):
    """Linear-interpolation quantile (type 7), matching the usual IQR convention."""
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def iqr(xs):
    return quantile(xs, 0.75) - quantile(xs, 0.25)


def paired_bootstrap_ci(diffs, alpha=0.05, iters=2000, seed=0, stat=median):
    """Percentile bootstrap CI for a statistic of the paired differences."""
    rng = random.Random(seed)
    n = len(diffs)
    reps = []
    for _ in range(iters):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        reps.append(stat(sample))
    reps.sort()
    lo = reps[int(math.floor((alpha / 2.0) * (iters - 1)))]
    hi = reps[int(math.ceil((1.0 - alpha / 2.0) * (iters - 1)))]
    return lo, hi


def sign_test_p(diffs):
    """Two-sided exact sign test on paired differences; ties are dropped.

    Chosen over a t-test because labels-to-target is right-censored at the
    budget and its distribution is visibly skewed, so a mean-based test would
    make a normality assumption the data do not support.
    """
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def normal_quantile(p):
    """Inverse standard normal CDF by bisection on math.erf.

    Bisection rather than a rational approximation: 80 iterations is
    instantaneous at this call volume and there is no approximation error to
    reason about.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        cdf = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
        if cdf < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def seeds_needed(sd_diff, delta, alpha=0.05, power=0.8):
    """Paired-design sample size: how many seeds to resolve a difference `delta`.

    n = (z_{1-alpha/2} + z_{power})^2 * sd^2 / delta^2

    This is the calculation nearly every published learning curve omits. It is
    the reason a 30-seed benchmark can still be underpowered.
    """
    if delta <= 0:
        raise ValueError("delta must be positive")
    if sd_diff <= 0:
        return 1
    za = normal_quantile(1.0 - alpha / 2.0)
    zb = normal_quantile(power)
    return int(math.ceil(((za + zb) ** 2) * (sd_diff ** 2) / (delta ** 2)))


def detectable_delta(sd_diff, n, alpha=0.05, power=0.8):
    """Inverse of seeds_needed: the smallest effect n seeds can resolve."""
    za = normal_quantile(1.0 - alpha / 2.0)
    zb = normal_quantile(power)
    return (za + zb) * sd_diff / math.sqrt(n)


def describe(xs):
    return {
        "median": median(xs),
        "q1": quantile(xs, 0.25),
        "q3": quantile(xs, 0.75),
        "iqr": iqr(xs),
        "mean": statistics.fmean(xs),
        "sd": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        "min": min(xs),
        "max": max(xs),
        "n": len(xs),
    }
