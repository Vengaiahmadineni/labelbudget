"""Comparison guard, label-budget recommendation, cost report, ASCII bands.

The central rule of the benchmark lives here: NO COMPARATIVE CLAIM IS EMITTED
unless (a) at least `min_seeds` paired seeds exist and (b) the paired bootstrap
confidence interval of the mean label difference excludes zero AND the exact
sign test agrees. Otherwise the verdict is 'indistinguishable at N seeds'.
Requiring both is conservative on purpose: the bootstrap CI is slightly
anti-conservative at small n, and the sign test ignores magnitude.
"""

from dataclasses import dataclass

from .stats import paired_bootstrap_ci, sign_test_p, describe, quantile
from .runner import paired_diffs
from .strategies import BASELINE

import statistics

# Below this many seeds no ranking claim is made at all. 10 is the smallest n
# for which the two-sided exact sign test can reach p < 0.05 with a few ties
# (n=6 all-one-direction gives p=0.031; ties at labels_to_target are common
# because label counts are multiples of the batch size).
MIN_SEEDS = 10


@dataclass
class Verdict:
    a: str
    b: str
    n: int
    mean_diff: float      # labels(a) - labels(b); negative = a needs fewer
    ci: tuple
    sign_p: float
    verdict: str          # 'a_better' | 'b_better' | 'indistinguishable' | 'insufficient_seeds'

    def sentence(self):
        if self.verdict == "insufficient_seeds":
            return "%s vs %s: only %d seeds (< %d) - no comparative claim emitted" % (
                self.a, self.b, self.n, MIN_SEEDS)
        if self.verdict == "indistinguishable":
            return ("%s vs %s: indistinguishable at %d seeds "
                    "(mean diff %+.1f labels, 95%% CI [%+.1f, %+.1f], sign p=%.3f)" % (
                        self.a, self.b, self.n, self.mean_diff, self.ci[0], self.ci[1], self.sign_p))
        win, lose = (self.a, self.b) if self.verdict == "a_better" else (self.b, self.a)
        return ("%s needs fewer labels than %s at %d seeds "
                "(mean diff %+.1f labels, 95%% CI [%+.1f, %+.1f], sign p=%.3f)" % (
                    win, lose, self.n, self.mean_diff, self.ci[0], self.ci[1], self.sign_p))


def compare_diffs(diffs, a="A", b="B", alpha=0.05, min_seeds=MIN_SEEDS, boot_seed=0):
    n = len(diffs)
    mean = statistics.fmean(diffs) if diffs else 0.0
    if n < min_seeds:
        return Verdict(a, b, n, mean, (float("nan"), float("nan")), 1.0, "insufficient_seeds")
    ci = paired_bootstrap_ci(diffs, alpha=alpha, seed=boot_seed, stat=statistics.fmean)
    p = sign_test_p(diffs)
    if ci[1] < 0 and p < alpha:
        v = "a_better"
    elif ci[0] > 0 and p < alpha:
        v = "b_better"
    else:
        v = "indistinguishable"
    return Verdict(a, b, n, mean, ci, p, v)


def compare(results, a, b=BASELINE, **kw):
    return compare_diffs(paired_diffs(results, a, b), a, b, **kw)


def single_seed_claim(results, a, b, seed):
    """What a one-run benchmark would report: whoever needed fewer labels wins.

    This is the GENERIC approach the benchmark exists to replace. It always
    names a winner (unless exactly tied), however large the seed noise is.
    """
    la = next(r.labels_to_target for r in results[a] if r.seed == seed)
    lb = next(r.labels_to_target for r in results[b] if r.seed == seed)
    if la == lb:
        return None, la, lb
    return (a if la < lb else b), la, lb


def budget_recommendation(runs, coverage=0.9):
    """Labels that reached the target in `coverage` of seeds.

    Returns (labels or None, n_censored). None means the target was not reached
    within the budget on enough seeds: the honest answer is 'buy more than the
    budget', not a number computed from censored values.
    """
    ltt = [r.labels_to_target for r in runs]
    cens = sum(1 for r in runs if r.censored)
    # Epsilon: (1 - 0.9) * 10 is 0.99999..., which made ONE censored seed out
    # of ten count as 'too many' (caught by test_budget_recommendation_and_cost).
    if cens > (1.0 - coverage) * len(runs) + 1e-9:
        return None, cens
    return int(quantile(ltt, coverage) + 0.999999), cens


def cost(labels, minutes_per_label=6.0, hourly_rate=90.0):
    """Money for `labels` expert labels. 6 min per clinical note is an assumed
    default, exposed as a parameter; the README states it as an assumption."""
    return labels * minutes_per_label / 60.0 * hourly_rate


def summary_rows(results):
    rows = []
    for name, runs in results.items():
        d = describe([r.labels_to_target for r in runs])
        rows.append((name, d, sum(1 for r in runs if r.censored)))
    return rows


def ascii_bands(results, names, width=40, lo=0.5, hi=0.9):
    """Accuracy-vs-labels with an inter-quartile band, one line per checkpoint.

    '-' spans Q1..Q3 across seeds, the letter marks the median. Text, because
    the demo must run on a plain Windows console (Plotly is not available).
    """
    lines = []
    n_points = len(next(iter(results.values()))[0].curve)
    for name in names:
        tag = name[0].upper()
        lines.append("  %s = %s" % (tag, name))
    for step in range(0, n_points, 2):
        n_labels = results[names[0]][0].curve[step][0]
        for name in names:
            accs = [r.curve[step][1] for r in results[name]]
            q1, med, q3 = quantile(accs, .25), quantile(accs, .5), quantile(accs, .75)
            pos = lambda v: max(0, min(width - 1, int((v - lo) / (hi - lo) * (width - 1))))
            bar = [" "] * width
            for i in range(pos(q1), pos(q3) + 1):
                bar[i] = "-"
            bar[pos(med)] = name[0].upper()
            lines.append("  %4d |%s| med %.3f IQR [%.3f,%.3f]" % (n_labels, "".join(bar), med, q1, q3))
        lines.append("")
    return lines
