"""Multi-seed active-learning experiment runner.

The headline metric is LABELS TO REACH A TARGET SCORE, not score at a fixed
label count. A practitioner budgets in labels, so the benchmark has to answer in
labels. That choice has a consequence the accuracy-curve framing hides: a run
that never reaches the target inside the budget is RIGHT-CENSORED, and censored
runs must be counted and reported, not quietly dropped.
"""

import random
import zlib
from dataclasses import dataclass, field

from .dataset import make_dataset
from .model import LogisticRegression, balanced_accuracy, CEILING_EPOCHS
from .strategies import STRATEGIES, BASELINE


@dataclass
class RunConfig:
    regime: str = "balanced"
    seed_size: int = 4          # initial random labels, before any strategy acts
    batch: int = 8              # labels acquired per acquisition round
    budget: int = 164           # total label budget
    target_frac: float = 0.97   # target = target_frac * full-pool ceiling
    n_pool: int = 300
    n_test: int = 400
    stratified_seed: bool = False


@dataclass
class RunResult:
    strategy: str
    seed: int
    curve: list                  # [(n_labels, balanced accuracy on test)]
    labels_to_target: int
    censored: bool
    target: float
    ceiling: float
    positives_labelled: int
    label_counts: list = field(default_factory=list)


def _initial_labels(ds, cfg, rng):
    idx = list(range(ds.n_pool))
    rng.shuffle(idx)
    if not cfg.stratified_seed:
        return idx[:cfg.seed_size]
    # Optional: force one of each class into the seed set. Off by default
    # because forcing a positive into the seed set is exactly the information
    # active learning is supposed to have to find for itself, and switching it
    # on removes most of the cold-start effect we are trying to measure.
    pos = [i for i in idx if ds.pool_y[i] == 1][:1]
    neg = [i for i in idx if ds.pool_y[i] == 0][:1]
    rest = [i for i in idx if i not in pos and i not in neg]
    return (pos + neg + rest)[:cfg.seed_size]


def ceiling_for(ds):
    """Score of the learner trained on the FULL labelled pool.

    The target accuracy is derived from this rather than picked as a round
    number: 'reach 95% of what you would get by labelling everything' is a
    statement a practitioner can act on, and it adapts to each seed's dataset
    difficulty instead of being unreachable on hard draws.
    """
    # Cold fit with a long budget: this number defines the target, so it must
    # be converged, unlike the warm-started per-round fits.
    m = LogisticRegression(ds.dim).fit(ds.pool_x, ds.pool_y, epochs=CEILING_EPOCHS)
    return balanced_accuracy(m, ds.test_x, ds.test_y)


def run_single(strategy_name, seed, cfg, ds=None, target=None, ceiling=None):
    """One (strategy, seed) run. Deterministic given (strategy_name, seed, cfg)."""
    if ds is None:
        ds = make_dataset(seed, cfg.regime, n_pool=cfg.n_pool, n_test=cfg.n_test)
    if ceiling is None:
        ceiling = ceiling_for(ds)
    if target is None:
        target = cfg.target_frac * ceiling

    # Two independent streams: the initial labelled set must be IDENTICAL across
    # strategies for a given seed (that is what makes the comparison paired),
    # while each strategy's own tie-breaking must not consume from that stream.
    init_rng = random.Random(seed * 104729 + 7)
    strat_rng = random.Random(zlib.crc32(strategy_name.encode()) * 1000003 + seed)
    # NOTE: not hash(): str hashing is salted per process (PYTHONHASHSEED), which
    # made 'same seed, same result' false across two invocations.

    select = STRATEGIES[strategy_name]
    labeled = _initial_labels(ds, cfg, init_rng)
    unlabeled = [i for i in range(ds.n_pool) if i not in set(labeled)]

    model = LogisticRegression(ds.dim)
    curve = []
    hit = None

    def refit_and_score():
        model.fit([ds.pool_x[i] for i in labeled], [ds.pool_y[i] for i in labeled])
        return balanced_accuracy(model, ds.test_x, ds.test_y)

    score = refit_and_score()
    curve.append((len(labeled), score))
    if score >= target:
        hit = len(labeled)

    while len(labeled) < cfg.budget and unlabeled:
        k = min(cfg.batch, cfg.budget - len(labeled), len(unlabeled))
        picked = select(k, unlabeled, labeled, ds, model, strat_rng)
        pset = set(picked)
        assert len(pset) == k and not pset & set(labeled), "strategy returned bad batch"
        labeled.extend(picked)
        unlabeled = [i for i in unlabeled if i not in pset]
        score = refit_and_score()
        curve.append((len(labeled), score))
        if hit is None and score >= target:
            hit = len(labeled)

    censored = hit is None
    # Censored runs are recorded at budget + batch, i.e. "more than the budget".
    # This is a lower bound on the truth, so it BIASES THE COMPARISON TOWARDS
    # whichever strategy censors more; the censor count is reported alongside
    # every median so the reader can see when that bias is in play.
    ltt = hit if hit is not None else cfg.budget + cfg.batch
    return RunResult(
        strategy=strategy_name,
        seed=seed,
        curve=curve,
        labels_to_target=ltt,
        censored=censored,
        target=target,
        ceiling=ceiling,
        positives_labelled=sum(ds.pool_y[i] for i in labeled),
        label_counts=[n for n, _ in curve],
    )


def run_benchmark(strategies, seeds, cfg):
    """Run every strategy on every seed. Returns {strategy: [RunResult, ...]}.

    The random baseline is ALWAYS run, whether or not the caller asked for it -
    every number in the report is relative to it, and a benchmark that can be
    invoked without its baseline will eventually be invoked without its
    baseline.
    """
    names = list(strategies)
    if BASELINE not in names:
        names.insert(0, BASELINE)
    out = {n: [] for n in names}
    for seed in seeds:
        ds = make_dataset(seed, cfg.regime, n_pool=cfg.n_pool, n_test=cfg.n_test)
        ceiling = ceiling_for(ds)
        target = cfg.target_frac * ceiling
        for n in names:
            out[n].append(run_single(n, seed, cfg, ds=ds, target=target, ceiling=ceiling))
    return out


def paired_diffs(results, strategy, baseline=BASELINE):
    """labels_to_target(strategy) - labels_to_target(baseline), seed by seed.

    Negative means the strategy needed FEWER labels than random.
    """
    a = {r.seed: r.labels_to_target for r in results[strategy]}
    b = {r.seed: r.labels_to_target for r in results[baseline]}
    seeds = sorted(set(a) & set(b))
    return [a[s] - b[s] for s in seeds]


def cold_start_scores(results, strategy, rounds=2):
    """Score after the first `rounds` acquisition rounds, per seed.

    Round 0 is the shared random seed set, so it is identical across strategies
    by construction and carries no information about the acquisition function.
    """
    return [r.curve[rounds][1] for r in results[strategy]]
