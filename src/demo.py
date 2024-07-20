"""60-second artefact: one seed says one thing, thirty seeds say another.

Run:  python -m src.demo
ASCII-only output (Windows cp1252 consoles).
"""

import time

from .runner import RunConfig, run_benchmark
from .report import (compare, single_seed_claim, budget_recommendation, cost,
                     summary_rows, ascii_bands)
from .stats import seeds_needed, detectable_delta
from .runner import paired_diffs

import statistics

N_SEEDS = 30
# margin is omitted from the timed run: for binary problems it selects exactly
# the same points as uncertainty (asserted in tests), so it would be a
# duplicate row. density is omitted to keep the demo under 60s.
NAMES = ["random", "uncertainty", "coreset"]


def main():
    t0 = time.time()
    cfg = RunConfig()
    res = run_benchmark(NAMES, range(N_SEEDS), cfg)
    elapsed = time.time() - t0

    print("labelbudget - labels-to-target across %d seeds (synthetic planted data)" % N_SEEDS)
    print("pool %d, seed set %d, batch %d, budget %d, target = %.0f%% of full-pool score"
          % (cfg.n_pool, cfg.seed_size, cfg.batch, cfg.budget, cfg.target_frac * 100))
    ceil = statistics.fmean(r.ceiling for r in res["random"])
    print("mean full-pool balanced accuracy %.3f (planted Bayes ceiling 0.841)" % ceil)
    print()

    # 1. The single-seed story.
    print("[1] What ONE seed would have told you")
    shown = 0
    for seed in range(N_SEEDS):
        win, la, lb = single_seed_claim(res, "uncertainty", "random", seed)
        if win is not None and max(la, lb) >= 2 * min(la, lb):
            print("  seed %2d: uncertainty %3d labels vs random %3d labels -> '%s is %.1fx better'"
                  % (seed, la, lb, win, max(la, lb) / min(la, lb)))
            shown += 1
            if shown == 3:
                break
    wins = sum(1 for s in range(N_SEEDS) if single_seed_claim(res, "uncertainty", "random", s)[0] == "uncertainty")
    loss = sum(1 for s in range(N_SEEDS) if single_seed_claim(res, "uncertainty", "random", s)[0] == "random")
    print("  over %d seeds: uncertainty 'wins' %d, random 'wins' %d, ties %d"
          % (N_SEEDS, wins, loss, N_SEEDS - wins - loss))
    print()

    # 2. Distribution per strategy.
    print("[2] Labels to target per strategy (censored = never reached target in budget)")
    print("  %-12s %7s %7s %7s %6s %9s" % ("strategy", "median", "Q1", "Q3", "sd", "censored"))
    for name, d, cens in summary_rows(res):
        print("  %-12s %7.1f %7.1f %7.1f %6.1f %6d/%d" % (name, d["median"], d["q1"], d["q3"], d["sd"], cens, d["n"]))
    print()

    # 3. The guard.
    print("[3] Comparative claims (paired bootstrap CI + exact sign test, alpha 0.05)")
    for name in NAMES[1:]:
        print("  " + compare(res, name, "random").sentence())
        print("  " + compare({k: v[:5] for k, v in res.items()}, name, "random").sentence())
    print()

    # 4. How many seeds would it take?
    print("[4] Power: seeds needed to resolve a difference")
    for name in NAMES[1:]:
        diffs = paired_diffs(res, name, "random")
        sd = statistics.stdev(diffs)
        print("  %s - random: sd of paired diff %.1f labels; %d seeds detect >= %.1f labels; "
              "a 10-label effect needs %d seeds"
              % (name, sd, N_SEEDS, detectable_delta(sd, N_SEEDS), seeds_needed(sd, 10)))
    print()

    # 5. Bands.
    print("[5] Balanced accuracy vs labels: median and IQR band across seeds")
    for line in ascii_bands(res, ["random", "uncertainty"]):
        print(line)

    # 6. Budget.
    print("[6] Label budget that reached target on 90% of seeds, at $90/hour, 6 min/label")
    for name in NAMES:
        labels, cens = budget_recommendation(res[name])
        if labels is None:
            print("  %-12s target missed within budget on %d/%d seeds -> buy more than %d labels"
                  % (name, cens, N_SEEDS, cfg.budget))
        else:
            print("  %-12s %4d labels = $%7.0f  (censored %d)" % (name, labels, cost(labels), cens))
    print()
    print("runtime %.1fs" % elapsed)


if __name__ == "__main__":
    main()
