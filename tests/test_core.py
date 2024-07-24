import math
import os
import random
import subprocess
import sys
import unittest

from src.dataset import make_dataset, bayes_balanced_accuracy
from src.model import LogisticRegression, balanced_accuracy, CEILING_EPOCHS
from src.strategies import (STRATEGIES, select_uncertainty, select_margin,
                            select_coreset, least_confidence_score, margin_score)
from src.runner import RunConfig, run_single, run_benchmark, paired_diffs
from src.stats import sign_test_p, seeds_needed, quantile, paired_bootstrap_ci
from src import report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHI1 = 0.5 * (1 + math.erf(1 / math.sqrt(2)))  # Bayes acc for sep 2.0: Phi(1)


class TestDataset(unittest.TestCase):
    def test_same_seed_same_data_different_seed_different(self):
        a, b, c = make_dataset(3), make_dataset(3), make_dataset(4)
        self.assertEqual(a.pool_x, b.pool_x)
        self.assertEqual(a.pool_y, b.pool_y)
        self.assertNotEqual(a.pool_x, c.pool_x)

    def test_planted_boundary_scores_the_analytic_bayes_accuracy(self):
        ds = make_dataset(1, n_pool=50, n_test=4000)
        se = math.sqrt(PHI1 * (1 - PHI1) / 4000)
        # 3 binomial standard errors around the closed-form Phi(sep/2).
        self.assertAlmostEqual(bayes_balanced_accuracy(ds), PHI1, delta=3 * se)

    def test_imbalanced_regime_positive_rate(self):
        ds = make_dataset(0, regime="imbalanced", n_pool=3000, n_test=10)
        # 0.04 +- 3 SE of a binomial proportion with n=3000.
        self.assertAlmostEqual(ds.positive_rate(), 0.04, delta=3 * math.sqrt(.04 * .96 / 3000))


class TestModel(unittest.TestCase):
    def test_full_data_fit_reaches_planted_ceiling(self):
        ds = make_dataset(2, n_pool=1000, n_test=4000)
        m = LogisticRegression(ds.dim).fit(ds.pool_x, ds.pool_y, epochs=CEILING_EPOCHS)
        acc = balanced_accuracy(m, ds.test_x, ds.test_y)
        se = math.sqrt(PHI1 * (1 - PHI1) / 4000)
        # Cannot beat Bayes beyond sampling noise; estimation loss from 13
        # parameters on 1000 rows is well under 2 points.
        self.assertLess(acc, PHI1 + 3 * se)
        self.assertGreater(acc, PHI1 - 0.02 - 3 * se)
        # Recovered direction aligns with the planted weight vector.
        nw = math.sqrt(sum(v * v for v in m.w))
        cos = sum(a * b for a, b in zip(m.w, ds.true_w)) / nw
        self.assertGreater(cos, 0.95)

    def test_class_weighting_prevents_majority_collapse(self):
        rng = random.Random(0)
        xs = [[rng.gauss(2 if i < 3 else -2, 1)] for i in range(60)]
        ys = [1 if i < 3 else 0 for i in range(60)]
        weighted = LogisticRegression(1).fit(xs, ys)
        self.assertEqual(weighted.predict([2.0]), 1)


class TestStrategies(unittest.TestCase):
    def setUp(self):
        self.ds = make_dataset(5, n_pool=120, n_test=10)
        self.model = LogisticRegression(self.ds.dim)
        self.model.w, self.model.b = list(self.ds.true_w), 0.0

    def test_uncertainty_picks_points_closest_to_planted_boundary(self):
        cand = list(range(self.ds.n_pool))
        picked = select_uncertainty(5, cand, [], self.ds, self.model, random.Random(0))
        dist = sorted(cand, key=lambda i: abs(self.model.decision(self.ds.pool_x[i])))
        self.assertEqual(set(picked), set(dist[:5]))

    def test_margin_equals_uncertainty_for_binary(self):
        for p in [0.01, 0.3, 0.5, 0.77, 0.99]:
            self.assertAlmostEqual(margin_score(p), 2 * least_confidence_score(p) - 1)
        cand = list(range(self.ds.n_pool))
        a = select_uncertainty(10, cand, [], self.ds, self.model, random.Random(1))
        b = select_margin(10, cand, [], self.ds, self.model, random.Random(1))
        self.assertEqual(set(a), set(b))

    def test_coreset_picks_planted_outlier_first(self):
        self.ds.pool_x[17] = [50.0] * self.ds.dim
        self.ds._d2 = None
        cand = [i for i in range(self.ds.n_pool) if i != 0]
        picked = select_coreset(1, cand, [0], self.ds, self.model, random.Random(0))
        self.assertEqual(picked, [17])

    def test_every_strategy_returns_k_fresh_indices(self):
        labeled = list(range(10))
        cand = list(range(10, self.ds.n_pool))
        for name, fn in STRATEGIES.items():
            got = fn(7, cand, labeled, self.ds, self.model, random.Random(3))
            self.assertEqual(len(set(got)), 7, name)
            self.assertFalse(set(got) & set(labeled), name)


class TestRunner(unittest.TestCase):
    def test_reproducible_in_process(self):
        cfg = RunConfig(budget=40)
        a = run_single("uncertainty", 7, cfg)
        b = run_single("uncertainty", 7, cfg)
        self.assertEqual(a.curve, b.curve)

    def test_reproducible_across_processes_with_different_hash_seeds(self):
        # Caught a real bug: the per-strategy RNG was seeded with hash(str),
        # which Python salts per process, so 'same seed' gave different runs.
        code = ("from src.runner import *; r=run_single('uncertainty', 3, RunConfig(budget=36));"
                "print(r.curve)")
        outs = []
        for hs in ("1", "2"):
            env = dict(os.environ, PYTHONHASHSEED=hs)
            outs.append(subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                                       capture_output=True, text=True, check=True).stdout)
        self.assertEqual(outs[0], outs[1])

    def test_random_baseline_always_run_and_seed_sets_are_paired(self):
        res = run_benchmark(["uncertainty"], [0, 1], RunConfig(budget=20))
        self.assertIn("random", res)
        self.assertEqual(len(res["random"]), 2)
        for a, b in zip(res["random"], res["uncertainty"]):
            self.assertEqual(a.curve[0], b.curve[0])  # identical initial labelled set

    def test_unreachable_target_is_censored_not_dropped(self):
        cfg = RunConfig(budget=20, target_frac=1.5)
        r = run_single("random", 0, cfg)
        self.assertTrue(r.censored)
        self.assertEqual(r.labels_to_target, cfg.budget + cfg.batch)


class TestStats(unittest.TestCase):
    def test_sign_test_exact(self):
        self.assertAlmostEqual(sign_test_p([-1] * 10), 2 / 1024)
        self.assertEqual(sign_test_p([0, 0, 0]), 1.0)
        self.assertAlmostEqual(sign_test_p([1, -1, 1, -1]), 1.0)

    def test_seeds_needed_known_value(self):
        # (1.95996 + 0.84162)^2 = 7.849 -> 8 seeds for delta = sd.
        self.assertEqual(seeds_needed(1.0, 1.0), 8)
        self.assertEqual(seeds_needed(50.0, 10.0), 197)

    def test_quantile_type7(self):
        self.assertEqual(quantile([1, 2, 3, 4], 0.5), 2.5)
        self.assertAlmostEqual(quantile([10, 20, 30, 40, 50], 0.9), 46.0)

    def test_bootstrap_ci_covers_planted_mean(self):
        import statistics
        covered = 0
        trials = 60
        for t in range(trials):
            rng = random.Random(t)
            d = [rng.gauss(-20, 40) for _ in range(30)]
            lo, hi = paired_bootstrap_ci(d, iters=400, seed=t, stat=statistics.fmean)
            covered += lo <= -20 <= hi
        # Nominal 95%; percentile bootstrap at n=30 undercovers slightly.
        # 60 trials: 3 SE of a proportion at .95 is ~0.085.
        self.assertGreater(covered / trials, 0.95 - 0.085)


class TestGuard(unittest.TestCase):
    """The guard must not declare a winner from seed noise."""

    NOISE_SD = 50.0  # close to the measured sd of paired diffs (42-53 labels)

    def _diffs(self, rng, mean, n):
        return [8 * round(rng.gauss(mean, self.NOISE_SD) / 8) for _ in range(n)]

    def test_null_false_positive_rate_vs_naive_single_seed(self):
        trials, guard_fp, naive_fp = 200, 0, 0
        for t in range(trials):
            rng = random.Random(1000 + t)
            d = self._diffs(rng, 0.0, 30)
            v = report.compare_diffs(d, boot_seed=t)
            guard_fp += v.verdict in ("a_better", "b_better")
            naive_fp += d[0] != 0  # a one-seed benchmark names a winner
        se = math.sqrt(0.05 * 0.95 / trials)
        self.assertLessEqual(guard_fp / trials, 0.05 + 3 * se)
        self.assertGreater(naive_fp / trials, 0.8)

    def test_real_effect_is_detected(self):
        # 40 labels at sd 50 and n=30: standardised effect 0.8, well above the
        # 80%-power detectable 0.51 sd; the sign test costs some power.
        hits = sum(report.compare_diffs(self._diffs(random.Random(t), -40.0, 30), boot_seed=t).verdict
                   == "a_better" for t in range(100))
        self.assertGreater(hits, 80)

    def test_refuses_claim_below_min_seeds(self):
        v = report.compare_diffs([-100, -120, -90], "x", "random")
        self.assertEqual(v.verdict, "insufficient_seeds")
        self.assertIn("no comparative claim", v.sentence())

    def test_overlapping_interval_reports_indistinguishable(self):
        v = report.compare_diffs([-8, 8] * 10, "x", "random")
        self.assertEqual(v.verdict, "indistinguishable")
        self.assertIn("indistinguishable at 20 seeds", v.sentence())


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = run_benchmark(["uncertainty", "margin", "most_confident"], range(10), RunConfig())

    def test_planted_bad_strategy_detected_as_worse_than_random(self):
        v = report.compare(self.res, "most_confident", "random")
        self.assertEqual(v.verdict, "b_better", v.sentence())

    def test_identical_strategies_are_never_declared_different(self):
        self.assertEqual(paired_diffs(self.res, "uncertainty", "margin"), [0] * 10)
        v = report.compare(self.res, "uncertainty", "margin")
        self.assertEqual(v.verdict, "indistinguishable")

    def test_budget_recommendation_and_cost(self):
        labels, cens = report.budget_recommendation(self.res["most_confident"])
        self.assertIsNone(labels)  # censored everywhere -> no fake number
        self.assertEqual(cens, 10)
        labels, _ = report.budget_recommendation(self.res["random"])
        ltt = sorted(r.labels_to_target for r in self.res["random"])
        self.assertGreaterEqual(labels, ltt[8])
        self.assertEqual(report.cost(100), 900.0)

    def test_report_text_is_ascii(self):
        text = "\n".join(report.ascii_bands(self.res, ["random", "uncertainty"]))
        text += report.compare(self.res, "uncertainty").sentence()
        text.encode("ascii")


if __name__ == "__main__":
    unittest.main()
