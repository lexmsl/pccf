from pccf.pccf_performance import *
from pccf.pccf_obj import Pccf
from pccf.roi_obj import Roi
import random
import matplotlib.pyplot as plt
import numpy as np
from pccf.utils_pccf import min_max_mu


class TestPccfUtilityFunctions:
    def test_find_mu(self):
        changes = [2, 41, 45]
        min_mu, max_mu = min_max_mu(changes)
        assert min_mu == 2
        assert max_mu == 39


class TestPerfMetricsPccf:
    def test1(self):
        perf = PccfPerformanceMetrics(changes=[10, 11, 20, 30])
        assert perf.tps([Roi(10, 2), Roi(35, 1)]) == 1
        assert perf.fns([Roi(10, 2), Roi(35, 1)]) == 3
        assert perf.fps([Roi(10, 2), Roi(35, 1)]) == 1
        assert perf.f1_score([Roi(10, 2), Roi(35, 1)]) == 1.0/(1.0+0.5*(3.0+1.0))
        assert PccfPerformanceMetrics([10, 20]).fns([Roi(9, 5), Roi(17, 1)]) == 1

    def test2_pccf_predictions_and_performance(self):
        changes = [10, 20, 30, 40, 50]
        pccf_perf = PccfPerformanceMetrics(changes)
        pccf_predictions = Pccf(10, radius=1).roi_intervals(len(changes))
        assert pccf_perf.f1_score(pccf_predictions) == 1

    def test3_one_line_tests(self):
        assert PccfPerformanceMetrics([10]).f1_score([Roi(10, 1)]) == 1
        assert PccfPerformanceMetrics([10, 12]).f1_score([Roi(10, 5)]) == 1.0/1.5
        assert PccfPerformanceMetrics([10, 20]).f1_score([Roi(9, 5), Roi(17, 1)]) == 1.0/2


def test_collect_performance_pccf_function():
    mu_true = 10
    changes = [mu_true * i + random.randint(-2, 2) for i in range(1, 100)]
    mu_vec, f1_vec = collect_pccf_performance(changes, mu_range=(1, 20), radius_range=(3, 7))
    should_be_best = list(filter(lambda x: x[0] == mu_true, zip(mu_vec, f1_vec)))[0][1]
    for mu, f1 in zip(mu_vec, f1_vec):
        assert f1 <= should_be_best
    assert f1_vec[5] <= should_be_best
    assert f1_vec[15] <= should_be_best


def test_pccf_performance_curve_convexity():
    """
    Test that max corresponds to mu and test convex shape
    """
    mu_true = 10
    changes = [mu_true * i + random.randint(-2, 2) for i in range(1, 100)]
    pccf_perf = PccfPerformanceMetrics(changes)
    performance = {}
    for mu in range(1, 30):
        f1_per_mu = []
        for r in range(3, 7):
            pccf_predictions = Pccf(mu, radius=r).roi_intervals(len(changes))
            f1 = pccf_perf.f1_score(pccf_predictions)
            f1_per_mu.append(f1)
        performance[mu] = np.mean(f1_per_mu)
    for k, v in performance.items():
        assert performance[mu_true] >= v
    assert performance[5] <= performance[mu_true]
    assert performance[15] <= performance[mu_true]
    make_plot = False
    if make_plot:
        x, y = zip(* sorted(performance.items()))
        plt.plot(x, y)
        plt.show()
