from pccf.detector_performance import collect_performance, collect_arl
import numpy as np
from numpy.random import randn
from pccf.detector import make_cusum_pccf, cusum, make_cusum_single_ref, \
    make_cusum_single_pccf_ref
from pccf.utils_perf import *
from pccf.roi_obj import Roi


def test_collect_arl():
    mu0 = 0.0
    cusum_single_ref = make_cusum_single_ref(mu0)
    cusum_single_pccf_ref = make_cusum_single_pccf_ref(mu0, Roi(100, radius=25))
    n_points = 100
    sigma = 1.1
    mu1 = 1.1
    sig = np.concatenate((randn(n_points) * sigma + mu0,
                          randn(n_points) * sigma + mu1), axis=0)
    theta_vec = np.linspace(0.01, 150.0, 300)
    arls_stat = collect_arl(cusum_single_ref, sig, theta_vec)
    arls_dyn = collect_arl(cusum_single_pccf_ref, sig, theta_vec)
    assert len(arls_dyn) == len(arls_stat)


def test_collect_performance():
    n_points = 15
    ROI_RADIUS = 4
    MIN_THETA = 0.1
    MAX_THETA = 7.0
    THETA_STEP = 0.7
    changes = [n_points + 1]
    np.random.seed(11)
    mu0 = 0.0

    theta_vec = [0.1, 0.8, 1.5, 2.2, 2.9, 3.6, 4.3, 5.0, 5.7, 6.4, 7.5]

    sig = [1.92440022, -0.3146803,  -0.53302165, -2.91865042, -0.00911309,
           -0.3515945, -0.5902923, 0.34694294, 0.46315579, -1.17216328,
           -0.97486364, -0.52330684, 0.75865054, 0.61731139, -1.43610336,
           0.86857721,
           2.91052113,
           3.83209748,
           2.0658174,
           1.34820871,
           3.30519267,
           1.7594657,
           2.89832745,
           3.8039788,
           2.7930878,
           2.18084256,
           2.90549849,
           1.39316707,
           1.90409751,
           1.46864998]

    cusum_single_ref = make_cusum_single_ref(mu0)
    rois = [Roi(n_points, ROI_RADIUS)]

    cusum_single_pccf_ref = make_cusum_single_pccf_ref(mu0, rois[0])

    delays_stat, fps_stat, fns_stat, tps_stat, f1_stat = \
        collect_performance(cusum_single_ref, sig, changes, theta_vec)

    delays_dyn, fps_dyn, fns_dyn, tps_dyn, f1_dyn = \
        collect_performance(cusum_single_pccf_ref, sig, changes, theta_vec,
                            rois=rois)

    np.testing.assert_equal(delays_stat, [np.nan, np.nan, np.nan,
                                          1.0,
                                          2.0, 2.0, 2.0,
                                          3.0, 3.0,
                                          4.0, 4.0])

    assert tps_stat == [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    assert fps_stat == [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert fns_stat == [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert f1_stat == list([tp / (tp + 0.5*(fp+fn)) for (tp, fp, fn) in zip(
        tps_stat, fps_stat, fns_stat)])

    assert tps_dyn == [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    assert fns_dyn == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    # assert fps_dyn == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1] - old behaviour.
    # After the fix we do not detect outside ROI
    assert fps_dyn == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    np.testing.assert_equal(delays_dyn, [1.0, 1.0, 1.0, 1.0,
                                         2.0, 2.0, 2.0,
                                         3.0, 3.0,
                                         np.nan, np.nan])

test_collect_performance()