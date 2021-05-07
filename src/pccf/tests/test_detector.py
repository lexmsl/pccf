# Test that detection delay is increasing when threshold is increased.
#
from pccf.detector import cusum, cusum_single, cusum_pccf, make_cusum_pccf, \
    make_cusum_single_ref, make_cusum_single_pccf_ref, Detector
from pccf.detector_performance import *
from pccf.roi_obj import Roi
from pccf.utils_perf import *
from numpy.random import randn


def gen_sig_two_changes():
    n = 500
    sigma = 1.1
    return np.concatenate(
        (
            np.random.rand(n) * sigma,
            np.random.rand(n) * sigma + 2.0,
            np.random.rand(n) * sigma,
        ),
        axis=0,
    )


def test_tps_num_behaviour_cusum_single():
    n_points = 100
    sigma = 1.0
    mu0 = 0.0
    mu1 = 1.1
    sig = np.concatenate((randn(n_points) * sigma + mu0,
                          randn(n_points) * sigma + mu1,
                          ), axis=0)
    changes = [100]
    max_theta = 20
    theta_vec = np.linspace(0.001, max_theta, 100)  # gen_theta_vec(0.001, 0.1, max_theta)
    cusum_loc = make_cusum_single_ref(mu0)
    delays_stat, fps_stat, fns_stat, tps_stat, f1_stat = \
        collect_performance(cusum_loc, sig, changes, theta_vec)

    for i, d in enumerate(tps_stat):
        if i > 0:
            assert d >= tps_stat[i-1]


def test_detection_delay_behaviour_cusum_pccf():
    """
    Test that when we increase threshold (sensitivity)
    then detection delay for Pccf Cusum decreases and becomes nan.

    Also test that if delay == nan then tps_num == 0
    """
    n_points = 100
    sigma = 1.0
    mu0 = 0.0
    mu1 = 1.1
    sig = np.concatenate((randn(n_points) * sigma + mu0,
                          randn(n_points) * sigma + mu1,
                          ), axis=0)
    changes = [100]
    max_theta = 40
    theta_vec = np.linspace(1, max_theta, 100)  # gen_theta_vec(1, 1, max_theta)
    rois = [Roi(100, radius=25)]
    cusum_pccf_ref = make_cusum_pccf(rois)

    delays_dyn, fps_dyn, fns_dyn, tps_dyn, f1_dyn = \
        collect_performance(cusum_pccf_ref, sig, changes, theta_vec, rois=rois)

    for i, d in enumerate(delays_dyn):
        if i == 0:
            continue
        else:
            assert d >= delays_dyn[i] or math.isnan(d)
            if math.isnan(d):
                assert tps_dyn[i] == 0


def test_cusum_single_nan_outputs():
    mu0 = 0.0
    n_points = 150
    sigma = 1.1
    mu1 = 2.1
    cusum_single_ref = make_cusum_single_ref(mu0)
    cusum_single_pccf_ref = make_cusum_single_pccf_ref(mu0, Roi(n_points+1, 15))
    sig = np.concatenate((randn(n_points) * sigma + mu0,
                          randn(n_points) * sigma + mu1), axis=0)
    r_stat = cusum_single_ref(sig, 1)
    r_dyn = cusum_single_pccf_ref(sig, 1)
    perf_stat = tp_fp_fn_delays([n_points], r_stat.detections)
    perf_dyn = tp_fp_fn_delays([n_points], r_dyn.detections)
    assert np.isnan(perf_stat.delays[0])
    assert np.isnan(perf_dyn.delays[0])
    r_dyn2 = cusum_single_pccf_ref(sig, 2000)
    perf_dyn2 = tp_fp_fn_delays([n_points], r_dyn.detections)
    assert np.isnan(perf_dyn2.delays[0])


def test_cusum_single():
    """
    Test behaviour of delay for one change case.
    Delay at the beginning is -1.0, then it must increase, and -1 if threshold is very large.
    """
    n = 30
    chp = 500
    sigma = 1.1
    avg_delays = []
    avg_fps = []
    for threshold in [30.0, 260.0, 280.0, 300.0, 320.0, 330.0, 10000]:
        delays = []
        fps = []
        for _ in range(n):
            sig = np.concatenate(
                (np.random.rand(chp)*sigma,
                 np.random.rand(chp)*sigma + 2.0), axis=0
            )
            r = cusum_single(sig, 0.0, threshold)
            delays_new = fn_delays([chp], r.detections)
            fps_new = fn_fps([chp], r.detections)
            delays.extend(delays_new)
            fps.append(len(fps_new))
        if len(delays) > 0:
            avg_delays.append(np.mean(delays))
        avg_fps.append(np.mean(fps))
    for i, e in enumerate(avg_delays):
        if i == 0:
            continue
        if not np.isnan(e) and not np.isnan(avg_delays[i - 1]):
            assert e >= avg_delays[i - 1]


def test_cusum_single_ref():
    """
    Test for detection location
    """
    mu0 = 0.0
    cusum_single_ref = make_cusum_single_ref(mu0)
    n_points = 150
    sigma = 1.1
    mu1 = 2.1
    detections = []
    for _ in range(30):
        sig = np.concatenate((randn(n_points) * sigma + mu0,
                              randn(n_points) * sigma + mu1), axis=0)
        r = cusum_single_ref(sig, 20)
        detections.append(r.detections[0])
    cde_average = np.mean(detections)
    assert 140 <= cde_average <= 170


def test_cusum_single_pccf_ref():
    """
    Test for detection location
    """
    mu0 = 0.0
    n_points = 150
    ROI_RADIUS = 5
    cusum_single_pccf_ref = make_cusum_single_pccf_ref(mu0,
                                                       Roi(n_points+1,
                                                           ROI_RADIUS))
    sigma = 1.1
    mu1 = 2.1
    for _ in range(30):
        sig = np.concatenate((randn(n_points) * sigma + mu0,
                              randn(n_points) * sigma + mu1), axis=0)
        r = cusum_single_pccf_ref(sig, 20)
        if len(r.detections) > 0 and not np.isnan(r.detections[0]):
            assert n_points - ROI_RADIUS <= r.detections[0] <= n_points + \
                   ROI_RADIUS + 1


def test_cusum_multi():
    """
    Test delay behaviour in case of multi-changes and when using cusum with
    re-setting / update.  In this case if threshold is low, then delay is also
    small - UNLIKE one change cusum, which detects first detection before
    actual change and stops and it is FA.
    """
    n = 30
    chp = 500
    sigma = 1.1
    avg_delays = []
    avg_fps = []
    thresholds = [1.0, 10.0, 30.0, 1e4]
    for threshold in thresholds:
        delays = []
        fps = []

        for _ in range(n):
            sig = np.concatenate(
                (
                    np.random.rand(chp) * sigma,
                    np.random.rand(chp) * sigma + 2.0,
                    np.random.rand(chp) * sigma + 0.0,
                ),
                axis=0,
            )
            r = cusum(sig, threshold)
            delays_new = fn_delays([chp, 2 * chp], r.detections)
            if threshold == thresholds[-1]:
                print(88888)
                print(delays_new)
            fps_new = fn_fps([chp, 2 * chp], r.detections)
            if len(delays_new) > 0:
                delays.extend(delays_new)
            fps.append(len(fps_new))
        if len(delays) > 0:
            avg_delays.append(np.mean(delays))
        else:
            print(delays)
        avg_fps.append(np.mean(fps))

    for i, e in enumerate(avg_delays):
        if i > 0:
            assert e >= avg_delays[i - 1]


def test_cusum_pccf_one_change():
    sig = np.concatenate((np.random.randn(100),
                          np.random.randn(100) + 2.1), axis=0)
    rois = [Roi(100, 3)]
    for theta in np.linspace(0.1, 10, 100):
        r = cusum_pccf(sig, theta, rois)
        assert all(rois[0].left <= e <= rois[0].right for e in r.detections)


def test_cusum_pccf_multi_changes():
    sig_earth_quakes = [13.0, 14.0, 8.0, 10.0, 16.0, 26.0, 32.0, 27.0, 18.0,
                        32.0, 36.0, 24.0, 22.0, 23.0, 22.0, 18.0, 25.0,
                        21.0, 21.0, 14.0, 8.0, 11.0, 14.0, 23.0, 18.0, 17.0,
                        19.0, 20.0, 22.0, 19.0, 13.0, 26.0, 13.0, 14.0,
                        22.0, 24.0, 21.0, 22.0, 26.0, 21.0, 23.0, 24.0,
                        27.0, 41.0, 31.0, 27.0, 35.0, 26.0, 28.0, 36.0,
                        39.0, 21.0, 17.0, 22.0, 17.0, 19.0, 15.0, 34.0,
                        10.0, 15.0, 22.0, 18.0, 15.0, 20.0, 15.0, 22.0,
                        19.0, 16.0, 30.0, 27.0, 29.0, 23.0, 20.0, 16.0,
                        21.0, 21.0, 25.0, 16.0, 18.0, 15.0, 18.0, 14.0,
                        10.0, 15.0, 8.0, 15.0, 6.0, 11.0, 8.0, 7.0, 13.0,
                        10.0, 23.0, 16.0, 15.0, 25.0, 22.0, 20.0, 16.0]
    sig_changes = [20.0, 40.0, 53.0, 69.0, 79.0, 95.0]
    # rois = list(pccf_rois_adhoc(sig_changes, 3))
    rois = [Roi(20, 3), Roi(36, 7), Roi(52, 2), Roi(68, 4), Roi(84, 8),
            Roi(100, 8)]
    changes_in_rois = []
    # Check if all changes are within ROIs
    for change in sig_changes:
        for roi in rois:
            if roi.left <= change <= roi.right:
                changes_in_rois.append(change)
    assert set(changes_in_rois) == set(sig_changes)
    cusum_pccf_ref = make_cusum_pccf(rois)
    all_detections = []
    for theta in np.linspace(0.01, 40, 100):
        detections = cusum_pccf_ref(sig_earth_quakes, theta).detections
        all_detections.extend(detections)
    all_detections = list(set(all_detections))
    print(all_detections)
    detections_with_rois = []
    # Check if all CDEs are within ROIs
    for cde in all_detections:
        for roi in rois:
            if roi.left <= cde <= roi.right:
                detections_with_rois.append(cde)
                print(f"left: {roi.left} < cde: {cde} > right: {roi.right} |"
                      f" center: {roi.center}")
    assert set(detections_with_rois) == set(all_detections)


def test_fns_behaviour():
    """
    Catches cases when detection is after ROI and no detections inside.
    And => FN.
    """
    n = 10
    half_length = 100
    change = half_length
    changes = [change]
    rois = [Roi(change, 5)]
    count_pccf_fns = 0
    count_cusum_fns = 0
    for _ in range(n):
        for theta in np.linspace(0.1, 10, 30):
            sig = np.concatenate((np.random.rand(half_length) * 1.1,
                                  np.random.rand(half_length) * 1.1 + 2.0),
                                 axis=0)
            r_cusum = cusum(sig, theta)
            r_pccf = cusum_pccf(sig, theta, rois)
            fns_cusum = fn_fns(changes, r_cusum.detections)
            fns_pccf = fn_fns(changes, r_pccf.detections)
            count_cusum_fns += len(fns_cusum)
            count_pccf_fns += len(fns_pccf)
            if len(fns_pccf) > 0:
                print(fns_cusum, fns_pccf, ' -- ', r_pccf.detections, r_cusum.detections)
    assert count_pccf_fns > count_cusum_fns


def test_two_detectors_outputs():
    change = 30
    threshold = 1.0
    sig = np.concatenate((np.random.rand(change) * 1.1,
                          np.random.rand(change) * 1.1 + 2.0),
                         axis=0)

    r_stat = cusum(sig, threshold)
    r_dyn = cusum_pccf(sig, threshold, [Roi(change, 3)])

    r_obj_stat = Detector(sig, threshold).run()
    r_obj_dyn = Detector(sig, threshold, [Roi(change, 3)]).run()

    assert r_stat.detections == r_obj_stat.detections
    assert r_dyn.detections == r_obj_dyn.detections
    np.testing.assert_almost_equal(r_stat.statistic, r_obj_stat.statistic)
    np.testing.assert_almost_equal(r_dyn.statistic, r_obj_dyn.statistic)
