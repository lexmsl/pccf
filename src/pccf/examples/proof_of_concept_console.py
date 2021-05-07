from pccf.detector_performance import *
from pccf.roi_obj import Roi
from pccf.settings import *
from pccf.examples.common import SignalSettings
from pccf.detector import cusum, cusum_pccf
from numpy.random import randn


def results_example(sig_params=SignalSettings(0.0, 1.1, 1.0, 100),
                    theta=0.5,
                    output_fig=None):
    """
    Maybe the most important code / simulation.
    See sample of signal and detections to check what is being measured in
    simulation.
    """
    mu0 = sig_params.mu0
    mu1 = sig_params.mu1
    sigma = sig_params.sigma
    half_len = sig_params.half_len
    changes = [half_len]
    rois = [Roi(half_len, ROI_RADIUS_ARTIFICIAL_SIGNAL)]
    sig = np.concatenate((randn(half_len)*sigma+mu0,
                          randn(half_len)*sigma+mu1), axis=0)
    r_stat = cusum(sig, theta)
    r_dyn = cusum_pccf(sig, theta, rois)

    stat_tps = fn_tps(changes, r_stat.detections)
    stat_fps = fn_fps(changes, r_stat.detections)
    stat_fns = fn_fns(changes, r_stat.detections)

    pccf_tps = fn_tps(changes, r_dyn.detections, rois)
    pccf_fps = fn_fps(changes, r_dyn.detections, rois)
    pccf_fns = fn_fns(changes, r_dyn.detections, rois)

    logger.info(f"""
    \nStart simulation example. Repeat several times to see difference in performance.
    
    Input signal settings: mu0 = {sig_params.mu0}, mu1 = {sig_params.mu1}, sigma = {sig_params.sigma}, change = {half_len}, length = {len(sig)}
    Prediction interval  : roi = {rois[0]}
    Detector's threshold : theta = {theta}
    """)
    if output_fig:
        fig = plt.figure(111, figsize=(6, 4))
        ax1 = fig.add_subplot(111)
        ax1.plot(sig, color='black')
        plt.savefig(output_fig, format='eps')
        logger.info(f"Save signal sample into {output_fig}")
    else:
        print("CUSUM")
        print(" {0:11s} {1:}".format("Detections", r_stat.detections))
        print(" {0:11s} {1:}".format('Delays', fn_delays(changes, r_stat.detections)))
        print(" {0:11s} {1:}".format('FPs', stat_fps))
        print(" {0:11s} {1:}".format('FNs', stat_fns))
        print(" {0:11s} {1:}".format('TPs', stat_tps))
        print(" {0:11s} {1:.2f}".format('F1', f1_score(tps_count=len(stat_tps),
                                                       fps_count=len(stat_fps),
                                                       fns_count=len(stat_fns))))

        print('\nPCCF')
        print(" {0:11s} {1:}".format("Detections", r_dyn.detections))
        print(" {0:11s} {1:}".format('Delays', fn_delays(changes, r_dyn.detections, rois)))
        print(" {0:11s} {1:}".format('FPs', pccf_fps))
        print(" {0:11s} {1:}".format('FNs', pccf_fns))
        print(" {0:11s} {1:}".format('TPs', pccf_tps))
        print(" {0:11s} {1:.2f}".format('F1', f1_score(tps_count=len(pccf_tps),
                                                       fps_count=len(pccf_fps),
                                                       fns_count=len(pccf_fns))))
        print('\n')


if __name__ == "__main__":
    sig_settings = SignalSettings(0.0, 1.1, 1.0, 100)
    results_example(sig_settings, 5.0)
