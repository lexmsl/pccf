from pccf.data import *
import json
from pccf.detector import make_cusum_pccf, cusum
from pccf.settings import *
from pccf.utils_general import path_ext_low_with_dot
from pccf.detector_performance import *
from pccf.pccf_performance import *
from pccf.examples.common import save_json_to_temp_file

MAX_THETA = 500


def performance_signal(sig,
                       sig_changes,
                       rois,
                       theta_vec=np.linspace(0.1, 1.0, 20),
                       ):
    """ Prediction and detection steps """

    delays_stat, fps_stat, fns_stat, tps_stat, f1_stat = collect_performance(cusum, sig, sig_changes, theta_vec)

    cusum_pccf_ref = make_cusum_pccf(rois)

    delays_dyn, fps_dyn, fns_dyn, tps_dyn, f1_dyn = collect_performance(cusum_pccf_ref, sig, sig_changes, theta_vec, rois=rois)

    results = {
        'theta_vec': list(theta_vec),
        'delays_stat': delays_stat,
        'fps_stat': fps_stat,
        'fns_stat': fns_stat,
        'tps_stat': tps_stat,
        "f1_stat": f1_stat,
        "delays_dyn": delays_dyn,
        "fps_dyn": fps_dyn,
        "fns_dyn": fns_dyn,
        "tps_dyn": tps_dyn,
        "f1_dyn": f1_dyn
    }
    save_path = save_json_to_temp_file(results, 'performance_detection_signals_')
    return save_path


def plot_results_from_file(path_json_results, save_fig=None, plot_all_metrics=False):
    logger.info(f"Plot results from file {path_json_results}")
    with open(path_json_results, 'r') as f:
        r = json.load(f)
    theta_vec = r['theta_vec']
    delays_stat = r["delays_stat"]
    delays_dyn = r['delays_dyn']
    tps_stat = r['tps_stat']
    tps_dyn = r["tps_dyn"]
    fps_stat = r["fps_stat"]
    fps_dyn = r["fps_dyn"]
    fns_stat = r["fns_stat"]
    fns_dyn = r["fns_dyn"]
    f1_stat = r["f1_stat"]
    f1_dyn = r["f1_dyn"]

    common_font_size = 30
    common_line_width = 4
    common_tick_size = 30

    logger.info(f"Plot all metrics = {plot_all_metrics}")

    if plot_all_metrics:
        fig = plt.figure(figsize=(30, 8))
        ax1 = fig.add_subplot(1, 5, 1)
        ax2 = fig.add_subplot(1, 5, 2)
        ax3 = fig.add_subplot(1, 5, 3)
        ax4 = fig.add_subplot(1, 5, 4)
        ax5 = fig.add_subplot(1, 5, 5)
    else:
        fig = plt.figure(figsize=(18, 8))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = None
        ax3 = None
        ax4 = None
        ax5 = fig.add_subplot(1, 2, 2)

    ax1.plot(theta_vec, delays_stat, ls="-", color="black", linewidth=common_line_width, label='Fixed settings')
    ax1.plot(theta_vec, delays_dyn,  ls="--", color="black", linewidth=common_line_width, label='Pccf')
    ax1.set_xlabel('Threshold', fontsize=common_font_size)
    ax1.set_ylabel('Detection delay', fontsize=common_font_size)
    ax1.xaxis.set_tick_params(labelsize=common_tick_size)
    ax1.yaxis.set_tick_params(labelsize=common_tick_size)
    ax1.legend(loc='upper center', shadow=False, fontsize=30,  bbox_to_anchor=(0.5, 1.2), prop={'size': 30}, ncol=2)

    if plot_all_metrics:
        ax2.plot(theta_vec, tps_stat, ls="-", color="black", linewidth=common_line_width)
        ax2.plot(theta_vec, tps_dyn, ls="--", color="black", linewidth=common_line_width)
        ax2.set_xlabel('Threshold', fontsize=common_font_size)
        ax2.set_ylabel('TP', fontsize=common_font_size)
        ax2.xaxis.set_tick_params(labelsize=common_tick_size)
        ax2.yaxis.set_tick_params(labelsize=common_tick_size)

        ax3.plot(theta_vec, fps_stat, ls="-", color="black", linewidth=common_line_width)
        ax3.plot(theta_vec, fps_dyn, ls="--", color="black", linewidth=common_line_width)
        ax3.set_xlabel('Threshold', fontsize=common_font_size)
        ax3.set_ylabel('FP', fontsize=common_font_size)
        ax3.xaxis.set_tick_params(labelsize=common_tick_size)
        ax3.yaxis.set_tick_params(labelsize=common_tick_size)

        ax4.plot(theta_vec, fns_stat, ls="-", color="black", linewidth=common_line_width)
        ax4.plot(theta_vec, fns_dyn, ls="--", color="black", linewidth=common_line_width)
        ax4.set_xlabel('Threshold', fontsize=common_font_size)
        ax4.set_ylabel('FN', fontsize=common_font_size)
        ax4.xaxis.set_tick_params(labelsize=common_tick_size)
        ax4.yaxis.set_tick_params(labelsize=common_tick_size)

    ax5.plot(theta_vec, f1_stat, ls="-", color="black", linewidth=common_line_width)
    ax5.plot(theta_vec, f1_dyn, ls="--", color="black", linewidth=common_line_width)
    ax5.set_xlabel('Threshold', fontsize=common_font_size)
    ax5.set_ylabel('F1', fontsize=common_font_size)
    ax5.xaxis.set_tick_params(labelsize=common_tick_size)
    ax5.yaxis.set_tick_params(labelsize=common_tick_size)

    plt.legend(frameon=False)
    plt.tight_layout()

    if save_fig:
        if path_ext_low_with_dot(save_fig) == '.eps':
            plt.savefig(save_fig, format='eps')
        elif path_ext_low_with_dot(save_fig) == '.png':
            plt.savefig(save_fig, format='png', dpi=300)
        else:
            raise NotImplementedError
        logger.info(f"Save Figure into {save_fig}")
        plt.close()
    else:
        plt.show()


def get_perfect_rois(changes, radius, left_margin=None, right_margin=None):
    """
    Use only for artificial signal.
    Or for benchmarking
    """
    rois_perfect = []
    for change in changes:
        new_roi = Roi(change, radius)
        if left_margin and right_margin:
            new_roi.set_left_right(change - left_margin, change + right_margin)
        rois_perfect.append(new_roi)
    return rois_perfect


# def run_earth_quakes():
#
#     predicted_rois = pccf_rois_adhoc(changes_earth_quakes,
#                                      ROI_RADIUS_EARTH_QUAKES)
#     performance_signal(sig_earth_quakes,
#                        changes_earth_quakes,
#                        predicted_rois,
#                        save_fig=p('performance_earth_quakes.PNG'))  # '../../tex/img/performance_earth_quakes.eps'
#
#     perfect_rois = get_perfect_rois(changes_earth_quakes,
#                                     ROI_RADIUS_EARTH_QUAKES)
#     performance_signal(sig_earth_quakes,
#                        changes_earth_quakes,
#                        perfect_rois,
#                        save_fig=p('performance_earth_quakes_perfect.PNG'))
#

# def run_twi():
#     logger.info("Twi changes:")
#     logger.info(CHANGES_TWI)
#     predicted_rois = pccf_rois_adhoc(CHANGES_TWI, ROI_RADIUS_TWI)
#     logger.info("Twi predicted ROIs:")
#     for roi in predicted_rois:
#         logger.info(roi)
#     performance_signal(sig_twi,
#                        CHANGES_TWI,
#                        predicted_rois,
#                        save_fig=p('performance_twi.PNG'))
#
#     perfect_rois = get_perfect_rois(CHANGES_TWI, ROI_RADIUS_TWI)
#     logger.info("Twi perfect ROIs:")
#     for roi in perfect_rois:
#         logger.info(roi)
#     performance_signal(sig_twi,
#                        CHANGES_TWI,
#                        perfect_rois,
#                        save_fig=p('performance_twi_perfect.PNG'))


# def run_lake_level():
#     predicted_rois = pccf_rois_adhoc(changes_lake_level, ROI_RADIUS_LAKE)
#     performance_signal(sig_lake_level,
#                        changes_lake_level,
#                        predicted_rois,
#                        save_fig=p('performance_lake.PNG'))
#     perfect_rois = get_perfect_rois(changes_lake_level, ROI_RADIUS_LAKE)
#     performance_signal(sig_lake_level,
#                        changes_lake_level,
#                        perfect_rois,
#                        save_fig=p('performance_lake_perfect.PNG'))


# def just_test():
#     """
#     The same as run_artificial_signal_avg() but single run and 2 changes.
#     """
#     n_points = 350
#     sigma = 1.1
#     mu0 = 0.0
#     mu1 = 1.1
#     sig = np.concatenate((randn(n_points) * sigma + mu0,
#                           randn(n_points) * sigma + mu1,
#                           randn(n_points) * sigma
#                           ), axis=0)
#     changes_art_sig = [n_points+1, 2*n_points+1]
#     perfect_rois = get_perfect_rois(changes_art_sig,
#                                     ROI_RADIUS_ARTIFICIAL_SIGNAL)
#     performance_signal(sig,
#                        changes_art_sig,
#                        perfect_rois,
#                        save_fig=p('performance_art_sig.PNG'))


def run_temperature_signal():
    perf = PccfPerformanceMetrics(changes_temperature_indices)
    mu_vec, f1_vec = collect_pccf_performance(changes_temperature_indices,
                                              mu_range=(950, 1350),
                                              radius_range=(300, 600))
    mu_opt = mu_vec[np.argmax(f1_vec)]
    pccf = Pccf(mu_opt, radius=500)
    # rois = pccf.roi_intervals(len(changes_temperature))
    rois = rois_temperature_signal()
    logger.info(f"Pccf F1 score for temperature signal ={perf.f1_score(rois)}")

    res_path_all = performance_signal(sig_temperature,
                                      changes_temperature_indices,
                                      rois,
                                      theta_vec=np.linspace(0.1, MAX_THETA, 100),
                                      )

    res_path_seven = performance_signal(sig_temperature,
                                        changes_temperature_indices[:7],
                                        rois,
                                        theta_vec=np.linspace(0.1, MAX_THETA, 100),
                                        )
    return res_path_all, res_path_seven


if __name__ == "__main__":
    recalc_results = False
    if recalc_results:
        res_file_all, res_file_seven = run_temperature_signal()
        print(f"File all: {res_file_all}")
        print(f"File 7: {res_file_seven}")
    else:
        res_file_all = os.path.join(SRC_TEMP_DIR, "performance_detection_signals_yxrulbfv")
        res_file_seven = os.path.join(SRC_TEMP_DIR, "performance_detection_signals_s8d9u3g6")
    plot_results_from_file(res_file_all,
                           save_fig=os.path.join(TEX_FOLDER_IMG, 'performance_temperature.eps'),
                           plot_all_metrics=False)
    plot_results_from_file(res_file_seven,
                           save_fig=os.path.join(TEX_FOLDER_IMG, 'performance_temperature_seven.eps'),
                           plot_all_metrics=False)
