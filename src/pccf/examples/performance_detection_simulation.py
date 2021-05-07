#
# Obtain smooth performance curves using artificial signals.
#
import os
from pccf.detector import make_cusum_pccf, cusum, make_cusum_single_ref, \
    make_cusum_single_pccf_ref
from numpy.random import randn
from pccf.examples.proof_of_concept_console import results_example
from pccf.examples.common import save_json_to_temp_file
from tqdm import tqdm
from pccf.detector_performance import *
from pccf.settings import TEX_FOLDER_IMG
from loguru import logger
from pccf.examples.common import SignalSettings
from pccf.utils_general import path_ext_low_with_dot
import matplotlib.pyplot as plt
import json
from pccf.settings import *
# np.random.seed(1602)
from pccf.examples.performance_arls_simulation import FILE_RESULTS1, FILE_RESULTS2, FILE_RESULTS3


# moved to settings.py sSIG_HALF_LEN = 100
ROI_RADIUS_ARTIFICIAL_SIGNAL = 25
MAX_THETA = 100  # 25
N_SIM = 1000
WITH_RESETTING = False


@dataclass()
class SimulationResults:
    theta_vec: list
    averaged_delays_stat: list
    averaged_delays_dyn: list
    averaged_tps_stat: list
    averaged_tps_dyn: list
    averaged_fps_stat: list
    averaged_fps_dyn: list
    averaged_fns_stat: list
    averaged_fns_dyn: list
    averaged_f1_stat: list
    averaged_f1_dyn: list


@dataclass
class SampleSignalAndSettings:
    sig: np.ndarray
    changes: list
    rois: List[Roi]


def run_artificial_signal_avg(max_theta=10,
                              signal_settings: SignalSettings = None,
                              one_change_point=True,
                              with_resetting=False):
    """
    The same as run_artificial_signal() in performance_detection_signals.py but
    averaged results and 1 change.
    """
    n_sim = N_SIM
    n_points = signal_settings.half_len
    sigma = signal_settings.sigma
    mu0 = signal_settings.mu0
    mu1 = signal_settings.mu1
    if one_change_point:
        changes = [n_points]
    else:
        changes = [n_points, 2 * n_points]

    rois = [Roi(change, ROI_RADIUS_ARTIFICIAL_SIGNAL) for change in changes]

    tps_vec_stat = []
    tps_vec_dyn = []

    fns_vec_stat = []
    fns_vec_dyn = []

    fps_vec_stat = []
    fps_vec_dyn = []

    delays_vec_stat = []
    delays_vec_dyn = []

    f1_vec_stat = []
    f1_vec_dyn = []

    theta_vec = list(np.linspace(0.1, max_theta, 100))

    if with_resetting:
        detector_fn = cusum
        cusum_pccf_ref = make_cusum_pccf(rois)
        detector_pccf_fn = cusum_pccf_ref
    else:
        cusum_single_ref = make_cusum_single_ref(mu0)
        cusum_single_pccf_ref = make_cusum_single_pccf_ref(mu0, rois[0])
        detector_fn = cusum_single_ref
        detector_pccf_fn = cusum_single_pccf_ref

    for _ in tqdm(range(n_sim)):
        if one_change_point:
            sig = np.concatenate((randn(n_points) * sigma + mu0,
                                  randn(n_points) * sigma + mu1,
                                  ), axis=0)
        else:
            sig = np.concatenate((randn(n_points) * sigma + mu0,
                                  randn(n_points) * sigma + mu1,
                                  randn(n_points) * sigma + mu0
                                  ), axis=0)
        # Vectors with counts
        delays_stat, fps_stat, fns_stat, tps_stat, f1_stat = \
            collect_performance(detector_fn, sig, changes, theta_vec)

        delays_dyn, fps_dyn, fns_dyn, tps_dyn, f1_dyn = \
            collect_performance(detector_pccf_fn, sig, changes, theta_vec, rois=rois)

        delays_vec_stat.append(delays_stat)
        delays_vec_dyn.append(delays_dyn)

        fps_vec_stat.append(fps_stat)
        fps_vec_dyn.append(fps_dyn)

        fns_vec_stat.append(fns_stat)
        fns_vec_dyn.append(fns_dyn)

        tps_vec_stat.append(tps_stat)
        tps_vec_dyn.append(tps_dyn)

        f1_vec_stat.append(f1_stat)
        f1_vec_dyn.append(f1_dyn)

    delays_stat_m = np.vstack(delays_vec_stat)
    delays_dyn_m = np.vstack(delays_vec_dyn)

    fps_stat_m = np.vstack(fps_vec_stat)
    fps_dyn_m = np.vstack(fps_vec_dyn)

    fns_stat_m = np.vstack(fns_vec_stat)
    fns_dyn_m = np.vstack(fns_vec_dyn)

    tps_stat_m = np.vstack(tps_vec_stat)
    tps_dyn_m = np.vstack(tps_vec_dyn)

    f1_stat_m = np.vstack(f1_vec_stat)
    f1_dyn_m = np.vstack(f1_vec_dyn)

    averaged_delays_stat = average_results_matrix_rows(delays_stat_m)
    averaged_delays_dyn = average_results_matrix_rows(delays_dyn_m)

    averaged_fps_stat = average_results_matrix_rows(fps_stat_m, count_nans_as_zero=False)
    averaged_fps_dyn = average_results_matrix_rows(fps_dyn_m, count_nans_as_zero=False)

    averaged_fns_stat = average_results_matrix_rows(fns_stat_m, count_nans_as_zero=False)
    averaged_fns_dyn = average_results_matrix_rows(fns_dyn_m, count_nans_as_zero=False)

    averaged_tps_stat = average_results_matrix_rows(tps_stat_m, count_nans_as_zero=False)
    averaged_tps_dyn = average_results_matrix_rows(tps_dyn_m, count_nans_as_zero=False)

    averaged_f1_stat = average_results_matrix_rows(f1_stat_m, count_nans_as_zero=False)
    averaged_f1_dyn = average_results_matrix_rows(f1_dyn_m, count_nans_as_zero=False)

    # Normalize
   #  averaged_fps_stat /= len(sig)
   #  averaged_fps_dyn /= len(sig)
   #  averaged_fns_stat /= len(changes)
   #  averaged_fns_dyn /= len(changes)
   #  averaged_tps_stat /= len(changes)
   #  averaged_tps_dyn /= len(changes)
   #  #
   #  averaged_f1_stat /= len(changes)
   #  averaged_f1_dyn /= len(changes)

    results = SimulationResults(theta_vec,
                                averaged_delays_stat, averaged_delays_dyn,
                                averaged_tps_stat, averaged_tps_dyn,
                                averaged_fps_stat, averaged_fps_dyn,
                                averaged_fns_stat, averaged_fns_dyn,
                                averaged_f1_stat, averaged_f1_dyn
                                )
    return results


def plot_results_from_file(results_temp_file,
                           output_fig=None,
                           plot_all_metrics=True):

    with open(results_temp_file, 'r') as fp:
        json_cont = json.load(fp)
        logger.info(f"Load results from {results_temp_file}")

    # Read results from ARL simulation
    with open(FILE_RESULTS1, 'r') as f1, \
            open(FILE_RESULTS2) as f2, \
            open(FILE_RESULTS3) as f3:
        results_arl1 = json.load(f1)
        results_arl2 = json.load(f2)
        results_arl3 = json.load(f3)

    common_font_size = 15
    common_font_size_axis = 15
    common_line_width = 2
    common_tick_size = 15

    if plot_all_metrics:
        fig = plt.figure('art sig sim', figsize=(24, 14))

        ax1 = fig.add_subplot(3, 5, 1)
        ax2 = fig.add_subplot(3, 5, 2)
        ax3 = fig.add_subplot(3, 5, 3)
        ax4 = fig.add_subplot(3, 5, 4)

        ax5 = fig.add_subplot(3, 5, 5)
        ax6 = fig.add_subplot(3, 5, 6)
        ax7 = fig.add_subplot(3, 5, 7)
        ax8 = fig.add_subplot(3, 5, 8)

        ax9 = fig.add_subplot(3, 5, 9)
        ax10 = fig.add_subplot(3, 5, 10)
        ax11 = fig.add_subplot(3, 5, 11)
        ax12 = fig.add_subplot(3, 5, 12)

        ax13 = fig.add_subplot(3, 5, 13)
        ax14 = fig.add_subplot(3, 5, 14)
        ax15 = fig.add_subplot(3, 5, 15)

    else:
        # Add 1 more column for ARLs
        fig = plt.figure('art sig sim', figsize=(10, 14))  #  figsize=(15, 14) for 3 columns
        ax1 = fig.add_subplot(3, 2, 1)
        ax2 = fig.add_subplot(3, 2, 2)
        #ax3 = fig.add_subplot(3, 3, 3)
        ax4 = fig.add_subplot(3, 2, 3)
        ax5 = fig.add_subplot(3, 2, 4)
        #ax6 = fig.add_subplot(3, 3, 6)
        ax7 = fig.add_subplot(3, 2, 5)
        ax8 = fig.add_subplot(3, 2, 6)
        #ax9 = fig.add_subplot(3, 3, 9)

    r = json_cont

    def plot_delay_to_ax(ax_ref, r_item, title):
        theta_vec = r_item['theta_vec']
        ax_ref.plot(theta_vec, r_item['averaged_delays_stat'], ls="-", color="black", linewidth=common_line_width)
        ax_ref.plot(theta_vec, r_item['averaged_delays_dyn'], ls="--", color="black", linewidth=common_line_width)
        ax_ref.set_title(title, fontsize=common_font_size)
        ax_ref.set_xlabel('Threshold', fontsize=common_font_size_axis)
        ax_ref.set_ylabel('Average detection delay', fontsize=common_font_size_axis)
        ax_ref.xaxis.set_tick_params(labelsize=common_tick_size)
        ax_ref.yaxis.set_tick_params(labelsize=common_tick_size)
        ax_ref.xaxis.grid(True, linestyle='--')
        ax_ref.yaxis.grid(True, linestyle='--')
        ax_ref.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80])

    def plot_tp_to_ax(ax_ref, r_item, title):
        ax_ref.plot(r_item['theta_vec'], r_item['averaged_tps_stat'], ls="-", color="black", linewidth=common_line_width)
        ax_ref.plot(r_item['theta_vec'], r_item['averaged_tps_dyn'], ls="--", color="black", linewidth=common_line_width)
        ax_ref.set_title(title, fontsize=common_font_size)
        ax_ref.set_xlabel('Threshold', fontsize=common_font_size_axis)
        ax_ref.set_ylabel('Average TP', fontsize=common_font_size_axis)
        ax_ref.xaxis.set_tick_params(labelsize=common_tick_size)
        ax_ref.yaxis.set_tick_params(labelsize=common_tick_size)
        ax_ref.xaxis.grid(True, linestyle='--')
        ax_ref.yaxis.grid(True, linestyle='--')

    def plot_fp_to_ax(ax_ref, r_item, title):
        ax_ref.plot(r_item['theta_vec'], r_item['averaged_fps_stat'], ls="-", color="black", linewidth=common_line_width)
        ax_ref.plot(r_item['theta_vec'], r_item['averaged_fps_dyn'], ls="--", color="black", linewidth=common_line_width)
        ax_ref.set_title(title, fontsize=common_font_size)
        ax_ref.set_xlabel('Threshold', fontsize=common_font_size_axis)
        ax_ref.set_ylabel('Average FP', fontsize=common_font_size_axis)
        ax_ref.xaxis.set_tick_params(labelsize=common_tick_size)
        ax_ref.yaxis.set_tick_params(labelsize=common_tick_size)
        ax_ref.xaxis.grid(True, linestyle='--')
        ax_ref.yaxis.grid(True, linestyle='--')

    def plot_fn_to_ax(ax_ref, r_item, title):
        ax_ref.plot(r_item['theta_vec'], r_item['averaged_fns_stat'], ls="-", color="black", linewidth=common_line_width)
        ax_ref.plot(r_item['theta_vec'], r_item['averaged_fns_dyn'], ls="--", color="black", linewidth=common_line_width)
        ax_ref.set_title(title, fontsize=common_font_size)
        ax_ref.set_xlabel('Threshold', fontsize=common_font_size_axis)
        ax_ref.set_ylabel('Average FN', fontsize=common_font_size_axis)
        ax_ref.xaxis.set_tick_params(labelsize=common_tick_size)
        ax_ref.yaxis.set_tick_params(labelsize=common_tick_size)
        ax_ref.xaxis.grid(True, linestyle='--')
        ax_ref.yaxis.grid(True, linestyle='--')

    def plot_f1_to_ax(ax_ref, r_item, title, xticks):
        ax_ref.plot(r_item['theta_vec'], r_item['averaged_f1_stat'], ls="-", color="black", linewidth=common_line_width)
        ax_ref.plot(r_item['theta_vec'], r_item['averaged_f1_dyn'], ls="--", color="black", linewidth=common_line_width)
        ax_ref.set_title(title, fontsize=common_font_size)
        ax_ref.set_xlabel('Threshold', fontsize=common_font_size_axis)
        ax_ref.set_ylabel('Average F1', fontsize=common_font_size_axis)
        ax_ref.xaxis.set_tick_params(labelsize=common_tick_size)
        ax_ref.yaxis.set_tick_params(labelsize=common_tick_size)
        ax_ref.xaxis.grid(True, linestyle='--')
        ax_ref.yaxis.grid(True, linestyle='--')
        ax_ref.set_xticks(xticks)

    def plot_arl_to_ax(ax_ref, results, title, xlim=(10, 10), ylim=(10, 10)):
        ax_ref.plot(results['theta_vec'], results['avg_arl_stat'], ls="-", color="black", linewidth=common_line_width)
        ax_ref.plot(results['theta_vec'], results['avg_arl_dyn'], ls="--", color="black", linewidth=common_line_width+2)
        ax_ref.axhline([75], color="black", linewidth=common_line_width+1, ls='--')
        ax_ref.axhline([125], color="black", linewidth=common_line_width+1, ls='--')
        ax_ref.axhline(100, color="black", linewidth=common_line_width-1, ls='-')
        ax_ref.set_title('ARL', fontsize=common_font_size)
        ax_ref.set_title(title, fontsize=common_font_size)
        ax_ref.set_xlabel('Threshold', fontsize=common_font_size_axis)
        ax_ref.set_ylabel('Average Run Length', fontsize=common_font_size_axis)
        ax_ref.xaxis.set_tick_params(labelsize=common_tick_size)
        ax_ref.yaxis.set_tick_params(labelsize=common_tick_size)
        ax_ref.xaxis.grid(True, linestyle='--')
        ax_ref.yaxis.grid(True, linestyle='--')
        # Insert zoomed
        axins = ax_ref.inset_axes([0.55, 0.07, 0.4, 0.4])
        axins.set_xlim(xlim)
        axins.set_ylim(ylim)
        axins.plot(results['theta_vec'], results['avg_arl_stat'], ls="-", color="black", linewidth=common_line_width)
        axins.plot(results['theta_vec'], results['avg_arl_dyn'], ls="--", color="black", linewidth=common_line_width+2)
        axins.axhline([75], color="black", linewidth=common_line_width+1, ls='--')
        axins.axhline([125], color="black", linewidth=common_line_width+1, ls='--')
        axins.axhline(100, color="black", linewidth=common_line_width-1, ls='-')
        axins.indicate_inset_zoom(axins, edgecolor="black")
        axins.set_xticks([10, 15, 20])
        axins.xaxis.grid(True, linestyle='--')
        axins.set_yticklabels('')

    def plot_tp_fp_to_ax(ax_ref, r_item, title):
        averaged_fps_stat = r_item['averaged_fps_stat']
        averaged_tps_stat = r_item['averaged_tps_stat']
        averaged_tps_dyn = r_item['averaged_tps_dyn']
        averaged_fps_dyn = r_item['averaged_fps_dyn']
        ax_ref.plot(averaged_fps_stat, averaged_tps_stat, ls="-", color="black", linewidth=common_line_width)
        ax_ref.plot(averaged_fps_dyn, averaged_tps_dyn, ls="--", color="black", linewidth=common_line_width+2)
        ax_ref.set_title(title, fontsize=common_font_size)
        ax_ref.set_xlabel('Average FP', fontsize=common_font_size_axis)
        ax_ref.set_ylabel('Average TP', fontsize=common_font_size_axis)
        ax_ref.xaxis.set_tick_params(labelsize=common_tick_size)
        ax_ref.yaxis.set_tick_params(labelsize=common_tick_size)
        ax_ref.xaxis.grid(True, linestyle='--')
        ax_ref.yaxis.grid(True, linestyle='--')

    if plot_all_metrics:
        plot_tp_to_ax(ax2, r[0],  'Average TP for $\delta=1.1$')
        plot_tp_to_ax(ax7, r[1],  'Average TP for $\delta=2.1$')
        plot_tp_to_ax(ax12, r[2], 'Average TP for $\delta=3.1$')

        plot_fp_to_ax(ax3, r[0],  'Average FP for $\delta=1.1$')
        plot_fp_to_ax(ax8, r[1],  'Average FP for $\delta=2.1$')
        plot_fp_to_ax(ax13, r[2], 'Average FP for $\delta=3.1$')

        plot_fn_to_ax(ax4, r[0],  'Average FN for $\delta=1.1$')
        plot_fn_to_ax(ax9, r[1],  'Average FN for $\delta=2.1$')
        plot_fn_to_ax(ax14, r[2], 'Average FN for $\delta=3.1$')

    plot_delay_to_ax(ax1, r[0], 'Detection delays for $\delta=1.1$')
    plot_delay_to_ax(ax4, r[1], 'Detection delays for $\delta=2.1$')
    plot_delay_to_ax(ax7, r[2], 'Detection delays for $\delta=3.1$')
    plot_f1_to_ax(ax2, r[0],  'Average F1 for $\delta=1.1$', [0, 10, 20, 30, 40, 50, 60, 70, 80])
    plot_f1_to_ax(ax5, r[1],  'Average F1 for $\delta=2.1$', [0, 10, 20, 30, 40, 50, 60, 70, 80])
    plot_f1_to_ax(ax8, r[2], 'Average F1 for $\delta=3.1$', [0, 10, 20, 30, 40, 50, 60, 70, 80])

    # plot_arl_to_ax(ax3, results_arl1, 'ARL for $\delta=1.1$', xlim=(0, 35), ylim=(74, 130))
    # plot_arl_to_ax(ax6, results_arl2, 'ARL for $\delta=2.1$', xlim=(8, 23), ylim=(74, 130))
    # plot_arl_to_ax(ax9, results_arl3, 'ARL for $\delta=3.1$', xlim=(5, 30), ylim=(74, 130))

    for ax_obj in [ax1, ax4, ax7]:
        ax_obj.set_xlim((0, 80))

    for ax_obj in [ax2, ax5, ax8]:
        ax_obj.set_xlim((0, 80))

    # for ax_obj in [ax3, ax6, ax9]:
    #     ax_obj.set_ylim((0, 140))

    # plt.text(-100, 400, 'Detection delay', fontsize=22)

    plt.tight_layout()

    if output_fig:
        if path_ext_low_with_dot(output_fig) == '.eps':
            plt.savefig(output_fig, format='eps')
        elif path_ext_low_with_dot(output_fig) == '.png':
            plt.savefig(output_fig, format='png', dpi=300)
        logger.info(f"Save Figure into {output_fig}")
        plt.close()
    else:
        plt.show()
        plt.close()


def simulation_iterate_mu2(mu2_list,
                           with_resetting=False,
                           ):
    """ Variate mu2-mu1 and therefore ARLs """
    results_ = []
    for mu2 in mu2_list:
        sig_settings = SignalSettings(0.0, mu2, 1.0, SIG_HALF_LEN)
        r = run_artificial_signal_avg(max_theta=MAX_THETA,
                                      signal_settings=sig_settings,
                                      one_change_point=True,
                                      with_resetting=with_resetting)
        results_.append({'mu': mu2,
                         'theta_vec': list(r.theta_vec),

                         'averaged_tps_stat': list(r.averaged_tps_stat),
                         'averaged_tps_dyn': list(r.averaged_tps_dyn),

                         'averaged_fps_stat': list(r.averaged_fps_stat),
                         'averaged_fps_dyn': list(r.averaged_fps_dyn),

                         'averaged_fns_stat': list(r.averaged_fns_stat),
                         'averaged_fns_dyn': list(r.averaged_fns_dyn),

                         'averaged_delays_stat': list(r.averaged_delays_stat),
                         'averaged_delays_dyn': list(r.averaged_delays_dyn),

                         'averaged_f1_stat': list(r.averaged_f1_stat),
                         'averaged_f1_dyn': list(r.averaged_f1_dyn),

                         })
    extra_prefix = '_with_resetting' if with_resetting else '_no_resetting'
    save_path = save_json_to_temp_file(results_,
                                       'performance_detection_sim_results' + extra_prefix,
                                       )
    return save_path


if __name__ == "__main__":
    # results_file = None
    # if 1 == 1:
    #     # results_file_with_resetting = run_simulations_for_mu2s(with_resetting=False)
    #     results_file_no_resetting = iterate_mu2([1.1, 2.1, 3.1], with_resetting=True)
    # else:
    #     # results_file_with_resetting = "./temp/performance_detection_sim_results_no_resettinghoo_qjuy"
    #     results_file_no_resetting = "/home/ms314/1/phd/src/pccf/temp/performance_detection_sim_results_no_resettinghoo_qjuy"
    # plot_simulation_results(results_file_no_resetting, output_fig='../../tex/img/' + plot_name + '_no_resetting.eps')
    plot_name = 'performance_detection_sim'
    recalc_results = False
    if recalc_results:
        results_file = simulation_iterate_mu2([1.1, 2.1, 3.1], with_resetting=WITH_RESETTING)
    else:
        # results_file = "/home/ms314/1/phd/src/pccf/temp/performance_detection_sim_results_no_resettingitna2g7h"
        results_file = os.path.join(SRC_TEMP_DIR,
                                    'performance_detection_sim_results_no_resettingv4f2oc_f')
    plot_results_from_file(results_file,
                           output_fig=os.path.join(TEX_FOLDER_IMG, plot_name + '.eps'),
                           plot_all_metrics=False)
