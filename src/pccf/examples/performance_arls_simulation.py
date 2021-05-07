"""
Measure ARL for different mu2 and save results to json files.
"""
import os
from pccf.detector import make_cusum_pccf, cusum, make_cusum_single_ref, make_cusum_single_pccf_ref
from tqdm import tqdm
from pccf.detector_performance import *
from pccf.settings import *
import json
from pccf.examples.common import SignalSettings, save_json_to_temp_file
import matplotlib.pyplot as plt
from numpy.random import randn
from pccf.examples.common import SignalSettings
from pccf.settings import SIG_HALF_LEN


FILE_RESULTS1 = os.path.join(SRC_TEMP_DIR, "performance_sim_arl_mu1_test.json")
FILE_RESULTS2 = os.path.join(SRC_TEMP_DIR, "performance_sim_arl_mu2_test.json")
FILE_RESULTS3 = os.path.join(SRC_TEMP_DIR, "performance_sim_arl_mu3_test.json")


def run_simulation_arl_perf(signal_settings: SignalSettings,
                            save_path):
    n_sim = 100
    n_points = signal_settings.half_len
    roi = Roi(signal_settings.half_len, ROI_RADIUS_ARTIFICIAL_SIGNAL)
    sigma = signal_settings.sigma
    mu0 = signal_settings.mu0
    mu1 = signal_settings.mu1

    cusum_single_ref = make_cusum_single_ref(mu0)
    cusum_single_pccf_ref = make_cusum_single_pccf_ref(mu0, roi)
    detector_fn = cusum_single_ref
    detector_pccf_fn = cusum_single_pccf_ref

    theta_vec = np.linspace(0.01, 40.0, 100)

    arl_vec_stat = []
    arl_vec_dyn = []
    for _ in tqdm(range(n_sim)):
        sig = np.concatenate((randn(n_points) * sigma + mu0,
                              randn(n_points) * sigma + mu1,
                              ), axis=0)

        arl_stat = collect_arl(detector_fn, sig, theta_vec)
        arl_dyn = collect_arl(detector_pccf_fn, sig, theta_vec)

        arl_vec_stat.append(arl_stat)
        arl_vec_dyn.append(arl_dyn)

    m_arl_stat = np.vstack(arl_vec_stat)
    m_arl_dyn = np.vstack(arl_vec_dyn)

    count_nans_as_zero = False
    # If ARL is NaN then we disregard it
    avg_arl_stat = average_results_matrix_rows(m_arl_stat, count_nans_as_zero=count_nans_as_zero)
    avg_arl_dyn = average_results_matrix_rows(m_arl_dyn, count_nans_as_zero=count_nans_as_zero)
    # avg_arl_stat = np.sum(m_arl_stat, axis=0)/m_arl_stat.shape[0]
    # avg_arl_dyn = np.nansum(m_arl_dyn, axis=0)/m_arl_dyn.shape[0]

    results = {'theta_vec': list(theta_vec),
               'avg_arl_stat': list(avg_arl_stat),
               'avg_arl_dyn': list(avg_arl_dyn),
               }
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Save into {save_path}")
    return save_path


def plot_results():
    with open(FILE_RESULTS1, 'r') as f1, \
            open(FILE_RESULTS2) as f2, \
            open(FILE_RESULTS3) as f3:
        results1 = json.load(f1)
        results2 = json.load(f2)
        results3 = json.load(f3)
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    def find_intersection(arl_stat, arl_dyn):
        for i, dyn in enumerate(arl_dyn):
            if dyn <= arl_stat[i]:
                return i
        return None

    v1 = find_intersection(results1['avg_arl_stat'], results1['avg_arl_dyn'])
    v2 = find_intersection(results2['avg_arl_stat'], results2['avg_arl_dyn'])
    v3 = find_intersection(results3['avg_arl_stat'], results3['avg_arl_dyn'])
    v1 = results1['theta_vec'][v1]
    v2 = results2['theta_vec'][v2]
    v3 = results3['theta_vec'][v3]
    logger.info(f"ARL intersection 1 = {v1}")
    logger.info(f"ARL intersection 2 = {v2}")
    logger.info(f"ARL intersection 3 = {v3}")

    def plot_to_ax(ax, results):
        ax.set_ylim((70, 130))
        ax.plot(results['theta_vec'], results['avg_arl_stat'], '', color='black', linewidth=1, ls='-')
        ax.plot(results['theta_vec'], results['avg_arl_dyn'], '', color='black', ls='--', linewidth=2)
        ax.axhline(100, color='black', ls='--', linewidth=0.5)
        ax.axhline(100 - ROI_RADIUS_ARTIFICIAL_SIGNAL, color='black', ls='--', linewidth=0.5)
        ax.axhline(100 + ROI_RADIUS_ARTIFICIAL_SIGNAL, color='black', ls='--', linewidth=0.5)

    plot_to_ax(ax1, results1)
    plot_to_ax(ax2, results2)
    plot_to_ax(ax3, results3)
    plt.show()


def run_simulation():
    sig_settings1 = SignalSettings(0.0, 1.1, 1.0, SIG_HALF_LEN)
    sig_settings2 = SignalSettings(0.0, 2.1, 1.0, SIG_HALF_LEN)
    sig_settings3 = SignalSettings(0.0, 3.1, 1.0, SIG_HALF_LEN)
    run_simulation_arl_perf(sig_settings1, save_path=FILE_RESULTS1)
    run_simulation_arl_perf(sig_settings2, save_path=FILE_RESULTS2)
    run_simulation_arl_perf(sig_settings3, save_path=FILE_RESULTS3)


if __name__ == "__main__":
    run_simulation()
    plot_results()
