import matplotlib.pyplot as plt
from pccf.data import *
from pccf.utils_pccf import *
from pccf.pccf_performance import *


def pccf_performance_signals_changes(changes_in,
                                     save_figure=None,
                                     signal_title=None,
                                     mu_range=None,
                                     radius_range=(1, 10)):
    """Returns Pccf() object with only mu set"""
    changes = changes_in.copy()
    changes = [int(c) for c in changes]
    min_mu, max_mu = min_max_mu(changes)
    logger.info(f"Input changes: {changes}")
    logger.info(f"min_mu={min_mu}, max_mu={max_mu}")
    logger.info(f"diff1={diff1(changes)}")
    if mu_range is None:
        mu_vec, f1_vec = collect_pccf_performance(changes, mu_range=(min_mu, max_mu), radius_range=radius_range)
    else:
        mu_vec, f1_vec = collect_pccf_performance(changes, mu_range=mu_range, radius_range=radius_range)
    idx_optimal = np.argmax(f1_vec)
    mu_optimal = mu_vec[idx_optimal]
    print(f"Optimal mu = {mu_optimal}, | f1 = {f1_vec[idx_optimal]}")
    # Construct corresponding Pccf - the same as used in collect_pccf_performance()
    pccf = Pccf(mu=mu_optimal)
    if save_figure:
        common_font_size = 30
        common_line_width = 4
        common_tick_size = 30
        fig = plt.figure(111, figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(mu_vec, f1_vec, ls="-", color="black", linewidth=common_line_width)
        ax.set_xlabel('Mu estimation', fontsize=common_font_size)
        ax.set_ylabel('F1 score', fontsize=common_font_size)
        ax.set_title(str(signal_title), fontsize=common_font_size)
        ax.set_ylim((-0.1, 1.1))
        ax.xaxis.set_tick_params(labelsize=common_tick_size)
        ax.yaxis.set_tick_params(labelsize=common_tick_size)

        if path_ext_low_with_dot(save_figure) == '.eps':
            plt.savefig(save_figure, format='eps')
        elif path_ext_low_with_dot(save_figure) == '.png':
            plt.savefig(save_figure, format='png', dpi=300)
        else:
            raise NotImplementedError
        logger.info(f"Save Figure into {save_figure}")
        plt.savefig(save_figure)

        plt.close()
    return pccf


if __name__ == "__main__":
    # pccf_performance_signals_changes(changes_twi, output_figure='./img/pccf_performance_twi.PNG', signal_name="Twi")
    # pccf_performance_signals_changes(changes_lake_level, output_figure='./img/pccf_performance_lake.PNG', signal_name="Lake level")
    # pccf_performance_signals_changes(changes_sunspots, output_figure='./img/pccf_performance_sunspots.PNG', signal_name="Sunspots")
    # pccf_performance_signals_changes(changes_earth_quakes, output_figure='./img/pccf_performance_earth_quakes.PNG', signal_name="Earth Quakes")
    # pccf_performance_signals_changes(changes_internet_traffic, output_figure='./img/pccf_performance_internet_traffic.PNG', signal_name="Internet traffic")

    pccf_performance_signals_changes(changes_temperature_indices,
                                     save_figure='../../tex/img/performance_pccf_temperature.eps',
                                     signal_title='Pccf performance for temperature signal',
                                     radius_range=(490, 510),
                                     mu_range=(950, 1350))
