import numpy as np
import matplotlib.pyplot as plt
from loguru import logger


def arl(delta=10.0, h=4.0):
    k = 0.5
    h_prime = h + 1.166
    num = np.exp(-2.0 * (delta - k) * h_prime) + 2 * (delta - k) * h_prime - 1.0
    den = 2.0 * (delta - k)**2
    return num/den


def plot_arl_delta():
    delta_vec = np.linspace(1.0, 20.0, 100)
    h_vec = np.linspace(0.0, 10.0, 100)
    arl_delta_vec1 = [arl(d, 4) for d in delta_vec]
    arl_delta_vec2 = [arl(d, 2) for d in delta_vec]
    delta1 = 10
    delta2 = 5
    arl_h_vec1 = [arl(delta1, h) for h in h_vec]
    arl_h_vec2 = [arl(delta2, h) for h in h_vec]

    fig = plt.figure(figsize=(24, 10))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.plot(delta_vec, arl_delta_vec1, color='black', linewidth=4, label='h=4')
    ax1.plot(delta_vec, arl_delta_vec2, color='black', linewidth=4, ls='--', label='h=2')
    ax1.legend(prop={'size': 30}, ncol=2, loc='upper center')
    ax1.xaxis.set_tick_params(labelsize=30)
    ax1.yaxis.set_tick_params(labelsize=30)
    ax1.set_xlabel('$\delta$', fontsize=40)
    ax1.set_ylabel('$ARL_{\delta}$', fontsize=33, rotation=0, labelpad=40)

    ax2.plot(h_vec, arl_h_vec1, color='black', linewidth=4, label='$\delta=' + str(delta1)+'$')
    ax2.plot(h_vec, arl_h_vec2, color='black', linewidth=4, ls='--', label='$\delta=' + str(delta2) + '$')
    ax2.legend(prop={'size': 30}, ncol=2, loc='upper center')
    ax2.xaxis.set_tick_params(labelsize=30)
    ax2.yaxis.set_tick_params(labelsize=30)
    ax2.set_xlabel('$h$', fontsize=40)
    ax2.set_ylabel('$ARL_{h}$', fontsize=33, rotation=0, labelpad=40)
    plt.tight_layout()
    savefig = '../../tex/img/arl.eps'
    logger.info(f"Save Figure into {savefig}")

    plt.savefig(savefig)
    plt.close()
    # plt.show()


if __name__ == '__main__':
    plot_arl_delta()
