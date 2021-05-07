"""
Demonstrate that CUSUM statistic is Random walk with/without drift
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pccf.roi_obj import Roi
from loguru import logger
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('axes', titlesize=14)
plt.rc('axes', labelsize=16)
# plt.style.use('/Users/al314/gitlocal/phd-monitor-2/src/pccf/style.use')
# plt.style.use('seaborn-dark')


def randn():
    return np.random.randn(1)[0]


class Simulation1:
    """
    Show that CUSUM and Random Walk with drift are the same.
    """

    def __init__(self):
        self.n = 501
        self.roi = Roi(230, 300)
        self.chp_loc = 251
        self.mu0 = 0.0
        self.mu1 = 2.0
        self.sigma = 1.1
        self.signal = np.concatenate(
            (
                np.random.randn(self.chp_loc - 1) * self.sigma + self.mu0,
                np.random.randn(self.n - self.chp_loc + 1) * self.sigma + self.mu1,
            ),
            axis=0,
        )

    def run(self, outpath=None):
        logger.info("Run proof of concept1 example")
        result_rw = self.calculate_random_walk()
        result_cusum = self.calculate_cusum()
        fig = plt.figure(122, figsize=(15, 5))
        fig.subplots_adjust(bottom=0.15, left=0.1)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.plot(result_rw, color="black")
        ax1.set_title("Random Walk with drift")
        ax1.set_ylim([np.min(result_rw), 100])
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax2.plot(result_cusum, color="black")
        ax2.set_ylim([np.min(result_cusum), 100])
        ax2.set_title("CuSum output statistic")
        plt.savefig(outpath, format='eps')
        logger.info(f"Save figure into {outpath}")

    def calculate_random_walk(self):
        output_statistic = np.zeros(self.n)
        for i in range(self.n):
            if i == 0:
                output_statistic[i] = randn() - self.mu0
            elif i < self.chp_loc:
                output_statistic[i] = output_statistic[i - 1] + randn() - self.mu0
            else:
                output_statistic[i] = (
                    output_statistic[i - 1] + randn() + self.mu1 - self.mu0
                )
        return output_statistic

    def calculate_cusum(self):
        output_statistic = np.zeros(self.n)
        for i in range(self.n):
            if i == 0:
                output_statistic[i] = self.signal[i] - self.mu0
            else:
                output_statistic[i] = (
                    output_statistic[i - 1] + self.signal[i] - self.mu0
                )
        return output_statistic


sim1 = Simulation1()
# sim1.run('./img/proof_of_concept1.eps')
sim1.run('../../tex/img/proof_of_concept1.eps')
