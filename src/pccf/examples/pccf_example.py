"""
Just plot Pccf to see how it looks
"""
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

plt.rc('font', family='serif')
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('axes', titlesize=14)
plt.rc('axes', labelsize=16)
from pccf.pccf_obj import Pccf


def run_example(outpath=None):
    mu = 10.0
    expected_num_of_changes = 9
    r = Pccf(mu, 1.5)
    r = r.probabilities(100, expected_num_of_changes)
    PLOT = True
    if PLOT:
        fig = plt.figure(111, figsize=(6, 4))
        fig.subplots_adjust(bottom=0.2, left=0.15)
        ax1 = fig.add_subplot(111)
        ax1.set_xlim(0, 79)
        ax1.plot(r, color="black")
        for i in range(expected_num_of_changes):
            ax1.axvline((i + 1) * mu - 1, color="black")
        ax1.set_title("PCCF ~ N(10.0, 1.5)")
        ax1.set_xlabel('Time')
        ax1.set_xticks([10, 20, 30, 40, 50, 60, 70])
        if outpath:
            plt.savefig(outpath, format='eps')
            logger.info(f"Save Figure into {outpath}")
        else:
            plt.show()


run_example('../../tex/img/pccf_example.eps')
