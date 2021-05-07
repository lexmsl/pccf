"""
Catch cases when method works and doesn't

Simulation
 - generate signal with 1 change
 - set ROI around it
 - run dynamic and static CUSUM
 - stop when cde_dyn is after change (not FA) and within ROI and cde_stat < change (is FA)
 - also catch the case when method doesn't improve results - FN
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('axes', titlesize=14)
plt.rc('axes', labelsize=16)
from pccf.roi_obj import Roi
from loguru import logger
np.random.seed(41)
np.random.seed(1602)


class Simulation2:
    """ Proof of concept cases """

    def __init__(self):
        self.signal = np.zeros([500])
        self.changepoint = 250
        self.cusum_stat = np.zeros([])
        self.threshold = 20
        self.roi = Roi(250, 20)

    def run_catch_fn_case(self, output_path=None):
        self.threshold = 20
        max_iter = 100
        c = 0
        while True:
            c += 1
            if c > max_iter:
                break
            cde_dyn, cde_stat = self.get_simulation_result()
            if cde_dyn and cde_stat and cde_dyn > self.roi.right and cde_stat > self.roi.right:
                fig = plt.figure('fn_case', figsize=(6, 5))
                ax = fig.add_subplot(111)
                fig.subplots_adjust(bottom=0.2, left=0.15)
                ax.plot(self.cusum_stat, color="black")
                # ax.set_title("CUSUM output statistic")
                ax.set_xlabel('Time')
                ax.set_yticklabels([])
                ax.set_yticks([])
                # ax.set_ylabel('Statistic value')
                ax.axvline(self.changepoint, color="black", linewidth=2)
                ax.axvline(self.roi.left, color="black", ls="--", linewidth=2)
                ax.axvline(self.roi.right, color="black", ls="--", linewidth=2.5)
                ax.axvline(cde_dyn, color="red", linewidth=2)
                ax.axvline(cde_stat, color="red", linewidth=2)
                ax.axhline(self.threshold, color="black", linewidth=2)
                if output_path:
                    plt.savefig(output_path, format='eps')
                    logger.info(f"Save Figure into {output_path}")
                    plt.close()
                else:
                    plt.show()
                break

    def run(self, outpath=None):
        self.threshold = 20
        max_iter = 1000
        c = 0
        while True:
            c += 1
            if c > max_iter:
                break
            cde_dyn, cde_stat = self.get_simulation_result()
            if cde_dyn and cde_stat and cde_stat < self.changepoint < cde_dyn and 7 < self.roi.right - cde_dyn < 9:
                logger.info(f"Static detection delay: {cde_stat - self.changepoint},\nDynamic detection delay: {cde_dyn - self.changepoint}")
                fig = plt.figure('ok case?', figsize=(6, 5))
                ax = fig.add_subplot(111)
                fig.subplots_adjust(bottom=0.2, left=0.15)
                ax.plot(self.cusum_stat, color="black")
                # ax.set_title("CUSUM output statistic")
                ax.set_xlabel('Time')
                # ax.set_ylabel('Statistic value')
                ax.set_yticklabels([])
                ax.set_yticks([])
                ax.axvline(self.changepoint, color="black", linewidth=2)
                ax.axvline(self.roi.left, color="black", ls="--", linewidth=2)
                ax.axvline(self.roi.right, color="black", ls="--", linewidth=2.5)
                ax.axvline(cde_dyn, color="red", linewidth=2)
                ax.axvline(cde_stat, color="red", linewidth=2)
                ax.axhline(self.threshold, color="black")
                shift_left = 50
                y_pos = 240
                plt.text(cde_stat-shift_left-120, y_pos, 'Detection 1', fontsize=16)
                plt.text(cde_stat-shift_left-20, y_pos-30, '(FA)', fontsize=16)
                plt.text(cde_dyn + 60, y_pos+30, 'Detection 2', fontsize=16)
                plt.text(10, self.threshold+10, 'Threshold', fontsize=16)
                # arrow for CDE2
                x_pos_arrow = 60
                plt.arrow(cde_dyn + x_pos_arrow,
                          y_pos+30,
                          -x_pos_arrow,
                          -30,
                          head_width=0.3,
                          width=0.5,
                          color='red')
                text_x_stat = 350
                #
                # plt.text(text_x_stat + 20, self.cusum_stat[text_x_stat],
                #          'CUSUM statistic')
                # Change
                x_change = 310
                y_change = 190
                plt.text(x_change, y_change, 'Change ', fontsize=16)
                plt.text(x_change, y_change-20, 'point', fontsize=16)
                x_change_arrow = x_change
                y_change_arrow = y_change

                plt.arrow(x_change_arrow,
                          y_change_arrow,
                          self.changepoint - x_change_arrow,
                          -10,
                          linewidth=2)
                # ROI texts
                x_roi_text = 174
                y_roi_text = 160
                plt.text(x_roi_text, y_roi_text, 'ROI', fontsize=16)
                plt.arrow(x_roi_text+20,
                          y_roi_text,
                          self.roi.left - x_roi_text-20,
                          -20,
                          ls='--',
                          linewidth=2)
                plt.arrow(x_roi_text+20,
                          y_roi_text,
                          self.roi.right - x_roi_text-20,
                          -20,
                          ls='--',
                          linewidth=2)
                if outpath:
                    plt.savefig(outpath, format='eps')
                    logger.info(f"Save Figure into {outpath}")
                    plt.close()
                else:
                    plt.show()
                break

    def get_simulation_result(self):
        mu0 = 0.0
        self.signal = Simulation2.generate_signal(self.changepoint, mu0, 1.1, 1.1)
        self.cusum_stat = self.calculate_cusum(mu0)
        cde_stat = self.detector(self.threshold)
        cde_dyn = self.detector(self.threshold, self.roi)
        return cde_dyn, cde_stat

    @staticmethod
    def generate_signal(k, mu0, mu1, sigma):
        return np.concatenate(
            (np.random.randn(k) * sigma + mu0, np.random.randn(k) * sigma + mu1), axis=0
        )

    def calculate_cusum(self, mu0):
        output_statistic = np.zeros(len(self.signal))
        n = len(output_statistic)
        for i in range(n):
            if i == 0:
                output_statistic[i] = self.signal[i] - mu0
            else:
                output_statistic[i] = output_statistic[i-1] + self.signal[i] - mu0
        return output_statistic

    def detector(self, threshold: float, roi: Roi = None):
        n = len(self.cusum_stat)
        if roi:
            for i in range(roi.left, len(self.signal)):
                if self.cusum_stat[i] >= threshold:
                    return i
        else:
            for i in range(n):
                if self.cusum_stat[i] >= threshold:
                    return i
        return None


sim2 = Simulation2()
# logger.info("Run separately! - otherwise plots are not renewed")
sim2.run('../../tex/img/proof_of_concept2.eps')
sim2.run_catch_fn_case('../../tex/img/proof_of_concept2_fn_case.eps')
