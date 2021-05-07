"""
Data loader for repl
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from loguru import logger
import shutil
from pccf.utils_general import *
from pccf.roi_obj import Roi
from datetime import datetime, timedelta

MAIN_PATH = "/home/ms314/1/phd/datasets"
SIGNALS_PLOTS_PATH = '/home/ms314/1/phd/datasets/plots_signals'


def get_time(x):
    format_str = '%H:%M'
    format_str = '%Y-%m-%d %H:%M:%S'
    m_time = datetime.utcfromtimestamp(x) + timedelta(hours=3)
    return m_time # .strftime(format_str)


def read_signal(path):
    with open(os.path.join(MAIN_PATH, path), "r") as f:
        sig = [float(e) for e in f.readlines()]
        return sig


sig_temperature = pd.read_csv(os.path.join(MAIN_PATH, "temperature_home_office/temperature_data1.csv"), header=None, sep=',')
sig_temperature.columns = ['time', 'temperature', 'humidity']
sig_temperature_time = sig_temperature.time.values.tolist()
sig_temperature_time = [get_time(x) for x in sig_temperature_time]
# sig_temperature.time = sig_temperature.time - sig_temperature.time[0]
sig_temperature = sig_temperature.temperature.values.tolist()
changes_temperature_indices = [790, 2080, 3550, 4970, 6395, 7890, 9260, 10760, 12160, 13600, 15050, 16450, 17920, 19450]


def temperature_work_days_starts():
    return [datetime.fromtimestamp(sig_temperature_time[chp]) for chp in changes_temperature_indices]


def temperature_time_intervals_between_work_day_starts():
    starts = temperature_work_days_starts()
    intervals_hours = []
    for i, w_time_start in enumerate(starts):
        if i > 0:
            delta = (w_time_start-starts[i-1]).total_seconds()/(60*60)
            intervals_hours.append(delta)
    return intervals_hours

# changes_temperature_candidates = [797, 2027, 2387, 3495, 3856, 4922, 5219, 5460, 6346, 6744, 7823, 8123, 9223, 9498, 10712, 12126, 12407, 13546, 13863, 14136, 14987, 15323, 16408, 16717, 16958, 17882, 18150, 18391, 19398]
# changes_temperature_candidates = []
# for i, t in enumerate(sig_temperature):
#     if i < 60:
#         continue
#     else:
#         if t - sig_temperature[i-30] > 0.5:
#             if len(changes_temperature_candidates) == 0:
#                 changes_temperature_candidates.append(i)
#             elif len(changes_temperature_candidates) > 0 and i - changes_temperature_candidates[-1] > 60*5:
#                 changes_temperature_candidates.append(i - 60)
# logger.info(f"changes_temperature_candidates = {changes_temperature_candidates}")


def rois_temperature_signal():
    radius = 200
    mu = 1400
    rois = [Roi(changes_temperature_indices[0], radius=radius)]
    for k in range(5):
        rois.append(Roi(changes_temperature_indices[0] + (k + 1) * mu, radius))
    logger.info("Temp. sig. ROIs")
    for r in rois:
        print(r)
    return rois


sig_twi = read_signal("exchange-rate-bank-of-twi/data/sigTwi.dat")
changes_twi = read_signal("exchange-rate-bank-of-twi/data/changesTwi.dat")


sig_internet_traffic = read_signal("internet-traffic-data-in-bits-fr/data/sigInternetTraffic.dat")
changes_internet_traffic = read_signal(
    "internet-traffic-data-in-bits-fr/data/changesInternetTraffic.dat"
)

sig_lake_level = read_signal("monthly-lake-erie-levels/data/sigLakeLevel.dat")
changes_lake_level = read_signal(
    "monthly-lake-erie-levels/data/changesErieLake.dat"
)

sig_earth_quakes = read_signal(
    "number-of-earthquakes-per-year/data/sigNumOfEarthQuakes.dat"
)
changes_earth_quakes = read_signal(
    "number-of-earthquakes-per-year/data/changesNumOfEarthQuakes.dat"
)

sig_sunspots = read_signal("wolfer-sunspot-numbers/data/sigSunspotNumber.dat")
changes_sunspots = read_signal("wolfer-sunspot-numbers/data/changes.dat")


signals_all = {
    "sig_twi": sig_twi,
    "sig_internet_traffic": sig_internet_traffic,
    "sig_lake_level": sig_lake_level,
    "sig_earth_quakes": sig_earth_quakes,
    "sig_sunspots": sig_sunspots,
}

changes_all = {
    "changes_twi": changes_twi,
    "changes_internet_traffic": changes_internet_traffic,
    "changes_lake_level": changes_lake_level,
    "changes_earth_quakes": changes_earth_quakes,
    "changes_sunspots": changes_sunspots,
}


def plot(sig, changes, title='', save_path=None, y_min=None, y_max=None):
    fig = plt.figure(111, figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.plot(sig, color="black")
    for c in changes:
        ax.axvline(c, color="red")
    if y_min and y_max:
        ax.set_ylim(y_min, y_max)
    elif y_min:
        ax.set_ylim(y_min, np.max(sig))
    else:
        ax.set_ylim(0, np.max(sig))
    ax.set_xlim(0, len(sig))
    ax.set_title(title)
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Save into {save_path}")
        plt.close()
    else:
        plt.show()


def plot_earth_quakes():
    plot(sig_earth_quakes,
         changes_earth_quakes,
         title='Earth quakes',
         save_path=os.path.join(SIGNALS_PLOTS_PATH, 'sig_earth_quakes.PNG'))


def plot_lake_level():
    plot(sig_lake_level,
         changes_lake_level,
         title='Lake level',
         save_path=os.path.join(SIGNALS_PLOTS_PATH, 'sig_lake_level.PNG'))


def plot_internet_traffic():
    plot(sig_internet_traffic,
         changes_internet_traffic,
         title='Internet traffic',
         save_path=os.path.join(SIGNALS_PLOTS_PATH, 'sig_internet_traffic.PNG'))


def plot_twi():
    plot(sig_twi,
         changes_twi,
         title='Twi',
         save_path=os.path.join(SIGNALS_PLOTS_PATH, 'sig_twi.PNG'))


def plot_sunspots():
    plot(sig_earth_quakes,
         changes_earth_quakes,
         title='Sunspots',
         save_path=os.path.join(SIGNALS_PLOTS_PATH, 'sig_sunspots.PNG'))


def plot_temperature():
    plot(sig_temperature,
         changes_temperature_indices,
         title='Temperature',
         y_min=10,
         y_max=25,
         save_path=os.path.join(SIGNALS_PLOTS_PATH, 'sig_temperature.PNG'))


def update_temperature_data():
    in_file = "/home/ms314/1/phd/src/pccf/temperature_data1.csv"
    out_file = "/home/ms314/1/phd/datasets/temperature_home_office/temperature_data1.csv"
    shutil.copy(in_file, out_file)
    logger.info(f"Copy {in_file} to {out_file}")


def plot_all_signals_to_folder():
    plot_earth_quakes()
    plot_lake_level()
    plot_internet_traffic()
    plot_twi()
    plot_sunspots()
    plot_temperature()


if __name__ == "__main__":
    pass
