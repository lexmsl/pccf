from data import *
from loguru import logger
import numpy as np
from datetime import datetime
import json

logger.info(f"Temp. data collectio\nStart: {datetime.fromtimestamp(sig_temperature_time[0])}\nEnd  : {datetime.fromtimestamp(sig_temperature_time[-1])}")

deltas_hours = temperature_time_intervals_between_work_day_starts()
logger.info(f"\nMean={np.mean(deltas_hours)}; Std={np.std(deltas_hours)}")
savepath = './temp/temperature_time_intervals.txt'
with open(savepath, 'w') as f:
    for t_interval in deltas_hours:
        f.write("{0:1.1f}, ".format(t_interval))
    f.write('\n')
    f.write(f'mean={np.mean(deltas_hours)}\n')
    f.write(f'std={np.std(deltas_hours)}\n')
logger.info(f"Save time intervals into {savepath}")