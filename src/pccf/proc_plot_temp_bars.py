import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

intervals = [21.5, 24.5, 23.7, 23.8, 24.9, 22.9, 25.0, 23.4, 24.0, 24.2, 23.4, 24.5, 25.5]
data = pd.DataFrame({'Intervals between changes': intervals,
                     'Values': intervals})

if __name__ == '__main__':
    fig, ax = plt.subplots()
    ax.bar(x=[i for i in range(len(intervals))], height=intervals, width=1.0, color='whitesmoke', edgecolor='black')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    for i, v in enumerate(intervals):
        ax.text(i-0.3, v+0.2, str(v), fontsize=12, rotation=90)
    ax.set_ylim(21, 27)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_title('Time intervals between changes')
    savefig = '../../tex/img/temperature_bars.eps'
    logger.info(f"Save Figure into {savefig}")
    plt.savefig(savefig, transparent=True)


