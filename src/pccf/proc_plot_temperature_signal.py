import matplotlib.pyplot as plt
from pccf.data import *
from debug_temperature_data import *

import matplotlib.dates as md

xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
xfmt = md.DateFormatter('%H:%M:%S')
xfmt = md.DateFormatter('%H:%M')


def plot_temperature_signal(save_fig=''):
    fig = plt.figure(figsize=(24, 12))
    ax1 = fig.add_subplot(211)
    ax1.xaxis.set_major_formatter(xfmt)
    ax1.plot(sig_temperature_time, sig_temperature, ls="-", color="black", linewidth=4)
    # ax1.set_ylabel('C', fontsize=40, rotation=0,loc='top')
    ax1.set_xlabel('Changes [hh:mm]', fontsize=35)
    ax1.xaxis.set_ticks([sig_temperature_time[c] for c in changes_temperature_indices])

    for c in changes_temperature_indices:
        ax1.axvline(sig_temperature_time[c], color="black")

    ax1.xaxis.set_tick_params(labelsize=30, rotation=45)
    ax1.yaxis.set_tick_params(labelsize=30)
    yticks = [15, 20, 25]
    ax1.set_yticks(yticks)
    ax1.set_yticklabels([str(t) + ' C°' for t in yticks])

    ax2 = fig.add_subplot(212)
    for i, r in enumerate(rois_temperature):
        cl = 'lightgrey' if i <= 6 else 'whitesmoke'
        ax2.axvspan(sig_temperature_time[r.left],
                    sig_temperature_time[r.right], color=cl)
        #     ax2.axvline(r.left, color='black', ls='-', linewidth=2)
        #     ax2.axvline(r.right, color='black', ls='-', linewidth=2)
    ax2.xaxis.set_major_formatter(xfmt)
    ax2.xaxis.set_tick_params(labelsize=30)
    ax2.yaxis.set_tick_params(labelsize=30)
    probs_padded = np.hstack([probs[600:], np.zeros(len(sig_temperature_time) - len(probs[600:]))])
    ax2.plot(sig_temperature_time,
             probs_padded,
             color='black', linewidth=4)

    for c in changes_temperature_indices:
        ax2.axvline(sig_temperature_time[c], color='black', linewidth=4)

    xticks = [sig_temperature_time[c] for c in changes_temperature_indices]
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([i+1 for i in range(len(xticks))])

    ax2.axhline(0.0015, ls='--')
    ax2.set_yticklabels([])
    ax2.set_ylabel('PCCF', fontsize=35)
    ax2.set_xlabel('Prediction interval №', fontsize=35)
    plt.tight_layout()

    if save_fig:
        if path_ext_low_with_dot(save_fig) == '.eps':
            plt.savefig(save_fig, format='eps')
        elif path_ext_low_with_dot(save_fig) == '.png':
            plt.savefig(save_fig, format='png', dpi=300)
        else:
            raise NotImplementedError
        logger.info(f"Save Figure into {save_fig}")
    plt.close()


if __name__ == "__main__":
    plot_temperature_signal(save_fig='../../tex/img/temperature_signal.eps')
