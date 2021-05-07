import matplotlib.pyplot as plt
from pccf.data import sig_temperature, changes_temperature_indices
from pccf.detector import *
from pccf.pccf_obj import *
from pccf.pccf_performance import *
from pccf.utils_pccf import *


def shifted_changes(changes):
    first_change = changes[0]
    return first_change, [chp-first_change for chp in changes[1:]]


change0, changes_temperature_s = shifted_changes(changes_temperature_indices)

MU = 1400
RADIUS = 200
SIGMA = 100


def rois_temperature_signal():
    radius = RADIUS
    mu = MU  # 1400
    rois = [Roi(changes_temperature_indices[0], radius=radius)]
    for k in range(13):
        rois.append(Roi(change0 + (k+1) * mu, radius))
    return rois


rois_temperature = rois_temperature_signal()

pccf = Pccf(1400, sigma=SIGMA, radius=RADIUS)
probs = pccf.probabilities(len(sig_temperature), len(changes_temperature_indices))

if __name__ == "__main__":
    logger.info(f"changes={changes_temperature_s}")
    logger.info(f"diff1={diff1(changes_temperature_s)}")
    logger.info(np.mean(diff1(changes_temperature_s)))
    perf = PccfPerformanceMetrics(changes_temperature_indices)
    logger.info(f"Pccf F1 score for temperature signal ={perf.f1_score(rois_temperature)}")


    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(211)
    ax1.plot(sig_temperature, color='black')
    for c in changes_temperature_indices:
        ax1.axvline(c, color='black')

    ax2 = fig.add_subplot(212)
    ax2.plot(probs[600:], color='black')
    for r in rois_temperature:
        ax2.axvline(r.left, color='black', ls='--')
        ax2.axvline(r.right, color='black', ls='--')
    for c in changes_temperature_indices:
        ax2.axvline(c, color='black')
    plt.show()

