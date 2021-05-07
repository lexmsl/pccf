# Plot CUSUM output statistic to see its behaviour
#
#
#
from pccf.detector import cusum
from pccf.detector import cusum_statistic
import numpy as np
import matplotlib.pyplot as plt


def gen_sig():
    n = 100
    sigma = 1.1
    return np.concatenate((np.random.randn(n) * sigma + 0.0,
                           np.random.randn(n) * sigma + 2.0,
                           np.random.randn(n) * sigma + 0.0,
                           ), axis=0)


sig = gen_sig()
mu0_estim = 0.0
r_single = cusum_statistic(sig, mu0_estim)
r_multi = cusum(sig, 20)

fig = plt.figure(313, figsize=(12, 8))
ax1 = fig.add_subplot(311)
ax1.plot(sig, color="black")
for c in r_multi.detections:
    ax1.axvline(c, color="red")
ax2 = fig.add_subplot(312)
ax2.plot(r_multi.statistic, color="black")
ax2.set_title('Cusum output statistic multi')
ax3 = fig.add_subplot(313)
ax3.plot(r_single)
ax3.set_title('Cusum output statistic')
ax3.set_ylim([-20, 70])
plt.show()
