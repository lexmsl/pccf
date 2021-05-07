import numpy as np
from abc import ABC, abstractmethod


class PerformanceMetrics(ABC):
    @abstractmethod
    def tps(self):
        pass

    @abstractmethod
    def fps(self):
        pass

    @abstractmethod
    def fns(self):
        pass

    @abstractmethod
    def f1_score(self):
        pass


def f1_score(tps_count=1, fps_count=1, fns_count=1):
    # TODO: there is another f1 score function in the module perf_detections
    return tps_count / (tps_count + 0.5 * (fps_count + fns_count))


def scale01(vec: np.ndarray) -> np.ndarray:
    return (vec - np.min(vec)) / (np.max(vec) - np.min(vec))


def rmse(a, b):
    n = len(a)
    assert n == len(b)
    return np.sum(np.sqrt((a - b) ** 2)) / float(n)

# def cdf(changes_list, n=10):
#     """
#     https://stackoverflow.com/questions/5328556/histogram-matplotlib
#     """
#     distances = list(map(lambda x: int(x), diff1(changes_list)))
#     max_dist = max(distances)
#     min_dist = min(distances)
#     print("Changes       : ", list(map(lambda x: int(x), changes_list)))
#     print("Distances     : ", distances)
#     print(f"Max distance  : {max_dist}")
#     print(f"Min distance  : {min_dist}")
#     hist, bin_edges = np.histogram(distances, bins=n)
#     hist = hist / np.sum(hist)
#     width = 0.7 * (bin_edges[1] - bin_edges[0])
#     centers = (bin_edges[1:] + bin_edges[:-1]) / 2
#     cdf = np.cumsum(hist)
#     bin_pairs = []
#     for e in list(zip(bin_edges[:-1], bin_edges[1:])):
#         bin_pairs.append((int(e[0]), int(e[1])))
#     for i, e in enumerate(bin_pairs):
#         print("Distance range: ", e, hist[i])
#     return cdf, centers


# def plot_cdf(changes_list, n=10):
#     cdf_, centers = cdf(changes_list, n)
#     plt.plot(centers, cdf_)
#     plt.show()
#
#

