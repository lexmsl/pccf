from .detector_performance import *


# @dataclass
# class RegionOfInterest:
#     left: int
#     right: int


@dataclass
class DetectorOutput:
    detections: List[int]
    statistic: List[float]


def make_cusum_single_ref(mu0):
    def f(s, t):
        return cusum_single(s, mu0, t)
    return f


def make_cusum_single_pccf_ref(mu0, roi):
    def f(s, t):
        return cusum_single(s, mu0, t, roi=roi)
    return f


def cusum_single(sig, mu0, threshold, roi: Roi = None):
    adaptive_statistic = False
    if adaptive_statistic:
        stat = cusum_statistic(sig, mu0, roi=roi)
    else:
        stat = cusum_statistic(sig, mu0, roi=None)
    detection = alarm_change(stat, threshold, roi)
    return DetectorOutput([detection], stat)


def alarm_change(output_stat: List[float],
                 threshold: float,
                 roi: Optional[Roi] = None,
                 ) -> Optional[int]:
    if roi:
        for i in range(len(output_stat)):
            if i in roi and output_stat[i] >= threshold:
                return i
    else:
        for i in range(len(output_stat)):
            if output_stat[i] >= threshold:
                return i
    return np.nan


def cusum_statistic(sig: List[float],
                    mu0: float,
                    roi: Roi = None) -> List[float]:
    """ I ROI is provided that statistic is calculated only within ROI
    """
    n = len(sig)
    stat = np.zeros(n)
    for i in range(n):
        if roi and i in roi:
            stat[i] = stat[i - 1] + sig[i] - mu0
        else:
            if i == 0:
                stat[i] = sig[i] - mu0
            else:
                stat[i] = stat[i - 1] + sig[i] - mu0
    return stat


def cusum(signal, threshold=1.0):
    n = len(signal)
    stat = np.zeros(n)
    mu = 0.0
    detections = []
    k = 1
    for i in range(n):
        mu = ((k - 1) / k) * mu + signal[i] / k
        k += 1
        if i == 0:
            stat[i] = signal[i] - mu
        else:
            stat[i] = stat[i - 1] + signal[i] - mu
        if abs(stat[i]) > threshold:
            mu = signal[i]
            k = 1
            stat[i] = 0.0
            detections.append(i)
    return DetectorOutput(detections, stat)


def cusum_pccf(signal, threshold, rois: List[Roi] = None):
    def entered_roi():
        return i == rois[next_roi_index].right + 1 and \
               next_roi_index + 1 < len(rois)

    def within_roi():
        return rois[next_roi_index].left <= i <= rois[
            next_roi_index].right

    n = len(signal)
    stat = np.zeros(n)
    mu = 0.0
    detections = []
    k = 1
    next_roi_index = 0
    for i in range(n):
        mu = ((k - 1) / k) * mu + signal[i] / k
        k += 1
        if i == 0:
            stat[i] = signal[i] - mu
        else:
            stat[i] = stat[i - 1] + signal[i] - mu
        if entered_roi():
            next_roi_index += 1
        if within_roi():
            if abs(stat[i]) > threshold:
                mu = signal[i]
                k = 1
                stat[i] = 0.0
                detections.append(i)
    return DetectorOutput(detections, stat)


class Detector:
    """
    In order to get 1 pseudocode for both detectors in paper.
    If rois is None then it is normal Cusum,
    otherwise it is Pccf detector.
    """

    def __init__(self, signal, threshold, rois: List[Roi] = None):
        self.threshold = threshold
        self.rois = rois
        self.k = 1
        self.t = 0
        self.next_roi_index = 0
        self.mu = 0.0
        self.signal = signal
        self.stat = np.zeros(len(signal))
        self.pccf = bool(self.rois)

    def run(self):
        n = len(self.signal)
        detections = []
        while self.t < n:
            self.update_mu(self.signal[self.t])
            self.update_cusum_statistic()
            if self.pccf and self.entered_roi():
                self.next_roi_index += 1
            if not self.pccf or \
                    (self.pccf and self.within_roi()):
                if self.crossed_threshold():
                    detections.append(self.t)
                    self.reset_detector()
            self.t += 1
        return DetectorOutput(detections, self.stat)

    def update_mu(self, new_observation):
        self.mu = ((self.k - 1) / self.k) * self.mu + new_observation / self.k

    def update_cusum_statistic(self):
        self.k += 1
        delta = self.signal[self.t] - self.mu
        if self.t == 0:
            self.stat[self.t] = delta
        else:
            self.stat[self.t] = self.stat[self.t - 1] + delta

    def crossed_threshold(self):
        return abs(self.stat[self.t]) > self.threshold

    def reset_detector(self):
        self.mu = self.signal[self.t]
        self.k = 1
        self.stat[self.t] = 0.0

    def entered_roi(self):
        return self.t == self.rois[self.next_roi_index].right + 1 and \
               self.next_roi_index + 1 < len(self.rois)

    def within_roi(self):
        return self.rois[self.next_roi_index].left <= self.t <= self.rois[
            self.next_roi_index].right


def make_cusum_pccf(roi_intervals: List[Roi]):
    """
    Create detector function accepting signal and sensitivity parameters only.
    """

    def func(sig, theta):
        return cusum_pccf(sig, theta, rois=roi_intervals)

    return func


def gen_sig(n=250, mu0=0.0, mu1=2.0, sigma=1.1) -> np.ndarray:
    return np.concatenate(
        (np.random.randn(n) * sigma + mu0, np.random.randn(n) * sigma +
         mu1), axis=0
    )
