from pccf.roi_obj import Roi
from typing import List
from pccf.utils_perf import f1_score
from pccf.utils_perf import PerformanceMetrics
import numpy as np
from pccf.pccf_obj import Pccf
from loguru import logger


class PccfPerformanceMetrics(PerformanceMetrics):
    """
    Strategy: Create mapping ROI : changes.
              We can count ROIs or changes - doesn't matter - result is len() of the list with ROIs or detections.
    One ROI predicts one change.
    Don't confuse changes and detections!

    Number of ROI should be the same as number of changes <= last ROI

    If there are changes outside last ROI - we don't consider them when calculating performance, because it means that
    we didn't try to predict them.
    NO - if change is in constructor, then we attempt to predict it! - done
    """
    def __init__(self, changes: List[int] = []):
        self.changes = changes

    def f1_score(self, rois: List[Roi] = []):
        num_tps = self.tps(rois)
        num_fps = self.fps(rois)
        num_fns = self.fns(rois)
        return f1_score(tps_count=num_tps, fps_count=num_fps, fns_count=num_fns)

    def tps(self, rois: List[Roi] = []):
        """ Number if TPs """
        return len([roi for roi, changes_in_roi in self.mapping_rois_changes(rois).items() if len(changes_in_roi) > 0])

    def fps(self, rois: List[Roi] = []):
        """ Number of FPs"""
        return len([roi for roi, changes_in_roi in self.mapping_rois_changes(rois).items() if len(changes_in_roi) == 0])

    def fns(self, rois: List[Roi] = []):
        """ Number of FNs """
        captured_changes = []
        num_of_changes_within_same_roi = 0
        for roi, changes_in_roi in self.mapping_rois_changes(rois).items():
            captured_changes.extend(changes_in_roi)
            if len(changes_in_roi) > 1:
                num_of_changes_within_same_roi += len(changes_in_roi) - 1
        missed_changes = set(self.changes_attempted_to_predict(rois)).difference(set(captured_changes))
        return len(sorted(list(missed_changes))) + num_of_changes_within_same_roi

    def mapping_rois_changes(self, rois: List[Roi] = []):
        mapping = {}
        for roi in rois:
            mapping[roi] = []
            for change in self.changes_attempted_to_predict(rois):
                if change in roi:
                    mapping[roi].append(change)
        return mapping

    def changes_attempted_to_predict(self, rois: List[Roi] = []):
        turn_it_off = True
        if turn_it_off:
            return self.changes
        else:
            last_roi = sorted(rois)[-1]
            return [c for c in self.changes if c <= last_roi.right]

    def changes_not_attempted_to_predict(self, rois: List[Roi] = []):
        last_roi = sorted(rois)[-1]
        return [c for c in self.changes if c > last_roi.right]


def collect_pccf_performance(changes_in, mu_range=(5, 10), radius_range=(5, 10), save_plot=''):
    pccf_perf = PccfPerformanceMetrics(changes_in)
    mu_values = []
    f1_values = []
    for mu in range(mu_range[0], mu_range[1]):
        f1_per_mu = []
        for r in range(radius_range[0], radius_range[1]):
            pccf_predictions = Pccf(mu, radius=r).roi_intervals(len(changes_in))
            f1 = pccf_perf.f1_score(pccf_predictions)
            f1_per_mu.append(f1)
        f1_avg_over_r = np.mean(f1_per_mu)
        mu_values.append(mu)
        f1_values.append(f1_avg_over_r)
    mu_values = np.array(mu_values)
    f1_values = np.array(f1_values)
    return mu_values, f1_values
