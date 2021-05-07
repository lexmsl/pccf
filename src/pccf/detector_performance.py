"""
Performance metrics for changes and detections.
"""
import math
import numpy as np
from typing import List, Tuple, Any, Optional
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple, Union
from pccf.utils_perf import *
from pccf.roi_obj import Roi
from loguru import logger
from pccf.utils_perf import f1_score, PerformanceMetrics

# done? TODO: functions accepting ROIs should not consider detections outside ROIs


class DetectionsPerformanceMetrics(PerformanceMetrics):
    def __init__(self, changes: List[int], detections: List[int], rois: List[Roi] = None):
        self.changes = changes
        self.detections = detections
        self.rois = rois

    def tps(self):
        tps_list = fn_tps(self.changes, self.detections, self.rois)
        return len(tps_list)

    def fps(self):
        fps_list = fn_fps(self.changes, self.detections, self.rois)
        return len(fps_list)

    def fns(self):
        fns_list = fn_fns(self.changes, self.detections, self.rois)
        return len(fns_list)

    def f1_score(self):
        f1 = f1_score(self.tps(), self.fps(), self.fns())
        return f1


@dataclass
class BinaryMetrics:
    tps: List[int]
    fps: List[int]
    fns: List[int]
    delays: List[int]


def tp_fp_fn_delays(changes: List[int],
                    detections: List[int]) -> BinaryMetrics:
    changes.sort()
    detections.sort()
    n = len(detections)
    change_to_cde_map = {}
    if n == 0 or detections[0] is None:
        return BinaryMetrics([], [], changes, [])
    for i in range(0, n):
        cde = detections[i]
        lt = closest_left_change(changes, cde)
        if lt != -1:
            if lt not in change_to_cde_map:
                change_to_cde_map[lt] = cde
        tps = list(change_to_cde_map.values())
        fps = [e for e in detections if e not in tps]
        fns = [e for e in changes if e not in change_to_cde_map.keys()]
        delays = []
        for chp, cde in change_to_cde_map.items():
            delays.append(cde - chp)
        if len(delays) == 0:
            delays = [np.nan for _ in range(len(detections))]
    return BinaryMetrics(tps, fps, fns, delays)


def fn_delays(changes: List[int],
              detections: List[int],
              rois: List[Roi] = None) -> List[int]:
    if rois is not None:
        delays = []
        for roi in rois:
            r = tp_fp_fn_delays([e for e in changes if
                                 roi.left <= e <= roi.right],
                                [e for e in detections if
                                 roi.left <= e <= roi.right])
            delays.extend(r.delays)
        return delays
    else:
        r = tp_fp_fn_delays(changes, detections)
        return r.delays


def fn_tps(changes: List[int],
           detections: List[int],
           rois: List[Roi] = None) -> List[int]:
    if rois is not None:
        detections_outside_rois_exception(detections, rois)
        tps = []
        for roi in rois:
            r = tp_fp_fn_delays([e for e in changes if
                                 roi.left <= e <= roi.right],
                                [e for e in detections if
                                 roi.left <= e <= roi.right])

            tps.extend(r.tps)
        tps = sorted(list(set(tps)))
        try:
            assert len(tps) <= len(changes)
        except AssertionError:
            import pdb
            pdb.set_trace()
        return tps
    else:
        r = tp_fp_fn_delays(changes, detections)
        tps = r.tps
        try:
            assert len(tps) <= len(changes)
        except AssertionError:
            import pdb
            pdb.set_trace()
        return r.tps


def fn_fps(changes: List[int],
           detections: List[int],
           rois: List[Roi] = None) -> List[int]:
    if rois is not None:
        fps = []
        for roi in rois:
            r = tp_fp_fn_delays([e for e in changes if
                                 roi.left <= e <= roi.right],
                                [e for e in detections if
                                 roi.left <= e <= roi.right])
            if r.fps:
                fps.extend(r.fps)

        detections_outside_rois_exception(detections, rois)

        if len(fps) > 1:
            fps = list(set(fps))
        return sorted(fps)
    else:
        r = tp_fp_fn_delays(changes, detections)
        return r.fps


def fn_fns(changes: List[int],
           detections: List[int],
           rois: List[Roi] = None) -> List[int]:
    if rois is not None:
        detections_outside_rois_exception(detections, rois)
        fns = []
        for roi in rois:
            r = tp_fp_fn_delays([e for e in changes if
                                 roi.left <= e <= roi.right],
                                [e for e in detections if
                                 roi.left <= e <= roi.right])

            fns.extend(r.fns)
        changes_not_predicted = events_outside_rois(changes, rois)
        if changes_not_predicted:
            fns.extend(changes_not_predicted)
        if len(fns) > 1:
            fns = sorted(list(set(fns)))
        return fns
    else:
        r = tp_fp_fn_delays(changes, detections)
        return r.fns


def fn_f1score_vec(tps_vec: List[float],
                   fps_vec: List[float],
                   fns_vec: List[float]) -> List[float]:
    tps_vec = np.array(tps_vec)
    fps_vec = np.array(fps_vec)
    fns_vec = np.array(fns_vec)
    return tps_vec / (tps_vec + 0.5 * (fps_vec + fns_vec))


def events_outside_rois(events: List[int], rois: List[Roi]) -> \
        List[int]:
    detections_outside = events.copy()
    for cde in detections_outside:
        if any(r.left <= cde <= r.right for r in rois):
            detections_outside = list(filter(lambda x: x != cde,
                                             detections_outside))
    return sorted(list(detections_outside))


def detections_outside_rois_exception(detections: List[int], rois: List[Roi]):
    cdes_outside_rois = events_outside_rois(detections, rois)
    if len(cdes_outside_rois) > 0 and not np.isnan(cdes_outside_rois[0]):
        raise ValueError(f"There are should not be detections"
                         f" {cdes_outside_rois} outside ROIs: {rois}")


def closest_left_change(changes: List[int], detection: int):
    left = 0
    right = len(changes) - 1
    ans = -1
    while left <= right:
        mid = math.floor(left + (right - left) / 2)
        if changes[mid] <= detection:
            ans = changes[mid]
            left = mid + 1
        else:
            right = mid - 1
    return ans


def closest_right_change(changes: List[int], detection):
    left = 0
    right = len(changes) - 1
    ans = -1
    while left <= right:
        mid = math.floor(left + (right - left) / 2)
        if changes[mid] >= detection:
            ans = changes[mid]
            right = mid - 1
        else:
            left = mid + 1
    return ans


def cost_function(changes: List[int], detections: List[int]):
    """ minimize it """
    binary_metrics = tp_fp_fn_delays(changes, detections)
    num_of_detected_changes = len(changes) - len(binary_metrics.tps)
    sum_delays = sum(binary_metrics.delays)
    alpha = 0.5
    cost = alpha * num_of_detected_changes + (1.0 - alpha) * (
            len(binary_metrics.fns) + len(binary_metrics.fps) + sum_delays
    )
    num_fa = len(binary_metrics.fps)
    return cost, binary_metrics  # num_fa, sum_delays


# def gen_theta_vec(theta_min, theta_step, theta_max):
#     theta_vec = []
#     theta = theta_min
#     while theta <= theta_max:
#         theta_vec.append(theta)
#         theta += theta_step
#     return theta_vec


def collect_performance(
        detector_func: Callable,
        signal: List,
        changes: List,
        theta_vec: Union[List[float], np.ndarray],
        rois: List[Roi] = None
) -> Tuple[List[float], List[int], List[int], List[int], List[int],
           List[float]]:
    # cost_vec = []
    delays_vec = []
    fps_vec_counts = []
    fns_vec_counts = []
    tps_vec_counts = []
    f1_vec = []
    # best_so_far = 1e6
    for theta in theta_vec:
        r = detector_func(signal, theta)
        # cost, bm = cost_function(changes, r.detections)
        delays = fn_delays(changes, r.detections, rois=rois)
        fps = fn_fps(changes, r.detections, rois)
        fns = fn_fns(changes, r.detections, rois)
        tps = fn_tps(changes, r.detections, rois)
        # if cost < best_so_far:
        #     best_so_far = cost
        # cost_vec.append(cost)
        if len(delays) > 0:
            count_nonzero = np.count_nonzero(~np.isnan(delays))
            if count_nonzero > 0:
                avg_delay = np.nansum(delays) / \
                            np.count_nonzero(~np.isnan(delays))
            else:
                avg_delay = np.nan
            delays_vec.append(avg_delay)
        else:
            delays_vec.append(np.nan)
        fps_num = len(fps)
        fns_num = len(fns)
        tps_num = len(tps)
        f1 = f1_score(tps_count=tps_num, fps_count=fps_num, fns_count=fns_num)
        fps_vec_counts.append(fps_num)
        fns_vec_counts.append(fns_num)
        tps_vec_counts.append(tps_num)
        f1_vec.append(f1)

    assert len(theta_vec) == len(delays_vec)
    assert len(theta_vec) == len(fps_vec_counts)
    assert len(theta_vec) == len(fns_vec_counts)
    assert len(theta_vec) == len(tps_vec_counts)
    return delays_vec, fps_vec_counts, fns_vec_counts, tps_vec_counts, f1_vec


def collect_arl(detector_fn: Callable,
                signal: List[float],
                theta_vec: List[float],
                ):
    runs_vec = []
    for theta in theta_vec:
        r = detector_fn(signal, theta)
        detection = r.detections[0]
        runs_vec.append(detection)
    return runs_vec


# Matrix operations
#
def average_results_matrix_rows(results_matrix, count_nans_as_zero=False):
    """
    Simulation results are in rows.
    Columns correspond to theta values.
    """
    if count_nans_as_zero:
        rows_number = results_matrix.shape[0]
        average_values = np.nansum(results_matrix, axis=0) / rows_number
        # average_values = np.sum(results_matrix, axis=0) / rows_number
        return average_values
    else:
        count_not_nans_in_columns = np.sum(~np.isnan(results_matrix), axis=0)
        sum_rows = np.nansum(results_matrix, axis=0)
        average_values = sum_rows / count_not_nans_in_columns
        return average_values


def count_nans_in_columns(mat):
    return np.sum(np.isnan(mat), axis=0)
