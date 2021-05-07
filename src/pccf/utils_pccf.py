from pccf.utils_general import diff1
from pccf.utils_perf import *
from pccf.roi_obj import Roi
from loguru import logger
from typing import List, Tuple


def pccf_rois_adhoc(changes_list: List[int], delta: int = 10):
    """
    NOTE: To find best Pccf use collect_performance() results.


    Ad-hoc because change will be included into ROIs aposteriori,
    so that we don't need to run search / optimization.
    Given PCCF ROI locations add widths - return list of Roi objects
    """
    pccf_predictions, mu = pccf_optimal_locations(changes_list)
    rois = []
    for c, p in zip(changes_list, pccf_predictions):
        rois.append(Roi(p, radius=abs(p - c)))
    return rois


def pccf_optimal_locations(changes_list: List[int]) -> np.ndarray:
    """
    NOTE: To find best Pccf use collect_performance() results.


    Find optimal Pccf locations without ROI widths
    """
    n = len(changes_list)
    mu_optimal, mu_avg = find_mu(changes_list)
    predicted = n_points_spaced_by_mu(n, mu_optimal) + changes_list[0] - mu_optimal
    return predicted, mu_optimal


def find_mu(changes_list: List[int]) -> Tuple[float, float]:
    """ Find mu for Pccf  """
    n = len(changes_list)
    mu_analytic = np.mean(diff1(changes_list))
    min_mu, max_mu = min_max_mu(changes_list)
    print(f"find_mu(): min/max mu {min_mu}, {max_mu}")
    rmse_optimal = 99999.0
    mu_optimal = 0
    for m in range(1, int(max_mu) + 1):
        predictions = n_points_spaced_by_mu(n, m)
        error = rmse(predictions, changes_list)
        if error < rmse_optimal:
            rmse_optimal = error
            mu_optimal = m
    print_message = False
    if print_message:
        logger.info(f'Minimal RMSE: {rmse_optimal:2.2f}, mu optimal : {mu_optimal:2.2f}')
    return mu_optimal, mu_analytic


def min_max_mu(changes):
    changes_aug = changes.copy()
    changes_aug.insert(0, 0)
    min_mu = np.min(diff1(changes_aug))
    max_mu = np.max(diff1(changes_aug))
    return min_mu, max_mu


def n_points_spaced_by_mu(n: int, mu: float) -> np.ndarray:
    return mu * np.linspace(1, n, n, dtype=int)


def pad_pccf_probabilities_vector_to_rois(pccf_vector: np.ndarray,
                                          pccf_mu: float,
                                          first_roi_center: int) -> np.ndarray:
    """
    Given list of equally spaced locations return corresponding padded Pccf
    vector.
    centers[0] = 4 <=> mu = 5
    centers[0] = 6 => pccf_vec[2:] = pccf_vec[:-2]; pccf_vec[:2] = 0
    """
    padded_pccf = pccf_vector
    shift = int(first_roi_center - int(pccf_mu) + 1)
    padded_pccf[shift:] = padded_pccf[:-shift]
    padded_pccf[:shift] = 0
    return padded_pccf
