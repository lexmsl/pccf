import pytest
from pccf.pccf_obj import Pccf
from pccf.roi_obj import Roi
from pccf.utils_pccf import pad_pccf_probabilities_vector_to_rois


def test_roi_objects():
    r = Roi(9, 3)
    assert 9 in r
    assert 6 in r
    assert 12 in r
    assert 13 not in r


def test_pccf_obj():
    pccf = Pccf(10.0)
    assert pccf.radius is None
    pccf.radius = 1
    assert pccf.radius == 1


def find_first_extremum(pccf_vec):
    """
    Find first extremum location.
    """
    for i in range(len(pccf_vec)):
        if i == 0:
            continue
        if pccf_vec[i] < pccf_vec[i-1]:
            return i - 1


def test_pccf_properties():
    """
    test Pccf properties
    """
    p = Pccf(10, 1).probabilities(100, 5)
    assert len(p) == 100
    assert find_first_extremum(p) == 9


def test_padded_pccf_props():
    p = Pccf(10, 1).probabilities(100, 5)
    rois_centers = [12, 22, 32]
    padded = pad_pccf_probabilities_vector_to_rois(p, 10, rois_centers[0])
    assert find_first_extremum(padded) == 12


def test_pccf_predictions():
    pccf = Pccf(mu=10, radius=3)
    rois = pccf.roi_intervals(n=1)
    assert rois == [Roi(10, 3)]
    rois = pccf.roi_intervals(n=2)
    assert rois == [Roi(10, 3), Roi(20, 3)]


def test_probabilities_predictions():
    pccf = Pccf(10)
    with pytest.raises(TypeError):
        pccf.probabilities(100, 7)
