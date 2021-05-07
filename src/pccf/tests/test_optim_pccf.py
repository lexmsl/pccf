from pccf.utils_pccf import *
from pccf.roi_obj import Roi


def test_optim():
    """
    Chech that changes are always within ROIs.
    :return:
    """
    input_examples = [[10, 19, 31],
                      [100, 200, 300]]
    for input_example in input_examples:
        r = pccf_rois_adhoc(input_example, delta=3)
        for i, roi in enumerate(r):
            assert roi.left <= input_example[i] <= roi.right




