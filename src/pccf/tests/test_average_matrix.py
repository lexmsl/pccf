import numpy as np
from pccf.detector_performance import average_results_matrix_rows, count_nans_in_columns


def test_m_avg():
    input = np.array([[1, 2, 3], [2, 3, 4]])
    out = average_results_matrix_rows(input)
    np.testing.assert_equal(out, np.array([1.5, 2.5, 3.5]))
    np.testing.assert_equal(average_results_matrix_rows(np.array([[1, 2, 3], [2, 3, np.nan]])),
                            np.array([1.5, 2.5, 3])
                            )

    c_nans = count_nans_in_columns(np.array([[1, 2, 3],
                                             [2, 3, np.nan]]))
    np.testing.assert_equal(c_nans, np.array([0, 0, 1]))

