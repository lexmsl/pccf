from pccf.detector_performance import *
import numpy as np
import pytest
from pccf.roi_obj import Roi


class TestPerfMetricsDetections:
    def test_f1(self):
        assert f1_score(1, 1, 1) == 0.5
        assert DetectionsPerformanceMetrics([10, 20], [12, 21]).f1_score() > \
               DetectionsPerformanceMetrics([10, 20], [21, 22]).f1_score()

    def test_tp_fp_fn_delays(self):
        r = tp_fp_fn_delays([10, 20, 30, 40, 50], [0, 1, 11, 19, 20, 21])
        assert r == BinaryMetrics([11, 20], [0, 1, 19, 21], [30, 40, 50], [1, 0])

    def test_fps(self):
        assert fn_fps([10, 20, 30], [1]) == [1]
        assert DetectionsPerformanceMetrics([10, 20, 30], [1]).fps() == 1
        assert fn_fps([10, 20, 30], [11]) == []
        assert fn_fps([10, 20, 30], []) == []
        assert fn_fps([10, 20, 30], [0]) == [0]
        assert fn_fps([10, 20, 30], [2, 11, 12]) == [2, 12]
        assert fn_fps([10, 20, 30], [2, 11, 12, 31, 35, 37]) == [2, 12, 35, 37]
        with pytest.raises(ValueError):
            assert fn_fps([10, 20, 30], [1, 15, 16, 21], [Roi(20, 3)]) == [1, 15, 16]
            assert fn_fps([10], [5], [Roi(10, 3)]) == [5]
            assert fn_fps([10], [16], [Roi(10, 3)]) == [16]
            assert fn_fps([10], [11, 16], [Roi(10, 3)]) == [16]
            assert fn_fps([10], [13, 16], [Roi(10, 3)]) == [16]
            assert fn_fps([10], [14, 16], [Roi(10, 3)]) == [14, 16]
        assert fn_fps([10], [9], [Roi(10, 3)]) == [9]
        assert fn_fps([10], [7, 8, 9], [Roi(10, 3)]) == [7, 8, 9]
        assert fn_fps([10], [11], [Roi(10, 3)]) == []
        assert fn_fps([10], [11, 12, 13], [Roi(10, 5)]) == [12, 13]

    def test_fns(self):
        assert fn_fns([10, 20, 30], [1]) == [10, 20, 30]
        assert DetectionsPerformanceMetrics([10, 20, 30], [1]).fns() == 3
        assert fn_fns([10, 20, 30], []) == [10, 20, 30]
        assert fn_fns([10, 20, 30], [11]) == [20, 30]
        assert fn_fns([10, 20, 30], [11, 29]) == [30]
        assert fn_fns([150], [11]) == [150]
        assert fn_fns([150], [151]) == []

        with pytest.raises(ValueError):
            assert fn_fns([10], [16], [Roi(10, 3)]) == [10]
        assert fn_fns([10], [13], [Roi(10, 3)]) == []
        assert fn_fns([10], [9], [Roi(10, 3)]) == [10]
        assert fn_fns([10], [7, 8, 9], [Roi(10, 3)]) == [10]
        assert fn_fns([10], [7, 8, 9, 11], [Roi(10, 3)]) == []
        assert fn_fns([10, 20], [7, 8, 9, 11], [Roi(10, 3)]) == [20]
        assert fn_fns([10, 20, 30], [7, 8, 9, 11], [Roi(10, 3)]) == [20, 30]

    def test_detections_out_rois(self):
        assert events_outside_rois([1, 2, 3, 4, 5, 6, 7],
                                   [Roi(5, 3)]) == [1]
        r = events_outside_rois([1, 6, 9, 14, 19, 21, 24, 25],
                                [Roi(10, 3), Roi(20, 4)])
        assert r == [1, 6, 14, 25]

    def test_tps(self):
        assert fn_tps([10, 20, 30], [1]) == []
        assert fn_tps([10, 20, 30], [9]) == []
        assert fn_tps([10, 20, 30],
                      [11, 12, 13, 20, 21, 29, 34]) == [11, 20, 34]

        with pytest.raises(ValueError):
            assert fn_tps([10, 20, 30], [22, 25], [Roi(20, 2)]) == [22]

        assert fn_tps([10, 20, 30], [19, 21, 22], [Roi(20, 2)]) == [21]
        assert DetectionsPerformanceMetrics([10, 20, 30], [19, 21, 22], [Roi(20, 2)]).tps() == 1
        assert fn_tps([10, 20, 30],
                      [7, 9, 12, 18, 20, 21],
                      [Roi(10, 3), Roi(20, 2)]) == [12, 20]

    def test_paper_examples(self):
        changes = [100, 200]
        detections = [80, 90, 110, 120, 216]
        detections_roi = [80, 90, 110, 120, 216]
        roi = [Roi(100, 15), Roi(200, 15)]

        assert fn_tps(changes, detections) == [110, 216]
        assert fn_fps(changes, detections) == [80, 90, 120]
        assert fn_fns(changes, detections) == []

        f1_score = fn_f1score_vec(len(fn_tps(changes, detections)),
                                  len(fn_fps(changes, detections)),
                                  len(fn_fns(changes, detections)))

        assert f1_score == 2.0 / (2.0 + 0.5 * (3 + 0))

    def test_roi_obj(self):
        roi = Roi(100, 2)
        assert roi.left == 98
        assert roi.right == 102

    def test_sequence(self):
        changes = [10]
        detections = [1, 2, 3, 4, 5, 6, 7, 8, 9,
                      12, 13, 14, 15, 16, 17, 18, 19, 20]
        assert fn_tps(changes, detections) == [12]

        assert fn_fps(changes, detections) == [1, 2, 3, 4, 5, 6, 7, 8, 9,
                                               13, 14, 15, 16, 17, 18, 19, 20]

        assert fn_fns(changes, detections) == []

    def test_delays(self):
        assert fn_delays([351, 701], [341, 691, 706]) == [340, 5]
        assert fn_delays([351, 701], [341, 691, 706],
                         rois=[Roi(701, 10)]) == [5]
        assert fn_delays([10, 20, 30], [1]) == [np.nan]
        assert fn_delays([10, 20, 30], []) == []
        assert fn_delays([10, 20, 30], [1, 2, 3]) == [np.nan, np.nan, np.nan]
        assert fn_delays([10, 20, 30], [1, 10]) == [0]
        assert fn_delays([10, 20, 30], [1, 11]) == [1]
        assert fn_delays([10, 20, 30], [10, 20, 30]) == [0, 0, 0]
        assert fn_delays([10, 20, 30], []) == []
        assert fn_delays([10, 20, 30], [11, 12, 13]) == [1]
        print(fn_delays([351, 701], [4, 6, 23, 35, 38, 51, 68, 76, 86, 90,
                                     97, 113, 128, 132, 158, 172, 176, 186,
                                     198, 208, 227, 235, 247, 255, 263, 308,
                                     314, 324, 330, 350, 360, 373, 395, 400,
                                     414, 420, 427, 444, 473, 477, 485, 495,
                                     503, 522, 532, 544, 547, 583, 602, 617,
                                     631, 636, 642, 655, 662, 675, 686, 691,
                                     700, 724, 738, 746, 751, 767, 777, 783,
                                     787, 793, 815, 837, 879, 883, 888, 899,
                                     912, 927, 958, 968, 988, 1009, 1021,
                                     1032, 1041, 1047]))


def test_average_matrix():
    input_example = np.vstack([[1, 2, np.nan],
                               [2, 4, 3]])
    expected = np.array([[1.5, 3, 3]])
    expected_nan_zero = np.array([[1.5, 3, 1.5]])
    assert (average_results_matrix_rows(input_example) == expected).all()
    assert (average_results_matrix_rows(input_example,
                                        count_nans_as_zero=True) ==
            expected_nan_zero).all()

# TestPerfMetrics().test_paper_examples()
# TestPerfMetrics().test_tps()
# TestPerfMetrics().test_fps()
# TestPerfMetrics().test_fns()
# TestPerfMetrics().test_detections_out_rois()
