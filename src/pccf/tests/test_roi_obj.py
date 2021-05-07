from pccf.roi_obj import Roi


def test_roi_obj():
    r1 = Roi(10, radius=3)
    assert r1.left == 7
    assert r1.right == 13
    r2 = Roi(12, radius=3)
    assert r2 in r1
    assert Roi(10, 1) in Roi(12, 1)
    assert Roi(10, 1) not in Roi(13, 1)
    assert Roi.from_left_right(5, 10) in Roi.from_left_right(9, 20)
    assert Roi.from_left_right(5, 10) not in Roi.from_left_right(11, 20)
    assert Roi.from_left_right(5, 10).radius == 2.5


def test_roi_setters():
    roi0 = Roi(10)
    roi0.radius = 5
    assert roi0.center == 10
    assert roi0.left == 5
    roi0.center = 20
    assert roi0.center == 20
    assert roi0.radius == 5
    assert roi0.right == 25
