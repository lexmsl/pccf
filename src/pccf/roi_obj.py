import numpy as np


class Roi:
    def __init__(self, center, radius=3):
        self._center = center
        self._radius = radius
        self._left = None
        self._right = None

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, value):
        self._center = value
        self._left = self._center - self._radius
        self._right = self.center + self._radius

    @property
    def left(self):
        return self._center - self._radius

    @property
    def right(self):
        return self._center + self._radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value

    @classmethod
    def from_left_right(cls, left, right):
        c = np.mean([left, right])
        return cls(c, right - c)

    def __repr__(self):
        return (
            f"(l={self.left}, c={self.center},"
            f" r={self.right})"
        )

    def __hash__(self):
        return hash((self.center, self.radius, self.left, self.right))

    def __contains__(self, item):
        if isinstance(item, Roi):
            if item - self == 0:
                return True
            else:
                return False
        elif isinstance(item, int) or isinstance(item, float):
            return self.left <= item <= self.right
        else:
            raise TypeError(f"Implement __contains__ method for type {type(item)}")

    def __eq__(self, other):
        return self.left == other.left and self.center == other.center and self.right == other.right

    def __le__(self, other):
        return self.center <= other.center

    def __lt__(self, other):
        return self.center <= other.center

    def __sub__(self, other):
        if self.left >= other.right:
            return self.left - other.right
        elif self.right <= other.left:
            return other.left - self.right
        elif abs(self.center - other.center) <= self.right + other.radius:
            return 0
        else:
            return 0
