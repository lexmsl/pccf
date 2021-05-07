import numpy as np
from scipy.stats import norm
from typing import List
from pccf.roi_obj import Roi


class Pccf:
    def __init__(self, mu: float, sigma: float = None, radius=None):
        """
        sigma is optional in case if we don't calculate probabilities
        """
        self.mu = mu
        self.sigma = sigma
        self._radius = radius

    def probabilities(self, n: int, k: int) -> np.ndarray:
        """
        Calculate Pccf as a vector of probabilities.
        :param n: number of time moments which Pccf lasts
        :param k: number of changes, determines how many N(mu, sigma) are added up
        todo: add option for starting point
        """
        if self.sigma is not None:
            change_probabilities = np.zeros(n)
            ts = np.linspace(1, n, n)
            for k in range(1, k+1):
                dist = norm(k * self.mu, np.sqrt(k) * self.sigma)
                change_probabilities += dist.pdf(ts)
            return change_probabilities
        else:
            raise TypeError(f"pccf: provide sigma estimate for Gaussian pdf")

    def roi_intervals(self, n: int, starting_point=0) -> List[Roi]:
        """
        k: number of changes to predict
        """
        if self.radius is not None:
            return [Roi(self.mu * (i + 1) + starting_point, self.radius) for i in range(n)]
        else:
            raise TypeError("pccf: provide radius")

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value

