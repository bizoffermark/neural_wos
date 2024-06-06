"""
    This repository contains a Laplace PDE with annulus domain and boundary
    
    \delta u = 0, x \in \Omega
    \delta u = 4 sin(5 \theta), x \in \partial \Omega

    \Omega = {x: 2 < |x| < 4}
"""

import numpy as np
import torch

from wos.pdes.base_pde import BasePDEEquation
from wos.utils.bound import (find_closest_point_on_annulus, test_annulus_bound,
                             test_annulus_domain)
from wos.utils.sampling import sample_from_annulus, sample_in_annulus


class LaplaceAnnulus(BasePDEEquation):
    """
    \delta u = 0, x \in \Omega
    \delta u = 4 sin(5 \theta), x \in \partial \Omega

    \Omega = {x: 2 < |x| < 4}
    """

    def __init__(self, stop_tol: float, r_low: float = 2.0, r_high: float = 4.0, n_dim: int = 2):
        super(LaplaceAnnulus, self).__init__(n_dim=2, stop_tol=stop_tol)
        self.r_low = r_low
        self.r_high = r_high
        mcd5 = np.array([[2**5, 2 ** (-5)], [4**5, 4 ** (-5)]])
        b5 = np.array([0, 4])
        c5, d5 = np.linalg.solve(mcd5, b5)
        self.c5, self.d5 = torch.tensor(c5), torch.tensor(d5)

    def domain_test_fn(self, x: torch.Tensor) -> torch.Tensor:
        return test_annulus_domain(
            x, r_low=self.r_low, r_high=self.r_high, stop_tol=self.stop_tol
        )

    def get_closest_bound_fn(self,
                             x: torch.Tensor,
                             signed: bool=True) -> torch.Tensor:
        return find_closest_point_on_annulus(x, 
                                             r_low=self.r_low, 
                                             r_high=self.r_high,
                                             signed=signed)

    def domain_fn(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros([x.shape[0],1], device=x.device)

    def bound_fn(self, x: torch.Tensor) -> torch.Tensor:
        cond, _, cond_high = test_annulus_bound(
            x, self.r_low, self.r_high, stop_tol=self.stop_tol, verbose=True
        )

        assert (
            torch.logical_not(cond).sum() == 0
        )  # all points should be on the boundary
        theta = torch.atan2(x[:, 1], x[:, 0])
        y = torch.zeros([x.shape[0], 1], device=x.device)

        angle = 5 * theta[cond_high]
        y[cond_high] = 4 * torch.sin(angle).unsqueeze(-1)
        return y

    def domain_sample_fn(self, n_sample: int, device: str = "cpu") -> torch.Tensor:
        return sample_in_annulus(
            n_sample, self.n_dim, self.r_low, self.r_high, device=device
        )

    def bound_sample_fn(self, n_sample: int, device: str = "cpu") -> torch.Tensor:

        return sample_from_annulus(
            n_sample, self.n_dim, self.r_low, self.r_high, device=device
        )

    @staticmethod
    def get_true_solution_fn(x: torch.Tensor) -> torch.Tensor:
        c5 = 4 / 1023
        d5 = -4096 / 1023

        r = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
        theta = torch.atan2(x[:, 1], x[:, 0])

        y = (c5 * r**5 + d5 * r ** (-5)) * torch.sin(5 * theta)
        y = y.unsqueeze(-1)
        return y.to(x.device)
