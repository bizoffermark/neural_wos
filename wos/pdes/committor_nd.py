"""
    Simple ND poisson equation from "The Deep Ritz method: A deep learning-based
    numerical algorithm for solving variational problems"

    -\Delta u = 1 in \Omega
    u = 0 on \partial \Omega

    \Omega = [0,1]  [0,1] \ [0, 1)  {0}
"""

import torch

from wos.pdes.base_pde import BasePDEEquation
from wos.utils.bound import (find_closest_point_on_annulus, test_annulus_domain)
from wos.utils.sampling import sample_from_annulus, sample_in_annulus


class CommittorNd(BasePDEEquation):
    def __init__(self, n_dim: int, stop_tol: float, r_low: float, r_high: float):
        super(CommittorNd, self).__init__(n_dim=n_dim, stop_tol=stop_tol)
        assert n_dim >= 3, "n_dim must be at least 3"
        self.r_low = r_low
        self.r_high = r_high

    def domain_test_fn(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return test_annulus_domain(
            x, r_low=self.r_low, r_high=self.r_high, stop_tol=self.stop_tol
        )

    def domain_sample_fn(self, n_sample: int, device: str = "cpu") -> torch.Tensor:
        return sample_in_annulus(
            n_sample, self.n_dim, self.r_low, self.r_high, device=device
        )

    def get_closest_bound_fn(self,
                             x: torch.Tensor,
                             signed: bool=True) -> torch.Tensor:
        # get the closest boundary point
        x_closest_corner, d = find_closest_point_on_annulus(
            x, r_low=self.r_low, r_high=self.r_high, signed=signed
        )

        return x_closest_corner, d

    def domain_fn(self, x: torch.Tensor):
        return torch.zeros(x.shape[0], 1, device=x.device)

    def bound_fn(self, x: torch.Tensor) -> torch.tensor:
        r = x.norm(dim=-1, keepdim=True)
        return torch.where(r < (self.r_low + self.r_high) / 2.0, 0.0, 1.0)

    def bound_sample_fn(self, n_sample: int, device: str = "cpu") -> torch.Tensor:

        # sample from the corner
        return sample_from_annulus(
            n_sample, self.n_dim, self.r_low, self.r_high, device=device
        )

    @staticmethod
    def get_true_solution_fn(
        x: torch.Tensor, r_low: float = 1, r_high: float = 2
    ) -> torch.Tensor:
        assert r_low < r_high, "r_low must be smaller than r_high"
        n_dim = x.shape[1]
        r_low2 = r_low**2
        vx = (r_low2 - x.norm(dim=-1, keepdim=True) ** (2 - n_dim) * r_low2) / (
            r_low2 - r_high ** (2 - n_dim) * r_low2
        )
        return vx
