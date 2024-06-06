"""
    Simple 10d poisson equation from "The Deep Ritz method: A deep learning-based
    numerical algorithm for solving variational problems"

    -\Delta u = 0 in \Omega
    u = \Sum_{k=1:5} x_{2k-1}x_{2k} on \partial \Omega

    \Omega = (0,1)^10
    \partial \Omega = \{x: x_i = 0 or x_i = 1 for some i\}
"""

import torch

from wos.pdes.base_pde import BasePDEEquation
from wos.utils.bound import find_closest_point_on_cube, test_rect_bound
from wos.utils.sampling import sample_from_box, sample_in_box


class LaplaceNd(BasePDEEquation):
    def __init__(self, n_dim: int, stop_tol: float):
        super(LaplaceNd, self).__init__(n_dim=n_dim, stop_tol=stop_tol)
        assert self.n_dim % 2 == 0, "n_dim must be even"
        self.pde_name = "laplace_nd"
        self.min_max = [0.0, 1.0]

    def domain_test_fn(self, x: torch.Tensor) -> torch.Tensor:

        cond_in_rectangle = test_rect_bound(x, self.min_max, self.stop_tol)
        return cond_in_rectangle

    def domain_sample_fn(self, n_sample: int, device: str = "cpu") -> torch.Tensor:
        # perform rejection sampling
        x_domain = sample_in_box(n_sample, self.n_dim, self.min_max, device=device)
        return x_domain

    def get_closest_bound_fn(self, x: torch.Tensor,
                             signed: bool=True) -> torch.Tensor:
        # get the closest boundary point
        x_closest_corner, d = find_closest_point_on_cube(x,
                                                         self.min_max,
                                                         signed=signed)

        return x_closest_corner, d

    def domain_fn(self, x: torch.Tensor):
        return torch.zeros(x.shape[0], 1, device=x.device)

    def bound_fn(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        u = torch.zeros([x.shape[0], 1], device=x.device)
        for k in range(self.n_dim // 2):
            u += (x[:, 2 * k] * x[:, 2 * k + 1]).unsqueeze(1)
        return u

    def bound_sample_fn(self, n_sample: int, device: str = "cpu") -> torch.Tensor:
        # sample from the boundary
        # choose points to be either 0 or 1
        x = sample_from_box(n_sample, self.n_dim, [0.0, 1.0], device=device)
        return x

    @staticmethod
    def get_true_solution_fn(x: torch.Tensor) -> torch.Tensor:
        n_dim = x.shape[1]
        u = torch.zeros(x.shape[0], 1, device=x.device)
        for k in range(n_dim // 2):
            u += (x[:, 2 * k] * x[:, 2 * k + 1]).unsqueeze(1)
        return u
