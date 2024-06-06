"""
    Simple 2D poisson equation from "The Deep Ritz method: A deep learning-based
    numerical algorithm for solving variational problems"

    -\Delta u = 1 in \Omega
    u = 0 on \partial \Omega

    \Omega = [0,1] x [0,1] \ [0, 1) x {0}
"""

import torch

from wos.pdes.base_pde import BasePDEEquation
from wos.utils.bound import find_closest_point_on_cube, test_rect_bound
from wos.utils.sampling import sample_from_box, sample_in_box


class Poisson2d(BasePDEEquation):
    def __init__(self, n_dim: int, stop_tol: float):
        super(Poisson2d, self).__init__(n_dim=n_dim, stop_tol=stop_tol)
        self.n_dim = 2
        self.pde_name = "poisson_2d"
        self.min_max = [-1.0, 1.0]

    def domain_test_fn(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Fix this to be the same as bound_test_fn, which take in just
        cond_in_rectangle = test_rect_bound(x, self.min_max, self.stop_tol)
        cond_not_on_boundary = torch.logical_not(self.bound_test_fn(x))
        cond = torch.logical_and(cond_in_rectangle, cond_not_on_boundary)
        return cond

    def domain_sample_fn(self, n_sample: int, device: str = "cpu") -> torch.Tensor:
        # perform rejection sampling
        x_domain = sample_in_box(n_sample, self.n_dim, self.min_max, device=device)
        return x_domain

    def get_closest_bound_fn(
        self,
        x: torch.Tensor,
        signed: bool = True
    ) -> torch.Tensor:
        # project x to [0, 1] x [0]
        x_closest_line = x.clone()
        x_closest_line[:, 0] = torch.clamp(x_closest_line[:, 0], 0.0, 1.0)
        x_closest_line[:, 1] = 0.0
        d_line = torch.norm(x - x_closest_line, dim=1)

        # project x to [-1, 1] x [-1, 1]
        x_closest_corner, d_corner = find_closest_point_on_cube(x, self.min_max,signed=signed)

        x_closest = torch.empty_like(x_closest_line, device=x.device)
        for i in range(x.shape[1]):
            x_closest[:, i] = torch.where(
                d_line < d_corner, x_closest_line[:, i], x_closest_corner[:, i]
            )
        d = torch.where(d_line < d_corner, d_line, d_corner)

        return x_closest, d

    def domain_fn(self, x: torch.Tensor):
        return -torch.ones(x.shape[0], 1, device=x.device, dtype=torch.float32)

    def bound_fn(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 1, device=x.device, dtype=torch.float32)

    def bound_sample_fn(self, n_sample: int, device: str = "cpu") -> torch.Tensor:

        # sample from the boundary
        x = sample_from_box(n_sample, self.n_dim, self.min_max, device=device)
        return x

    @staticmethod
    def get_true_solution_fn(x: torch.Tensor) -> torch.Tensor:
        raise "No true solution for Poisson2d"
