"""
    Simple ND poisson equation from "The Deep Ritz method: A deep learning-based
    numerical algorithm for solving variational problems"

    -\Delta u = 1 in \Omega
    u = 0 on \partial \Omega

    \Omega = [0,1]  [0,1] \ [0, 1)  {0}
"""

import torch

from wos.pdes.base_pde import BasePDEEquation
from wos.utils.bound import find_closest_point_on_cube, test_rect_bound
from wos.utils.sampling import sample_from_box, sample_in_box


class PoissonNd(BasePDEEquation):
    """
    N-dimensional Poisson equation from DeepRitz paper
    """

    def __init__(self, n_dim: int, stop_tol: float):

        super(PoissonNd, self).__init__(n_dim=n_dim, stop_tol=stop_tol)
        self.pde_name = "poisson_nd"
        self.min_max = [0.0, 1.0]

    def domain_test_fn(self, x: torch.Tensor) -> torch.Tensor:

        cond_in_rectangle = test_rect_bound(x, self.min_max, self.stop_tol)
        cond_not_on_boundary = torch.logical_not(self.bound_test_fn(x))
        cond = torch.logical_and(cond_in_rectangle, cond_not_on_boundary)
        return cond

    def domain_sample_fn(self, n_sample: int, device: str = "cpu") -> torch.Tensor:
        # perform rejection sampling
        x_domain = sample_in_box(n_sample, self.n_dim, self.min_max, device=device)
        return x_domain

    def get_closest_bound_fn(self, x: torch.Tensor, signed: bool=True) -> torch.Tensor:
        # get the closest boundary point
        x_closest_corner, d = find_closest_point_on_cube(x, self.min_max, signed=signed)

        return x_closest_corner, d

    def domain_fn(self, x: torch.Tensor):
        return 2 * self.n_dim * torch.ones([x.shape[0], 1], device=x.device)

    def bound_fn(self, x: torch.Tensor) -> torch.Tensor:
        return (x**2).sum(dim=1, keepdim=True)

    def bound_sample_fn(self, n_sample: int, device: str = "cpu") -> torch.Tensor:

        # sample from the corner
        x = sample_from_box(n_sample, self.n_dim, self.min_max, device=device)

        return x

    @staticmethod
    def get_true_solution_fn(x: torch.Tensor) -> torch.Tensor:
        ux = (x**2).sum(dim=-1, keepdim=True)
        return ux
