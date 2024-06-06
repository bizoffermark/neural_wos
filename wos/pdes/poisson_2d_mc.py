"""
    2D poisson equation from WoS paper
"""

import math

import torch

from wos.pdes.base_pde import BasePDEEquation
from wos.utils.bound import find_closest_point_on_cube, test_rect_bound
from wos.utils.sampling import sample_from_box, sample_in_box


class Poisson2d_MC(BasePDEEquation):
    '''
        2D function from MC geometry processing paper
    '''
    def __init__(self, n_dim: int, stop_tol: float):
        super(Poisson2d_MC, self).__init__(n_dim=n_dim, stop_tol=stop_tol)
        self.pde_name = "poisson_2d_mc"
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

    def get_closest_bound_fn(self,
                             x: torch.Tensor,
                             signed: bool=True) -> torch.Tensor:
        # get the closest boundary point
        x_closest_corner, d = find_closest_point_on_cube(x, self.min_max,
                                                         signed=signed)

        return x_closest_corner, d

    def domain_fn(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == 2
        return -(
            8
            * math.pi**2
            * torch.cos(2 * math.pi * x[:, 0])
            * torch.sin(2 * math.pi * x[:, 1])
        ).view(-1, 1)

    def bound_fn(self, x: torch.Tensor) -> torch.Tensor:

        assert x.shape[1] == 2
        return (
            torch.cos(2 * math.pi * x[:, 0]) * torch.sin(2 * math.pi * x[:, 1])
        ).view(-1, 1)

    def bound_sample_fn(self, n_sample: int, device: str = "cpu") -> torch.Tensor:

        # sample from the corner
        x = sample_from_box(n_sample, self.n_dim, self.min_max, device=device)
        return x

    @staticmethod
    def get_true_solution_fn(x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == 2
        ux = torch.cos(2 * math.pi * x[:, 0]) * torch.sin(2 * math.pi * x[:, 1])
        return ux.view(-1, 1)
