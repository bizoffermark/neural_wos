import math

import torch
from torch._tensor import Tensor

from wos.pdes.base_pde import BasePDEEquation
from wos.utils.bound import find_closest_point_on_cube, test_rect_bound
from wos.utils.sampling import sample_from_box, sample_in_box


class TrajPDE(BasePDEEquation):
    def __init__(self, 
                 n_dim: int, 
                 stop_tol: float, 
                 alpha: float = 1e-6, 
                 scale=10):
        super(TrajPDE, self).__init__(n_dim=n_dim, stop_tol=stop_tol)
        self.alpha = alpha
        self.n_dim = 2
        self.min_max = [0.0, 1.0]

        # define range of parameters
        if self.alpha < 9e-4:
            self.gamma_range = [0.5, 1.5]
        else:
            self.gamma_range = [0.0, 1.0]
        self.delta_range = [2.5, 3.5]

        self.optim_gamma = 1.0 / (1 + 4 * self.alpha * math.pi**4)
        self.optim_delta = math.pi
        self.scale = scale
        self.optim_params = torch.tensor(
            [self.optim_gamma, self.optim_delta, self.optim_delta], device="cuda"
        ).unsqueeze(0)

    def target_u(self, x: torch.Tensor) -> torch.Tensor:
        #    gamma: float,
        #    delta: float) -> torch.Tensor:
        """
        Target function to optimize
        """
        ux = (
            self.scale
            / (2 * math.pi**2)
            * torch.sin(math.pi * x[:, 0])
            * torch.sin(math.pi * x[:, 1])
        )
        ux = ux.unsqueeze(-1)
        return ux

    def optimal_m(self,
                  x: torch.Tensor) -> torch.Tensor:
        m = self.optim_gamma * torch.sin(self.optim_delta * x[:, 0]) * torch.sin(self.optim_delta * x[:, 1])
        return m.unsqueeze(-1)

    def target_m(
        self, 
        x: torch.Tensor,
        gamma: torch.Tensor,
        delta_1: torch.Tensor,
        delta_2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Target function to optimize
        """

        # return torch.sin(delta * x[:, 0]) * torch.sin(delta * x[:, 1])
        y = self.scale * gamma * torch.sin(delta_1 * x[:, 0]) * torch.sin(delta_2 * x[:, 1])
        y = y.unsqueeze(-1)
        return y

    def domain_test_fn(self, x: torch.Tensor) -> torch.Tensor:

        cond_in_rectangle = test_rect_bound(x, self.min_max, self.stop_tol)
        cond_not_on_boundary = torch.logical_not(self.bound_test_fn(x))
        cond = torch.logical_and(cond_in_rectangle, cond_not_on_boundary)
        return cond

    def domain_sample_fn(self, n_sample: int, device: str = "cpu") -> torch.Tensor:
        # perform rejection sampling
        x_domain = sample_in_box(n_sample, self.n_dim, self.min_max, device=device)
        return x_domain

    def bound_sample_fn(self, n_sample: int, device: str = "cpu") -> torch.Tensor:

        # sample from the corner
        x = sample_from_box(n_sample, self.n_dim, self.min_max, device=device)
        return x

    def get_closest_bound_fn(self, 
                             x: torch.Tensor,
                             signed: bool=False) -> torch.Tensor:
        # get the closest boundary point
        x_closest_corner, d = find_closest_point_on_cube(x, self.min_max)

        return x_closest_corner, d

    def domain_fn(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        return -self.target_m(x, params[:, 0], params[:, 1], params[:,2])

    def bound_fn(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros([x.shape[0], 1], device=x.device)

    def sample_gamma(self, n_sample: int, device: str = "cpu") -> torch.Tensor:
        """
        Sample points from the domain
        """
        return (
            torch.rand([n_sample, 1], device=device)
            * (self.gamma_range[1] - self.gamma_range[0])
            + self.gamma_range[0]
        )

    def sample_delta(self, n_sample: int, device: str = "cpu") -> torch.Tensor:
        """
        Sample points from the domain
        """
        return (
            torch.rand([n_sample, 1], device=device)
            * (self.delta_range[1] - self.delta_range[0])
            + self.delta_range[0]
        )

    def get_true_solution_fn(self, x: torch.Tensor, alpha: float):
        #  gamma: float,
        #  delta: float) -> Tensor:

        # m_true = 1/(1 + alpha * (1/gamma)**2) * torch.sin(delta * x[:, 0]) * torch.sin(delta * x[:, 1])
        # u_true = gamma * m_true
        m_true = (
            self.scale
            / (1 + 4 * alpha * math.pi**4)
            * torch.sin(math.pi * x[:, 0])
            * torch.sin(math.pi * x[:, 1])
        )
        u_true = 1 / (2 * math.pi**2) * m_true
        return u_true.unsqueeze(-1), m_true.unsqueeze(-1)
