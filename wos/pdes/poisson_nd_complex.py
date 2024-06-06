import torch

from wos.pdes.base_pde import BasePDEEquation
from wos.utils.bound import find_closest_point_on_cube, find_closest_point_on_cube_from_outside,test_rect_bound
from wos.utils.sampling import sample_from_box, sample_in_box


class PoissonNd_Complex(BasePDEEquation):
    """
    N-dimensional Poisson equation with rectangular torus domain
    """

    def __init__(self, n_dim: int, stop_tol: float, k: torch.Tensor=None):

        super(PoissonNd_Complex, self).__init__(n_dim=n_dim, stop_tol=stop_tol)
        self.pde_name = "poisson_nd_complex"
        self.L_out = 1.0
        self.p = 0.75
        self.L_in = self.L_out * (1 - self.p)**(1/self.n_dim)
        self.min_max = [-self.L_out, self.L_out]
        
        self.inner_min_max = [-self.L_in, self.L_in]
        if k is None:
            self.k = 2*torch.pi*torch.ones([self.n_dim], device="cpu")
        assert self.k.shape[0] == self.n_dim

    def domain_test_fn(self, x: torch.Tensor) -> torch.Tensor:
        d = torch.abs(x)
        cond_in = test_rect_bound(x, self.inner_min_max, -self.stop_tol)
        cond_out = test_rect_bound(x, self.min_max, self.stop_tol)
        cond = torch.logical_and(torch.logical_not(cond_in), cond_out)
        return cond

    def domain_sample_fn(self, n_sample: int, device: str = "cpu") -> torch.Tensor:
        # perform rejection sampling
        x_domain = sample_in_box(n_sample, self.n_dim, self.min_max, device=device)
        in_box_indicator = (torch.abs(x_domain) < self.L_in).all(dim=-1)
        while in_box_indicator.sum() > 0:
            x_domain[in_box_indicator] = sample_in_box(in_box_indicator.sum(), self.n_dim, self.min_max, device=device)
            in_box_indicator = (torch.abs(x_domain) < self.L_in).all(dim=-1)
        return x_domain

    def get_closest_bound_fn(self, x: torch.Tensor, signed: bool=True) -> torch.Tensor:
        # get the closest boundary point
        x_closest_corner_out, d_out = find_closest_point_on_cube(x, self.min_max, signed=signed)
        x_closest_corner_in, d_in = find_closest_point_on_cube_from_outside(x, self.inner_min_max, signed=signed)
        cond = d_out < d_in
        x_closest_corner = x_closest_corner_in
        x_closest_corner[cond] = x_closest_corner_out[cond]
        d = torch.where(cond, d_out, d_in)
        return x_closest_corner, d

    def domain_fn(self, x: torch.Tensor):
        if self.k.device != x.device:
            self.k = self.k.to(x.device)
        
        return -1/self.n_dim * (self.k * self.k *torch.sin(self.k * x)).sum(dim=-1, keepdim=True)

    def bound_fn(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_true_solution_fn(x)

    def bound_sample_fn(self, n_sample: int, device: str = "cpu") -> torch.Tensor:

        # sample from the corner
        x_out = sample_from_box(n_sample, self.n_dim, self.min_max, device=device)
        x_in = sample_from_box(n_sample, self.n_dim, self.inner_min_max, device=device)

        out_weight = (self.L_out)**(self.n_dim - 1)
        in_weight = (self.L_in)**(self.n_dim - 1)
        p_out = out_weight / (out_weight + in_weight)
        cond = torch.rand(n_sample, device=device) < p_out,
        x = x_in
        x[cond] = x_out[cond]
        assert x.shape[0] == n_sample
        return x

    def get_true_solution_fn(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.n_dim
        assert self.k.shape[0] == self.n_dim
        self.k = self.k.to(x.device)
        ux = 1/self.n_dim * torch.sin(self.k * x).sum(dim=-1, keepdim=True)
        return ux
