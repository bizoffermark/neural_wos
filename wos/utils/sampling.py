"""
    Sampling methods
"""

from typing import Callable, List

import torch

from wos.utils.bound import find_closest_point_on_cube


def sample_from_sphere(
    n_sample: int,
    n_dim: int,
    r: torch.Tensor | float,
    device: str
) -> torch.Tensor:
    """
    Sample points from the sphere
    """

    if isinstance(r, float):
        r = torch.ones(n_sample, device=device) * r

    p = torch.randn([n_sample, n_dim], device=device)
    p = p / torch.norm(p, dim=-1, keepdim=True)
    assert p.shape == (n_sample, n_dim)

    return p * r.view(-1, 1)


def sample_in_sphere(
    n_sample: int, 
    n_dim: int, 
    r: torch.Tensor | float,
    device: str
) -> torch.Tensor:
    """
    Sample points in the n-dimensional hypersphere

    Implementation algorithm is from:
    http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf

    """
    if isinstance(r, float):
        r = torch.ones(n_sample, device=device) * r

    assert r.shape == (n_sample,)
    u = torch.randn(n_sample, n_dim + 2, device=device)
    norm = torch.norm(u, dim=-1, keepdim=True)
    u = u / norm
    u = u * r.view(-1, 1)

    # The first N coordinates are uniform in a unit N ball
    if n_sample == 1:
        return u[0, :n_dim]
    return u[:, :n_dim]


def sample_in_box(
    n_sample: int, n_dim: int, min_max: List[float], device: str
) -> torch.Tensor:
    """
    Sample points inside a box
    """
    x = (
        torch.rand([n_sample, n_dim], device=device) * (min_max[1] - min_max[0])
        + min_max[0]
    )
    return x


def sample_from_box(
    n_sample: int, n_dim: int, min_max: List[float], device: str
) -> torch.Tensor:
    """
    Sample points from the box (on the boundary)
    """

    x = sample_in_box(n_sample, n_dim, min_max, device=device)
    # project the points to the surface of the box

    x = find_closest_point_on_cube(x, min_max)[0]
    return x

def sample_in_annulus(
    n_sample: int, n_dim: int, r_low: float, r_high: float, device: str
) -> torch.Tensor:
    """
    Sample points inside annulus of radius r_low and r_high
    in n_dim dimensions
    """

    x = sample_from_sphere(n_sample, n_dim, 1.0, device=device)
    u = torch.rand([n_sample, 1], device=device)  # random number
    r = (u * (r_high) ** n_dim + (1 - u) * (r_low) ** n_dim) ** (1 / n_dim)

    x = x * r

    return x


def sample_from_annulus(
    n_sample: int, n_dim: int, r_low: float, r_high: float, device: str
) -> torch.Tensor:
    """
    Sample points from the annulus of radius r_low and r_high
    in n_dim dimensions
    """
    x = sample_from_sphere(n_sample, n_dim, 1.0, device=device)
    x = x / torch.norm(x, dim=-1, keepdim=True)
    u = torch.rand([n_sample, 1], device=device)  # random number
    p = (r_low ** (n_dim - 1)) / ((r_high ** (n_dim - 1)) + (r_low ** (n_dim - 1)))
    x = torch.where(u < p, r_low, r_high) * x
    return x
