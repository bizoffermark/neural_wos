"""
    Define generic harmonic Green's function
"""

import math

import torch

from wos.utils.volume import volumn_sphere


def harmonic_green2d(
    x: torch.Tensor, y: torch.Tensor, R: torch.Tensor or float, eps: float = 1e-4
):
    """
    2D harmonic Green's function
    """
    assert x.shape[1] == 2 and y.shape[1] == 2
    if isinstance(R, float):
        R = R * torch.ones(x.shape[0])

    r = torch.norm(x - y, p=2, dim=1)

    g = torch.log(R / (r + eps)) / (2 * math.pi)

    g[g.isnan()] = 0
    return g.unsqueeze(1)


def compute_vol_green_offset(
    x: torch.Tensor, y: torch.Tensor, r: torch.Tensor or float, eps: float = 1e-8
):
    """
    Compute the offset of the Green's function
    together with the volume of the sphere.
    It is computed together to avoid numerical error
    """
    n_dim = x.shape[1]
    if isinstance(r, float):
        r = r * torch.ones(x.shape[0], device=x.device)

    if n_dim == 2:
        vol = volumn_sphere(r, n_dim)
        green = harmonic_green2d(x, y, r, eps)
        return -vol * green
    
    const = 1 / (n_dim * (n_dim - 2))
    dist = torch.norm(x - y, p=2, dim=1)
    dist_square = dist**2
    r_square = r**2
    offset = const * (
        dist_square * (r / (dist + eps)) ** n_dim - r_square
    ).unsqueeze(1)

    return -offset
