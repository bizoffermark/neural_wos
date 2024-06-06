"""
    This repository contains volumns of different geometries
"""

import math

import torch


def volumn_sphere(
    r: torch.Tensor or float, n_dim: float, devcie: str = "cpu"
) -> torch.Tensor:
    """
    Calculate the volumn of a n-dimensional sphere
    """
    if isinstance(r, float):
        r = r * torch.ones(1, device=devcie)
    if n_dim == 1:
        vol = 2 * r
    elif n_dim == 2:
        vol = math.pi * r**2
    elif n_dim == 3:
        vol = 4 / 3 * math.pi * r**3
    else:
        vol = (math.pi ** (n_dim / 2) / math.gamma(n_dim / 2 + 1)) * r**n_dim
    return vol.unsqueeze(1)
