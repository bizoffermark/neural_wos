"""
    This repository contains general functions needed for finding the closest boundary point
    and testing whether a point is near the boundary
"""

from typing import List, Tuple
import torch
import math


def find_closest_point_on_rectangle(
    x: torch.Tensor, x_bound: List[float] | torch.Tensor,
    signed: bool = True # TODO: This can be just removed later
):
    """
    Find the closest point on a rectangle
    from a given point.


    The idea is to project the dimension with the closest distance
    to the boundary.
    """

    if isinstance(x_bound, list):
        assert len(x_bound) == 2
        x_bound = torch.tensor(x_bound, device=x.device)
        x_bound = torch.ones([x.shape[-1], 2], device=x.device) * x_bound

    # assert that all points are inside the rectangle

    x_mid = (x_bound[:, 0] + x_bound[:, 1]) / 2

    dist_to_mid = x - x_mid

    x_closest_each_dim = torch.empty_like(x)
    for i in range(x.shape[-1]):
        x_closest_each_dim[:, i] = torch.where(
            dist_to_mid[:, i] > 0.0, x_bound[i, 1], x_bound[i, 0]
        )

    dist_closest, dist_closest_idx = (x - x_closest_each_dim).abs().min(dim=-1)

    mask = torch.zeros_like(x, device=x.device)
    mask = mask.scatter_(1, dist_closest_idx.unsqueeze(1), 1.0)

    x_closest = x * (1 - mask) + x_closest_each_dim * mask
    # project the point to the boundary

    return x_closest, dist_closest


def find_closest_point_on_cube(x: torch.Tensor, x_bound: List[float] = [0, 1.0], signed=True):
    """
    Find the closest point on a cube
    from a given point.


    The idea is to project the dimension with the closest distance
    to the boundary.
    """
    x_closest = x.clone()

    a = torch.tensor(x_bound[0], device=x.device)
    b = torch.tensor(x_bound[1], device=x.device)
    min_val, min_idx = (x - a).min(dim=-1)
    max_val, max_idx = (b - x).min(dim=-1)
    idx = torch.where(min_val < max_val, min_idx, max_idx)
    val = torch.where(min_val < max_val, a, b)
    dist_closest = torch.where(min_val < max_val, min_val, max_val)
    x_closest[torch.arange(0, x.size(0), dtype=torch.long), idx] = val

    return x_closest, dist_closest

def find_closest_point_on_cube_from_outside(x: torch.Tensor, x_bound: List[float] = [0, 1.0], signed=True):
    '''
    Find the closest point on a cube from outside
    '''

    x_closest = x.clone()

    x_closest = torch.clamp(x_closest, x_bound[0], x_bound[1])
    dist_closest = torch.norm(x - x_closest, dim=-1)

    return x_closest, dist_closest

def find_closest_point_on_annulus(
    x: torch.Tensor, r_low: float = 2.0, r_high: float = 4.0, signed=True
):
    """
    Get the closest boundary point (signed distance naturally)
    """

    r = torch.norm(x, dim=-1)

    mid = (r_low + r_high) / 2.0

    low_idx = r < mid
    high_idx = torch.logical_not(low_idx)

    x_out = torch.empty_like(x)
    x_out[low_idx] = x[low_idx] / r[low_idx].unsqueeze(-1) * r_low
    x_out[high_idx] = x[high_idx] / r[high_idx].unsqueeze(-1) * r_high

    dist_closest = torch.empty_like(r)
    dist_closest[low_idx] = r[low_idx] - r_low  # r - r_low
    dist_closest[high_idx] = r_high - r[high_idx] # r_high - r

    return x_out, dist_closest


def test_rect_bound(
    x: torch.Tensor, min_max: List[float] | torch.Tensor, stop_tol: float = 1e-4
):
    """
    Test if a point is in the rectangle
    """
    # min_val, max_val = min_max[0], min_max[1]
    if isinstance(min_max, list):
        assert len(min_max) == 2
        min_max = torch.tensor(min_max, device=x.device)
        min_max = torch.ones([x.shape[-1], 2], device=x.device) * min_max
    # Adjust boundaries by the tolerance
    min_val, max_val = min_max[:, 0] + stop_tol, min_max[:, 1] - stop_tol

    # Check if points are within the adjusted boundaries for each dimension
    within_bounds = (x > min_val) & (x < max_val)

    # Combine conditions across all dimensions
    cond = within_bounds.all(dim=-1)

    return cond
    
def test_annulus_bound(
    x: torch.Tensor,
    r_low: float = 2.0,
    r_high: float = 4.0,
    stop_tol: float = 1e-4,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] or torch.Tensor:
    """
    Check if a point is on the boundary
    and return a boolean tensor
    """
    r = x.norm(p=2, dim=1)
    cond_low = (r - r_low) < stop_tol  # |r - r_low| < eps
    cond_high = (r_high - r) < stop_tol  # | r - r_high| < eps

    cond = torch.logical_or(cond_low, cond_high)

    if verbose:
        return cond, cond_low, cond_high
    else:
        return cond


def test_annulus_domain(
    x: torch.Tensor, r_low: float = 2.0, r_high: float = 4.0, stop_tol: float = 1e-4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] or torch.Tensor:
    """
    Check if a point is in the annulus domain
    and return a boolean tensor
    """
    r = x.norm(p=2, dim=1, keepdim=True)
    cond = torch.logical_and(
        r > r_low + stop_tol, r < r_high - stop_tol
    )  # r_low < r < r_high -> with stop_tol as tolerance range

    return cond
