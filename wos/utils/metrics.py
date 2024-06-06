"""
    This repository contains all metrics used for evaluation
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from wos.utils.plots import plot_hist


def compute_relative_error(
    y_domain_true: torch.Tensor,
    y_domain_pred: torch.Tensor,
    y_bound_true: torch.Tensor = None,
    y_bound_pred: torch.Tensor = None,
    eps: float = 1e-6,
) -> float:

    y_domain_pred = y_domain_pred.squeeze()
    y_domain_true = y_domain_true.squeeze()
    assert y_domain_pred.shape == y_domain_true.shape

    error_domain = torch.linalg.vector_norm(y_domain_pred - y_domain_true) / (
        torch.linalg.vector_norm(y_domain_true) + eps
    )

    if y_bound_pred is not None:
        y_bound_pred = y_bound_pred.squeeze()
        y_bound_true = y_bound_true.squeeze()

        assert y_bound_pred.shape == y_bound_true.shape
        error_bound = torch.linalg.vector_norm(y_bound_pred - y_bound_true) / (
            torch.linalg.vector_norm(y_bound_true) + eps
        )

        return error_domain, error_bound
    else:
        return error_domain

def compute_max_error(
    y_domain_true: torch.Tensor,
    y_domain_pred: torch.Tensor,
    y_bound_true: torch.Tensor = None,
    y_bound_pred: torch.Tensor = None,
) -> float:

    y_domain_pred = y_domain_pred.squeeze()
    y_domain_true = y_domain_true.squeeze()
    assert y_domain_pred.shape == y_domain_true.shape
    error_domain = (y_domain_pred - y_domain_true).abs().max()

    if y_bound_pred is not None:
        y_bound_pred = y_bound_pred.squeeze()
        y_bound_true = y_bound_true.squeeze()

        assert y_bound_pred.shape == y_bound_true.shape
        error_bound = (y_bound_pred - y_bound_true).abs().max()

        return error_domain, error_bound
    else:
        return error_domain
    


def compute_max_rel_error(
    y_domain_true: torch.Tensor,
    y_bound_true: torch.Tensor,
    y_domain_pred: torch.Tensor,
    y_bound_pred: torch.Tensor,
    eps: float = 1.0,
) -> float:

    y_domain_pred = y_domain_pred.squeeze()
    y_bound_pred = y_bound_pred.squeeze()
    y_domain_true = y_domain_true.squeeze()
    y_bound_true = y_bound_true.squeeze()

    assert y_domain_pred.shape == y_domain_true.shape
    assert y_bound_pred.shape == y_bound_true.shape
    error_domain = (
        (y_domain_pred - y_domain_true).abs() / (y_domain_true.abs() + eps)
    ).max()
    error_bound = (
        (y_bound_pred - y_bound_true).abs() / (y_bound_true.abs() + eps)
    ).max()
    return error_domain, error_bound

def compute_l2_error(
    y_domain_true: torch.Tensor,
    y_domain_pred: torch.Tensor,
    y_bound_pred: torch.Tensor = None,
    y_bound_true: torch.Tensor = None,
):

    y_domain_pred = y_domain_pred.squeeze()
    y_domain_true = y_domain_true.squeeze()
    assert y_domain_pred.shape == y_domain_true.shape
    error_domain = ((y_domain_pred - y_domain_true) ** 2).mean().sqrt()

    if y_bound_pred is not None:
        y_bound_pred = y_bound_pred.squeeze()
        y_bound_true = y_bound_true.squeeze()
        assert y_bound_pred.shape == y_bound_true.shape
        error_bound = ((y_bound_pred - y_bound_true) ** 2).mean().sqrt()

        return error_domain, error_bound
    else:
        return error_domain



def compute_density(
    y_true: torch.Tensor, y_pred: torch.Tensor, n_bins: int = 100
) -> float:
    """
    Compute histogram over random samples and compare
    density between true and predicted solutions
    """

    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
    assert y_true.shape == y_pred.shape
    bin_width = (max(y_true) - min(y_true)) / n_bins

    bins_min, bins_max = min(torch.min(y_true), torch.min(y_pred)), max(
        torch.max(y_true), torch.max(y_pred)
    )

    bins = np.arange(bins_min, bins_max + bin_width, bin_width)
    hist_true, true_edges = np.histogram(
        y_true, bins=bins, density=True
    )
    hist_pred, pred_edges = np.histogram(y_pred, bins=bins, density=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.stairs(hist_true, true_edges, label="True")
    ax.stairs(hist_pred, pred_edges, label="Pred")
    ax.legend()
    ax.grid()

    error = ((hist_true - hist_pred) ** 2).mean() ** 0.5

    return error, fig
