from typing import Callable, Dict

import torch
import torch.nn as nn
# define a new loss function called DeepRitzLoss
# from pytorch_memlab import profile

from wos.solvers.base_solver import BaseSolver

# from line_profiler import profile
def DeepRitzLoss(
    model: nn.Module,
    domain_fn: Callable,  # domain function
    bound_fn: Callable,  # boundary function
    x_domain: torch.Tensor,
    x_bound: torch.Tensor,
    beta: float = 400,
):
    """Summary
    Estimate Deep-Ritz method loss function

    g = Int_{\Omega} 1/2 |grad u|^2 - f u dx

    Args:
        u (nn.Module): neural network function estimator
        f (Callable): domain function
        x (torch.Tensor): input data

    """
    x_domain.requires_grad_(True)
    ux = model(x_domain)
    fux = domain_fn(x_domain) * ux
    du_dx = torch.autograd.grad(ux.sum(), x_domain, create_graph=True)[0]
    norm_du_dx = (du_dx**2).sum(dim=-1)
    x_domain.requires_grad_(False)

    ux_bound = model(x_bound)
    fx_bound = bound_fn(x_bound)
    g = torch.mean(1 / 2 * norm_du_dx + fux) + beta * torch.mean(
        (ux_bound - fx_bound) ** 2
    )

    return g


def SDELoss(
    x: torch.Tensor,
    sde_solver: BaseSolver,
    model: nn.Module | None = None,
    grad_and_model: Callable | None = None,
    path_aug: bool = False,
    grad_target: bool = True,
    x_bound: torch.Tensor | None = None,
    bound_fn: Callable | None = None,
    beta: float | None = None,
) -> float:
    """
    Estimate the loss function for SDE
    """
    with torch.set_grad_enabled(grad_target):
        xs, *val = sde_solver.evaluate(
            x, model=model, grad_and_model=grad_and_model, evaluate_mode=False
        )

        y_pred = model(xs)
        y_true = val[0]
    loss = torch.mean((y_pred - y_true) ** 2)

    # bound loss
    if x_bound is not None:
        y_bound = model(x_bound)
        bound_val = bound_fn(x_bound)
        assert y_bound.shape == bound_val.shape
        bound_loss = torch.mean((y_bound - bound_val) ** 2)

        # weighted sum
        loss += beta * bound_loss

    return loss


def WoSLoss(
    x_domain: torch.Tensor,
    x_bound: torch.Tensor,
    wos_solver: BaseSolver,
    bound_fn: Callable,
    model: nn.Module,
    grad_and_model: nn.Module = None,
    path_aug: bool = False,
    beta: float = 500.0,
    params_domain: torch.Tensor = None,
    params_bound: torch.Tensor = None,
) -> float:
    """
    Estimate the loss function for WOS
    """

    grads = None
    if not path_aug:
        if grad_and_model is None:
            y_pred = model(x_domain, params_domain)
        else:
            grads, y_pred = grad_and_model(x_domain, params_domain)
    with torch.no_grad():
        xs, *val = wos_solver.evaluate(
            x_domain,
            model=model,
            grads=grads,
            evaluate_mode=False,
            params=params_domain,
        )
    if path_aug:
        y_true, x_random_flat = val
        if grad_and_model is not None:  # path augmentation + control variate
            grad, y_pred = grad_and_model(xs, params_domain)
            y_true -= (grad * x_random_flat).sum(dim=1, keepdim=True)
        else:
            y_pred = model(xs)
    else:
        y_true = val[0]
    if beta != 1.0:
        loss = torch.mean((y_pred - y_true) ** 2) 
    else:
        loss = (y_pred - y_true)**2
        loss = loss.sum()
    if x_bound.shape[0] > 3:
        bound_val = bound_fn(x_bound)
        y_bound = model(x_bound, params_bound)
        assert y_bound.shape == bound_val.shape
        if beta == 1.0:
            bound_loss = (y_bound - bound_val) ** 2
            loss += bound_loss.sum()
            n = x_bound.shape[0] + x_domain.shape[0]
            loss /= n # normalize the loss
        else:
            bound_loss = torch.mean((y_bound - bound_val) ** 2)
            loss += beta * bound_loss
    return loss


def PINNLoss(
    x_domain: torch.Tensor,
    x_bound: torch.Tensor,
    model: nn.Module,
    domain_fn: Callable,
    bound_fn: Callable,
    beta: float,
    y_true: torch.Tensor = None,
) -> torch.Tensor:
    """
    Estimate the loss function for PINN

    This is done by performing
    L = MSE(y, y_pred) + \int_{\Omega} (dy2/dx2 - dy_pred2/dx2)^2 dx + \int_{\partial \Omega} (y_bound - y_pred)^2 dx
    """

    x_domain.requires_grad_(True)

    y_domain = model(x_domain)
    dydx = torch.autograd.grad(y_domain.sum(), x_domain, create_graph=True)[0]
    dydx2 = 0.0
    for i in range(dydx.shape[-1]):
        dydx2 += torch.autograd.grad(
            dydx[:, i].sum(), x_domain, create_graph=True, retain_graph=True
        )[0][:, i : i + 1]
    x_domain.requires_grad_(False)

    # MSE_loss
    if y_true is not None:
        mse_loss = torch.mean((y_domain - y_true) ** 2)
    else:
        mse_loss = 0.0

    # PDE loss
    # - domain loss
    domain_val = domain_fn(x_domain)
    assert dydx2.shape == domain_val.shape
    domain_loss = torch.mean((dydx2 - domain_val) ** 2)

    # - bound loss
    y_bound = model(x_bound)
    bound_val = bound_fn(x_bound)
    assert y_bound.shape == bound_val.shape
    bound_loss = torch.mean((y_bound - bound_val) ** 2)

    # weighted sum
    loss = mse_loss + domain_loss + beta * bound_loss
    return loss


def TrajLoss(
    x_domain: torch.Tensor,
    params: torch.Tensor,
    target_u: Callable,
    target_m: Callable,
    model_u: nn.Module,
    alpha: float = 1e-6,
):
    ux = model_u(x_domain, params)  # state
    target_u = target_u(x_domain)  # target state
    mx = target_m(x_domain, params[:, 0], params[:, 1], params[:, 2])  # control

    loss = 0.5 * ((ux - target_u) ** 2).mean() + alpha / 2 * (mx**2).mean()

    return loss

