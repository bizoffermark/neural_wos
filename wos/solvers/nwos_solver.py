"""
    Use PINN (Physics Informed Neural Network) to solve the PDE

    This solver builds a neural network to approximate the solution of the PDE
    while the PDE condition is used as a loss function to train the neural network

"""

import torch
from hydra.utils import instantiate
# from wos.models.mlp import MLP
from omegaconf import DictConfig

from wos.pdes.base_pde import BasePDEEquation
from wos.solvers.model_base_solver import ModelBaseSolver
from wos.utils.losses import WoSLoss


class NWoSSolver(ModelBaseSolver):
    """
    NWoS Solver
    """

    def __init__(self, cfg: DictConfig) -> None:
        super(NWoSSolver, self).__init__(cfg)

        self.wos_solver = instantiate(
            self.cfg.solver.wos_solver, self.cfg
        )  # WoSSolver(self.cfg, pde)

        self.nn_target = self.cfg.solver.nn_target  # use nn_target to train the model
        # if not converged

        self.path_aug = (
            self.cfg.solver.path_aug
        )  # use data augmentation by adding intermediate
        # points on the path
        self.control_variate = self.cfg.solver.control_variate

        self.beta = self.cfg.solver.beta

    def grad_and_model(self, 
                       x, 
                       *args, 
                       create_graph=False, 
                       retain_graph=True, 
                       **kwargs):
        requires_grad = x.requires_grad
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            outputs = self.model(x, *args, **kwargs)
            gradient = torch.autograd.grad(
                outputs.sum(),
                x,
                create_graph=create_graph,
                retain_graph=retain_graph,
            )[0]
        x.requires_grad_(requires_grad)
        return gradient, outputs

    def loss(self, x: torch.Tensor):
        """
        Compute the loss function
        """
        assert self.model.training
        if self.control_variate:
            grad_and_model = self.grad_and_model
        else:
            grad_and_model = None

        loss = WoSLoss(
            x_domain=x[0],
            x_bound=x[1],
            wos_solver=self.wos_solver,
            bound_fn=self.pde.bound_fn,
            model=self.model,
            grad_and_model=grad_and_model,
            path_aug=self.path_aug,
            beta=self.beta,
        )
        return loss
