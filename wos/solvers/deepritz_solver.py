"""
    Use PINN (Physics Informed Neural Network) to solve the PDE

    This solver builds a neural network to approximate the solution of the PDE
    while the PDE condition is used as a loss function to train the neural network

"""

from typing import List

import torch
from omegaconf import DictConfig

from wos.solvers.model_base_solver import ModelBaseSolver
from wos.utils.losses import DeepRitzLoss


class DeepRitzSolver(ModelBaseSolver):
    """
    Deep Ritz Solver
    """

    def __init__(self, cfg: DictConfig) -> None:
        super(DeepRitzSolver, self).__init__(cfg)
        self.beta = self.cfg.solver.beta

    def loss(self, x: List[torch.Tensor]):
        """
        Compute the loss
        """
        assert self.model.training
        loss = DeepRitzLoss(
            model=self.model,
            domain_fn=self.pde.domain_fn,
            bound_fn=self.pde.bound_fn,
            x_domain=x[0],
            x_bound=x[1],
            beta=self.beta,
        )
        return loss
