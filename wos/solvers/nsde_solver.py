"""
    Use PINN (Physics Informed Neural Network) to solve the PDE

    This solver builds a neural network to approximate the solution of the PDE
    while the PDE condition is used as a loss function to train the neural network

"""

import torch
from hydra.utils import instantiate
# from wos.models.mlp import MLP
from omegaconf import DictConfig

from wos.solvers.model_base_solver import ModelBaseSolver
from wos.utils.losses import SDELoss
# from line_profiler import profile
class NSDESolver(ModelBaseSolver):
    """
    PINN Solver
    """

    def __init__(self, cfg: DictConfig) -> None:
        super(NSDESolver, self).__init__(cfg)

        self.sde_solver = instantiate(
            self.cfg.solver.sde_solver, self.cfg
        )  # WoSSolver(self.cfg, pde)

        self.nn_target = self.cfg.solver.nn_target  # use nn_target to train the model
        # if not converged

        self.path_aug = (
            self.cfg.solver.path_aug
        )  # use data augmentation by adding intermediate
        # points on the path
        self.grad_target = self.cfg.solver.get(
            "grad_target", False
        )  # use data augmentation by adding intermediate
        self.control_variate = self.cfg.solver.get("control_variate", False)
        self.beta = self.cfg.solver.get("beta")

        if self.grad_target and not (self.control_variate or self.nn_target):
            print(
                "`grad_target` does not have any effect without `control_variate` or `self.nn_target`"
            )
    
    def grad_and_model(self, x, *args, retain_graph=True, **kwargs):
        requires_grad = x.requires_grad
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            outputs = self.model(x, *args, **kwargs)
            gradient = torch.autograd.grad(
                outputs.sum(),
                x,
                create_graph=self.grad_target,
                retain_graph=retain_graph,
            )[0]
        x.requires_grad_(requires_grad)
        assert self.grad_target # this has to be always True
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
        loss = SDELoss(
            x=x[0],
            sde_solver=self.sde_solver,
            model=self.model,
            grad_and_model=grad_and_model,
            path_aug=self.path_aug,
            grad_target=self.grad_target,
            x_bound=x[1],
            bound_fn=self.pde.bound_fn,
            beta=self.beta,
        )

        return loss
