"""
    This file contains the implementation of the walk on spheres solver
"""

from typing import Callable

import torch
from omegaconf import DictConfig
# from pytorch_memlab import profile
from torch import nn
import numpy as np

from wos.solvers.base_solver import BaseSolver
# from line_profiler import profile

class SDESolver(BaseSolver):
    """
    Euler-Maruyama SDE solver
    """

    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        super().__init__(cfg)
        self.n_traj = self.cfg.solver.n_traj
        self.max_step = self.cfg.solver.max_step
        self.time_step = self.cfg.solver.time_step  # discretization step size 1e-2

        self.n_traj_max_per_shard = self.cfg.solver.n_traj_max_per_shard

        self.path_aug = self.cfg.solver.path_aug
        self.nn_target = self.cfg.solver.nn_target

        total_n = self.n_traj * self.n_traj_max_per_shard
        total_max = int(1e7)
        if total_n > total_max:
            self.n_traj_max_per_shard = total_max // self.n_traj
        self.sigma = self.cfg.solver.get("sigma", 2**0.5)
        self.count_grad = 0
        
        self.check_dist = True
    # @profile
    def evaluate(
        self,
        xs: torch.Tensor,
        return_std: bool = False,
        model: nn.Module = None,
        evaluate_mode: bool = True,
        grad_and_model: Callable | None = None,
    ):
        """
        Evaluate with WoS
        """
        n_sample, n_dim = xs.shape
        # check if the points are in the domain for sanity check

        n_shard = max(self.n_traj // self.n_traj_max_per_shard, 1)


        val_mean = torch.zeros([n_sample, 1], device=self.device)  # initialize the integral value
        
        if self.counter_mode:
            n_bins = 1000
            bins = np.linspace(0, 2000, n_bins+1)
            hist = np.zeros(n_bins, dtype=int)

        if return_std:
            val_mean2 = torch.zeros(
                [n_sample, 1], device=self.device
            )  # for computing std

        for i in range(n_shard):
            if i == n_shard - 1:
                n_traj_per_shard = self.n_traj - i * self.n_traj_max_per_shard
            else:
                n_traj_per_shard = self.n_traj_max_per_shard

            x_it_shard = xs.repeat(1, n_traj_per_shard).view(-1, n_dim)
            val_shard = torch.zeros(
                [x_it_shard.shape[0], 1], device=self.device
            )  # initialize the integral value
            dist_shard = torch.ones(x_it_shard.shape[0], device=self.device) * 100
            conv_shard = torch.zeros(
                x_it_shard.shape[0], dtype=torch.bool, device=self.device
            )
            conv_not_shard = torch.logical_not(conv_shard)
            if self.counter_mode:
                counter_shard = torch.zeros(
                    x_it_shard.shape[0], 1, dtype=torch.int, device=self.device
                )

            for i_step in range(self.max_step):
                _, dist_shard[conv_not_shard] = self.pde.get_closest_bound_fn(
                    x_it_shard[conv_not_shard]
                )  # initialize distance to the boundary
                conv_shard = self.pde.bound_test_fn(x=None, d=dist_shard)
                conv_not_shard = torch.logical_not(conv_shard)
                # break if all points are on the boundary
                if self.counter_mode:
                    counter_shard[conv_not_shard] += 1
                if conv_not_shard.sum() == 0:
                    break
                dW = (
                    torch.randn_like(x_it_shard[conv_not_shard], device=self.device)
                    * self.time_step**0.5
                )  # Wiener process
                val_shard[conv_not_shard] -= (
                    2
                    / self.sigma**2
                    * self.time_step
                    * self.pde.domain_fn(x_it_shard[conv_not_shard])
                )  # update the integral value
                if grad_and_model is not None:
                    self.count_grad +=1 
                    grad, _ = grad_and_model(x_it_shard[conv_not_shard])
                    val_shard[conv_not_shard] -= self.sigma * (grad * dW).sum(
                        dim=-1, keepdim=True
                    )
                x_it_shard[conv_not_shard] += dW * self.sigma  # Euler-Maruyama method
                # if self.check_dist:
                #     import wandb
                #     # breakpoint()
                #     wandb.log({"Convergence rate": conv_not_shard.sum().item() / x_it_shard.shape[0],
                #                "Step": i_step})

            if self.nn_target:
                if model is not None:
                    model.eval()
                xs_closest, ds = self.pde.get_closest_bound_fn(x_it_shard[conv_shard])
                val_shard[conv_shard] += self.pde.bound_fn(
                    xs_closest
                ) 

                val_shard[conv_not_shard] += model(x_it_shard[conv_not_shard])

            else:
                xs_closest, ds = self.pde.get_closest_bound_fn(x_it_shard)
                val_shard += self.pde.bound_fn(xs_closest)
            if return_std:
                val_mean2 += (
                    val_shard.view(-1, n_traj_per_shard).sum(dim=-1, keepdim=True) ** 2
                )
            val_mean += val_shard.view(-1, n_traj_per_shard).sum(dim=-1, keepdim=True)
            
            if self.counter_mode:
                hist += np.histogram(counter_shard.cpu().numpy(), bins)[0]

        mu = val_mean / self.n_traj
        self.count_grad = 0 # reset
        if self.counter_mode:
            return mu, hist, bins
        # check if points are outside the domain by getting statiscs of distance
        elif return_std:
            std = (val_mean2 / self.n_traj - mu**2) ** 0.5
            return xs, mu, std
        elif evaluate_mode:
            return mu
        else:
            return xs, mu
