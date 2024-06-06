"""
    This file contains the implementation of the walk on spheres solver
"""


# from pytorch_memlab import profile
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from wos.solvers.base_solver import BaseSolver
from wos.utils.green_fn import compute_vol_green_offset
from wos.utils.sampling import sample_from_sphere, sample_in_sphere


# from line_profiler import profile
class WoSSolver(BaseSolver):
    """
    Walk on spheres Laplace solver
    """

    def __init__(self, cfg: DictConfig) -> None:
        super(WoSSolver, self).__init__(cfg)

        self.n_traj = self.cfg.solver.n_traj
        self.max_step = self.cfg.solver.max_step

        self.nn_target = self.cfg.solver.nn_target
        self.n_traj_max_per_shard = self.cfg.solver.n_traj_max_per_shard


        # Only needed to test how much neural target would help
        # self.use_target_true = self.cfg.solver.get("use_target_true", False)
    def evaluate(
        self,
        xs: torch.Tensor,
        return_std: bool = False,
        grads: torch.Tensor = None,
        model: nn.Module = None,
        evaluate_mode: bool = True,
        params: torch.Tensor = None,
    ):
        """
        Evaluate with WoS
        """

        n_sample, n_dim = xs.shape
        # check if the points are in the domain for sanity check
        # _ = pde.domain_test_fn(xs)
        n_shard = max(1, self.n_traj // self.n_traj_max_per_shard)
        val_mean = torch.zeros(
            [n_sample, 1], device=self.device
        )  # initialize the integral value

        if self.counter_mode:
            n_bins = 1000
            bins = np.linspace(0, 2000, n_bins + 1)
            hist = np.zeros(n_bins, dtype=int)

        if return_std:
            val_mean2 = torch.zeros(
                [n_sample, 1], device=self.device
            )  # for computing std

        for i in range(n_shard):
            # iterate until all points are on the boundary or the
            # maximum number of steps is reached
            if i > 0 and self.use_mem_test:
                break
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
            if params is not None:
                params_shard = params.repeat(1, n_traj_per_shard).view(
                    -1, params.shape[1]
                )
            if self.counter_mode:
                counter_shard = torch.zeros(
                    x_it_shard.shape[0], 1, dtype=torch.int, device=self.device
                )
            for step in range(self.max_step):
                _, dist_shard[conv_not_shard] = self.pde.get_closest_bound_fn(
                    x_it_shard[conv_not_shard],
                    signed = False
                )  # initialize distance to the boundary
                conv_shard = self.pde.bound_test_fn(d=dist_shard)
                conv_not_shard = torch.logical_not(conv_shard)
                if self.counter_mode:
                    counter_shard[conv_not_shard.cpu()] += 1
                # break if all points are on the boundary
                if conv_not_shard.sum() == 0:
                    break

                # sample a random direction inside the sphere
                y_random_dir = sample_in_sphere(
                    conv_not_shard.sum(),
                    n_dim,
                    dist_shard[conv_not_shard],
                    device=self.device,
                )  # sample from the sphere

                y_random = (
                    y_random_dir + x_it_shard[conv_not_shard]
                )  # move the point to the
                # boundary
                # compute source term contribution
                if params is not None:
                    domain = self.pde.domain_fn(y_random, params_shard[conv_not_shard])
                else:
                    domain = self.pde.domain_fn(y_random)

                if domain.abs().sum() != 0:
                    # compute Green's function offset => only do it if there is a contribution so that we save a little bit of computation
                    vol_green = compute_vol_green_offset(
                        x_it_shard[conv_not_shard],
                        y_random,
                        dist_shard[conv_not_shard],
                        eps=self.eps,
                    )
                    green_offset = domain * vol_green
                    val_shard[
                        conv_not_shard
                    ] += green_offset  # add the contribution to the integral

                # move the points to the boundary
                x_random_dir = sample_from_sphere(
                    conv_not_shard.sum(),
                    n_dim,
                    dist_shard[conv_not_shard],
                    device=self.device,
                )

                if step == 0 and (grads is not None):
                    val_shard[conv_not_shard] -= (
                        grads.repeat(1, n_traj_per_shard).view(-1, n_dim)[
                            conv_not_shard
                        ]
                        * x_random_dir
                    ).sum(dim=1, keepdim=True)
                x_it_shard[conv_not_shard] += x_random_dir

            if self.nn_target:
                model.eval()
                xs_closest, _ = self.pde.get_closest_bound_fn(x_it_shard[conv_shard], signed = False)
                val_shard[conv_shard] += self.pde.bound_fn(
                    xs_closest
                )  # add the boundary value to the integral

                if params is not None:
                    val_shard[conv_not_shard] += model(
                        x_it_shard[conv_not_shard], params_shard[conv_not_shard]
                    ) # only relevant for the trajectory optimization
                else:
                    val_shard[conv_not_shard] += model(x_it_shard[conv_not_shard])
            else:
                xs_closest, _ = self.pde.get_closest_bound_fn(x_it_shard, signed = False)
                u_bound = self.pde.bound_fn(xs_closest)

                val_shard += u_bound
            val_mean += val_shard.view(-1, n_traj_per_shard).sum(
                dim=1, keepdim=True
            )

            if return_std:
                val_mean2 += (val_shard.view(-1, n_traj_per_shard) ** 2).sum(
                    dim=1, keepdim=True
                )

            if self.counter_mode:
                hist += np.histogram(counter_shard.cpu().numpy(), bins)[0]
        mu = val_mean / self.n_traj

        if self.counter_mode:
            return mu, hist, bins
        # Return std as well as mean
        elif return_std:
            assert mu.shape == val_mean2.shape
            std = (val_mean2 / self.n_traj - mu**2) ** 0.5
            return xs, mu, std

        # Return only mean
        elif evaluate_mode:
            return mu

        # Default
        return xs, mu
