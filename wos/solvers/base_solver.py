import json
import logging
import math
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig

import wandb
from wos.utils import wandb as wandb_utils
from wos.utils.common import CKPT_DIR
from wos.utils.metrics import (compute_density, compute_l2_error,
                               compute_max_error, compute_max_rel_error,
                               compute_relative_error)
from wos.utils.plots import plot_step_of_wos_with_bins, plot_3d,plot_3d_sns

from wos.utils import memory
class BaseSolver:
    def __init__(self, cfg: DictConfig) -> None:
        """
        A base solver class
        """
        self.cfg = deepcopy(cfg)
        if cfg.get("num_threads"):
            torch.set_num_threads(cfg.num_threads)

        self.pde = instantiate(self.cfg.pde)
        self.solver_name = self.cfg.solver.solver_name

        self.eps = self.cfg.solver.eps
        self.device = self.cfg.solver.device
        self.stop_tol = self.cfg.solver.stop_tol

        # Device
        self.device = self.cfg.solver.get("device")
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device: torch.device | str

        self.n_batch = self.cfg.solver.n_batch
        self.n_batch_domain = self.cfg.solver.n_batch_domain
        self.n_batch_bound = self.cfg.solver.n_batch_bound

        self.model = None

        self.counter_mode = self.cfg.solver.get("counter_mode", False)

        # Set output directory
        if self.cfg.get("out_dir") is None:
            self.out_dir = Path.cwd()
        else:
            self.out_dir = Path(cfg.out_dir)

        # Seed pytorch and numpy (e.g., torchsde uses the numpy seed)
        if "seed" in self.cfg:
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

        # Logging and checkpoints
        self.plot_results: bool = self.cfg.get("plot_results", False)
        self.store_last_ckpt: bool = self.cfg.get("store_last_ckpt", False)
        self.restore_ckpt_from_wandb: bool | None = self.cfg.get(
            "restore_ckpt_from_wandb"
        )
        self.upload_ckpt_to_wandb: str | bool | None = self.cfg.get(
            "upload_ckpt_to_wandb"
        )
        if (
            isinstance(self.upload_ckpt_to_wandb, str)
            and self.upload_ckpt_to_wandb != "last"
        ):
            raise ValueError("Unknown upload mode.")
        self.eval_marginal_dims: list = self.cfg.get("eval_marginal_dims", [])

        # Paths
        self.ckpt_file: str | None = self.cfg.get("ckpt_file")
        self.ckpt_dir = self.out_dir / CKPT_DIR
        logging.info("Checkpoint directory: %s", self.ckpt_dir)
        # See https://jsonlines.org/ for the JSON Lines text file format
        self.metrics_file = self.out_dir / "metrics.json"

        # Weights & Biases
        self.wandb = wandb.run
        if self.wandb is None:
            wandb.init(mode="disabled")
        else:
            self.wandb.summary["device"] = str(self.device)

        # time
        self.initialized = True
        self.initial_time = time.time()
        self.cumu_time = 0.0

        self.n_sample_eval = self.cfg.solver.get("n_sample_eval", int(1e6))

        self.cuda_max_mem_train_mb = self.cfg.solver.get("cuda_max_mem_train_mb")

        self.use_mem_test = self.cfg.solver.get("use_mem_test", False)
        self.flop_profile = self.cfg.solver.get("flop_profile", False)

        self.make_3d_plot = self.cfg.solver.get("make_3d_plot", False)
    def evaluate(self, x: torch.Tensor, *args, **kwargs):
        """
        Evaluate the model at the given points
        """
        raise NotImplementedError


    def compute_l2_error_random(self, n_sample: int):
        """
        Compute L2 error
        """

        x_domain = self.pde.domain_sample_fn(n_sample, device=self.device)
        x_bound = self.pde.bound_sample_fn(n_sample, device=self.device)
        y_domain_true = self.pde.get_true_solution_fn(x_domain)
        y_domain_pred = self.evaluate(x_domain, evaluate_mode=True)

        y_bound_true = self.pde.get_true_solution_fn(x_bound)
        y_bound_pred = self.evaluate(x_bound, evaluate_mode=True)

        l2_error_domain, l2_error_bound = compute_l2_error(
            y_domain_true=y_domain_true,
            y_domain_pred=y_domain_pred,
            y_bound_true=y_bound_true,
            y_bound_pred=y_bound_pred,
        )
        return l2_error_domain, l2_error_bound

    def compute_relative_error_random(self, n_sample: int):
        """
        Compute random relative error
        """
        x_domain = self.pde.domain_sample_fn(n_sample, device=self.device)
        x_bound = self.pde.bound_sample_fn(n_sample, device=self.device)

        y_domain_true = self.pde.get_true_solution_fn(x_domain)
        y_domain_pred = self.evaluate(x_domain, evaluate_mode=True)

        y_bound_true = self.pde.get_true_solution_fn(x_bound)
        y_bound_pred = self.evaluate(x_bound, evaluate_mode=True)

        relative_error_domain, relative_error_bound = compute_relative_error(
            y_domain_true=y_domain_true,
            y_domain_pred=y_domain_pred,
            y_bound_true=y_bound_true,
            y_bound_pred=y_bound_pred,
        )

        return relative_error_domain, relative_error_bound

    def compute_density_random(self, n_sample: int):
        """
        Compute random density error
        """
        x_domain = self.pde.domain_sample_fn(n_sample, device=self.device)
        y_true = self.pde.get_true_solution_fn(x_domain)
        y_pred = self.evaluate(x_domain, evaluate_mode=True)

        density_error, density_fig = compute_density(y_true=y_true, y_pred=y_pred)

        return density_error, density_fig

    def compute_metrics(self, x_domain: torch.Tensor, x_bound: torch.Tensor):
        """
        Record metrics => evaluate the model at the given points
        """
        y_domain_pred = self.evaluate(x_domain, evaluate_mode=True)
        y_domain_true = self.pde.get_true_solution_fn(x_domain).cpu()

        y_bound_pred = self.evaluate(x_bound, evaluate_mode=True)
        y_bound_true = self.pde.get_true_solution_fn(x_bound).cpu()

        if self.counter_mode:
            y_domain_pred, *domain_step_counter = y_domain_pred
            y_bound_pred, *nan = y_bound_pred
        y_domain_pred = y_domain_pred.cpu()
        y_bound_pred = y_bound_pred.cpu()

        # compute L2 error, relative error, and density
        l2_error_domain, l2_error_bound = compute_l2_error(
            y_domain_true=y_domain_true,
            y_domain_pred=y_domain_pred,
            y_bound_true=y_bound_true,
            y_bound_pred=y_bound_pred,
        )

        relative_error_domain, relative_error_bound = compute_relative_error(
            y_domain_true=y_domain_true,
            y_domain_pred=y_domain_pred,
            y_bound_true=y_bound_true,
            y_bound_pred=y_bound_pred,
            eps=1e-6,
        )

        mixed_error_domain, mixed_error_bound = compute_relative_error(
            y_domain_true=y_domain_true,
            y_domain_pred=y_domain_pred,
            y_bound_true=y_bound_true,
            y_bound_pred=y_bound_pred,
            eps=1.0,
        )

        max_error_domain, max_error_bound = compute_max_error(
            y_domain_true=y_domain_true,
            y_domain_pred=y_domain_pred,
            y_bound_true=y_bound_true,
            y_bound_pred=y_bound_pred,
        )

        max_rel_error_domain, max_rel_error_bound = compute_max_rel_error(
            y_domain_true=y_domain_true,
            y_domain_pred=y_domain_pred,
            y_bound_true=y_bound_true,
            y_bound_pred=y_bound_pred,
        )

        density_error, density_fig = compute_density(
            y_true=y_domain_true, y_pred=y_domain_pred
        )

        metrics = {
            "l2_error_domain": l2_error_domain.item(),
            "l2_error_bound": l2_error_bound.item(),
            "relative_error_domain": relative_error_domain.item(),
            "relative_error_bound": relative_error_bound.item(),
            "max_error_domain": max_error_domain.item(),
            "max_error_bound": max_error_bound.item(),
            "max_rel_error_domain": max_rel_error_domain.item(),
            "max_rel_error_bound": max_rel_error_bound.item(),
            "mixed_error_domain": mixed_error_domain.item(),
            "mixed_error_bound": mixed_error_bound.item(),
            "density_error": density_error,
        }
        plots = {"density_plot": density_fig}
        if self.make_3d_plot:
            assert self.pde.n_dim == 2
            fig = plot_3d_sns(x_domain.cpu(), y_domain_pred.cpu(), save_path=None, title=self.solver_name)
            plots['3d'] = fig

        if self.counter_mode:
            counts, bins = domain_step_counter
            fig_log = plot_step_of_wos_with_bins(counts, bins, log=True)
            fig_no_log = plot_step_of_wos_with_bins(counts, bins, log=False)
            bin_midpoints = (bins[:-1] + bins[1:]) / 2

            # Calculate total number of observations
            total_count = np.sum(counts)

            # Approximate mean
            mean = np.sum(counts * bin_midpoints) / total_count

            # Approximate standard deviation
            variance = np.sum(counts * (bin_midpoints - mean) ** 2) / total_count
            std = np.sqrt(variance)

            # Approximate quantiles
            cdf = np.cumsum(counts) / total_count
            quantiles = {}
            for q in [0.1, 0.3, 0.5, 0.7, 0.9]:
                quantile_index = np.argmax(cdf >= q)
                quantiles[q] = bin_midpoints[quantile_index]
            metrics.update(
                {
                    "step_0.1_quantile": quantiles[0.1],
                    "step_0.3_quantile": quantiles[0.3],
                    "step_0.5_quantile": quantiles[0.5],
                    "step_0.7_quantile": quantiles[0.7],
                    "step_0.9_quantile": quantiles[0.9],
                }
            )
            metrics.update({"step_mean": mean, "step_std": std})
            plots.update({"step_of_wos_log": fig_log, "step_of_wos_no_log": fig_no_log})

        return metrics, plots

    def mem_test_wos(self):
        
        self.validate(n_sample=self.n_sample_eval, iteration=0, cumu_time=0)
        self.use_mem_test = False

    def check_train_mem(self, fn=None, mem_bound_mb=None, rel_margin=0.1, throw_error=True):
        mem_bound = None if mem_bound_mb is None else mem_bound_mb * 1024**2
        if fn is None:
            fn = self.mem_test_wos
        # check the memory size of a variable self.buffer
        if torch.device(self.device).type == "cuda":
            mem_train, _ = memory.max_mem_allocation(fn, device=self.device)
            memory.check_mem(
                mem_train,
                mem_bound=mem_bound,
                rel_margin=rel_margin,
                device=self.device,
                throw_error=throw_error,
            )

            # log
            if wandb.run is not None:
                total_memory = torch.cuda.get_device_properties(
                    self.device
                ).total_memory
                wandb.run.summary["cuda_mem_avail"] = memory.byte2mb(total_memory)
                wandb.run.summary["cuda_mem_train"] = memory.byte2mb(mem_train)
        elif wandb.run is not None:
            wandb.run.summary["cuda_mem_avail"] = "cpuonly"

    def validate(
        self,
        n_sample: int,
        iteration: int,
        cumu_time: float,
        params: torch.Tensor = None,
        oracle: torch.Tensor = None,
        mode: str = "valid",
    ):            
        if self.model is not None:
            self.model.eval()

        with torch.no_grad():
            if params is not None:
                # This is for parametric PDE
                if params.shape[0] == 1:
                    # using optimal params
                    params = params.expand(n_sample, -1)
                assert params.shape[0] == n_sample

                x_domain = self.pde.domain_sample_fn(n_sample, device=self.device)
                if oracle is None:
                    # using 1 optim param for testing
                    u_true, _ = self.pde.get_true_solution_fn(
                        x_domain, self.pde.alpha
                    )  # , gamma, delta)
                else:
                    u_true = oracle(x_domain, params=params)
                    # m_true = self.pde.target_m(x_domain, params[:,0], params[:,1])
                u_pred, m_pred = self.evaluate(x_domain, params, evaluate_mode=True)
                assert u_true.shape == u_pred.shape
                l2_loss_traj = (
                    0.5 * (u_pred - u_true) ** 2 + self.pde.alpha / 2 * m_pred
                ).mean()
                l2_loss_u = ((u_true - u_pred) ** 2).mean().sqrt()
                abs_loss_u = torch.abs(u_true - u_pred).mean()

                relative_loss_u = torch.linalg.vector_norm(
                    u_true - u_pred
                ) / torch.linalg.vector_norm(u_true)
                # relative_loss_m = torch.linalg.vector_norm(m_true - m_pred) / torch.linalg.vector_norm(m_true)

                metrics = {
                    f"{mode}_l2_loss_traj": l2_loss_traj.item(),
                    f"{mode}_l2_loss_u": l2_loss_u.item(),
                    #    f'{mode}_l2_loss_m': l2_loss_m.item(),
                    f"{mode}_relative_loss_u": relative_loss_u.item(),
                    #    f'{mode}_relative_loss_m': relative_loss_m.item(),
                    f"{mode}_abs_loss_u": abs_loss_u.item(),
                }
                plots = None
                if self.plot_results:
                    assert self.pde.n_dim == 2
                    fig = plot_3d(x_domain, u_pred, save_path=None)
                    plots = fig

            else:
                x_domain = self.pde.domain_sample_fn(n_sample, device=self.device)
                x_bound = self.pde.bound_sample_fn(n_sample, device=self.device)
                metrics, plots = self.compute_metrics(x_domain, x_bound)
        metrics = self.make_valid_report(iteration=iteration, metrics=metrics,
                                         cumulative_time=cumu_time)

        self.log(metrics, plots)

    def make_train_report(
        self,
        train_loss: float,
        iteration: int,
        cumulative_time: float,
        time_per_iteration: float,
        opt_loss: float = None,
    ):
        """
        Make a record for logging
        """
        report = {
            "train_loss": train_loss,
            "iteration": iteration,
            "cumulative_time": cumulative_time,
            "time_per_iteration": time_per_iteration,
        }
        if opt_loss is not None:
            report["opt_loss"] = opt_loss
        return report

    def make_valid_report(self, 
                          iteration: int, 
                          metrics: dict,
                          cumulative_time: float):
        report = {"iteration": iteration}
        report["cumulative_time"] = cumulative_time
        report.update(metrics)

        return report

    def log(self, metrics: dict, plots: dict = None):
        """
        Log the metrics
        """
        # save metrics to disk
        with self.metrics_file.open("a") as f:
            f.write(json.dumps(metrics) + "\n")

        # save plots to disk
        if plots == None:
            wandb.log(metrics)
        else:
            plots = {k: wandb_utils.format_fig(fig) for k, fig in plots.items()}
            wandb.log({**metrics, **plots})
        logging.info("Metrics:\n%s", yaml.dump(metrics))

        return metrics
