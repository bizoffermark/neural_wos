import time
from typing import List

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from wos.models.mlp import MLP
from wos.solvers.base_solver import BaseSolver
import wandb
from wos.utils import memory

class ModelBaseSolver(BaseSolver):
    def __init__(self, cfg: DictConfig) -> None:
        """
        A base solver class
        """
        super(ModelBaseSolver, self).__init__(cfg)

        # self.cfg.model_params.in_dim = self.cfg.pde_params.n_dim
        self.n_iters = self.cfg.solver.n_iters
        if self.solver_name == "traj_nwos":
            self.model = instantiate(
                self.cfg.solver.model, in_dim=self.cfg.solver.model.in_dim + 3
            ).to(
                self.device
            )  # insantiating the model
        else:
            if self.cfg.solver.get("test_memory", False):
                self.model = MLP(
                    in_dim = self.cfg.solver.model.in_dim,
                    hid_dim = 256,
                    out_dim=1,
                    n_hidden=6,
                    act_fn = torch.nn.GELU(),
                    use_resnet=True,
                ).to(self.device)
            else:
                self.model = instantiate(self.cfg.solver.model).to(
                    self.device
                )  # insantiating the model
        if self.cfg.optimizer.get("betas"):
            self.optimizer = instantiate(
                self.cfg.optimizer,
                params=self.model.parameters(),
                betas = tuple(self.cfg.optimizer.betas)
            )  # insantiating the optimizer
        elif self.cfg.optimizer.get("momentum"):
            self.optimizer = instantiate(
                self.cfg.optimizer,
                params=self.model.parameters(),
                momentum = self.cfg.optimizer.momentum
            )
        else:
            self.optimizer = instantiate(
                self.cfg.optimizer, params=self.model.parameters()
            )
        self.scheduler = None
        if self.cfg.get("lr_scheduler"):
            self.scheduler = instantiate(
                self.cfg.lr_scheduler, optimizer=self.optimizer
            )
        self.use_early_stopping = self.cfg.solver.use_early_stopping
        self.early_stopping = self.cfg.solver.early_stopping
        self.best_loss = float("inf")
        self.counter = 0

        self.logging_period = self.cfg.solver.logging_period

        self.best_model = self.model.state_dict()

        self.time_limit = self.cfg.solver.get("time_limit", -1)
        self.iter = 0

    def load_model(self, dir_path: str):
        """
        Load pre-trained model
        """
        with open(dir_path, "rb") as f:
            self.model = torch.load(f)

    def evaluate(self, x: torch.Tensor, *args, **kwargs):
        """
        Evaluate the model at the given points
        """
        y = self.model(x)
        return y

    def save_model(self, dir_path: str):
        """
        Save the model
        """
        with open(dir_path, "wb") as f:
            torch.save(self.model, f)

    def sample(self, n_batch: int = None) -> List[torch.Tensor]:
        """
        Sample the data from both domain and boundary
        """
        if n_batch is None:
            n_batch_domain = self.n_batch_domain
        else:
            n_batch_domain = n_batch

        x_domain = self.pde.domain_sample_fn(n_batch_domain, device=self.device)
        if self.n_batch_bound > 0:
            x_bound = self.pde.bound_sample_fn(self.n_batch_bound, device=self.device)
        else:
            x_bound = None
        return [x_domain, x_bound]

    def loss(self, x: torch.Tensor | List[torch.Tensor]):
        """
        Compute the loss
        """
        raise NotImplementedError

    def backward(self):
        x = self.sample()
        loss = self.loss(x)
        loss.backward()
        return loss

    def train_iters(self):
        """
        Train for 1 iteration
        """
        assert self.model.training
        self.optimizer.zero_grad()
        loss = self.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item()

    def check_train_mem(self, fn=None, mem_bound_mb=None, rel_margin=0.1):
        mem_bound = None if mem_bound_mb is None else mem_bound_mb * 1024**2
        if fn is None:
            fn = self.backward
        # check the memory size of a variable self.buffer
        if torch.device(self.device).type == "cuda":
            mem_train, _ = memory.max_mem_allocation(fn, device=self.device)
            memory.check_mem(
                mem_train,
                mem_bound=mem_bound,
                rel_margin=rel_margin,
                device=self.device,
            )

            # log
            # TODO: should this be moved?
            if wandb.run is not None:
                total_memory = torch.cuda.get_device_properties(
                    self.device
                ).total_memory
                wandb.run.summary["cuda_mem_avail"] = memory.byte2mb(total_memory)
                wandb.run.summary["cuda_mem_train"] = memory.byte2mb(mem_train)
        elif wandb.run is not None:
            wandb.run.summary["cuda_mem_avail"] = "cpuonly"


    def check_early_stopping(self, loss: float):
        """
        Check if the early stopping condition is satisfied
        """

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_model = self.model.state_dict()
            self.counter = 0
        else:
            if self.counter >= self.early_stopping:
                return True
            self.counter += 1
            return False

    def train(self):
        """
        Train the model
        """
        # check train memory usage
        if self.cuda_max_mem_train_mb is not None:
            self.check_train_mem(mem_bound_mb=self.cuda_max_mem_train_mb)

        p = 0
        final_iter = self.n_iters
        if self.time_limit > -1:
            n_iters = 1000000 # just a large number
        else:
            n_iters = self.n_iters
        for i in tqdm(range(n_iters)):
            self.iter = i  ## this is needed for nwos to perform warm start...
            self.model.train()
            start_time = time.time()
            loss_train = self.train_iters()
            end_time = time.time()

            self.cumu_time += end_time - start_time
            
            self.model.eval()
            train_report = self.make_train_report(
                train_loss=loss_train,
                iteration=i,
                cumulative_time=self.cumu_time,
                time_per_iteration=(self.cumu_time / (i + 1)),
            )

            if self.scheduler is not None:
                train_report["lr"] = self.optimizer.param_groups[0]["lr"]
            self.log(train_report, plots=None)

            if self.use_early_stopping:
                if self.check_early_stopping(loss_train):
                    print("Early stopping at iteration {}".format(i))
                    break
            
            if self.time_limit > -1:
                if self.cumu_time > p:
                    p += (self.time_limit / 100) # recording every 1% of the time
                    print(f"Updating time threshold to {p} at iteration {i}")
                    self.validate(n_sample=self.n_sample_eval, iteration=i, cumu_time=self.cumu_time)
            elif i % self.logging_period == 0: #or (self.time_limit > -1 and self.cumu_time > p):
                torch.cuda.empty_cache()
                self.validate(n_sample=self.n_sample_eval, iteration=i, cumu_time=self.cumu_time)

            if self.cumu_time > self.time_limit & self.time_limit > 0:
                print("Time limit reached")
                final_iter = i
                break
        
        # final update
        torch.cuda.empty_cache()
        self.validate(n_sample=self.n_sample_eval, iteration=final_iter, cumu_time=self.cumu_time)

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
