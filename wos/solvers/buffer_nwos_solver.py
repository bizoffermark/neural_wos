"""
    Use PINN (Physics Informed Neural Network) to solve the PDE

    This solver builds a neural network to approximate the solution of the PDE
    while the PDE condition is used as a loss function to train the neural network

"""
from typing import List
import torch
from hydra.utils import instantiate
# from wos.models.mlp import MLP
from omegaconf import DictConfig

from wos.solvers.model_base_solver import ModelBaseSolver
# from wos.utils.losses import WoSLoss
from wos.utils.buffer import TrainBuffer
from tqdm import tqdm
import time
from wos.utils.metrics import compute_l2_error, compute_max_error, compute_max_rel_error, compute_relative_error
# import wandb
# from line_profiler import profile
class BufferNWoSSolver(ModelBaseSolver):
    """
    NWoS Solver using buffer
    """

    def __init__(self, cfg: DictConfig) -> None:
        super(BufferNWoSSolver, self).__init__(cfg)

        # instantiate the WoS solver
        self.wos_solver = instantiate(
            self.cfg.solver.wos_solver, self.cfg
        )  

        # use nn_target to train the model if not
        # converged
        self.nn_target = self.cfg.solver.nn_target

        # perform control varaite using 1st order Taylor
        # expansion to reduce the variance
        self.control_variate = self.cfg.solver.control_variate

        # beta term to balance the boundary and domain loss
        self.beta = self.cfg.solver.beta

        # use boundary points to initialize the buffer
        self.bound_init = self.cfg.solver.get("bound_init", False)

        # train the model every train_period iterations 
        # => refill afte the train_period - 1 iterations
        self.train_period = self.cfg.solver.get("train_period", 100)
        # size of the buffer
        self.buffer_size = self.cfg.solver.get("buffer_size", int(4e5))
        # proportion of the buffer size to the batch size
        self.n_batch_buffer_factor = self.cfg.solver.get("n_batch_buffer_factor", 0.5)
        # proportion of the bound size to the batch size
        self.n_batch_bound_factor = self.cfg.solver.get("n_batch_bound_factor", 0.1)
        # proportion of the domain size to the batch size
        self.n_batch_domain_factor = self.cfg.solver.get(
            "n_batch_domain_factor",
            max(1.0 - self.n_batch_bound_factor - self.n_batch_buffer_factor, 0.0))

        # use true solution to fill up the buffer

        # number of trajectories
        self.n_traj = self.cfg.solver.n_traj
        # number of trajectories to use for the initial filling of the buffer
        self.n_traj_init =self.cfg.solver.n_traj_init

        self.priority_evict = self.cfg.solver.get("priority_evict", False)
        self.priority_walk = self.cfg.solver.get("priority_walk", False)
        # buffer initialization
        self.buffer = TrainBuffer(n_sample=self.buffer_size,
                                  max_step=self.wos_solver.max_step,
                                  n_traj=self.wos_solver.n_traj,
                                  n_traj_init=self.n_traj_init,
                                  n_dim=self.pde.n_dim,
                                  device=self.device,
                                  bound_init=self.bound_init,
                                  priority_evict=self.priority_evict,
                                  priority_walk=self.priority_walk)

        # number of points to train the model
        self.n_batch_train = self.cfg.solver.get("n_batch_train", 65536)

        # use top-k worst points in the buffer for walk
        self.use_topk = self.cfg.solver.get("use_topk", False)

        # initailize the training iteration
        self.iter = 0

        # use neural cache instead of nwos
        self.use_neural_cache = self.cfg.solver.get("use_neural_cache", False)

        # use warmup before using neural target
        self.warm_up_period = self.cfg.solver.get("warm_up_period", 0)

    def grad_and_model(self,
                       x: torch.tensor,
                       *args,
                       create_graph=False,
                       retain_graph=True,
                       **kwargs):
        '''
            Compute both gradient and model output

            Return dydx, y
        '''
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

    def loss(self,
             x: torch.Tensor,
             y: torch.Tensor,
             domain_mask: torch.Tensor):
        """
        Compute the loss function
        """
        assert self.model.training
        assert domain_mask is not None
        assert domain_mask.shape[0] == x.shape[0]
        y_pred = self.model(x)

        bound_mask = domain_mask.bitwise_not()
        loss = 0.0
        if domain_mask.sum() > 0:
            loss += ((y_pred[domain_mask] - y[domain_mask])**2).mean()
    
        if bound_mask.sum() > 0:
            loss_bound = ((y_pred[bound_mask] - y[bound_mask])**2).mean()
            loss += self.beta * loss_bound

        return loss

    def solve_pde(self,
                  x: torch.Tensor,
                  domain_indicator: torch.Tensor = None):
        """
        Solve the PDE by using WoS solver
        Just a wrapper to make everything looks cleaner
        """
        if domain_indicator is not None:
            x_domain = x[domain_indicator]
            x_bound = x[~domain_indicator]
        else:
            x_domain = x
            x_bound = None

        grads = None
        if self.control_variate:
            grads, _ = self.grad_and_model(x_domain)
        with torch.no_grad():
            if self.iter < self.warm_up_period:
                # Do not use grads or nn_target during warm up
                self.wos_solver.nn_target = False

                _, *val = self.wos_solver.evaluate(
                                x_domain,
                                model=None,
                                grads=None,
                                evaluate_mode=False,
                                params=None,
                            )
            else:
                if self.iter == self.warm_up_period:
                    self.wos_solver.nn_target = self.nn_target
                # TODO: Expand to support params
                _, *val = self.wos_solver.evaluate(
                    x_domain,
                    model=self.model,
                    grads=grads,
                    evaluate_mode=False,
                    params=None,
                )
                self.model.train()
        if x_bound is not None:
            y_true = torch.empty([x.shape[0], 1], device=self.device)
            y_true[domain_indicator] = val[0]
            y_true[~domain_indicator] = self.pde.bound_fn(x_bound)
        else:
            y_true = val[0]
        return y_true

    def fill_up_buffer(self):
        """
        Fill up the buffer under 2 scenarios:
        1. The buffer is full -> sample from the buffer and domain and update the buffer with new evaluations
        2. The buffer is not full -> sample from the domain/bound and update the buffer with new evaluations
        """
        if self.buffer.full:
            x, _, buffer_index, domain_indicator = self.sample(self.n_batch, train_phase = False)
            y = self.solve_pde(x, domain_indicator)
        else:
            if self.bound_init:
                x = self.pde.bound_sample_fn(self.buffer_size, device=self.device)
                y = self.pde.bound_fn(x)
                domain_indicator = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
            else:
                x = self.pde.domain_sample_fn(self.buffer_size, device=self.device)
                # use fewer trajectories to fill up the buffer initially
                self.wos_solver.n_traj = self.n_traj_init

                y = self.solve_pde(x)

                # return to the original number of trajectories
                self.wos_solver.n_traj = self.n_traj
                domain_indicator = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            assert x.shape[0] == self.buffer_size
            buffer_index = None
        self.buffer.fill(x, y, buffer_index, domain_indicator)

    def _cat_with_none(self,
                       x_domain: torch.Tensor | None,
                       x_bound: torch.Tensor | None,
                       x_buffer: torch.Tensor | None,
                       y_buffer: torch.Tensor | None,
                       domain_indicator: torch.Tensor):
        '''
            Concatenate x_domain = domain, x_bound = bound, x_buffer = buffer
        '''
        if x_domain is None:
            if x_bound is None:
                if x_buffer is None:
                    raise ValueError("All x_domain, x_bound, x_buffer are None")
                else:
                    return x_buffer, y_buffer, domain_indicator
            else:
                out_domain_mask = torch.zeros(x_bound.shape[0], dtype=torch.bool, device=x_bound.device)
                if x_buffer is None:
                    y_bound = self.pde.bound_fn(x_bound)
                    return x_bound, y_bound, out_domain_mask
                else:
                    return torch.cat([x_bound, x_buffer], dim=0), torch.cat([self.pde.bound_fn(x_bound), y_buffer], dim=0), torch.cat([out_domain_mask, domain_indicator], dim=0)
        else:
            in_domain_mask = torch.ones(x_domain.shape[0], dtype=torch.bool, device=x_domain.device)
            if x_bound is None:
                if x_buffer is None:
                    y = self.solve_pde(x_domain, in_domain_mask)
                    return x_domain, y, in_domain_mask
                else:
                    y = self.solve_pde(x_domain, in_domain_mask)
                    return torch.cat([x_domain, x_buffer], dim=0), torch.cat([y, y_buffer], dim=0), torch.cat([in_domain_mask, domain_indicator], dim=0)
            else:
                out_domain_mask = torch.zeros(x_bound.shape[0], dtype=torch.bool, device=x_bound.device)
                if x_buffer is None:
                    y_domain = self.solve_pde(x_domain, in_domain_mask)
                    y_bound = self.pde.bound_fn(x_bound)
                    return torch.cat([x_domain, x_bound], dim=0), torch.cat([y_domain, y_bound], dim=0), torch.cat([in_domain_mask, out_domain_mask], dim=0)
                else:
                    y_domain = self.solve_pde(x_domain, in_domain_mask)
                    y_bound = self.pde.bound_fn(x_bound)
                    return torch.cat([x_domain, x_bound, x_buffer], dim=0), torch.cat([y_domain, y_bound, y_buffer], dim=0), torch.cat([in_domain_mask, out_domain_mask, domain_indicator], dim=0)

    def cat_with_none(self,
                      x_domain: torch.Tensor,
                      x_bound: torch.Tensor,
                      x_buffer: torch.Tensor,
                      y_buffer: torch.Tensor,
                      buffer_index: torch.Tensor,
                      domain_indicator: torch.Tensor):
        '''
            Concatenate x_domain = domain, x_bound = bound, x_buffer = buffer
        '''
        x, y, domain_indicator = self._cat_with_none(x_domain,
                                                  x_bound,
                                                  x_buffer,
                                                  y_buffer,
                                                  domain_indicator)
        

        if buffer_index is not None:
            if x_domain is not None:
                if x_bound is not None:
                    buffer_index = torch.cat([self.buffer_size*torch.ones(x_domain.shape[0]+x_bound.shape[0],
                                                                dtype=torch.long, device=x.device),
                                    buffer_index], dim=0)
                else:
                    buffer_index = torch.cat([self.buffer_size*torch.ones(x_domain.shape[0],
                                                                dtype=torch.long, device=x.device),
                                    buffer_index], dim=0)
            else:
                if x_bound is not None:
                    buffer_index = torch.cat([self.buffer_size*torch.ones(x_bound.shape[0],
                                                                dtype=torch.long, device=x.device),
                                    buffer_index], dim=0)


        return x, y, buffer_index, domain_indicator

    def _sample(self,
                n_batch_domain,
                n_batch_bound,
                n_batch_buffer,
                y_true=None,
                train_phase=False):
        '''
            Internal function to sample points from 3 possible sources: domain, boundary, buffer
        '''
        if n_batch_domain > 0:
            x_domain = self.pde.domain_sample_fn(n_batch_domain, device=self.device)
        else:
            x_domain = None

        if n_batch_bound > 0:
            x_bound = self.pde.bound_sample_fn(n_batch_bound, device=self.device)
        else:
            x_bound = None

        if n_batch_buffer > 0:
            x_buffer, y_buffer, buffer_index, domain_indicator = self.buffer.sample(
                n_batch_buffer, y_true=y_true, train_phase=train_phase
                )
        else:
            x_buffer = None
            y_buffer = None
            buffer_index = None
            domain_indicator = None

        if (x_domain is None) and (x_bound is None) and (x_buffer is None):
            raise ValueError("All x_domain, x_bound, x_buffer are None")
        return [x_domain, x_bound, x_buffer, y_buffer, buffer_index, domain_indicator]

    def sample(self,
               n_batch: int = None,
               train_phase: bool = True) -> List[torch.Tensor]:
        """
        Sample the data
        2 possibilitiies:
        1) sampling without buffer -> sample, fill and return
        2) sampling with buffer -> sample and return (no fill)

        3 different scenarios
        2) sampling at train phase -> sample boundary and buffer points
        3) sampling at walk phase -> sample domain and buffer points

        Returning orders by are just [x_domain, x_bound, x_buffer, buffer_index]
        if the bound is not used, x_bound is None
        """


        # First Scenario
        if train_phase:
            n_batch_bound= int(self.n_batch_bound_factor * n_batch)
            n_batch_buffer = min(n_batch - n_batch_bound, self.buffer.n_domain_pts)
            n_batch_bound = n_batch - n_batch_buffer
            n_batch_domain = 0

        # Second Scenario
        else:
            n_batch_buffer = min(int(self.n_batch_buffer_factor * n_batch), self.buffer.n_domain_pts)
            n_batch_domain = n_batch - n_batch_buffer
            n_batch_bound = 0

        x_domain, x_bound, x_buffer, y_buffer, buffer_index, domain_indicator = self._sample(
            n_batch_domain,
            n_batch_bound,
            n_batch_buffer,
            y_true = self.pde.get_true_solution_fn(self.buffer.x) if self.use_topk else None,
            train_phase=train_phase
            )

            

        x, y, buffer_index, domain_indicator = self.cat_with_none(x_domain,
                                                                  x_bound,
                                                                  x_buffer,
                                                                  y_buffer,
                                                                  buffer_index,
                                                                  domain_indicator,
                                                                  )

        return x, y, buffer_index, domain_indicator

    def backward(self,
                 ):
        x, y, _, domain_mask = self.sample(self.n_batch_train, train_phase=True)
        loss = self.loss(x, y, domain_mask).float()
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

    def mem_test(self):
        '''
            Memory yest is done by 
            1) instantiating the buffer
            2) filling up the buffer initially
            3) training the model
            4) filling up the buffer again
        '''
        # First we want to ensure that buffer size is not too large
        self.buffer = TrainBuffer(n_sample=self.buffer_size,
                                  max_step=self.wos_solver.max_step,
                                  n_traj=self.wos_solver.n_traj,
                                  n_traj_init=self.n_traj_init,
                                  n_dim=self.pde.n_dim,
                                  device=self.device,
                                  bound_init=self.bound_init)

        # Second we want to ensure that buffer initialization is not too large
        self.fill_up_buffer()
        self.model.train()
        # Third we want to ensure that training over the buffer is not bad
        self.backward()

        # Fourth we want to ensure that updating the buffer is not bad
        self.fill_up_buffer()

    def train(self):
        """
        Train the model
        """
        self.model.train()

        # if self.cuda_max_mem_train_mb is not None:
        #     self.check_train_mem(fn=self.mem_test,
        #                          mem_bound_mb=self.cuda_max_mem_train_mb)

        print("memory check passed")
        self.buffer.reset() # reset the buffer
        p = 0
        if self.time_limit > -1:
            n_iters = 10000000 # just a large number
        else:
            n_iters = self.n_iters
        for i in tqdm(range(n_iters)):
            self.iter = i  ## this is needed for nwos to perform warm start...
            self.model.train()
            start_time = time.time()

            if i % self.train_period == 0: # beginning of the training period is always the walk first!
                print(f"Walking with {self.wos_solver.n_traj} trajectories")
                self.fill_up_buffer()
                if i == 0:
                    self.fill_up_buffer()
            loss_train = self.train_iters()

            end_time = time.time()

            self.cumu_time += end_time - start_time
            # check buffer value accuracy
            y_buffer_true = self.pde.get_true_solution_fn(self.buffer.x)
            buffer_loss = compute_l2_error(y_domain_pred=self.buffer.y,
                                           y_domain_true=y_buffer_true)
            buffer_max_loss = compute_max_error(y_domain_pred = self.buffer.y,
                                                y_domain_true =y_buffer_true)
            buffer_relative_loss = compute_relative_error(y_domain_pred=self.buffer.y,
                                                        y_domain_true=y_buffer_true)
            idx = torch.argmax((self.buffer.y - y_buffer_true)**2)
            buffer_max_loss_val = self.buffer.y[idx]
            buffer_max_loss_val_true = y_buffer_true[idx]
            biffer_max_loss_freq = self.buffer.freq[idx]
            self.model.eval()

            train_report = self.make_train_report(
                train_loss=loss_train,
                iteration=i,
                cumulative_time=self.cumu_time,
                time_per_iteration=(self.cumu_time / (i + 1)),
            )
            domain_perc = self.buffer.domain_percentage()
            freq_report = self.buffer.average_walked()
            train_report["domain_perc"] = domain_perc.item()
            train_report.update(freq_report)
            train_report["buffer_loss"] = buffer_loss.item()
            train_report["buffer_max_loss"] = buffer_max_loss.item()
            train_report["buffer_max_loss_val"] = buffer_max_loss_val.item()
            train_report["buffer_max_loss_val_true"] = buffer_max_loss_val_true.item()
            train_report["buffer_max_loss_freq"] = biffer_max_loss_freq.item()
            train_report["buffer_relative_loss"] = buffer_relative_loss.item()
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

            if (self.cumu_time > self.time_limit) and (self.time_limit > 0):
                print("Time limit reached")
                torch.cuda.empty_cache()
                self.validate(n_sample=self.n_sample_eval, iteration=i, cumu_time=self.cumu_time)
                break

        torch.cuda.empty_cache()
        self.validate(n_sample=self.n_sample_eval, iteration=self.n_iters, cumu_time=self.cumu_time)
