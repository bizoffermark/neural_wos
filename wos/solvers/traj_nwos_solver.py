"""
    Use PINN (Physics Informed Neural Network) to solve the PDE

    This solver builds a neural network to approximate the solution of the PDE
    while the PDE condition is used as a loss function to train the neural network

"""

import time

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from wos.solvers.model_base_solver import ModelBaseSolver
from wos.utils.losses import TrajLoss, WoSLoss


class TrajNWoSSolver(ModelBaseSolver):
    """
    PINN Solver
    """

    def __init__(self, cfg: DictConfig) -> None:
        super(TrajNWoSSolver, self).__init__(cfg)

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

        self.alpha = self.cfg.solver.alpha

        self.n_iters_opt = self.cfg.solver.get("n_iters_opt", 10)

        self.optim_params = self.pde.optim_params

    def evaluate(self, x: torch.Tensor, params: torch.Tensor, *args, **kwargs):
        """
        Evaluate the model at the given points
        """
        u = self.model(x, params)
        m = self.pde.target_m(x, params[:, 0], params[:, 1], params[:, 2])
        return u, m

    def make_params(self, n_sample: int):
        gammas = self.pde.sample_gamma(n_sample, device=self.device)
        deltas_1 = self.pde.sample_delta(n_sample, device=self.device)
        deltas_2 = self.pde.sample_delta(n_sample, device=self.device)
        params = torch.cat([gammas, deltas_1, deltas_2], dim=-1)
        return params

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

        # First Stage
        for i in tqdm(range(n_iters)):
            self.iter = i  ## this is needed for nwos to perform warm start...
            self.model.train()
            start_time = time.time()
            loss_train = self.train_iters()
            end_time = time.time()

            self.cumu_time += end_time - start_time
            # y_true_domain= self.pde.get_true_solution_fn(x)

            train_report = self.make_train_report(
                train_loss=loss_train,
                iteration=i,
                cumulative_time=self.cumu_time,
                time_per_iteration=(self.cumu_time / (i + 1)),
            )
            if self.scheduler is not None:
                train_report["lr"] = self.optimizer.param_groups[0]["lr"]
            # train_report['gamma'] = self.params[0].item()
            # train_report['delta'] = self.params[1].item()
            self.log(train_report, plots=None)

            if self.time_limit > -1:
                if self.cumu_time > p:
                    p += (self.time_limit / 100) # recording every 1% of the time
                    print(f"Updating time threshold to {p} at iteration {i}")
                    self.validate(n_sample=self.n_sample_eval,
                                  iteration=i,
                                  cumu_time = self.cumu_time,
                                  params=self.optim_params,
                    )

            elif i % self.logging_period == 0:
                self.validate(
                    n_sample=self.n_sample_eval,
                    iteration=i,
                    cumu_time = self.cumu_time,
                    params=self.optim_params,
                )

            if self.cumu_time > self.time_limit & self.time_limit > 0:
                print("Time limit reached")
                final_iter = i
                break


        # clear cuda memory
        del self.optimizer
        self.optimizer = None
        torch.cuda.empty_cache()
        # evaluate the model over random parameters
        params = self.make_params(self.n_sample_eval)
        x_domain, _ = self.sample(self.n_sample_eval)
        ## This is needed since we only have true solution for only fixed parameters, but not 
        ## for the random parameters
        self.update_wos_for_true_estimaiton()

        self.validate(
            n_sample=self.n_sample_eval,
            iteration=final_iter,
            params=params,
            cumu_time=self.cumu_time,
            oracle=self.wos_solver.evaluate,
        )

        # clear cuda memory
        del params
        torch.cuda.empty_cache()

        # save self.moel
        with open('model.tb', "wb") as f:
            torch.save(self.model, f)
        # Second stage
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        self.params = torch.nn.Parameter(
            self.make_params(1).to(self.device), requires_grad=True
        )

        self.optimizer_params = torch.optim.LBFGS(
            params=[self.params], line_search_fn="strong_wolfe", max_iter=100
        )
        initial_params = self.params.detach().clone()
        print(f"self.params: {initial_params}")
        for j in tqdm(range(self.n_iters_opt)):
            def closure():
                self.optimizer_params.zero_grad()
                loss = self.loss_traj(x_domain=x_domain)
                loss.backward()
                return loss

            loss = self.optimizer_params.step(closure)
            self.log({"lbfgs_loss": loss.item()}, plots=None)
            print(
                f"at iteration {j}, loss: {self.loss_traj(x_domain=x_domain).item()}, params: {self.params}"
            )
        with torch.no_grad():
            loss_traj = self.loss_traj(x_domain=x_domain).item()

        assert self.params.shape == self.optim_params.shape
        abs_gamma_error = (self.params[0, 0] - self.optim_params[0, 0]).abs().item()
        abs_delta_1_error = (self.params[0, 1] - self.optim_params[0, 1]).abs().item()
        abs_delta_2_error = (self.params[0, 2] - self.optim_params[0, 2]).abs().item()
        params_error = ((self.params - self.optim_params) ** 2).sum().sqrt().item()
        params_raleative_error = (
            params_error / (self.optim_params**2).sum().sqrt().item()
        )

        report = {
            "loss_traj": loss_traj,
            "params_l2_error": params_error,
            "params_raleative_l2_error": params_raleative_error,
            "abs_gamma_error": abs_gamma_error,
            "abs_delta_1_error": abs_delta_1_error,
            "abs_delta_2_error": abs_delta_2_error,
            "gamma_pred": self.params[0, 0].item(),
            "delta_1_pred": self.params[0, 1].item(),
            "delta_2_pred": self.params[0, 2].item(),
            "gamma_init": initial_params[0, 0].item(),
            "delta_1_init": initial_params[0, 1].item(),
            "delta_2_init": initial_params[0, 2].item(),
        }
        self.log(report, plots=None)

    def update_wos_for_true_estimaiton(self):
        '''
            Update WoS solver to have much higher trajectory number and max steps
            to introduce unbiased esimation.
            
            This is necessary because we tehnically do not know the true optimal solution
            of the system
        
        '''
        self.wos_solver.n_traj = 10#50000
        self.wos_solver.max_step = 10#1000
        self.wos_solver.n_traj_max_per_shard = 10
        self.wos_solver.nn_target = False

    def loss(self, x: torch.Tensor):
        """
        Compute the loss function
        """
        assert self.model.training
        if self.control_variate:
            grad_and_model = self.grad_and_model
        else:
            grad_and_model = None
        x_domain = x[0]
        x_bound = x[1]

        params_domain = self.make_params(x_domain.shape[0])
        params_bound = self.make_params(x_bound.shape[0])

        loss_wos = WoSLoss(
            x_domain=x_domain,
            x_bound=x_bound,
            wos_solver=self.wos_solver,
            bound_fn=self.pde.bound_fn,
            model=self.model,
            grad_and_model=grad_and_model,
            path_aug=self.path_aug,
            beta=self.beta,
            params_domain=params_domain,
            params_bound=params_bound,
        )
        return loss_wos

    def loss_traj(self, x_domain: torch.Tensor):
        loss_traj = TrajLoss(
            x_domain=x_domain,
            params=self.params,
            target_u=self.pde.target_u,
            target_m=self.pde.target_m,
            model_u=self.model,
            alpha=self.alpha,
        )
        return loss_traj
