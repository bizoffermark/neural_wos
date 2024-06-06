# # Laplace 10D (Done)
python3 scripts/main.py -m solver=pinn solver.n_iters=1000 pde=laplace_nd pde.n_dim=10 solver.beta=500 solver.n_batch=4096 wandb.project=final_step_comparison
python3 scripts/main.py -m solver=deepritz solver.n_iters=1000 pde=laplace_nd pde.n_dim=10 solver.beta=5000 solver.n_batch=4096 wandb.project=final_step_comparison 
python3 scripts/main.py -m solver=nsde solver.n_iters=1000 pde=laplace_nd pde.n_dim=10 solver.beta=5000 solver.n_batch=4096 solver.max_step=5 solver.time_step=1e-3 wandb.project=final_step_comparison 
python3 scripts/main.py -m solver=nwos solver.n_iters=1000 pde=laplace_nd pde.n_dim=10 solver.beta=5000 solver.n_batch=4096 solver.max_step=1,5,10 solver.n_traj=1 wandb.project=final_step_comparison
