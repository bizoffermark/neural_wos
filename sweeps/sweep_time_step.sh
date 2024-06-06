python3 scripts/main.py -m solver=pinn pde=laplace_nd pde.n_dim=10 solver.beta=500 solver.n_batch=4096 solver.n_iters=1000 wandb.project=final_time_step
python3 scripts/main.py -m solver=deepritz pde=laplace_nd pde.n_dim=10 solver.beta=5000 solver.n_batch=8192 solver.n_iters=1000 wandb.project=final_time_step 
python3 scripts/main.py -m solver=nsde pde=laplace_nd pde.n_dim=10 solver.beta=5000 solver.n_batch=16384 solver.max_step=1 solver.time_step=1e-3 solver.n_iters=1000 wandb.project=final_time_step 
python3 scripts/main.py -m solver=wos pde=laplace_nd pde.n_dim=10 solver.max_step=100 solver.n_traj=10000 solver.n_iters=1000 wandb.project=final_time_step
python3 scripts/main.py -m solver=neural_cache pde=laplace_nd pde.n_dim=10 solver.buffer_size=100000 solver.n_batch=8192 solver.train_period=100 solver.n_traj=50 solver.n_iters=1000 wandb.project=final_time_step
python3 scripts/main.py -m solver=nwos pde=laplace_nd pde.n_dim=10 solver.beta=5000 solver.n_batch=16384 solver.max_step=1,5,10 solver.n_traj=10  solver.n_iters=1000 wandb.project=final_time_step
