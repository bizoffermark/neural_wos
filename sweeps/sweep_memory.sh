
# poisson 100d
python3 scripts/main.py -m solver=nwos pde=poisson_nd pde.n_dim=10,100,1000 solver.n_batch=512 solver.max_step=1 solver.n_traj=1 solver.nn_target=False solver.control_variate=False solver.n_iters=1 wandb.project=final_memory
python3 scripts/main.py -m solver=nsde pde=poisson_nd pde.n_dim=10,100,1000 solver.n_batch=512 solver.max_step=1 solver.time_step=0.0000001 solver.n_iters=1 wandb.project=final_memory
python3 scripts/main.py -m solver=deepritz pde=poisson_nd pde.n_dim=10,100,1000 +solver.test_memory=True solver.n_batch=512 solver.n_iters=1 wandb.project=final_memory
python3 scripts/main.py -m solver=pinn pde=poisson_nd pde.n_dim=10,100,1000 solver.n_batch=512 solver.n_iters=1 wandb.project=final_memory solver.cuda_max_mem_train_mb=1000000 
