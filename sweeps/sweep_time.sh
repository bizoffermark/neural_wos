export WANDB_ENTITY=nwos

# Laplace 10D (Done)
python3 scripts/main.py -m seed=1,2,3,4,5 solver.time_limit=1000 solver=pinn pde=laplace_nd pde.n_dim=10 solver.beta=500 solver.n_batch=4096 wandb.project=final_timed
python3 scripts/main.py -m seed=1,2,3,4,5 solver.time_limit=1000 solver=deepritz pde=laplace_nd pde.n_dim=10 solver.beta=5000 solver.n_batch=8192 wandb.project=final_timed 
python3 scripts/main.py -m seed=1,2,3,4,5 solver.time_limit=1000 solver=nsde pde=laplace_nd pde.n_dim=10 solver.beta=5000 solver.n_batch=16384 solver.max_step=1 solver.time_step=1e-3 wandb.project=final_timed 
python3 scripts/main.py -m seed=1,2,3,4,5 solver.time_limit=1000 solver=wos pde=laplace_nd pde.n_dim=10 solver.max_step=100 solver.n_traj=10000 wandb.project=final_timed
python3 scripts/main.py -m seed=1,2,3,4,5 solver.time_limit=1000 solver=neural_cache pde=laplace_nd pde.n_dim=10 solver.buffer_size=100000 solver.n_batch=8192 solver.train_period=100 solver.n_traj=50 wandb.project=final_timed 
python3 scripts/main.py -m seed=1,2,3,4,5 solver.time_limit=1000 solver=buffer_nwos pde=laplace_nd solver.bound_init=True pde.n_dim=10 solver.beta=5000 solver.n_batch=16384 solver.max_step=50 solver.n_traj=10 solver.train_period=1000 wandb.project=final_timed

# # Committor 10D (Done)
python3 scripts/main.py -m seed=1,2,3,4,5 solver.time_limit=1000 solver=pinn pde=committor_nd pde.n_dim=10 solver.beta=500 solver.n_batch=4096 wandb.project=final_timed 
python3 scripts/main.py -m seed=1,2,3,4,5 solver.time_limit=1000 solver=deepritz pde=committor_nd pde.n_dim=10 solver.beta=500 solver.n_batch=8192 wandb.project=final_timed
python3 scripts/main.py -m seed=1,2,3,4,5 solver.time_limit=1000 solver=nsde pde=committor_nd pde.n_dim=10 solver.beta=0.5 solver.n_batch=2048 solver.time_step=0.001 wandb.project=final_timed &
python3 scripts/main.py -m seed=1,2,3,4,5 solver.time_limit=1000 solver=wos pde=committor_nd pde.n_dim=10 solver.max_step=100 solver.n_traj=10000 wandb.project=final_timed
python3 scripts/main.py -m seed=1,2,3,4,5 solver.time_limit=1000 solver=neural_cache pde=committor_nd pde.n_dim=10 solver.buffer_size=100000 solver.n_batch=8192 solver.train_period=100 solver.n_traj=40 wandb.project=final_timed
python3 scripts/main.py -m seed=1,2,3,4,5 solver.time_limit=1000 solver=buffer_nwos pde=committor_nd pde.n_dim=10 solver.beta=1.0 solver.n_batch=65536 solver.max_step=500 solver.n_traj=500 solver.train_period=1000 solver.n_batch_bound_factor=0.5 solver.bound_init=True wandb.project=final_timed

# # Poisson 10D Complex (Done)
python3 scripts/main.py -m seed=1,2,3,4,5 solver.time_limit=1000 solver=pinn pde=poisson_nd_complex pde.n_dim=10 solver.beta=1000 solver.n_batch=4096 wandb.project=final_timed
python3 scripts/main.py -m seed=1,2,3,4,5 solver.time_limit=1000 solver=deepritz pde=poisson_nd_complex pde.n_dim=10 solver.beta=5000 solver.n_batch=8192 wandb.project=final_timed
python3 scripts/main.py -m seed=1,2,3,4,5 solver.time_limit=1000 solver=nsde pde=poisson_nd_complex pde.n_dim=10 solver.beta=100 solver.n_batch=2048 solver.time_step=0.001 wandb.project=final_timed
python3 scripts/main.py -m seed=1,2,3,4,5 solver.time_limit=1000 solver=wos pde=poisson_nd_complex pde.n_dim=10 solver.max_step=100 solver.n_traj=10000 wandb.project=final_timed
python3 scripts/main.py -m seed=1,2,3,4,5 solver.time_limit=1000 solver=neural_cache pde=poisson_nd_complex pde.n_dim=10 solver.buffer_size=100000 solver.n_batch=8192 solver.train_period=100 solver.n_traj=40 wandb.project=final_timed
python3 scripts/main.py -m seed=1,2,3,4,5 solver.time_limit=1000 solver=buffer_nwos pde=poisson_nd_complex pde.n_dim=10 solver.beta=500 solver.n_batch=4096 solver.max_step=10 solver.n_traj=100 wandb.project=final_timed

# # Poisson 50D
python3 scripts/main.py -m seed=1,2,3,4,5 solver.time_limit=2000 solver=pinn pde=poisson_nd pde.n_dim=50 solver.beta=5000 solver.n_batch=256 wandb.project=final_timed
python3 scripts/main.py -m seed=1,2,3,4,5 solver.time_limit=2000 solver=deepritz pde=poisson_nd pde.n_dim=50 solver.beta=5000 solver.n_batch=8192 wandb.project=final_timed 
python3 scripts/main.py -m seed=1,2,3,4,5 solver.time_limit=2000 solver=nsde pde=poisson_nd pde.n_dim=50 solver.beta=0.5 solver.n_batch=16384 solver.max_step=5 solver.time_step=0.00001 wandb.project=final_timed 
python3 scripts/main.py -m seed=1,2,3,4,5 solver.time_limit=2000 solver=wos pde=poisson_nd pde.n_dim=50 solver.n_batch=1000000 solver.max_step=10 solver.n_traj=1 wandb.project=final_timed 
python3 scripts/main.py -m seed=1,2,3,4,5 solver.time_limit=2000 solver=neural_cache pde=poisson_nd pde.n_dim=50 solver.buffer_size=20000 solver.n_batch_train=1024 solver.train_period=5000 solver.n_traj=10 wandb.project=final_timed
python3 scripts/main.py -m seed=1,2,3,4,5 solver.time_limit=2000 solver=nwos pde=poisson_nd pde.n_dim=50 solver.beta=5000 solver.n_batch=8192 solver.max_step=1 solver.n_traj=100 solver.nn_target=False solver.control_variate=True wandb.project=final_timed
