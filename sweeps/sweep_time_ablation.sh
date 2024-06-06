
python3 scripts/main.py -m solver=nwos pde=poisson_nd pde.n_dim=10 \
solver.max_step=1,5,10 solver.n_iters=1000 \
solver.n_traj=100 \
solver.n_batch=2048 \
solver.nn_target=True,False \
solver.control_variate=True,False \
wandb.project=final_time_ablation