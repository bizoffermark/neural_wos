
# Poisson 10D Complex
python3 scripts/main.py -m solver=nwos pde=poisson_nd_complex pde.n_dim=10 solver.beta=500 solver.nn_target=False solver.control_variate=False solver.model.hid_dim=256 solver.n_batch=2621 optimizer.lr=1e-3 solver.max_step=10 solver.n_traj=50 wandb.project=nwos_ablation &
python3 scripts/main.py -m solver=nwos pde=poisson_nd_complex pde.n_dim=10 solver.beta=500 solver.nn_target=False solver.control_variate=True solver.model.hid_dim=256 solver.n_batch=2621 optimizer.lr=1e-3 solver.max_step=10 solver.n_traj=50 wandb.project=nwos_ablation &
python3 scripts/main.py -m solver=nwos pde=poisson_nd_complex pde.n_dim=10 solver.beta=500 solver.nn_target=True solver.control_variate=False solver.model.hid_dim=256 solver.n_batch=2621 optimizer.lr=1e-3 solver.max_step=10 solver.n_traj=50 wandb.project=nwos_ablation &

python3 scripts/main.py -m solver=nwos solver.time_limit=1000 pde=laplace_nd pde.n_dim=10 solver.beta=500 solver.nn_target=False solver.control_variate=False solver.n_batch=655 solver.max_step=10 solver.n_traj=100 wandb.project=final_ablation &
python3 scripts/main.py -m solver=nwos solver.time_limit=1000 pde=laplace_nd pde.n_dim=10 solver.beta=500 solver.nn_target=False solver.control_variate=True solver.n_batch=655 solver.max_step=10 solver.n_traj=100 wandb.project=final_ablation &
python3 scripts/main.py -m solver=nwos solver.time_limit=1000 pde=laplace_nd pde.n_dim=10 solver.beta=500 solver.nn_target=True solver.control_variate=False solver.n_batch=655 solver.max_step=10 solver.n_traj=100 wandb.project=final_ablation &
python3 scripts/main.py -m solver=nwos solver.time_limit=1000 pde=laplace_nd pde.n_dim=10 solver.beta=500 solver.nn_target=True solver.control_variate=True solver.n_batch=655 solver.max_step=10 solver.n_traj=100 wandb.project=final_ablation &

# python3 scripts/main.py -m solver=nwos pde=committor_nd pde.n_dim=10 solver.beta=0.5 solver.nn_target=False solver.control_variate=False solver.n_batch=2621 solver.max_step=10 solver.n_traj=50 wandb.project=nwos_ablation &
# python3 scripts/main.py -m solver=nwos pde=committor_nd pde.n_dim=10 solver.beta=0.5 solver.nn_target=False solver.control_variate=True solver.n_batch=2621 solver.max_step=10 solver.n_traj=50 wandb.project=nwos_ablation &
# python3 scripts/main.py -m solver=nwos pde=committor_nd pde.n_dim=10 solver.beta=0.5 solver.nn_target=True solver.control_variate=False solver.n_batch=2621 solver.max_step=10 solver.n_traj=50 wandb.project=nwos_ablation &
# python3 scripts/main.py -m solver=nwos pde=committor_nd pde.n_dim=10 solver.beta=0.5 solver.nn_target=True solver.control_variate=True solver.n_batch=2621 solver.max_step=10 solver.n_traj=50 wandb.project=speed_nwos &

python3 scripts/main.py -m seed=1,2,3,4,5 solver=nwos pde=poisson_nd_complex pde.n_dim=10 solver.beta=500 solver.n_batch=4096 solver.max_step=10 solver.n_traj=100 solver.nn_target=True,False solver.control_variate=True,False wandb.project=final_ablation &
python3 scripts/main.py -m seed=1,2,3,4,5 solver=nwos pde=committor_nd pde.n_dim=10 solver.beta=1.0 solver.n_batch=65536 solver.max_step=500 solver.n_traj=100 solver.n_batch_bound_factor=0.5 solver.nn_target=True,False solver.control_variate=True,False wandb.project=final_ablation &

python3 scripts/main.py -m solver.time_limit=1000 solver=buffer_nwos pde=poisson_nd_complex pde.n_dim=10 solver.beta=500 solver.n_batch=4096 solver.max_step=10 solver.n_traj=100 solver.nn_target=True,False solver.control_variate=True,False wandb.project=final_ablation &
python3 scripts/main.py -m solver.time_limit=1000 solver=buffer_nwos pde=committor_nd pde.n_dim=10 solver.beta=1.0 solver.n_batch=65536 solver.max_step=500 solver.n_traj=500 solver.train_period=1000 solver.n_batch_bound_factor=0.5 solver.bound_init=True solver.nn_target=True,False solver.control_variate=True,False wandb.project=final_timed
python3 scripts/main.py -m solver.time_limit=1000 solver=buffer_nwos pde=laplace_nd solver.bound_init=True pde.n_dim=10 solver.beta=5000 solver.n_batch=512 solver.max_step=1 solver.n_traj=10 solver.train_period=1000 solver.nn_target=True,False solver.control_variate=True,False wandb.project=final_ablation &
python3 scripts/main.py -m solver.time_limit=1000 solver=buffer_nwos pde=committor_nd pde.n_dim=10 solver.beta=1.0 solver.n_batch=512 solver.max_step=10 solver.n_traj=10 solver.train_period=1000 solver.n_batch_bound_factor=0.5 solver.bound_init=True solver.nn_target=True,False solver.control_variate=True,False wandb.project=final_ablation &
