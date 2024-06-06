# Neural Walk-on-Spheres (NWoS)

This repository provides the code for our work Neural Walk-on-Spheres.

The exact details for implementations and experiments are described in our paper [Solving Poisson Equations Using Neural Walk-on-Spheres](https://arxiv.org/abs/2406.03494) published to ICML 2024 and ICLR 2024 Workshop.

Hong Chul Nam*, Julius Berner*, Anima Anandkumar <br>
*: Equal contribution

---
![Neural Walk-on-Sphere](images/logo.png "Overview of Neural Walk-on-Spheres (NWoS) ")


## Dependency & Installation
All dependent files are kept in `setup.py` and can be installed by simply running `pip install -e .`.


## WoS Implementation
WoS is implemented in `wos/solvers/wos.py`. The idea is to parallelize all different trajectories by sharding the total trajectories to smaller chunks and walking over each small chunk, thereby ensuring the trade-off between speed and memory.

## PDE Configuration
The function requires inputs of different Poisson equations with Dirichlet boundary conditions. The basic idea is to define source term, boundary term, domain sampling, and boundary sampling functions by inheriting the class `BasePDEEquation` defined in `wos/pdes/base_pde.py`. 

## Evaluations
The repository contains evaluations of DeepRitz, PINN, diffusion loss, and NWos using Hydra.

To run all baselines to reproduce the results from Table 1 of the paper, run the following script:
```bash
bash sweeps/sweep_time.sh
```
One critical thing is to run only one training per GPU to ensure that the timing condition is fair across each task.

## Replication of Figures
In order to replicate some figures from the paper, check the `pots` folder and follow the instructions written in the comment to reproduce all figures and results in the paper.

## Extending to Other Poisson Equations
In order to extend this framework to support other Poisson PDEs, check out the PDE configuration section and define a corresponding PDE. Afterward, define the pde config at `configs/pde` and call such a function by calling:
```bash
python scripts/main.py solver=nwos pde=YOUR_PDE_NAME pde.n_dim=YOUR_PDE_DIMENSION
```
You should specify more other parameters, as in `sweeps/sweep_time.py`, to get a better idea of what parameters to choose.

## Citation

If you find this repository or our paper useful, please consider giving a star :star: and citation:

```bibtex
@inproceedings{
nam2024solving,
title={Solving Poisson Equations Using Neural Walk-on-Spheres},
author={Hong Chul Nam and Julius Berner and Anima Anandkumar},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
}
```