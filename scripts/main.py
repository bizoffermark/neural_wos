"""
Script for running the experiments.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import random
from time import sleep
import subprocess

import hydra
import wandb
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import wos.utils.hydra # DONT REMOVE THIS!!
from omegaconf import DictConfig, OmegaConf
from wos.utils.wandb import merge_wandb_cfg

import torch

logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    sleep(random.randint(1, 10))
    print(f"cfg: {cfg}")
    logging.info("---------------------------------------------------------------")
    envs = {k: os.environ.get(k) for k in ["CUDA_VISIBLE_DEVICES", "PYTHONOPTIMIZE"]}
    logging.info("Env:\n%s", yaml.dump(envs))

    # Log overrides
    hydra_config = HydraConfig.get()
    logging.info("Command line args:\n%s", "\n".join(hydra_config.overrides.task))

    # Setup dir
    OmegaConf.set_struct(cfg, False)
    out_dir = Path(hydra_config.runtime.output_dir).absolute()
    logging.info("Hydra and wandb output path: %s", out_dir)
    if not cfg.get("out_dir"):
        cfg.out_dir = str(out_dir)
        cfg.solver.out_dir = str(out_dir)
    logging.info("Solver output path: %s", cfg.out_dir)

    # Setup wandb
    tags = [t for t in hydra_config.overrides.task if len(t) < 32]
    if "wandb" not in cfg:
        cfg.wandb = OmegaConf.create()
    if not cfg.wandb.get("tags"):
        cfg.wandb.tags = tags

    if not cfg.wandb.get("id"):
        # create id based on log directory for automatic (slurm) resuming
        sha = hashlib.sha256()
        sha.update(str(out_dir).encode())
        cfg.wandb.id = sha.hexdigest()

    if not cfg.wandb.get("name"):
        if hydra_config.mode is hydra.types.RunMode.RUN:
            name = str(out_dir.relative_to(out_dir.parents[1]))
        else:
            name = str(out_dir.parent.relative_to(out_dir.parents[2]))
        cfg.wandb.name = name + "," + ",".join([t.split("=")[-1] for t in tags])

    OmegaConf.set_struct(cfg, True)
    wandb.init(
        dir=out_dir,
        **cfg.wandb,
    )
    sleep(random.randint(1, 10))

    # Resume old wandb run
    if wandb.run is not None and wandb.run.resumed:
        logging.info("Resume wandb run %s", wandb.run.path)
        if cfg.get("merge_wandb_resume_cfg"):
            cfg = merge_wandb_cfg(cfg)

    # Log config and overrides
    logging.info("---------------------------------------------------------------")
    logging.info("Run config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    logging.info("---------------------------------------------------------------")

    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb_config["hydra"] = OmegaConf.to_container(hydra_config, resolve=True)

    for k in [
        "help",
        "hydra_help",
        "hydra_logging",
        "job_logging",
        "searchpath",
        "callbacks",
        "sweeper",
    ]:
        wandb_config["hydra"].pop(k, None)
    wandb.config.update(wandb_config, allow_val_change=True)

    # run solver
    logging.info("---------------------------------------------------------------")

    try:
        OmegaConf.resolve(cfg)
        solver = instantiate(cfg.solver.solver, cfg)
        if solver.solver_name in ["NWoS", "Deep Ritz", "PINN", "Diffusion Loss", 
                                  "Trajectory NWoS","neural_cache", "Buffer NWoS"]:
            solver.train()
        else:
            if solver.use_mem_test:
                solver.check_train_mem(fn=solver.mem_test_wos,
                                mem_bound_mb=solver.cuda_max_mem_train_mb,
                                throw_error=False)

            solver.validate(solver.n_sample_eval, 0, cumu_time=1000)

        wandb.run.summary["error"] = None
        logging.info("Completed ✅")
        wandb.finish()

    except Exception as e:
        logging.critical(e, exc_info=True)
        wandb.run.summary["error"] = str(e)
        wandb.finish(exit_code=1)


def sync_wandb(wandb_dir: Path | str):
    run_dirs = [f for f in Path(wandb_dir).iterdir() if "run-" in f.name]
    for run_dir in sorted(run_dirs, key=os.path.getmtime):
        logging.info("Syncing %s.", run_dir)
        subprocess.run(
            ["wandb", "sync", "--no-include-synced", "--mark-synced", str(run_dir)]
        )


if __name__ == "__main__":
    main()
