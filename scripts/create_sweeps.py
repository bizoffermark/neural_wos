from pathlib import Path
import yaml

import wandb


def exec_slurm_from_sweep(cfg_path, run_file, entity=None, project=None, n_runs=10):
    with open(cfg_path, "r") as f:
        sweep_config = yaml.safe_load(f)

    entity = entity or sweep_config.get("entity")
    project = project or sweep_config.get("project")
    sweep_id = wandb.sweep(sweep_config, entity=entity, project=project)

    bin_file = "bin/slurm_sweep.sh"
    with open(run_file, mode="a") as f:
        f.write(
            f"SWEEP_ID={entity}/{project}/{sweep_id} sbatch -a 0-{n_runs} {bin_file}"
        )


if __name__ == "__main__":
    run_file = Path(__file__).parents[1] / "bin" / "start_sweeps.sh"
    n_runs = 2
    project = "sweeps"
    entity = "nwos"

    base_path = Path(__file__).parents[1] / "configs" / "sweeps" / "grid"
    files = list(base_path.iterdir())

    for file in files:
        exec_slurm_from_sweep(
            files, run_file, entity=entity, project=project, n_runs=n_runs
        )
