import subprocess
import argparse

import wandb

from util import use_770_permissions

from config import REPO_ROOT
from probe.probe_config import SWEEP_CONFIGS


SLURM_JOB_BASE = r"""
#!/bin/bash
#SBATCH -p {slurm_partition} --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=63G
#SBATCH -t 24:00:00
#SBATCH -J probe_experiment_{sweep_id}
#SBATCH -e logs/probe/{sweep_id}/probe_experiment_{sweep_id}-%j.err
#SBATCH -o logs/probe/{sweep_id}/probe_experiment_{sweep_id}-%j.out

# Activate virtual environment
conda activate {conda_env_name}

# Run the script
python probe/main.py --sweep_id {sweep_id} --wandb_project {wandb_project}
"""

if __name__ == "__main__":
    """
    Define a sweep configuration to run in wandb. Call an agent script to run a sweep within the sweep ID.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--conda_env_name", type=str, default="syntheory")
    parser.add_argument("--sweep_config", type=str, default="jukebox")
    parser.add_argument("--slurm_jobs", type=int, default=1000)
    parser.add_argument("--gpu_partition", type=str)
    args = parser.parse_args()

    if args.sweep_config not in SWEEP_CONFIGS:
        raise ValueError(
            f"Unknown Sweep Configuration Given - Got: {args.sweep_config}"
        )

    sweep_config = SWEEP_CONFIGS[args.sweep_config]
    sweep_id_filename = args.sweep_config + ".txt"

    with use_770_permissions():
        sweep_info_file = REPO_ROOT / sweep_id_filename
        if sweep_info_file.exists() and sweep_info_file.is_file():
            # a sweep with this same name already exists, assuming a unique mapping from sweep file name and config
            sweep_id, wandb_project_name = sweep_info_file.read_text().split("\n")
            print(
                f"A wandb sweep ID already existed, using sweep id = {sweep_id}, wandb_project = {wandb_project_name}"
            )
        else:
            # create a new sweep with this configuration and project name
            wandb_sweep_parameters = sweep_config["wandb_sweep_parameters"]
            wandb_project_name = sweep_config["wandb_project_name"]

            # project should not have spaces
            sweep_id = wandb.sweep(
                sweep=wandb_sweep_parameters, project=wandb_project_name
            )

            print(
                f"A wandb sweep ID did NOT exist, created sweep id = {sweep_id}, wandb_project = {wandb_project_name}"
            )
            sweep_info_file.write_text(sweep_id + "\n" + wandb_project_name)

        # create script file to run
        script_contents = SLURM_JOB_BASE.format(
            conda_env_name=args.conda_env_name,
            sweep_id=sweep_id,
            wandb_project=wandb_project_name,
            slurm_partition=args.gpu_partition,
        ).strip()

        # write the slurm jobs to a shell script, change permissions of that file
        tmp_file = REPO_ROOT / f"{wandb_project_name}_sweep_probe.sh"
        tmp_file.write_text(script_contents)
        subprocess.run(f"chmod u+x {tmp_file.absolute()}", shell=True)

        # enqueue this script to slurm scheduler n times
        for _ in range(args.slurm_jobs):
            subprocess.run(f"sbatch {tmp_file.absolute()}", shell=True)
