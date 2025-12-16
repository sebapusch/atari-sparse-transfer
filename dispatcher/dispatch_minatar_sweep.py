import yaml
import itertools
import subprocess
import os
import sys
from pathlib import Path

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_command(command, dry_run=False):
    print(f"Executing: {command}")
    if not dry_run:
        result = subprocess.run(command, shell=True)
        if result.returncode != 0:
            print(f"Error running command: {command}", file=sys.stderr)
            return False
    return True

def main():
    # 1. Load the sweep configuration
    sweep_config_path = Path("configs/sweeps/minatar_sweep.yaml")
    if not sweep_config_path.exists():
        print(f"Error: Config file not found at {sweep_config_path}")
        return

    config = load_config(sweep_config_path)
    base_config = config.get("base_config", "minatar")
    params = config.get("params", {})
    
    seeds = params.get("seed", [1])
    # Handle single values if they aren't lists, just in case
    if not isinstance(seeds, list): seeds = [seeds]

    env_ids = params.get("env_id", [])
    if not isinstance(env_ids, list): env_ids = [env_ids]
    
    sparsities = params.get("sparsity", [])
    if not isinstance(sparsities, list): sparsities = [sparsities]

    # 2. Iterate over Cartesian product
    keys = ["seed", "env_id", "sparsity"]
    combinations = itertools.product(seeds, env_ids, sparsities)
    
    # Check if we verify dry run argument
    dry_run = "--dry-run" in sys.argv

    print(f"Starting sweep dispatch for base_config: {base_config}")
    print(f"Parameters: seeds={seeds}, envs={env_ids}, sparsities={sparsities}")
    print("-" * 50)

    count = 0
    for seed, env_id, sparsity in combinations:
        count += 1
        
        # Extract environment name for tagging (e.g., "MinAtar/breakout" -> "breakout")
        env_name = env_id.split("/")[-1]
        
        # Construct the job name and tags
        job_name = f"minatar-{env_name}-S{int(sparsity*100)}-s{seed}"
        tags = f"['minatar', 'ddqn', 'gmp', '{env_name}', 'sparsity-{sparsity}']"
        
        # Construct command arguments
        # base command
        cmd_parts = [
            "sbatch",
            "dispatcher/train.sh",
            base_config 
        ]
        
        # Add overrides
        overrides = [
            f"seed={seed}",
            f"env.id={env_id}",
            f"pruning.method=gmp",
            f"pruning.final_sparsity={sparsity}",
            f"wandb.group=minatar-sweep",
            f"wandb.tags=\"{tags}\"",
            f"hydra.job.name={job_name}",
             # We might want to set the experiment name in wandb too
            f"wandb.name={job_name}"
        ]
        
        # Join standard args
        cmd = " ".join(cmd_parts + overrides)
        
        print(f"Job {count}: {job_name}")
        run_command(cmd, dry_run=dry_run)
        print("-" * 20)
        
    print(f"Total jobs dispatched: {count}")

if __name__ == "__main__":
    main()
