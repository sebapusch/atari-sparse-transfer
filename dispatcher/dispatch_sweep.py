import yaml
import itertools
import subprocess
import sys
import argparse
from pathlib import Path

# Mapping from config file keys to valid Hydra override keys
KEY_MAPPING = {
    "env_id": "+env.id",
    "sparsity": "pruning.final_sparsity",
}

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_command(command, dry_run=False):
    if dry_run:
        print(f"[DRY RUN] {command}")
        return True
    
    print(f"Executing: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error running command: {command}", file=sys.stderr)
        return False
    return True

def format_value_for_name(key, value):
    """Helper to format values for job names compactly."""
    if key == "sparsity":
        return f"S{int(float(value)*100)}"
    if key == "env_id":
        # Extract environment name (e.g., "MinAtar/breakout" -> "breakout")
        return value.split("/")[-1]
    if key == "seed":
        return f"s{value}"
    return f"{value}"

def main():
    parser = argparse.ArgumentParser(description="Dispatch sweep jobs based on a YAML config.")
    parser.add_argument("config_name", nargs="?", help="Name of the sweep configuration file in configs/sweeps/.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args()

    # Construct path: configs/sweeps/<config_name>
    # logic: if user provides a path that exists, use it. 
    # Otherwise, assume it's in configs/sweeps
    
    config_arg = Path(args.config_name)
    
    # Check if it is a full path or relative path provided
    if config_arg.exists():
        sweep_config_path = config_arg
    else:
        # Check in configs/sweeps
        sweep_config_path = Path("configs/sweeps") / config_arg

        # If extension is missing, try adding .yaml
        if not sweep_config_path.exists() and not sweep_config_path.suffix:
             sweep_config_path = sweep_config_path.with_suffix(".yaml")

    if not sweep_config_path.exists():
        print(f"Error: Config file not found at {sweep_config_path}")
        sys.exit(1)

    print(f"Loading sweep config from: {sweep_config_path}")
    config = load_config(sweep_config_path)
    
    base_config = config.get("base_config")
    if not base_config:
        print("Error: 'base_config' not specified in the sweep configuration.")
        sys.exit(1)

    sweep_name = config.get("name", "sweep")
    params = config.get("params", {})
    
    # keys are the parameter names (e.g., seed, env_id, sparsity)
    keys = list(params.keys())
    # values_list contains the list of values for each key
    values_list = []
    
    for k in keys:
        val = params[k]
        if not isinstance(val, list):
            val = [val]
        values_list.append(val)

    # Cartesian product of all parameters
    combinations = list(itertools.product(*values_list))
    
    print(f"Starting sweep dispatch for base_config: {base_config}")
    print(f"Sweep Name: {sweep_name}")
    print(f"Parameters: {keys}")
    print(f"Total Combinations: {len(combinations)}")
    print("-" * 50)

    count = 0
    for combo in combinations:
        count += 1
        
        # Create a dictionary for this specific job's parameters
        job_params = dict(zip(keys, combo))
        
        # 1. Build overrides
        overrides = []
        tags = [sweep_name]
        
        # We also build parts of the name
        name_parts = [sweep_name]
        
        # Specifically handle env_id if present for better naming
        if "env_id" in job_params:
            env_name = job_params["env_id"].split("/")[-1]
            tags.append(env_name)
            # We add env name to name_parts separately or let the loop handle it?
            # The original script did: minatar-{env}-{S}-s{seed}
            # Let's try to replicate a sensible structure:
            # {sweep_name}-{env}-{params...}
        
        # Construct overrides and collect naming parts
        for k, v in job_params.items():
            # Map key if it exists in KEY_MAPPING, else use key directly
            hydra_key = KEY_MAPPING.get(k, k)
            overrides.append(f"{hydra_key}={v}")
            
            # Add to tags
            tags.append(str(v))
            if k == "sparsity":
                 tags.append(f"sparsity-{v}")

        # Build Job Name
        # Priority components for name: env_id, sparsity, seed. Others appended.
        # This is heuristics to match previous style but stay generic-ish.
        
        # Extract specific values for the name if they exist
        env_val = job_params.get("env_id")
        sparsity_val = job_params.get("sparsity")
        seed_val = job_params.get("seed")
        
        name_components = [sweep_name]
        if env_val:
            name_components.append(format_value_for_name("env_id", env_val))
        if sparsity_val is not None:
             name_components.append(format_value_for_name("sparsity", sparsity_val))
        if seed_val is not None:
             name_components.append(format_value_for_name("seed", seed_val))
             
        # Add any other keys not in [env_id, sparsity, seed] to the name to ensure uniqueness
        for k, v in job_params.items():
            if k not in ["env_id", "sparsity", "seed"]:
                name_components.append(f"{k}{v}")

        job_name = "-".join(name_components)
        
        # Add W&B specific overrides
        # We assume wandb.group is the sweep name
        overrides.append(f"wandb.group={sweep_name}")
        overrides.append(f"wandb.tags=\"{tags}\"")
        overrides.append(f"+wandb.name={job_name}")

        # Construct command
        # sbatch dispatcher/train.sh <base_config> <overrides>
        cmd_parts = [
            "sbatch",
            "dispatcher/train.sh",
            base_config
        ]
        
        full_cmd = " ".join(cmd_parts + overrides)
        
        print(f"Job {count}: {job_name}")
        run_command(full_cmd, dry_run=args.dry_run)
        print("-" * 20)
        
    print(f"Total jobs dispatched: {count}")

if __name__ == "__main__":
    main()
