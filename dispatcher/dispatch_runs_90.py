import sys
import os
import subprocess
import argparse
from pathlib import Path

# Fix for shadowing "wandb" directory in current path
# Remove current working directory from sys.path if it's there
# so that 'import wandb' finds the installed package, not the folder.
sys.path = [p for p in sys.path if p != os.getcwd()]

import wandb

# Constants
WANDB_PROJECT = "sebapusch-university-of-groningen/sparsity-rl"
RUNS_FILE = "runs-90.txt"
TRAIN_SCRIPT = "dispatcher/train.sh"
BASE_CONFIG = "minatar-lth"

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

def parse_runs_file(file_path):
    """
    Parses the runs file formatted as: <run_name> -> <artifact_name>
    Returns a list of tuples: (run_name, artifact_name)
    """
    runs = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or "->" not in line:
                continue
            parts = line.split("->")
            run_name = parts[0].strip()
            artifact_name = parts[1].strip()
            runs.append((run_name, artifact_name))
    return runs

def get_run_config(api, run_name):
    """
    Fetches the run configuration from WandB.
    """
    # First try filtering by display_name
    runs = api.runs(WANDB_PROJECT, filters={"display_name": run_name})
    if not runs:
        # Fallback to name
        runs = api.runs(WANDB_PROJECT, filters={"name": run_name})
    
    if not runs:
        print(f"Warning: Run '{run_name}' not found in project '{WANDB_PROJECT}'")
        return None
    
    # Assuming the first match is the correct one (names should be unique enough)
    return runs[0]

def main():
    parser = argparse.ArgumentParser(description="Dispatch jobs for runs specified in runs-90.txt using their original configs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args()

    # Check for runs-90.txt in the current directory or relative to script?
    # Expected execution is from root of repo
    runs_file_path = Path(RUNS_FILE)
    if not runs_file_path.exists():
        print(f"Error: {RUNS_FILE} not found in current directory.")
        sys.exit(1)

    print("Connecting to WandB API...")
    try:
        api = wandb.Api()
    except Exception as e:
        print(f"Error connecting to WandB API: {e}")
        sys.exit(1)

    print(f"Parsing {RUNS_FILE}...")
    runs_to_process = parse_runs_file(runs_file_path)
    print(f"Found {len(runs_to_process)} runs to process.")

    for run_name, artifact_name in runs_to_process:
        print(f"Processing: {run_name} -> {artifact_name}")
        
        run_obj = get_run_config(api, run_name)
        if not run_obj:
            print(f"Skipping {run_name} (details fetch failed)")
            continue

        config = run_obj.config
        
        # Extract required config values
        seed = config.get("seed")
        env_id = config.get("env", {}).get("id") or config.get("env_id") # Handling nested or flat structure if it varies
        sparsity = config.get("pruning", {}).get("final_sparsity")

        if seed is None or env_id is None or sparsity is None:
            print(f"Skipping {run_name}: Missing config values. (seed={seed}, env.id={env_id}, pruning.final_sparsity={sparsity})")
            continue

        # Construct overrides
        # dispatcher/train.sh minatar-lth +initial_artifact="sebapusch-university-of-groningen/sparsity-rl/<artifact_name>" +wandb.name="<run_name>-2" seed=3 +env.id=MinAtar/seaquest pruning.final_sparsity=0.75 wandb.group=minatar_ddqn_lth_sweep "wandb.tags=['minatar_ddqn_lth_sweep', 'seaquest', '3', 'MinAtar/seaquest', '0.75', 'sparsity-0.75']"
        
        new_run_name = f"{run_name}-2"
        full_artifact_path = f"{WANDB_PROJECT}/{artifact_name}"
        
        # Tags construction similar to requested format
        env_short = env_id.split("/")[-1] if "/" in str(env_id) else str(env_id)
        tags_list = [
            "minatar_ddqn_lth_sweep", # Keeping consistent with request example
            env_short,
            str(seed),
            str(env_id),
            str(sparsity),
            f"sparsity-{sparsity}"
        ]
        
        # Format tags as a python list string for the command line
        tags_str = str(tags_list).replace("'", "'") # Just to be explicit, though str() usually gives single quotes

        cmd_parts = [
            "sbatch" if not args.dry_run else "echo sbatch", # Or assume train.sh handles sbatch? 
            # The USER REQUEST says: "launches a batch script in the form: dispatcher/train.sh ..." 
            # Looking at existing dispatch_sweep.py: "sbatch dispatcher/train.sh"
            # So I will use "sbatch dispatcher/train.sh"
            TRAIN_SCRIPT,
            BASE_CONFIG,
            f'+initial_artifact="{full_artifact_path}"',
            f'+wandb.name="{new_run_name}"',
            f'seed={seed}',
            f'+env.id={env_id}',
            f'pruning.final_sparsity={sparsity}',
            'wandb.group=minatar_ddqn_lth_sweep',
            f'"wandb.tags={tags_list}"'
        ]

        full_cmd = " ".join(cmd_parts)
        
        # Use sbatch prefix if not dry run?
        # Actually in dispatch_sweep.py:
        # cmd_parts = ["sbatch", "dispatcher/train.sh", base_config]
        # full_cmd = " ".join(cmd_parts + overrides)
        # So I should construct it similarly.
        
        real_cmd_parts = ["sbatch", TRAIN_SCRIPT, BASE_CONFIG]
        overrides = [
             f'+initial_artifact="{full_artifact_path}"',
            f'+wandb.name="{new_run_name}"',
            f'seed={seed}',
            f'+env.id={env_id}',
            f'pruning.final_sparsity={sparsity}',
            'wandb.group=minatar_ddqn_lth_sweep',
            f'"wandb.tags={tags_list}"'
        ]
        
        final_cmd = " ".join(real_cmd_parts + overrides)
        
        run_command(final_cmd, dry_run=args.dry_run)
        print("-" * 20)

if __name__ == "__main__":
    main()
