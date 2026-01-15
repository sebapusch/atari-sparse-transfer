import sys
import os
import subprocess
import argparse
from pathlib import Path

# Fix for shadowing "wandb" directory in current path
sys.path = [p for p in sys.path if p != os.getcwd()]

import wandb

# Constants
WANDB_PROJECT_SOURCE = "sebapusch-university-of-groningen/atari-lottery"
WANDB_PROJECT_TARGET = "lth-transfer"
TARGET_ENVS = ["ALE/Breakout-v5", "ALE/Pong-v5", "ALE/SpaceInvaders-v5"]
BASE_CONFIG = "atari-gmp"
TRAIN_SCRIPT = "dispatcher/train.sh"

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
    Skips lines starting with [failed] or empty lines.
    Returns a list of tuples: (run_name, artifact_name)
    """
    runs = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("[failed]"):
                continue
            if "->" not in line:
                print(f"Skipping malformed line: {line}")
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
    runs = api.runs(WANDB_PROJECT_SOURCE, filters={"display_name": run_name})
    if not runs:
        # Fallback to name
        runs = api.runs(WANDB_PROJECT_SOURCE, filters={"name": run_name})
    
    if not runs:
        print(f"Warning: Run '{run_name}' not found in project '{WANDB_PROJECT_SOURCE}'")
        return None
    
    return runs[0]

def main():
    parser = argparse.ArgumentParser(description="Dispatch transfer runs for tickets.")
    parser.add_argument("file", help="Path to the file containing run -> artifact mappings")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--mock", action="store_true", help="Mock WandB runs for testing.")
    parser.add_argument("--load-encoder-only", default=True, action="store_true", help="Load only the encoder weights from the artifact.")
    args = parser.parse_args()

    runs_file_path = Path(args.file)
    if not runs_file_path.exists():
        print(f"Error: {args.file} not found.")
        sys.exit(1)

    if not args.mock:
        print("Connecting to WandB API...")
        try:
            api = wandb.Api()
        except Exception as e:
            print(f"Error connecting to WandB API: {e}")
            sys.exit(1)
    else:
        print("Using Mock WandB API.")
        api = None

    print(f"Parsing {args.file}...")
    runs_to_process = parse_runs_file(runs_file_path)
    print(f"Found {len(runs_to_process)} valid runs to process.")

    for run_name, artifact_name in runs_to_process:
        print("-" * 40)
        print(f"Processing Ticket: {run_name} -> {artifact_name}")
        
        if args.mock:
            # Mock logic
            class MockRun:
                def __init__(self, name):
                    self.config = {}
                    if "SpaceInvaders" in name:
                        self.config["env"] = {"id": "ALE/SpaceInvaders-v5"}
                    elif "Pong" in name:
                        self.config["env"] = {"id": "ALE/Pong-v5"}
                    elif "Breakout" in name:
                        self.config["env"] = {"id": "ALE/Breakout-v5"}
                    self.config["seed"] = 123
                    self.config["pruning"] = {"final_sparsity": 0.6}
            run_obj = MockRun(run_name)
        else:
            run_obj = get_run_config(api, run_name)
            
        if not run_obj:
            print(f"Skipping {run_name} (details fetch failed)")
            continue

        config = run_obj.config
        
        # Determine Source Environment
        source_env_id = config.get("env", {}).get("id") or config.get("env_id")
        seed = config.get("seed")
        source_sparsity = config.get("pruning", {}).get("final_sparsity")
        
        if not source_env_id:
            print(f"Skipping {run_name}: Could not determine source env_id.")
            continue
            
        print(f"  Source Env: {source_env_id}")
        if source_sparsity is not None:
            print(f"  Source Sparsity: {source_sparsity}")
        
        # Determine Destination Environments (All TARGET_ENVS except source)
        dest_envs = [env for env in TARGET_ENVS if env != source_env_id]
        
        if len(dest_envs) == len(TARGET_ENVS):
             print(f"  Warning: Source env {source_env_id} not in TARGET_ENVS {TARGET_ENVS}. Dispatching to all targets.")
        
        full_artifact_path = f"{WANDB_PROJECT_SOURCE}/{artifact_name}"

        # Dispatch 4 runs per ticket: 2 Envs * 2 Pruning Settings
        for dest_env in dest_envs:
            dest_env_short = dest_env.split("/")[-1].split("-")[0] # e.g., ALE/Breakout-v5 -> Breakout

            for pruning_setting in ["None", "GMP"]:
                
                # Construct Name and Tags
                pruning_tag = "dense" if pruning_setting == "None" else "gmp-0.9"
                new_run_name = f"transfer-{run_name}-{dest_env_short}-{pruning_tag}"
                
                tags = [
                    "transfer_ticket_sweep",
                    f"source-{source_env_id}",
                    f"dest-{dest_env}",
                    f"seed-{seed}",
                    pruning_tag
                ]
                
                cmd_parts = ["sbatch", TRAIN_SCRIPT, BASE_CONFIG]
                
                overrides = [
                    f'+initial_artifact="{full_artifact_path}"',
                    f'+wandb.name="{new_run_name}"',
                    f'wandb.project="{WANDB_PROJECT_TARGET}"',
                    f'seed={seed}',
                    f'+env.id={dest_env}',
                    'wandb.group=transfer_ticket_sweep',
                    f'"wandb.tags={tags}"'
                ]
                
                if args.load_encoder_only:
                    overrides.append("load_encoder_only=True")

                if pruning_setting == "None":
                    overrides.append("pruning=none")
                elif pruning_setting == "GMP":
                    overrides.append("pruning=gmp")
                    overrides.append("pruning.final_sparsity=0.9")
                    if source_sparsity is not None:
                        overrides.append(f"pruning.initial_sparsity={source_sparsity}")
                
                final_cmd = " ".join(cmd_parts + overrides)
                run_command(final_cmd, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
