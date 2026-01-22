import os
import glob
import csv
import argparse
import sys
import torch
import time
import datetime
import numpy as np

# Ensure we can unpickle custom classes
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    from rlp.core.trainer import TrainingConfig
except ImportError:
    # Fallback if specific import fails, though adding src to path should fix it
    pass

def get_latest_checkpoint(directory: str) -> str | None:
    """
    Helper to find the numerically highest checkpoint in a specific directory.
    Solves the 'checkpoint_100 vs checkpoint_99' sorting bug.
    """
    files = glob.glob(os.path.join(directory, "*.pt"))
    if not files:
        return None

    def extract_step(filename: str) -> int:
        try:
            return int(filename.split('_')[-1].replace('.pt', ''))
        except ValueError:
            return -1

    files = sorted(files, key=lambda x: extract_step(os.path.basename(x)))
    return files[-1]

def calculate_sparsity(state_dict: dict) -> float:
    """
    Calculates global sparsity from the network state dict.
    checks for *_mask keys.
    """
    total_params = 0
    total_zeros = 0
    
    # We look for 'network' or 'agent' key, or just the dict itself
    target_dict = state_dict
    if 'agent' in state_dict:
        target_dict = state_dict['agent']
    if 'network' in target_dict:
        target_dict = target_dict['network']
        
    for key, tensor in target_dict.items():
        if key.endswith('_mask'):
            # This is a pruning mask
            total_params += tensor.numel()
            total_zeros += (tensor == 0).sum().item()
            
    if total_params == 0:
        return 0.0
        
    return total_zeros / total_params

def infer_action_count(state_dict: dict) -> int | None:
    """
    Infers the number of actions from the output layer of the network.
    Assumes standard Linear head structures often used in these models.
    """
    target_dict = state_dict
    if 'agent' in state_dict:
        target_dict = state_dict['agent']
    
    # Try nested network dict
    if 'network' in target_dict:
        target_dict = target_dict['network']
        
    # Look for likely output layer keys
    # e.g., "head.layer.weight" or "head.value_branch.weight" (dueling)
    # Common pattern: The layer with the fewest output dimensions usually corresponds to actions (or value=1)
    
    # Generic approach: Search for "head" keys
    head_candidates = [k for k in target_dict.keys() if 'head' in k and 'weight' in k]
    
    possible_outputs = []
    
    for key in head_candidates:
        tensor = target_dict[key]
        if tensor.ndim == 2:
            # Linear layer: (out_features, in_features)
            out_features = tensor.shape[0]
            if out_features > 1: # exclude value=1 for dueling value stream if separate
                 possible_outputs.append(out_features)
                 
    if not possible_outputs:
        return None
        
    # Heuristic: The action count is likely one of [4, 6] for Atari
    # If dueling, we might have an advantage stream (N actions) and value stream (1)
    # So we take the max matching known spaces or just the value found
    
    # Return the last one found (often the final layer) or specific logic?
    # Usually head is last
    return possible_outputs[-1] 

def infer_env(action_count: int | None) -> str:
    if action_count == 4:
        return "ALE/Breakout-v5"
    elif action_count == 6:
        return "ALE/Pong-v5 or ALE/SpaceInvaders-v5"
    else:
        return f"Unknown (actions={action_count})"

def main():
    parser = argparse.ArgumentParser(description="Scrape model checkpoints from a directory.")
    parser.add_argument("--root", type=str, required=True, help="Root directory to search for checkpoints.")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV file.")
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root)
    output_file = os.path.abspath(args.output)
    
    if not os.path.isdir(root_dir):
        print(f"Error: Root directory '{root_dir}' does not exist.")
        sys.exit(1)

    print(f"Scanning '{root_dir}' for checkpoints...")
    
    # Columns
    fieldnames = [
        'run_id', 'wandb_name', 'pruning_method', 'sparsity', 
        'action_count', 'env_id', 'seed', 'creation_time', 'dir_creation_time', 'path'
    ]
    
    results = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(f.startswith("checkpoint_") and f.endswith(".pt") for f in filenames):
            run_id = os.path.basename(dirpath)
            latest_ckpt_path = get_latest_checkpoint(dirpath)
            
            if not latest_ckpt_path:
                continue

            abs_path = os.path.abspath(latest_ckpt_path)
            
            # Filter: Only include checkpoints >= 1,000,001
            try:
                step_str = os.path.basename(abs_path).split('_')[-1].replace('.pt', '')
                if int(step_str) < 1000001:
                    continue
            except ValueError:
                continue
            
            # Metadata default values
            wandb_name = "N/A"
            pruning_method = "Unknown"
            sparsity = 0.0
            action_count = None
            env_id = "Unknown"
            seed = -1
            creation_time = "Unknown"
            dir_creation_time = "Unknown"
            
            # File creation time
            
            # File creation time
            try:
                mtime = os.path.getmtime(abs_path)
                creation_time = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            except OSError:
                pass

            # Directory creation time
            try:
                ctime = os.path.getctime(dirpath)
                dir_creation_time = datetime.datetime.fromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S')
            except OSError:
                pass
                
            # Load Checkpoint
            try:
                # Use safe globals context if possible, or just standard load
                # We need TrainingConfig to be importable
                # We map to CPU to avoid CUDA OOM or errors
                checkpoint = torch.load(abs_path, map_location='cpu', weights_only=False)
                
                # Extract Config
                if 'cfg' in checkpoint:
                    cfg = checkpoint['cfg']
                    
                    # 1. WandB Name (try-except access)
                    try:
                        # Assuming cfg might be a dataclass or OmegaConf
                        # Check recursively
                        if hasattr(cfg, 'wandb') and hasattr(cfg.wandb, 'name'):
                            wandb_name = cfg.wandb.name
                        elif isinstance(cfg, dict) and 'wandb' in cfg and 'name' in cfg['wandb']:
                            wandb_name = cfg['wandb']['name']
                    except Exception:
                         pass
                         
                    # 2. Pruning Method from delegate_stopping
                    # User: "gmp vs lth based on cfg.deletegate_stopping"
                    # LTH -> delegate_stopping=True, GMP -> delegate_stopping=False
                    try: 
                        delegate_stopping = getattr(cfg, 'delegate_stopping', None)
                        if delegate_stopping is True:
                            pruning_method = "LTH"
                        elif delegate_stopping is False:
                            pruning_method = "GMP"
                        else:
                            # if None, inspect further or leave Unknown
                            pass
                    except Exception:
                        pass
                        
                    # 3. Seed
                    seed = getattr(cfg, 'seed', -1)
                        
                # 4. Sparsity & Action Count from State Dict
                # Checkpoint usually has 'state' or 'agent' key or is the dict itself
                # RLP checkpointer saves: {'agent': ..., 'cfg': ..., 'step': ...}
                sparsity = calculate_sparsity(checkpoint)
                action_count = infer_action_count(checkpoint)
                env_id = infer_env(action_count)
                
            except Exception as e:
                print(f"⚠️ Warning processing {run_id}: {e}")
                
            results.append({
                'run_id': run_id,
                'wandb_name': wandb_name,
                'pruning_method': pruning_method,
                'sparsity': f"{sparsity:.2f}",
                'action_count': action_count if action_count is not None else "",
                'env_id': env_id,
                'seed': seed,
                'seed': seed,
                'creation_time': creation_time,
                'dir_creation_time': dir_creation_time,
                'path': abs_path
            })
            
            print(f"Processed {run_id}: {pruning_method}, S={sparsity:.2f}, Env={env_id}, Name={wandb_name}")

    # Sort results by creation_time
    # Handle possible empty strings for creation_time by putting them first (default string sort)
    results.sort(key=lambda x: x.get('creation_time', ""))

    # Write to CSV
    print(f"Writing {len(results)} entries to '{output_file}'...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("Done.")

if __name__ == "__main__":
    main()
