import csv
import argparse
import sys
import os
from datetime import datetime, timedelta, timezone
import re

def parse_iso_z(date_str):
    """Parses ISO 8601 with Z, e.g. 2026-01-14T13:41:05.000Z"""
    if not date_str:
        return None
    # Python 3.11+ supports Z with fromisoformat, but to be safe for 3.10-:
    if date_str.endswith('Z'):
        date_str = date_str[:-1] + '+00:00'
    return datetime.fromisoformat(date_str)

def parse_local(date_str):
    """Parses YYYY-MM-DD HH:MM:SS assuming it is Local Time (+01:00 as per system metadata)"""
    # System reported: 2026-01-17T... +01:00
    # So we treat the string as +01:00
    if not date_str or date_str == "Unknown":
        return None
    dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    # Assign +01:00 timezone
    tz = timezone(timedelta(hours=1))
    return dt.replace(tzinfo=tz)

def main():
    parser = argparse.ArgumentParser(description="Match runs to checkpoints.")
    parser.add_argument("--runs", required=True, help="Path to runs CSV (e.g. runs-gmp.csv)")
    parser.add_argument("--checkpoints", required=True, help="Path to checkpoints CSV")
    parser.add_argument("--output", required=True, help="Path to output matched CSV")
    parser.add_argument("--tolerance", type=int, default=3600, help="Time tolerance in seconds for match (default 1h)")
    args = parser.parse_args()

    # Load Checkpoints
    print(f"Loading checkpoints from {args.checkpoints}...")
    checkpoints = []
    with open(args.checkpoints, 'r') as f:
        # Handle potential whitespace in headers
        header_line = f.readline()
        fieldnames = [h.strip() for h in header_line.split(',')]
        reader = csv.DictReader(f, fieldnames=fieldnames)
        
        for row in reader:
            # Strip whitespace from values
            row = {k: v.strip() for k, v in row.items() if v is not None}
            
            # Parse useful fields
            row['dt_creation'] = parse_local(row.get('creation_time'))
            row['dt_dir_creation'] = parse_local(row.get('dir_creation_time'))
            
            # Normalize env_id for matching
            # Checkpoint env_id might be "ALE/Pong-v5 or ALE/SpaceInvaders-v5"
            eid = row.get('env_id', '').lower()
            if 'pong' in eid:
                row['env_keys'] = ['pong']
            elif 'spaceinvaders' in eid or 'space_invaders' in eid: # handle both casing/spacing
                row['env_keys'] = ['spaceinvaders']
            elif 'breakout' in eid:
                row['env_keys'] = ['breakout']
            elif 'or' in eid:
                # Ambiguous, could be any of the listed
                # "ALE/Pong-v5 or ALE/SpaceInvaders-v5" -> ['pong', 'spaceinvaders']
                row['env_keys'] = []
                if 'pong' in eid: row['env_keys'].append('pong')
                if 'spaceinvaders' in eid: row['env_keys'].append('spaceinvaders')
            else:
                row['env_keys'] = []
            
            checkpoints.append(row)

    # Load Runs
    print(f"Loading runs from {args.runs}...")
    matches = []
    
    with open(args.runs, 'r') as f:
        reader = csv.DictReader(f)
        for run in reader:
            # Strip whitespace from run dict just in case
            run = {k: v.strip() for k, v in run.items() if v is not None}
            
            run_name = run.get('Name')
            run_seed = run.get('seed')
            run_env = run.get('env.id', '').lower()
            run_end_time_str = run.get('End Time')
            
            # Filter valid runs
            if not run_end_time_str:
                continue

            dt_end = parse_iso_z(run_end_time_str)
            
            # 1. Filter by Seed
            candidates = [c for c in checkpoints if c.get('seed') == run_seed]
            
            # 2. Filter by Env
            # Run env: "ale/breakout-v5" -> key "breakout"
            run_env_key = None
            if 'pong' in run_env: run_env_key = 'pong'
            elif 'breakout' in run_env: run_env_key = 'breakout'
            elif 'spaceinvaders' in run_env: run_env_key = 'spaceinvaders'
            
            if run_env_key:
                candidates = [c for c in candidates if run_env_key in c['env_keys']]
            
            # 3. Find closest time match
            # We compare checkpoint file creation time vs run end time
            best_match = None
            min_diff = float('inf')
            match_status = "No Match"
            matched_ckpt_path = ""
            
            for ckpt in candidates:
                if not ckpt['dt_creation']:
                    continue
                
                # timestamps might differ. Run end time is when WandB stopped recording.
                # Checkpoint creation time is file mtime.
                # Assuming they are close.
                diff = abs((ckpt['dt_creation'] - dt_end).total_seconds())
                
                if diff < args.tolerance: # Within tolerance
                    if diff < min_diff:
                        min_diff = diff
                        best_match = ckpt

            if best_match:
                matched_ckpt_path = best_match['path']
                match_status = "Match"
                # Check for ambiguity: are there other candidates with very similar time?
                # (Optional refinement)
                if min_diff > 600: # If diff is large (>10 mins), mark as "Weak Match"
                     match_status = f"Weak Match ({int(min_diff)}s diff)"
            else:
                # Fallback: check dir_creation vs run start? 
                # User specifically asked for creation_time matching End Time.
                pass
                
            matches.append({
                'run_name': run_name,
                'run_end_time': run_end_time_str,
                'match_status': match_status,
                'checkpoint_path': matched_ckpt_path,
                'time_diff_seconds': int(min_diff) if best_match else "",
                'env_id': run.get('env.id'),
                'seed': run_seed
            })

    # Write Output
    print(f"Writing matches to {args.output}...")
    headers = ['run_name', 'run_end_time', 'match_status', 'checkpoint_path', 'time_diff_seconds', 'env_id', 'seed']
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(matches)
        
    print("Done.")

if __name__ == "__main__":
    main()
