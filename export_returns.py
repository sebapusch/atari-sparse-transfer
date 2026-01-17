#!/usr/bin/env python3
"""
Export and aggregate W&B episodic return data.

Usage:
    python export_returns.py --input_dir <folder> --output <path.csv> \\
        [--entity sebapusch-university-of-groningen] \\
        [--metric charts/episodic_return] \\
        [--bin_size 1000] \\
        [--max_step 10000000]

This script reads metadata CSVs from a directory (one per project), fetches
run history from W&B, aggregates episodic returns into fixed step bins,
and outputs a single combined CSV.
"""

import argparse
import csv
import glob
import logging
import math
import os
import sys
import time
from collections import defaultdict
from typing import Optional, List, Dict, Any

import pandas as pd
import wandb
import wandb.errors

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Constants for robust column detection
RUN_NAME_CANDIDATES = ["run_name", "name", "Name", "run", "Run", "display_name"]
SEED_CANDIDATES = ["seed", "config.seed", "Seed", "run_seed"]
RUN_ID_CANDIDATES = ["id", "run_id"]

def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find the first matching column name from candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None

def get_bin(step: int, bin_size: int) -> int:
    """
    Calculate bin edge for a given step.
    Bins: (0, 1000], (1000, 2000], ...
    Returns upper edge: 1000, 2000, ...
    """
    if step <= 0:
        return 0  # Should not happen for valid steps, or handle as 0
    # math.ceil(step / bin_size) * bin_size equivalent
    # Using integer arithmetic:
    return ((step - 1) // bin_size + 1) * bin_size

def safe_api_call(func, *args, retries=3, backoff_factor=2, **kwargs):
    """Execute a function with exponential backoff retries."""
    last_exception = None
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except (wandb.errors.CommError, wandb.errors.UsageError, ConnectionError, TimeoutError) as e:
            last_exception = e
            wait_time = backoff_factor ** attempt
            logger.warning(f"API call failed: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            # Non-retriable error
            raise e
    raise last_exception

def process_history(
    run,
    metric_key: str,
    bin_size: int,
    max_step: int,
    progress_callback=None
) -> Dict[int, float]:
    """
    Scan run history and aggregate metric into bins.
    Returns: Dict[step_bin, aggregated_value]
    """
    # Hardcoded step key as identified from data
    step_key = "_step"
    
    # We aggregate into a dict: bin_edge -> list/sum of values
    # To save memory, we can store count and sum.
    bin_sums = defaultdict(float)
    bin_counts = defaultdict(int)
    
    # helper for formatting large numbers
    def fmt_steps(n):
        if n >= 1_000_000:
            return f"{n/1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n/1_000:.1f}K"
        return str(n)

    if progress_callback:
        progress_callback("Downloading history...")

    try:
        # Request only step key and metric
        # Use max_step as samples limit to ensure full coverage
        df = run.history(keys=[step_key, metric_key], pandas=True, samples=max_step)
        
        if df.empty:
            if progress_callback:
                progress_callback("History DF is empty.")
            return {}
            
        if progress_callback:
            progress_callback(f"Processing {len(df)} rows...")
            
        # Ensure numeric
        if step_key not in df.columns:
            if progress_callback:
                progress_callback(f"Step key '{step_key}' not found in data.")
            return {}
            
        df[step_key] = pd.to_numeric(df[step_key], errors='coerce')
        
        if metric_key in df.columns:
            df[metric_key] = pd.to_numeric(df[metric_key], errors='coerce')
        else:
            if progress_callback:
                progress_callback(f"Metric '{metric_key}' not found in data.")
            return {}
            
        # Filter rows
        df = df[df[step_key] <= max_step]
        df = df.dropna(subset=[step_key, metric_key])
        
        for _, row in df.iterrows():
            step = int(row[step_key])
            val = row[metric_key]
            
            bin_edge = get_bin(step, bin_size)
            if bin_edge > max_step:
                continue
                
            bin_sums[bin_edge] += val
            bin_counts[bin_edge] += 1
            
    except Exception as e:
        logger.warning(f"Error fetching history for run {run.name}: {e}")
        return {}
        logger.warning(f"Error scanning history for run {run.name}: {e}")
        return {}

    # Compute means
    c = {}
    for b, count in bin_counts.items():
        if count > 0:
            c[b] = bin_sums[b] / count
            
    return c

def parse_filename(filename: str) -> tuple[str, Dict[str, str]]:
    """
    Parse filename to extract project name and tags.
    Format: project[key:val,key2:val2].csv
    Returns: (project_name, tags_dict)
    """
    base = os.path.splitext(filename)[0]
    if "[" in base and base.endswith("]"):
        try:
            project_part, tags_part = base.split("[", 1)
            tags_part = tags_part[:-1] # Remove trailing ]
            
            tags = {}
            if tags_part:
                for item in tags_part.split(","):
                    if ":" in item:
                        k, v = item.split(":", 1)
                        tags[k.strip()] = v.strip()
            
            return project_part.strip(), tags
        except ValueError:
            pass # Fallback
            
    return base, {}

def main():
    parser = argparse.ArgumentParser(description="Export W&B returns to CSV.")
    parser.add_argument("--input_dir", required=True, help="Directory containing project CSVs")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--entity", default="sebapusch-university-of-groningen", help="W&B Entity")
    parser.add_argument("--metric", default="charts/episodic_return", help="Metric to export")
    parser.add_argument("--bin_size", type=int, default=1000, help="Step aggregation bin size")
    parser.add_argument("--max_step", type=int, default=10_000_000, help="Maximum step to process")
    parser.add_argument("--dry-run", action="store_true", help="Validate parsing without downloading")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)
        
    api = wandb.Api(timeout=20)
    
    # Find CSV files
    input_files = glob.glob(os.path.join(args.input_dir, "*.csv"))
    if not input_files:
        logger.warning(f"No CSV files found in {args.input_dir}")
        return

    logger.info(f"Found {len(input_files)} CSV files.")
    
    # Pre-scan for tags
    all_tag_keys = set()
    file_info = [] # List of (filepath, project_name, tags)
    
    for filepath in sorted(input_files):
        filename = os.path.basename(filepath)
        project_name, tags = parse_filename(filename)
        file_info.append((filepath, project_name, tags))
        all_tag_keys.update(tags.keys())
        
    sorted_tag_keys = sorted(list(all_tag_keys))
    if sorted_tag_keys:
        logger.info(f"Detected tags: {sorted_tag_keys}")
    
    # Initialize output CSV
    base_headers = ["run_name", "seed", "step", "episodic_return", "project"]
    output_headers = base_headers + sorted_tag_keys
    
    try:
        with open(args.output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(output_headers)
    except IOError as e:
        logger.error(f"Could not create output file {args.output}: {e}")
        sys.exit(1)
    
    total_runs_processed = 0
    
    for filepath, project_name, tags in file_info:
        filename = os.path.basename(filepath)
        logger.info(f"Processing project: {project_name} (from {filename})")
        
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            logger.error(f"Failed to read {filename}: {e}")
            continue
            
        # Detect columns
        run_col = find_column(df, RUN_NAME_CANDIDATES)
        seed_col = find_column(df, SEED_CANDIDATES)
        id_col = find_column(df, RUN_ID_CANDIDATES)
        
        if not run_col or not seed_col:
            logger.error(f"Missing columns in {filename}. "
                         f"Run candidates: {RUN_NAME_CANDIDATES}, "
                         f"Seed candidates: {SEED_CANDIDATES}")
            logger.error(f"Available columns: {list(df.columns)}")
            sys.exit(1)
            
        logger.info(f"  Columns detected -> Run: '{run_col}', Seed: '{seed_col}', ID: '{id_col}'")
        
        # Iterate rows
        rows = df.to_dict('records')
        
        # Simple progress tracking
        total = len(rows)
        print(f"  -> Found {total} runs in {filename}")
        
        for i, row in enumerate(rows):
            run_name = row.get(run_col, "Unknown")
            seed_val = row.get(seed_col, "Unknown")
            
            # Helper to update status line
            prefix = f"    [{i+1}/{total}] {run_name}: "
            def report(status):
                # Pad to overwrite previous text
                sys.stdout.write(f"\r{prefix}{status}".ljust(100))
                sys.stdout.flush()

            report("Starting...")
            
            # Cast seed to int if possible
            try:
                seed_val = int(seed_val)
            except (ValueError, TypeError):
                pass # Keep as is if string
                
            if args.dry_run:
                report("Dry Run - Skipped")
                # Small delay to make it visible in dry run
                time.sleep(0.05)
                continue

            run = None
            # Fetch run
            report("Fetching run metadata...")
            try:
                if id_col and pd.notna(row.get(id_col)):
                    run_id = row[id_col]
                    full_path = f"{args.entity}/{project_name}/{run_id}"
                    run = safe_api_call(api.run, full_path)
                else:
                    # Search by name
                    # Note: display_name in filters usually maps to run name in UI
                    runs = safe_api_call(api.runs, path=f"{args.entity}/{project_name}", filters={"display_name": run_name})
                    
                    if len(runs) == 0:
                        report("Run not found!")
                        # Log detailed error on new line so it isn't overwritten
                        sys.stdout.write("\n")
                        logger.warning(f"Run '{run_name}' not found in {project_name}.")
                        continue
                    elif len(runs) == 1:
                        run = runs[0]
                    else:
                        # Multiple matches: pick most recent
                        runs = sorted(runs, key=lambda r: r.created_at, reverse=True)
                        run = runs[0]
                        report("Multiple runs found (using newest)")
                        logger.warning(f"\n    Multiple runs found for '{run_name}'. Using most recent (ID: {run.id}).")
                        
            except Exception as e:
                report("Error fetching run")
                sys.stdout.write("\n")
                logger.error(f"Failed to fetch run '{run_name}': {e}")
                continue
            
            if not run:
                continue
                
            # Aggregate history
            report("Scanning history...")
            agg_data = process_history(run, args.metric, args.bin_size, args.max_step, progress_callback=report)
            
            if not agg_data:
                # No data found or error
                report("No valid data found")
                continue
                
            # Write to CSV
            report("Writing results...")
            # Sort by step
            sorted_steps = sorted(agg_data.keys())
            
            rows_to_write = []
            for s in sorted_steps:
                val = agg_data[s]
                # Base row + tags
                row_data = [run_name, seed_val, s, val, project_name]
                # Add tag values
                for k in sorted_tag_keys:
                    row_data.append(tags.get(k, ""))
                
                rows_to_write.append(row_data)
                
            if rows_to_write:
                with open(args.output, 'a', newline='') as f:
                    w = csv.writer(f)
                    w.writerows(rows_to_write)
            
            report("Done")
            total_runs_processed += 1

        print("") # Newline after progress bar

    logger.info(f"Done. Processed {total_runs_processed} runs successfully. Output at {args.output}")

if __name__ == "__main__":
    main()
