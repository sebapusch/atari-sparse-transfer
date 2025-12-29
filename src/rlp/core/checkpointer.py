import os
import glob
import time
import torch
import wandb
from typing import Any, List, Optional


def _latest_in_dir(directory: str) -> str | None:
    """
    Helper to find the numerically highest checkpoint in a specific directory.
    Solves the 'checkpoint_100 vs checkpoint_99' sorting bug.
    """
    files = glob.glob(os.path.join(directory, "*.pt"))
    if not files:
        return None

    def extract_step(filename: str) -> int:
        # Format: "checkpoint_123.pt" -> 123
        try:
            return int(filename.split('_')[-1].replace('.pt', ''))
        except ValueError:
            return -1  # Handle unexpected filenames gracefully

    # Sort by the integer step number
    files = sorted(files, key=lambda x: extract_step(os.path.basename(x)))
    return files[-1]


class Checkpointer:
    def __init__(self, checkpoint_base_dir: str, run_id: str | None, entity: str | None = None, project: str | None = None):
        # 1. Priority: Manual ID > WandB ID > Timestamp
        if run_id is not None:
            self.run_id = run_id
        elif wandb.run is not None:
            self.run_id = wandb.run.id
        else:
            self.run_id = str(int(time.time()))

        self.entity = entity
        self.project = project

        # 2. Only use WandB features if WandB is actually active,
        # and we are running the correct run ID.
        self.wandb_enabled = (wandb.run is not None) and (self.run_id == wandb.run.id)

        self.checkpoint_dir = os.path.join(checkpoint_base_dir, self.run_id)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(self,
             global_step: int,
             state: dict[str, Any],
             aliases: List[str] | None = None,
             metadata: dict[str, Any] = None) -> None:
        """
        Saves locally and optionally uploads to WandB.
        """
        if aliases is None:
            aliases = ['latest']

        # 1. Save Locally
        filename = f"checkpoint_{global_step}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)

        tmp_path = filepath + f".tmp.{os.getpid()}"
        torch.save(state, tmp_path)
        os.replace(tmp_path, filepath)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 2. Upload to WandB
        if self.wandb_enabled:
            artifact = wandb.Artifact(
                name=f"model-{self.run_id}",
                type="model",
                metadata=metadata or {"step": global_step}
            )
            artifact.add_file(filepath)
            wandb.log_artifact(artifact, aliases=aliases)

        # 3. Cleanup old local files (Keep top 3 to save disk space)
        self._cleanup_local()

    def load(self, device: torch.device) -> dict[str, Any] | None:
        """
        Loads latest checkpoint from WandB (if available) or local disk.
        """
        filepath = None

        # A. Try downloading from WandB
        # A. Try downloading from WandB
        # We attempt this if we have a run_id, using the static helper.
        # This handles both active runs (wandb_enabled) and inactive resumes.
        if self.run_id and (self.wandb_enabled or (self.project and self.entity)): 
             # Use the static download method which handles api/active run logic
             downloaded = Checkpointer.download_checkpoint(
                 run_id=self.run_id,
                 artifact_alias='latest',
                 download_root=os.path.dirname(self.checkpoint_dir), # download_checkpoint appends run_id
                 entity=self.entity,
                 project=self.project
             )
             if downloaded:
                 # Check if the downloaded file is newer than what we might have locally
                 # (If download_checkpoint returns a path, it just downloaded or found it)
                 filepath = downloaded

        # B. Fallback: Search local directory (if WandB failed or returned nothing)
        if filepath is None:
            filepath = _latest_in_dir(self.checkpoint_dir)
            if filepath:
                print(f"✅ Checkpointer: Found local checkpoint: {filepath}")

        # C. Load the file
        if filepath:
            # Local import to prevent circular dependency
            from rlp.core.trainer import TrainingConfig
            try:
                with torch.serialization.safe_globals([TrainingConfig]):
                    return torch.load(filepath, map_location=device)
            except AttributeError:
                # Fallback for older PyTorch versions without safe_globals
                 return torch.load(filepath, map_location=device)

        print("🆕 Checkpointer: No checkpoint found. Starting fresh.")
        return None

    def _cleanup_local(self, keep: int = 2):
        """Removes old checkpoints from the local directory."""
        # Reuse the robust sorting logic
        files = glob.glob(os.path.join(self.checkpoint_dir, "*.pt"))

        def extract_step(filename: str):
            try:
                return int(filename.split('_')[-1].replace('.pt', ''))
            except ValueError:
                return -1

        sorted_files = sorted(files, key=lambda x: extract_step(os.path.basename(x)))

        if len(sorted_files) > keep:
            for f in sorted_files[:-keep]:
                try:
                    os.remove(f)
                except OSError:
                    pass

    @staticmethod
    def download_checkpoint(run_id: str,
                            artifact_alias: str = 'latest',
                            download_root: str = '/tmp/checkpoints',
                            entity: Optional[str] = None,
                            project: Optional[str] = None) -> str | None:
        """
        Static helper to fetch a checkpoint from a remote WandB run.
        """
        if wandb is None:
            print("⚠️ Checkpointer: WandB not active, cannot download checkpoint.")
            return None

        # Determine entity/project/api
        api = wandb.Api() 
        
        if wandb.run:
             entity = entity or wandb.run.entity
             project = project or wandb.run.project
        
        if not entity or not project:
            print("⚠️ Checkpointer: Entity and Project must be specified (or active run) to download checkpoint.")
            return None

        try:
            print(f"🔄 Checkpointer: Fetching remote artifact {entity}/{project}/model-{run_id}:{artifact_alias}...")
            artifact_path = f"{entity}/{project}/model-{run_id}:{artifact_alias}"
            artifact = wandb.use_artifact(artifact_path)

            # Download returns the directory path
            save_dir = os.path.join(download_root, run_id)
            os.makedirs(save_dir, exist_ok=True)
            download_dir = artifact.download(root=save_dir)

            # Find latest in that dir
            filepath = _latest_in_dir(download_dir)
            
            if filepath:
                print(f"✅ Checkpointer: Downloaded checkpoint to: {filepath}")
                return filepath
            else:
                print(f"⚠️ Checkpointer: No .pt files found in downloaded artifact.")
                return None

        except (wandb.errors.CommError, wandb.errors.UsageError) as e:
            print(f"⚠️ Checkpointer: WandB download failed ({e}).")
            return None