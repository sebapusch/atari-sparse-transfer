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
        filepath, step = Checkpointer.download_checkpoint(
            run_id=self.run_id,
            artifact_alias='latest',
            download_root=os.path.dirname(self.checkpoint_dir), # download_checkpoint appends run_id
            entity=self.entity,
            project=self.project
        )

        # A. Try downloading from WandB
        # A. Try downloading from WandB
        # We attempt this if we have a run_id, using the static helper.
        # This handles both active runs (wandb_enabled) and inactive resumes.
        # if self.run_id and (self.wandb_enabled or (self.project and self.entity)): 
        #      # Use the static download method which handles api/active run logic
        #      downloaded = Checkpointer.download_checkpoint(
        #          run_id=self.run_id,
        #          artifact_alias='latest',
        #          download_root=os.path.dirname(self.checkpoint_dir), # download_checkpoint appends run_id
        #          entity=self.entity,
        #          project=self.project
        #      )
        #      if downloaded:
        #          # Check if the downloaded file is newer than what we might have locally
        #          # (If download_checkpoint returns a path, it just downloaded or found it)
        #          filepath = downloaded

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
                    state = torch.load(filepath, map_location=device)
                    state['step'] = step
                    return state
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
                            project: Optional[str] = None) -> tuple[str, int] | None:
        """
        Static helper to fetch a checkpoint from a remote WandB run.
        """
        if wandb is None:
            print("⚠️ Checkpointer: WandB not active, cannot download checkpoint.")
            return None

        # Determine entity/project/api
        api = wandb.Api()

        run = api.run(f"{project}/{run_id}")

        artifacts = list(run.logged_artifacts())
        artifacts_sorted = sorted(artifacts, key=lambda a: a.created_at)

        try:
            # Construct path parts
            path_prefix = f"{entity}/{project}" if entity else project
            artifact_path = f"{path_prefix}/model-{run_id}:{artifact_alias}"
            
            print(f"🔄 Checkpointer: Fetching remote artifact {artifact_path}...")
            
            # 1. Try exact match first
            # 1. Try exact match first
            try:
                # Find artifact matching alias
                found_artifact = artifacts_sorted[-2]
                
                if found_artifact:
                    artifact = found_artifact
                else:
                    # Default to latest if provided alias not found (or if alias is 'latest' but not tagged)
                    artifact = artifacts_sorted[-2]
                
                print(f"🔄 Checkpointer: Selected artifact: {artifact.name} (aliases: {artifact.aliases})")

                save_dir = os.path.join(download_root, run_id)
                os.makedirs(save_dir, exist_ok=True)
                download_dir = artifact.download(root=save_dir)
            except (wandb.errors.CommError, wandb.errors.UsageError) as e:
                print(f"⚠️ Checkpointer: Specific artifact {artifact_path} not found ({e}). Checking for ANY model artifact...")
                
                # 2. Fallback: Search for any 'model' type artifact in this run
                # access the run object via API
                run_path = f"{path_prefix}/{run_id}"
                run = api.run(run_path)
                artifacts = run.artifacts(type="model")
                
                if not artifacts:
                    print(f"⚠️ Checkpointer: No 'model' artifacts found for run {run_path}.")
                    return None
                    
                # Sort by updated_at to get the latest
                # artifacts is a collection, let's list it
                artifacts_list = list(artifacts)
                
                target_artifact = artifacts_list[0]
                print(f"✅ Checkpointer: Found alternative artifact: {target_artifact.name}")
                
                save_dir = os.path.join(download_root, run_id)
                os.makedirs(save_dir, exist_ok=True)
                download_dir = target_artifact.download(root=save_dir)

            # Find latest in that dir
            filepath = _latest_in_dir(download_dir)
            
            if filepath:
                print(f"✅ Checkpointer: Downloaded checkpoint to: {filepath}")
                return filepath, artifact.history_step
            else:
                print(f"⚠️ Checkpointer: No .pt files found in downloaded artifact.")
                return None

        except Exception as e:
            print(f"⚠️ Checkpointer: WandB download failed ({e}).")
            return None

    @staticmethod
    def download_artifact_by_path(path: str, download_root: str = '/tmp/checkpoints') -> tuple[str, int, dict] | None:
        """
        Downloads a WandB artifact by its full path (e.g., 'entity/project/name:alias').
        Returns (filepath, step, artifact_metadata).
        """
        if wandb is None:
            print("⚠️ Checkpointer: WandB not active, cannot download artifact.")
            return None
            
        api = wandb.Api()
        
        try:
            print(f"🔄 Checkpointer: Fetching artifact {path}...")
            artifact = api.artifact(path)
            
            # Use artifact ID to avoid collisions
            save_dir = os.path.join(download_root, artifact.id)
            os.makedirs(save_dir, exist_ok=True)
            
            download_dir = artifact.download(root=save_dir)
            
            # Find the actual .pt file
            filepath = _latest_in_dir(download_dir)
            
            if filepath:
                print(f"✅ Checkpointer: Downloaded artifact to: {filepath}")
                # Try to get step from metadata or aliases
                step = 0
                if hasattr(artifact, 'history_step'):
                    step = artifact.history_step
                elif artifact.metadata and 'step' in artifact.metadata:
                    step = artifact.metadata['step']
                    
                return filepath, step, artifact.metadata
            else:
                print(f"⚠️ Checkpointer: No .pt files found in artifact {path}.")
                return None
                
        except Exception as e:
            print(f"⚠️ Checkpointer: Failed to download artifact {path}: {e}")
            return None