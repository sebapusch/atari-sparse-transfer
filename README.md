# RLP-25: Deep Reinforcement Learning Scaffolding

This project provides a modular and decoupled scaffolding for Deep Reinforcement Learning research, focusing on sparse networks. It supports various algorithms, environments, and pruning techniques, with first-class Weights & Biases integration.

## Installation

### Prerequisites

*   Python 3.12 or higher
*   [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd rlp-25
    ```

2.  **Install dependencies:**

    Using `uv` (recommended):
    ```bash
    uv sync
    ```

    Using `pip`:
    ```bash
    pip install -e .
    ```

## Running Experiments

The main entry point for running experiments is `src/rlp/entry/train.py`. The project uses [Hydra](https://hydra.cc/) for configuration management.

### Basic Usage

To run an experiment with default settings:

```bash
python src/rlp/entry/train.py
To run experiments, you need to ensure the `rlp` package is in your python path. You can do this by installing it in editable mode:

```bash
pip install -e .
```

Or by setting `PYTHONPATH`:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```

### Basic Training

```bash
python src/rlp/entry/train.py algorithm=dqn env=atari env.id=BreakoutNoFrameskip-v4
```

> [!IMPORTANT]
> For Atari environments, you **must** use the `NoFrameskip-v4` versions (e.g., `PongNoFrameskip-v4`, `BreakoutNoFrameskip-v4`). 
> Other versions (like `Pong-v4` or `Pong-v5`) do not trigger the necessary preprocessing wrappers (resizing, grayscale), leading to massive memory usage and crashes.

### With Weights & Biases

To track experiments with WandB:

```bash
python src/rlp/entry/train.py algorithm=dqn env=atari env.id=PongNoFrameskip-v4 wandb.enabled=true wandb.project=rlp-25 wandb.group=dqn-pong
```

> [!IMPORTANT]
> When `wandb.enabled=true`, you **must** provide a `wandb.group` (e.g., `wandb.group=experiment-name`). This ensures that runs are organized and comparable. The logger will raise an error if the group is missing.

### Custom Configuration

You can launch experiments with custom configurations using YAML files.

**Option 1: Component Configs**
Create new config files in the respective directories (e.g., `configs/algorithm/my_dqn.yaml`) and select them via the command line:

```bash
python src/rlp/entry/train.py algorithm=my_dqn
```

**Option 2: Full Config File**
Create a new config file in `configs/` (e.g., `configs/my_experiment.yaml`) that overrides defaults:

```yaml
# configs/my_experiment.yaml
defaults:
  - config

algorithm:
  learning_starts: 1000
```

Then run with `--config-name`:

```bash
python src/rlp/entry/train.py --config-name my_experiment
```

### Overriding Configuration

You can override any configuration parameter from the command line.

**Example: Run DQN on Atari Breakout**

```bash
python src/rlp/entry/train.py algorithm=dqn env=atari env.id=BreakoutNoFrameskip-v4
```

**Example: Enable Weights & Biases**

```bash
python src/rlp/entry/train.py wandb.enabled=true wandb.project=my-rl-project
```

**Example: Change Seed and Device**

```bash
python src/rlp/entry/train.py seed=42 device=cpu
```

## Configuration Structure

The configuration files are located in the `configs/` directory:

*   `config.yaml`: Main configuration file with defaults.
*   `algorithm/`: Algorithm-specific configs (e.g., `dqn.yaml`).
*   `env/`: Environment configs (e.g., `atari.yaml`).
*   `pruning/`: Pruning technique configs.
*   `train/`: Training loop settings.

## Project Structure

*   `src/rlp`: Source code package.
    *   `agent/`: Agent implementations.
    *   `algorithm/`: Algorithm logic.
    *   `components/`: Reusable network components.
    *   `core/`: Core utilities (logger, trainer).
    *   `entry/`: Entry point scripts.
    *   `env/`: Environment wrappers and factory.
    *   `pruning/`: Pruning engine and schedulers.
*   `configs`: Hydra configuration files.

## Configuration Reference

### Global Settings (`config.yaml`)

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `seed` | `1` | Random seed for reproducibility. |
| `device` | `"cuda"` | Device to run on (`"cuda"` or `"cpu"`). |
| `wandb.enabled` | `false` | Enable Weights & Biases logging. |
| `wandb.project` | `"rlp-sparse"` | W&B project name. |
| `wandb.entity` | `null` | W&B entity (username or team). |
| `wandb.group` | `null` | W&B run group. |
| `wandb.tags` | `[]` | List of W&B tags. |
| `wandb.job_type` | `"train"` | W&B job type. |
| `wandb.name` | `null` | W&B run name (auto-generated if null). |
| `output_dir` | `${hydra:runtime.output_dir}` | Directory for output files. |

### Algorithm (`configs/algorithm/dqn.yaml`)

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `algorithm.name` | `"dqn"` | Name of the algorithm. |
| `algorithm.network.encoder` | `"nature_cnn"` | Encoder type (`"nature_cnn"`, `"minatar_cnn"`). |
| `algorithm.network.head` | `"linear"` | Head type (`"linear"`, `"dueling"`). |
| `algorithm.network.hidden_dim` | `512` | Hidden dimension size. |
| `algorithm.gamma` | `0.99` | Discount factor. |
| `algorithm.tau` | `1.0` | Soft update coefficient (1.0 = hard update). |
| `algorithm.target_network_frequency` | `1000` | Steps between target network updates. |
| `algorithm.batch_size` | `32` | Training batch size. |
| `algorithm.learning_starts` | `80000` | Steps before learning starts. |
| `algorithm.train_frequency` | `4` | Steps between training updates. |
| `algorithm.epsilon.start` | `1.0` | Starting epsilon for exploration. |
| `algorithm.epsilon.end` | `0.01` | Final epsilon. |
| `algorithm.epsilon.decay_fraction` | `0.10` | Fraction of total timesteps for epsilon decay. |
| `algorithm.optimizer.lr` | `1e-4` | Learning rate. |
| `algorithm.optimizer.eps` | `1e-8` | Optimizer epsilon. |

### Environment (`configs/env/atari.yaml`)

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `env.id` | `"BreakoutNoFrameskip-v4"` | Gym environment ID. |
| `env.num_envs` | `1` | Number of parallel environments. |
| `env.capture_video` | `false` | Whether to capture video of the agent. |
| `env.frame_stack` | `4` | Number of frames to stack. |
| `env.grayscale` | `true` | Convert observations to grayscale. |
| `env.resize` | `84` | Resize observations to this dimension (square). |
| `env.noop_max` | `30` | Maximum number of no-ops at start of episode. |
| `env.scale_obs` | `false` | Scale observations to [0, 1] (usually handled by network). |
| `env.clip_rewards` | `true` | Clip rewards to [-1, 1]. |

### Training (`configs/train/default.yaml`)

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `train.total_timesteps` | `10000000` | Total training timesteps. |
| `train.buffer_size` | `1000000` | Replay buffer size. |
| `train.save_model` | `false` | Whether to save the model. |
| `train.upload_model` | `false` | Whether to upload the model to Hugging Face. |
| `train.hf_entity` | `""` | Hugging Face entity for model upload. |
| `train.checkpoint_interval` | `100000` | Steps between checkpoints. |
| `train.log_interval` | `100` | Steps between logging. |

### Pruning (`configs/pruning/none.yaml`)

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `pruning.method` | `"none"` | Pruning method. |

### Pruning - GMP (`configs/pruning/gmp.yaml`)

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `pruning.method` | `"gmp"` | Pruning method. |
| `pruning.initial_sparsity` | `0.0` | Initial sparsity level (0.0 - 1.0). |
| `pruning.final_sparsity` | `0.8` | Final target sparsity level (0.0 - 1.0). |
| `pruning.start_step` | `0` | Step to start pruning. |
| `pruning.end_step` | `null` | Step to stop pruning (if null, uses total_timesteps). |
| `pruning.update_frequency` | `1000` | Frequency of pruning updates in steps. |
| `pruning.scheduler` | `"cubic"` | Sparsity scheduler (`"linear"`, `"cubic"`). |
