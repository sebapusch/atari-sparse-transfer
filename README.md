# RLP-25: Deep Reinforcement Learning Scaffolding

RLP-25 is a modular and decoupled scaffolding for Deep Reinforcement Learning (DRL) research, designed for flexibility and ease of use. It supports various algorithms (DQN, DDQN), environments (Atari, MinAtar), and advanced features like network pruning, with first-class integration for [Weights & Biases](https://wandb.ai/).

## 🚀 Getting Started

### Installation

**Prerequisites**: Python 3.12+

Recommended installation via [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Or via pip:

```bash
pip install -e .
```

### Quick Start

Run a default training session (DQN on BreakoutNoFrameskip-v4):

```bash
python src/rlp/entry/train.py
```

---

## ⚙️ Configuration Guide

This project uses [Hydra](https://hydra.cc/) for configuration management. All configurations are located in the `configs/` directory. You can override any parameter from the command line.

### Key Configuration Parameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| **Global** | | |
| `seed` | `1` | Random seed for reproducibility. |
| `device` | `"cuda"` | Computation device (`"cuda"` or `"cpu"`). |
| `output_dir` | (hydra) | Directory for logs and checkpoints. |
| **WandB** | | |
| `wandb.enabled` | `false` | Enable Weights & Biases logging. |
| `wandb.project` | `"rlp-sparse"` | W&B project name. |
| `wandb.entity` | `null` | W&B username or team. |
| `wandb.group` | `null` | Group name for organizing runs (Mandatory if enabled). |
| `wandb.id` | `null` | **Unique Run ID** (used for resuming). |
| **Training** | | |
| `train.total_timesteps` | `10M` | Total interaction steps. |
| `train.checkpoint_interval`| `100k` | Frequency of saving model checkpoints. |
| `train.save_model` | `false` | Save final model. |

### 🔄 Resuming Runs

To resume a training run, you must provide the **WandB Run ID**. This ensures that the logger attaches to the existing run and the checkpointer loads the correct state.

```bash
python src/rlp/entry/train.py wandb.enabled=true wandb.id=YOUR_RUN_ID
```

The system will:
1.  Initialize WandB with `resume="allow"` and the specified `id`.
2.  Look for checkpoints in the directory associated with that `id`.
3.  Resume training from the last saved step.

---

## 💡 Usage Examples

### 1. Basic Training (Atari)
Train a DQN agent on Pong. Note: Always use `NoFrameskip-v4` environments.

```bash
python src/rlp/entry/train.py algorithm=dqn env.id=PongNoFrameskip-v4
```

### 2. MinAtar Environments
Use the `minatar` config group and specify a MinAtar environment ID.

```bash
python src/rlp/entry/train.py --config-name=minatar env.id="MinAtar/breakout"
```

### 3. Training with Weights & Biases
Enable logging and specify a group for organization.

```bash
python src/rlp/entry/train.py \
    wandb.enabled=true \
    wandb.project=my-rl-project \
    wandb.group=experiment-v1 \
    wandb.name=dqn-breakout-run1
```

### 4. Resuming an Interrupted Run
If a run with ID `3xample1d` crashed or was stopped, resume it exactly where it left off:

```bash
python src/rlp/entry/train.py \
    wandb.enabled=true \
    wandb.id=3xample1d
```

### 5. Running with Custom Hyperparameters
Override specific parameters directly:

```bash
python src/rlp/entry/train.py \
    algorithm.learning_starts=1000 \
    algorithm.batch_size=64 \
    train.total_timesteps=500000
```
