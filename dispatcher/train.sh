#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err

cd /scratch/$USER/RLP-25
source .venv/bin/activate
module load CUDA/12.6.0
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# 1. Set the config name from the 1st argument, defaulting to the original if empty
CONFIG_NAME=${1:-config}

# 2. Shift the arguments so that $@ contains only the remaining flags (e.g., --x=y)
if [ "$#" -ge 1 ]; then
    shift
fi

# 3. Run python with the dynamic config and any extra arguments
python src/rlp/entry/train.py --config-name "$CONFIG_NAME" "$@"

sbatch dispatch/train.sh minatar