#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

ENV_NAME="kimi_env"
PYTHON_VERSION="3.10"

if conda info --envs | awk '{print $1}' | grep -Eq "^${ENV_NAME}$"; then
    echo "Environment '$ENV_NAME' already exists. Updating..."
else
    echo "Creating custom MLLM conda environment '$ENV_NAME'..."
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
fi

# Initialize conda for the script so 'conda activate' works
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo "Installing CUDA 12.4 Toolkit..."
conda install -c nvidia cuda-toolkit=12.4 -y
export CUDA_HOME=$CONDA_PREFIX

# NOTE: Pinned to 2.5.1 to prevent PyTorch 2.6+ masking engine conflicts
echo "Installing PyTorch 2.5.1 (Stable)..."
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# NOTE: Pinned to 4.48.2. This is the last major version before HF 
# removed FX tools and fundamentally changed the AutoConfig RoPE parsers.
echo "Installing Pinned Transformers (4.48.2)..."
pip install transformers==4.48.2

echo "Installing universal routing and tensor utilities..."
pip install accelerate pillow requests einops gdown black tqdm tiktoken pipreqs pandas

echo "Installing standard video/image processing tools..."
pip install av qwen-vl-utils[decord]==0.0.14

echo "Installing Flash Attention 2..."
pip install --no-cache-dir flash-attn --no-build-isolation

echo "==========================================================="
echo "Success! Your locked-down legacy MLLM environment is ready."
echo "To use it, run: conda activate $ENV_NAME"
echo "==========================================================="
