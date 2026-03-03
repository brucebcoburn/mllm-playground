#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

ENV_NAME="mllm_env"
PYTHON_VERSION="3.10"

# Check if environment already exists
if conda info --envs | awk '{print $1}' | grep -Eq "^${ENV_NAME}$"; then
    echo "Environment '$ENV_NAME' already exists. Updating it instead of creating..."
else
    echo "Creating versatile MLLM conda environment '$ENV_NAME'..."
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
fi

# Initialize conda for the script so 'conda activate' works
eval "$(conda shell.bash hook)"

echo "Activating environment..."
conda activate $ENV_NAME

echo "Installing isolated CUDA 12.4 Toolkit for compiling dependencies..."
conda install -c nvidia cuda-toolkit=12.4 -y

# Set CUDA_HOME so Flash Attention knows exactly where to find the compiler
export CUDA_HOME=$CONDA_PREFIX

echo "Installing PyTorch 2.6+ with CUDA 12.4 support..."
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "Installing HuggingFace Transformers from bleeding-edge source..."
pip install --upgrade git+https://github.com/huggingface/transformers.git

echo "Installing universal routing and tensor utilities..."
pip install --upgrade accelerate pillow requests einops gdown black tqdm pipreqs

echo "Installing standard video/image processing tools..."
pip install --upgrade av qwen-vl-utils[decord]==0.0.14

echo "Installing/Recompiling Flash Attention 2 (Crucial for VRAM efficiency)..."
pip install --upgrade --no-cache-dir flash-attn --no-build-isolation

echo "==========================================================="
echo "Success! Your universal MLLM environment is ready to go."
echo "To use it, run: conda activate $ENV_NAME"
echo "==========================================================="