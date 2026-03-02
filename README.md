# mllm-playground

A "hodgepodge" repository for setting up and using various Multimodal Large Language Models (MLLMs). This project serves as a practical reference for deploying and testing models across vastly different hardware environments.

We are currently building out examples and documentation for local multi-GPU servers (CUDA), Apple Silicon (Mac/MLX), and supercomputer clusters (Slurm).

---

## 📂 Repository Structure

To keep the codebase organized across different hardware backends, this repository is structured by **platform** rather than by model. 

```text
mllm-playground/
├── configs/                # Shared configurations (Model IDs, prompts, max tokens)
│   └── default_config.py
├── shared_utils/           # Hardware-agnostic helper functions (e.g., data downloading)
│   └── core_utils.py
├── CUDA/                   # Scripts & setups for standard PyTorch/CUDA environments
│   ├── setup_mllm_CUDA.sh
│   └── qwen3vl_test.py
├── Mac/                    # Scripts & setups for Apple Silicon using MLX
│   ├── setup_mllm_MLX.sh
│   └── qwen3vl_test_mlx.py
├── Slurm/                  # Batch scripts and setups for Supercomputer clusters
│   └── run_qwen3vl.slurm
└── test_images/            # Auto-generated directory for inference testing

## 🛠️ Future Development & File Structure Guide

As this project grows and new MLLMs are added, we aim to follow these structural guidelines to keep the repository clean and modular:

### 1. Global Configurations go in `configs/`
If a variable is used across multiple platforms (like a Hugging Face model ID, generation temperatures, or a default text prompt), we put it in `configs/default_config.py`. This prevents us from having to update the same string in three different scripts when a new model version drops.

### 2. Platform-Agnostic Logic goes in `shared_utils/`
If a function does not care about the hardware (e.g., downloading test files, parsing JSON outputs, or formatting strings), it belongs in `shared_utils/`. These should focus on being lightweight and on "pure" model inference.

### 3. Execution Scripts are Platform-Specific
Always place runnable scripts in their respective hardware folders (`CUDA/`, `Mac/`, `Slurm/`). 

* **Naming convention:** Use `<model_name>_test.py` for standard PyTorch/CUDA, and `<model_name>_test_mlx.py` for Mac.
* **Pathing:** Every execution script must dynamically append the root directory to `sys.path` at the top of the file so it can seamlessly import from `configs` and `shared_utils`.

We may include some "additional functionality" scripts (such as hidden state extraction, etc.) which will be reflected in the filenames.

### 4. Isolate Environment Setups
Keep a `setup_<env>.sh` script in each platform folder. A pip install for CUDA is fundamentally different from installing Apple's MLX, so we must maintain isolated dependency instructions.