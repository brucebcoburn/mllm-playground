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
│   ├── setup_mllm.sh
│   └── qwen3_test.py
├── Mac/                    # Scripts & setups for Apple Silicon using MLX
│   ├── setup_mlx.sh
│   └── qwen3_test_mlx.py
├── Slurm/                  # Batch scripts and setups for Supercomputer clusters
│   └── run_qwen3.slurm
└── test_images/            # Auto-generated directory for inference testing