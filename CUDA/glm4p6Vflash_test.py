import os
import sys
import time
import csv
import torch
import argparse
from tqdm import tqdm
from PIL import Image

# NOTE: We have an explicit GLM4v class (HF model card -> Files and Versions -> config.json)
from transformers import Glm4vForConditionalGeneration, AutoProcessor

# --- PATH SETUP ---
# Get the directory of this script (mllm-playground/<platform-type>)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the root 'mllm-playground/'
BASE_DIR = os.path.dirname(CURRENT_DIR)
# Add the root to Python's path so we can find configs/ and shared_utils/ directories
sys.path.append(BASE_DIR)

# Import shared modules
from shared_utils.core_utils import download_test_data, TEST_IMAGES_DIR, VALID_EXTS
from configs.default_config import (
    MAX_NEW_TOKENS,
    PROMPT_TEXT,
    TEMPERATURE,
    DO_SAMPLE,
    SEED_VAL,
    GLM_MODEL_ID,
)

if __name__ == "__main__":
    # --- 0. ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(
        description="Run GLM-4.6V-Flash inference over a directory of images."
    )
    parser.add_argument(
        "--gpu-id", type=int, default=1, help="ID of the GPU to use (default: 1)"
    )
    args = parser.parse_args()
    device = f"cuda:{args.gpu_id}"

    # NOTE: Enforcing determinism
    # We attempt to set a manual seed for reproducibility sake.
    torch.manual_seed(SEED_VAL)
    torch.cuda.manual_seed_all(SEED_VAL)

    # --- 1. AUTOMATIC DATA DOWNLOAD & FOLDER SETUP ---
    download_test_data()

    # Logic: Results will be stored dynamically in results/glm4_6v/
    RESULTS_DIR = os.path.join(CURRENT_DIR, "results", "glm4_6v")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    csv_path = os.path.join(RESULTS_DIR, "inference_results.csv")
    summary_path = os.path.join(RESULTS_DIR, "metrics_summary.txt")

    image_files = [
        f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(VALID_EXTS)
    ]

    if not image_files:
        print(f"Error: No images found in {TEST_IMAGES_DIR}")
        sys.exit(1)

    # --- 2. MODEL LOADING ---
    print(f"Loading model ({GLM_MODEL_ID}) onto {device}...")

    # NOTE: Using the explicit Glm4vForConditionalGeneration class found in config.json
    # We use 'dtype' instead of the deprecated 'torch_dtype' for Transformers v5.x
    model = Glm4vForConditionalGeneration.from_pretrained(
        GLM_MODEL_ID,
        dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(GLM_MODEL_ID, trust_remote_code=True)

    # --- 3. INFERENCE LOOP ---
    print(f"Starting inference on {len(image_files)} images...")

    total_time = 0.0
    total_tokens = 0

    with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "Filename",
                "Inference Results",
                "Inference Time (s)",
                "Tokens Generated",
                "Tokens/Sec",
            ]
        )

        pbar = tqdm(image_files, desc="Processing Images", unit="img")

        for filename in pbar:
            # --- 3A. INPUT SETUP ---
            image_path = os.path.join(TEST_IMAGES_DIR, filename)

            # NOTE: Loading the image via PIL directly for GLM's processor
            image = Image.open(image_path).convert("RGB")

            # NOTE: HF models are stateless
            # We do not need to "clear" the model's context between images because
            # the only context it sees is this brand new `messages` list...
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": PROMPT_TEXT},
                    ],
                }
            ]

            # NOTE: Applying the Chat Template
            # `add_generation_prompt=True` appends the final "Assistant:" trigger token
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to(device)

            # --- 4. GENERATION ---
            # NOTE: PyTorch GPU operations are asynchronous.
            # synchronize() forces Python to halt until the GPU is 100% idle before timing.
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    do_sample=DO_SAMPLE,
                    top_p=None,
                    top_k=None,
                )

            # NOTE: We synchronize again to wait until the GPU finishes the final token computation.
            torch.cuda.synchronize()
            end_time = time.perf_counter()

            # --- 5. METRICS & OUTPUT ---
            duration = end_time - start_time
            # Strip out the input prompt tokens to count only what the model newly generated
            new_tokens = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            token_count = len(new_tokens[0])
            tokens_per_sec = token_count / duration

            output_text = processor.batch_decode(new_tokens, skip_special_tokens=True)[
                0
            ].strip()

            total_time += duration
            total_tokens += token_count

            writer.writerow(
                [
                    filename,
                    output_text,
                    f"{duration:.4f}",
                    token_count,
                    f"{tokens_per_sec:.2f}",
                ]
            )
            pbar.set_postfix({"speed": f"{tokens_per_sec:.1f} t/s"})

            # NOTE: VRAM Management
            # We delete the Python references to allow PyTorch's caching allocator
            # to instantly reuse these blocks of VRAM for the next image.
            del inputs, generated_ids, new_tokens, image

    # --- 6. SUMMARY METRICS ---
    avg_time_per_image = total_time / len(image_files)
    avg_tokens_per_sec = total_tokens / total_time  # Global average

    summary_table = (
        f"\n{'='*40}\n"
        f" GLM-4.6V INFERENCE SUMMARY\n"
        f"{'='*40}\n"
        f"Total Images Processed : {len(image_files)}\n"
        f"Total Time Taken       : {total_time:.2f} seconds\n"
        f"Total Tokens Generated : {total_tokens}\n"
        f"{'-'*40}\n"
        f"Avg Time per Image     : {avg_time_per_image:.2f} seconds\n"
        f"Avg Speed              : {avg_tokens_per_sec:.2f} tokens/sec\n"
        f"{'='*40}\n"
        f"Results saved to: {RESULTS_DIR}\n"
    )

    print(summary_table)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_table)
