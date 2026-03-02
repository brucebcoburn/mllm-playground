import os
import sys
import time
import csv
import torch
import argparse
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

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
    QWEN3_VL_ID,
    MAX_NEW_TOKENS,
    PROMPT_TEXT,
    TEMPERATURE,
    DO_SAMPLE,
    SEED_VAL,
)

if __name__ == "__main__":
    # --- 0. ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(
        description="Run Qwen3-VL inference over a directory of images."
    )
    parser.add_argument(
        "--gpu-id", type=int, default=1, help="ID of the GPU to use (default: 1)"
    )
    args = parser.parse_args()
    device = f"cuda:{args.gpu_id}"

    # NOTE: Enforcing determinism
    # We attempt to set a manual seed for reproducibility sake. If DO_SAMPLE (later on) is False,
    # this help to suppress floating-point variations from Flash Attention. HOWEVER, Flash Attention 2
    # is notorious for NOT leading to reproducible results. Regardless, we do what we can...
    torch.manual_seed(SEED_VAL)
    torch.cuda.manual_seed_all(SEED_VAL)

    # --- 1. AUTOMATIC DATA DOWNLOAD & FOLDER SETUP ---
    download_test_data()

    # Logic: Results will be stored dynamically in results/qwen3vl/
    RESULTS_DIR = os.path.join(CURRENT_DIR, "results", "qwen3vl")
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
    print(f"Loading model ({QWEN3_VL_ID}) onto {device}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN3_VL_ID,
        dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(QWEN3_VL_ID)

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

            # NOTE: HF models are stateless
            # We do not need to "clear" the model's context between images because
            # the only context it sees is this brand new `messages` list generated here...
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": PROMPT_TEXT},
                    ],
                }
            ]

            # NOTE: Applying the Chat Template
            # MLLMs are trained on highly specific text structures (e.g., <|im_start|>user... vs [INST]...).
            # `apply_chat_template` takes our readable Python dictionary and automatically translates it
            # into the exact raw string format that the specific model expects
            # `add_generation_prompt=True` appends the final "Assistant:" trigger token so the model
            # knows it is its turn to start generating text
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)

            # --- 4. GENERATION ---
            # NOTE: PyTorch GPU operations are asynchronous.
            # Python (the CPU) issues the command and immediately moves to the next line
            # BUT synchronize() forces Python to halt and wait until the GPU is 100% idle
            # before we start the stopwatch
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,  # Our default is None (b/c do_sample=False)
                    do_sample=DO_SAMPLE,  # Our default is False (we want deterministic)
                    top_p=None,  # Overwriting Qwen's generation_config.json (otherwise warning
                    top_k=None,  # due to our "deterministic settings")
                )

            # NOTE: We synchronize again to force Python to wait until the GPU
            # finishes computing the absolute final token before we stop the stopwatch
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
            # We delete the Python references to the heavy tensors from this iteration
            # This allows PyTorch's built-in caching allocator to instantly reuse these
            # blocks of VRAM for the next image without needing to ask the OS for memory
            del inputs, generated_ids, new_tokens

    # --- 6. SUMMARY METRICS ---
    avg_time_per_image = total_time / len(image_files)
    avg_tokens_per_sec = total_tokens / total_time  # Global average

    summary_table = (
        f"\n{'='*40}\n"
        f" INFERENCE SUMMARY\n"
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
