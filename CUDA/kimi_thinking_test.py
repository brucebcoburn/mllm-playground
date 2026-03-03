import os
import sys
import time
import csv
import torch
import argparse
import gc
from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig

# --- PATH SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(BASE_DIR)

from shared_utils.core_utils import download_test_data, TEST_IMAGES_DIR, VALID_EXTS
from configs.default_config import (
    PROMPT_TEXT,
    TEMPERATURE,
    DO_SAMPLE,
    SEED_VAL,
    KIMI_VL_ID,
    THINKING_MAX_TOKENS,
)


def extract_thinking_and_summary(text: str, bot: str = "◁think▷", eot: str = "◁/think▷"):
    """Parses the raw output into a tuple of (thinking_process, final_summary)."""
    if bot in text and eot not in text:
        return text[text.index(bot) + len(bot):].strip(), "" # Caught mid-thought
    if eot in text:
        return text[text.index(bot) + len(bot):text.index(eot)].strip(), text[text.index(eot) + len(eot):].strip()
    return "", text # No thinking tags found, return whole text as summary

if __name__ == "__main__":
    # --- 0. ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description="Run Kimi-VL Thinking inference.")
    parser.add_argument("--gpu-id", type=int, default=1, help="ID of the GPU to use")
    args = parser.parse_args()
    device = f"cuda:{args.gpu_id}"

    torch.manual_seed(SEED_VAL)
    torch.cuda.manual_seed_all(SEED_VAL)

    # --- 1. DATA & FOLDER SETUP ---
    download_test_data()

    RESULTS_DIR = os.path.join(CURRENT_DIR, "results", "kimivl")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    csv_path = os.path.join(RESULTS_DIR, "inference_results.csv")
    summary_path = os.path.join(RESULTS_DIR, "metrics_summary.txt")

    image_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(VALID_EXTS)]
    if not image_files:
        print(f"Error: No images found in {TEST_IMAGES_DIR}")
        sys.exit(1)

    # --- 2. MODEL LOADING ---
    print(f"Loading thinking model ({KIMI_VL_ID}) onto {device}...")
    
    # NOTE: Monkey-patch fix for the bleeding-edge transformers environment.
    import transformers.utils.import_utils
    transformers.utils.import_utils.is_torch_fx_available = lambda: False
    
    # NOTE: Config Interception Fix
    # The HF config parser automatically renames the "type" key to "rope_type",
    # but Kimi's custom code still explicitly looks for "type".
    config = AutoConfig.from_pretrained(KIMI_VL_ID, trust_remote_code=True)
    
    # DeepSeek/Kimi nests the rope_scaling dict inside `text_config`
    if hasattr(config, "text_config") and getattr(config.text_config, "rope_scaling", None) is not None:
        if "type" not in config.text_config.rope_scaling:
            # Re-inject 'type' by mapping it back from 'rope_type' (or defaulting to "yarn")
            config.text_config.rope_scaling["type"] = config.text_config.rope_scaling.get("rope_type", "yarn")
            
    # NOTE: trust_remote_code=True is required for custom architectures
    model = AutoModelForCausalLM.from_pretrained(
        KIMI_VL_ID,
        config=config,  # <-- We pass our patched config in here
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True, 
        attn_implementation="flash_attention_2",
    ).eval()
    
    processor = AutoProcessor.from_pretrained(KIMI_VL_ID, trust_remote_code=True)

    # --- 3. INFERENCE LOOP ---
    print(f"Starting inference on {len(image_files)} images...")

    total_time = 0.0
    total_tokens = 0

    with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        # Split the results column into Thinking and Summary
        writer.writerow([
            "Filename", "Thinking Process", "Final Answer", 
            "Inference Time (s)", "Tokens Generated", "Tokens/Sec"
        ])

        pbar = tqdm(image_files, desc="Processing Images", unit="img")

        for filename in pbar:
            image_path = os.path.join(TEST_IMAGES_DIR, filename)
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS) # Need to resize otherwise image-based attention is too large (for A40)

            # NOTE: Kimi requires the image_path directly in the dict
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": PROMPT_TEXT},
                    ],
                }
            ]

            text_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = processor(
                text=[text_prompt],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to(device)

            # --- 4. GENERATION ---
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=THINKING_MAX_TOKENS,
                    temperature=TEMPERATURE,
                    do_sample=DO_SAMPLE,
                )

            torch.cuda.synchronize()
            end_time = time.perf_counter()

            # --- 5. METRICS & OUTPUT PARSING ---
            duration = end_time - start_time
            
            new_tokens = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            token_count = len(new_tokens[0])
            tokens_per_sec = token_count / duration

            raw_output = processor.batch_decode(
                new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            # Extract the thought process from the final answer
            thinking_process, final_answer = extract_thinking_and_summary(raw_output)

            # Fallback just in case the model failed to output tags
            if not thinking_process and not final_answer:
                final_answer = raw_output

            total_time += duration
            total_tokens += token_count

            writer.writerow([
                filename, thinking_process, final_answer,
                f"{duration:.4f}", token_count, f"{tokens_per_sec:.2f}"
            ])
            pbar.set_postfix({"speed": f"{tokens_per_sec:.1f} t/s"})

            # 1. Delete the Python references
            del inputs, generated_ids, new_tokens, image
            
            # 2. Force Python to run garbage collection immediately
            gc.collect()
            
            # 3. Force PyTorch to flush everything back to the OS (Possibly optional)
            # Originally done to make room
            torch.cuda.empty_cache()

    # --- 6. SUMMARY METRICS ---
    avg_time_per_image = total_time / len(image_files)
    avg_tokens_per_sec = total_tokens / total_time 

    summary_table = (
        f"\n{'='*40}\n INFERENCE SUMMARY\n{'='*40}\n"
        f"Total Images Processed : {len(image_files)}\n"
        f"Total Time Taken       : {total_time:.2f} seconds\n"
        f"Total Tokens Generated : {total_tokens}\n{'-'*40}\n"
        f"Avg Time per Image     : {avg_time_per_image:.2f} seconds\n"
        f"Avg Speed              : {avg_tokens_per_sec:.2f} tokens/sec\n{'='*40}\n"
        f"Results saved to: {RESULTS_DIR}\n"
    )

    print(summary_table)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_table)