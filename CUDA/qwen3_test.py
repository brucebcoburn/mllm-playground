import os
import sys
import time
import torch
import argparse
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# --- PATH SETUP ---
# Get the directory of this script (mllm-playground/<platform-type>)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the root 'mllm-playground/'
BASE_DIR = os.path.dirname(CURRENT_DIR)
# Add the root to Python's path so we can find configs/ and shared_utils/ directories
sys.path.append(BASE_DIR)

# Import our shared modules
from shared_utils.core_utils import download_test_data, TEST_IMAGES_DIR
from configs.default_config import QWEN3_VL_ID, MAX_NEW_TOKENS, PROMPT_TEXT

if __name__ == "__main__":
    # --- 0. ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(
        description="Run Qwen3-VL inference with specified GPU."
    )
    parser.add_argument(
        "--gpu-id", type=int, default=1, help="ID of the GPU to use (default: 1)"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to a specific image (overrides default haggis.png)",
    )

    args = parser.parse_args()
    device = f"cuda:{args.gpu_id}"

    # --- 1. AUTOMATIC DATA DOWNLOAD ---
    download_test_data()

    # --- 2. MODEL LOADING ---
    print(f"Loading model ({QWEN3_VL_ID}) onto {device}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN3_VL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="flash_attention_2",
    )

    processor = AutoProcessor.from_pretrained(QWEN3_VL_ID)

    # --- 3. INPUT SETUP ---
    local_image_path = args.image or os.path.join(TEST_IMAGES_DIR, "haggis.png")

    if not os.path.exists(local_image_path):
        print(f"Error: Image not found at {local_image_path}")
        sys.exit(1)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": local_image_path},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }
    ]

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
    print(f"Generating response for image: {os.path.basename(local_image_path)}...")
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    # --- 5. METRICS & OUTPUT ---
    duration = end_time - start_time
    new_tokens = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    token_count = len(new_tokens[0])
    tokens_per_sec = token_count / duration

    output_text = processor.batch_decode(new_tokens, skip_special_tokens=True)

    print("\n" + "=" * 30)
    print(f"MODEL RESPONSE:")
    print(output_text[0])
    print("=" * 30)
    print(f"GPU Used: {device}")
    print(f"Speed: {tokens_per_sec:.2f} tokens/sec")
    print("=" * 30)
