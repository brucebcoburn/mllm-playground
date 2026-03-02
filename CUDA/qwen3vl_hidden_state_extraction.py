"""
Hidden State Extraction Script for MLLMs

This script runs sequential inference on a directory of images and extracts the 
internal hidden states from the model during generation

AVAILABLE EXTRACTION SCHEMES:

1. --scheme mean_pooling (Smallest)
   * What it is: A single vector representing the mathematical average of the entire 
     generated response at the final transformer layer
   * Shape: (1, hidden_dimension)
   * Expected Size: ~8 KB per image
   * Best For: Basic clustering, concept probing, or similarity checks between images

2. --scheme last_token (Very Small)
   * What it is: The output from every single transformer layer, but ONLY for the 
     very last token (word) the model generated before stopping
   * Shape: (num_layers, 1, hidden_dimension)
   * Expected Size: ~262 KB per image
   * Best For: Probing the model's internal state exactly when it finishes its thought

3. --scheme last_layer (Medium - Industry Standard)
   * What it is: The final layer's output for EVERY single token generated in the response
   * Shape: (1, generated_sequence_length, hidden_dimension)
   * Expected Size: ~4 MB per image
   * Best For: Token-by-token analysis and understanding the trajectory of the response

4. --scheme all (Comprehensive Package)
   * What it is: Computes all three of the above metrics and saves them together 
     as a PyTorch dictionary in a single file.
   * Structure: {"mean_pooling": tensor, "last_token": tensor, "last_layer": tensor}
   * Expected Size: ~4.3 MB per image (the combined size of the three)
   * Best For: Giving yourself total analytical flexibility without needing to 
     re-run the expensive inference script later
"""

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
)

if __name__ == "__main__":
    # --- 0. ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description="Extract hidden states from Qwen3-VL.")
    parser.add_argument("--gpu-id", type=int, default=1, help="ID of the GPU to use (default: 1)")
    parser.add_argument(
        "--scheme", 
        type=str, 
        choices=["last_layer", "last_token", "mean_pooling", "all"],
        default="last_layer",
        help="Which hidden state representations to calculate and save."
    )
    args = parser.parse_args()
    device = f"cuda:{args.gpu_id}"

    # LEARNING NOTE: Enforcing determinism. 
    # We attempt to set a manual seed for reproducibility sake. Because DO_SAMPLE is False,
    # this helps to suppress floating-point variations. Flash Attention 2 is notorious 
    # for NOT leading to 100% reproducible results, but we do what we can...
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # --- 1. DATA & FOLDER SETUP ---
    download_test_data()
    
    # Store in <platform>/hidden_states/qwen3vl/
    HIDDEN_STATES_DIR = os.path.join(CURRENT_DIR, "hidden_states", "qwen3vl")
    os.makedirs(HIDDEN_STATES_DIR, exist_ok=True)
    
    csv_path = os.path.join(HIDDEN_STATES_DIR, f"extraction_log_{args.scheme}.csv")

    image_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(VALID_EXTS)]
    
    if not image_files:
        print(f"Error: No images found in {TEST_IMAGES_DIR}")
        sys.exit(1)

    # --- 2. MODEL LOADING ---
    print(f"Loading model ({QWEN3_VL_ID}) onto {device}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN3_VL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(QWEN3_VL_ID)

    # --- 3. INFERENCE & EXTRACTION LOOP ---
    print(f"Extracting hidden states ({args.scheme}) for {len(image_files)} images...")
    
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Filename", "Generated Text", "Saved Tensor File", "Tensor Shape/Type"])

        pbar = tqdm(image_files, desc="Processing Images", unit="img")
        
        for filename in pbar:
            image_path = os.path.join(TEST_IMAGES_DIR, filename)

            # LEARNING NOTE: HF models are stateless.
            # We do not need to "clear" the model's context between images because
            # the only context it sees is this brand new `messages` list generated here.
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": PROMPT_TEXT},
                    ],
                }
            ]

            # LEARNING NOTE: Applying the Chat Template.
            # MLLMs are trained on highly specific text structures (e.g., <|im_start|>user...).
            # `apply_chat_template` takes our readable Python dictionary and automatically 
            # translates it into the exact raw string format that the specific model expects. 
            # `add_generation_prompt=True` appends the final "Assistant:" trigger token so 
            # the model knows it is its turn to start generating text.
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

            torch.cuda.synchronize()

            # --- 4. GENERATION WITH HIDDEN STATES ---
            with torch.no_grad():
                # We explicitly add output_hidden_states and return_dict_in_generate.
                # Setting top_p and top_k to None guarantees greedy decoding is untouched.
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    do_sample=DO_SAMPLE,
                    top_p=None, 
                    top_k=None,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )

            torch.cuda.synchronize()

            # Decode the generated text for our logs
            generated_ids = outputs.sequences
            new_tokens = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

            # --- 5. HIDDEN STATE PARSING LOGIC ---
            # outputs.hidden_states is a tuple of length (generated_tokens).
            # Each element is a tuple of length (num_layers).
            raw_states = outputs.hidden_states
            
            # 1. Calculate Last Token (All layers for the very last generated token)
            final_step_layers = raw_states[-1]
            last_token_tensor = torch.stack([layer.cpu() for layer in final_step_layers])

            # 2. Calculate Last Layer (The final transformer layer for all generated tokens)
            # Step 0 contains the prompt + first generated token's contextual state.
            step_0_last_layer = raw_states[0][-1][:, -1:, :].cpu() 
            # Step 1+ contain the states for the subsequently generated tokens.
            subsequent_steps = [step[-1].cpu() for step in raw_states[1:]]
            last_layer_tensor = torch.cat([step_0_last_layer] + subsequent_steps, dim=1)

            # 3. Calculate Mean Pooling (Average across the sequence length of the last layer)
            mean_pooling_tensor = last_layer_tensor.mean(dim=1)

            # Route the correct tensor(s) based on the chosen scheme
            tensor_to_save = None
            shape_str = ""

            if args.scheme == "all":
                tensor_to_save = {
                    "last_token": last_token_tensor,
                    "last_layer": last_layer_tensor,
                    "mean_pooling": mean_pooling_tensor
                }
                shape_str = "Dict of 3 tensors"
            elif args.scheme == "last_token":
                tensor_to_save = last_token_tensor
                shape_str = str(list(tensor_to_save.shape))
            elif args.scheme == "last_layer":
                tensor_to_save = last_layer_tensor
                shape_str = str(list(tensor_to_save.shape))
            elif args.scheme == "mean_pooling":
                tensor_to_save = mean_pooling_tensor
                shape_str = str(list(tensor_to_save.shape))

            # --- 6. SAVING OFF THE TENSORS ---
            base_filename = os.path.splitext(filename)[0]
            save_name = f"{base_filename}_{args.scheme}.pt"
            save_path = os.path.join(HIDDEN_STATES_DIR, save_name)
            
            # Save using PyTorch's native serialization format
            torch.save(tensor_to_save, save_path)

            writer.writerow([filename, output_text, save_name, shape_str])

            # LEARNING NOTE: VRAM Management
            # We delete the Python references to the massive tuple and output tensors
            # This allows PyTorch's built-in caching allocator to instantly reuse these 
            # blocks of VRAM for the next image without needing to ask the OS for memory
            del inputs, outputs, generated_ids, new_tokens, raw_states 
            del last_token_tensor, last_layer_tensor, mean_pooling_tensor, tensor_to_save