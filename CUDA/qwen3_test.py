import os
import sys
import torch
import time
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 1. COMMAND LINE GPU SELECTION
# Usage: python qwen3_test.py 2 (to use GPU 2)
target_gpu_id = sys.argv[1] if len(sys.argv) > 1 else "1"
device = f"cuda:{target_gpu_id}"

model_id = "Qwen/Qwen3-VL-8B-Instruct"

print(f"Loading model onto {device}...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map={"": device}, 
    attn_implementation="flash_attention_2"
)

processor = AutoProcessor.from_pretrained(model_id)

# 2. INPUT SETUP
local_image_path = os.environ.get('HAGGIS', 'path/to/default.jpg')

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": local_image_path},
            {"type": "text", "text": "Describe this image in detail."},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to(device)

# 3. GENERATION WITH TIMING
print("Generating response...")

# Synchronize and start timer
torch.cuda.synchronize()
start_time = time.perf_counter()

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=512)

# Synchronize and stop timer
torch.cuda.synchronize()
end_time = time.perf_counter()

# 4. METRICS CALCULATION
duration = end_time - start_time
# We only count the newly generated tokens (output), not the input tokens
new_tokens = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
token_count = len(new_tokens[0])
tokens_per_sec = token_count / duration

# 5. OUTPUT
output_text = processor.batch_decode(
    new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print("\n" + "="*30)
print(f"MODEL RESPONSE:")
print(output_text[0])
print("="*30)
print(f"Time Taken: {duration:.2f} seconds")
print(f"Tokens Generated: {token_count}")
print(f"Speed: {tokens_per_sec:.2f} tokens/sec")
print("="*30)