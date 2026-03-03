# --- DEFAULT PARAMS ---
SEED_VAL = 8008135

# --- MODEL IDs ---
QWEN3_VL_ID = "Qwen/Qwen3-VL-8B-Instruct"
GLM_MODEL_ID = "zai-org/GLM-4.6V-Flash"
GEMMA3_ID = "google/gemma-3-12b-it"
KIMI_VL_ID = "moonshotai/Kimi-VL-A3B-Thinking-2506"

# --- THINKING TOKENS ---
# NOTE: Thinking models need massive max token limits for their inner monologue
THINKING_MAX_TOKENS = 32768

# --- GENERATION SETTINGS ---
MAX_NEW_TOKENS = 512
PROMPT_TEXT = "Describe this image in detail."
DO_SAMPLE = False
TEMPERATURE = None
