import yaml  
from pathlib import Path  
import os

def load_config(config_path: str) -> dict:  
    with open(config_path, 'r', encoding='utf-8') as f:  
        return yaml.safe_load(f)  

# Load configuration  
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_CONFIG = load_config(BASE_DIR / 'configs' / 'model_config.yaml')  
TRAIN_CONFIG = load_config(BASE_DIR / 'configs' / 'train_config.yaml')  
DPO_CONFIG = load_config(BASE_DIR / 'configs' / 'dpo_config.yaml')





BATCH_SIZE = 8
NUM_PROCESSES = 2


OUTPUT_DIR = "output/"
MODEL_PATH = "/root/src/models/Qwen3-0.6B"


REWARD_MODEL_PATH = "/root/src/models/reward-model-deberta-v3-large-v2"

EMBEDDING_MODEL_PATH = "/root/src/models/all-MiniLM-L6-v2"

EMBEDDING_MODEL_PATH_BPE = "/root/src/models/bge-small-en-v1.5"

SFT_MODEL_NAME = "qwen3_sft"
SFT_MODEL_PATH = os.path.join(OUTPUT_DIR, SFT_MODEL_NAME)


DPO_MODEL_NAME = "qwen3_dpo"
DPO_MODEL_PATH = os.path.join(OUTPUT_DIR, DPO_MODEL_NAME)


PPO_MODEL_NAME = "qwen3_ppo"
PPO_MODEL_PATH = os.path.join(OUTPUT_DIR, PPO_MODEL_NAME)


GRPO_MODEL_NAME = "qwen3_grpo"
GRPO_MODEL_PATH = os.path.join(OUTPUT_DIR, GRPO_MODEL_NAME)



SFT_DPO_MODEL_NAME = "qwen3_sft_dpo"
SFT_DPO_MODEL_PATH = os.path.join(OUTPUT_DIR, SFT_DPO_MODEL_NAME)



DATA_PATH = "test_travel_qa.json"
RAG_DATA_PATH = "src/data/crosswoz-sft"

DPO_DATA_PATH = "/root/work1/src/src/data/Human-Like-DPO-Dataset"

CACHED_SFT_DATA_PATH = "/root/src/data/sft_data_cached"
CACHED_DPO_DATA_PATH = "/root/src/data/dpo_data_cached"
CACHED_GRPO_DATA_PATH = "/root/src/data/grpo_data_cached"
CACHED_PPO_DATA_PATH = "/root/src/data/ppo_data_cached"

DEEPSPEED_CONFIG_PATH = "src/configs/ds_config.json"

PDF_FOLDER_PATH = "src/agents/travel_knowledge/tour_pdfs"
PAGE_FOLDER_PATH = "src/agents/travel_knowledge/tour_pages"

# # Using configuration in code
# model = TravelMind(
#     model_name=model_config['model']['name'],
#     lora_config=model_config['lora']
# )

# trainer = Trainer(
#     model=model,
#     args=TrainingArguments(**train_config['training'])
# )

# dpo_trainer = TravelMindDPOTrainer(
#     model=model,
#     config=DPOConfig(**dpo_config['dpo']['training'])
# )  