# Model and Data Download Guide

**Project:** TravelMind
**Author:** Frank Chen
**Version:** 1.0.0

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Model Downloads](#model-downloads)
4. [Dataset Downloads](#dataset-downloads)
5. [Configuration](#configuration)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

---

## Overview

This guide provides comprehensive instructions for downloading and configuring the models and datasets required for TravelMind. The system requires several pre-trained models and datasets for different functionalities:

- **Base Models:** Qwen3 series for language understanding
- **Reward Models:** For RLHF training
- **Embedding Models:** For RAG functionality
- **Training Datasets:** For fine-tuning and alignment

---

## Directory Structure

Create the following directory structure for organizing models and data:

```
TravelMind/
├── models/
│   ├── base/
│   │   ├── Qwen3-0.6B/
│   │   ├── Qwen3-1.5B/
│   │   └── Qwen3-3B/
│   ├── reward/
│   │   └── reward-model-deberta-v3-large-v2/
│   ├── embeddings/
│   │   ├── all-MiniLM-L6-v2/
│   │   └── bge-small-en-v1.5/
│   └── README.md
├── data/
│   ├── sft/
│   │   └── travel-QA/
│   ├── dpo/
│   │   └── Human-Like-DPO-Dataset/
│   ├── rag/
│   │   ├── crosswoz-sft/
│   │   └── travel_knowledge/
│   └── README.md
└── checkpoints/
    ├── sft/
    ├── dpo/
    ├── ppo/
    └── README.md
```

---

## Model Downloads

### Prerequisites

Install Hugging Face CLI:

```bash
pip install huggingface-hub
```

Authenticate (optional, for private models):

```bash
huggingface-cli login
```

### Base Models

#### Qwen3 Series

Download based on your hardware capabilities:

```bash
# Lightweight model (0.6B parameters) - 1.5GB
cd models/base/
huggingface-cli download \
    --resume-download \
    --local-dir Qwen3-0.6B \
    Qwen/Qwen3-0.6B

# Standard model (1.5B parameters) - 3GB (Recommended)
huggingface-cli download \
    --resume-download \
    --local-dir Qwen3-1.5B \
    Qwen/Qwen3-1.5B

# Large model (3B parameters) - 6GB
huggingface-cli download \
    --resume-download \
    --local-dir Qwen3-3B \
    Qwen/Qwen3-3B
```

### Reward Models

For RLHF training (PPO, DPO):

```bash
cd models/reward/
huggingface-cli download \
    --resume-download \
    --local-dir reward-model-deberta-v3-large-v2 \
    OpenAssistant/reward-model-deberta-v3-large-v2
```

### Embedding Models

For RAG functionality:

```bash
cd models/embeddings/

# Sentence Transformers model
huggingface-cli download \
    --resume-download \
    --local-dir all-MiniLM-L6-v2 \
    sentence-transformers/all-MiniLM-L6-v2

# BGE embeddings (alternative)
huggingface-cli download \
    --resume-download \
    --local-dir bge-small-en-v1.5 \
    BAAI/bge-small-en-v1.5
```

### Download Script

Use the automated download script:

```bash
#!/bin/bash
# scripts/download_models.sh

set -e

MODEL_DIR="models"
BASE_DIR="$MODEL_DIR/base"
REWARD_DIR="$MODEL_DIR/reward"
EMBED_DIR="$MODEL_DIR/embeddings"

# Create directories
mkdir -p "$BASE_DIR" "$REWARD_DIR" "$EMBED_DIR"

# Function to download model
download_model() {
    local repo=$1
    local dir=$2
    echo "Downloading $repo to $dir..."

    if [ -d "$dir" ] && [ "$(ls -A $dir)" ]; then
        echo "$dir already exists and is not empty, skipping..."
        return
    fi

    huggingface-cli download \
        --resume-download \
        --local-dir "$dir" \
        "$repo"
}

# Download base models
echo "=== Downloading Base Models ==="
download_model "Qwen/Qwen3-1.5B" "$BASE_DIR/Qwen3-1.5B"

# Download reward model
echo "=== Downloading Reward Model ==="
download_model "OpenAssistant/reward-model-deberta-v3-large-v2" \
    "$REWARD_DIR/reward-model-deberta-v3-large-v2"

# Download embedding models
echo "=== Downloading Embedding Models ==="
download_model "sentence-transformers/all-MiniLM-L6-v2" \
    "$EMBED_DIR/all-MiniLM-L6-v2"

echo "=== Download Complete ==="
```

---

## Dataset Downloads

### SFT Training Data

```bash
cd data/sft/

# Travel QA dataset
huggingface-cli download \
    --repo-type dataset \
    --resume-download \
    --local-dir travel-QA \
    JasleenSingh91/travel-QA
```

### DPO/GRPO Training Data

```bash
cd data/dpo/

# Human-Like DPO dataset
huggingface-cli download \
    --repo-type dataset \
    --resume-download \
    --local-dir Human-Like-DPO-Dataset \
    HumanLLMs/Human-Like-DPO-Dataset
```

### RAG Knowledge Base

```bash
cd data/rag/

# CrossWOZ dataset for dialogue
huggingface-cli download \
    --repo-type dataset \
    --resume-download \
    --local-dir crosswoz-sft \
    BruceNju/crosswoz-sft

# Create travel knowledge directory
mkdir -p travel_knowledge
```

### Custom Data Format

#### SFT Data Format (JSON)

```json
[
    {
        "conversations": [
            {
                "from": "user",
                "value": "Plan a 3-day trip to Paris"
            },
            {
                "from": "assistant",
                "value": "I'll help you plan a wonderful 3-day trip to Paris..."
            }
        ]
    }
]
```

#### DPO Data Format (JSON)

```json
[
    {
        "prompt": "What's the best time to visit Tokyo?",
        "chosen": "The best time to visit Tokyo is during spring (March-May) or autumn (September-November)...",
        "rejected": "Anytime is good to visit Tokyo."
    }
]
```

#### RAG Document Format

Place PDF or text files in `data/rag/travel_knowledge/`:

```
data/rag/travel_knowledge/
├── paris_guide.pdf
├── tokyo_attractions.txt
├── london_hotels.json
└── metadata.json
```

---

## Configuration

### Update Configuration Files

Edit `src/configs/config.py`:

```python
import os

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_BASE_PATH = os.path.join(PROJECT_ROOT, "models")
DATA_BASE_PATH = os.path.join(PROJECT_ROOT, "data")

# Model paths
MODEL_PATH = os.path.join(MODEL_BASE_PATH, "base", "Qwen3-1.5B")
REWARD_MODEL_PATH = os.path.join(MODEL_BASE_PATH, "reward", "reward-model-deberta-v3-large-v2")
EMBEDDING_MODEL_PATH = os.path.join(MODEL_BASE_PATH, "embeddings", "all-MiniLM-L6-v2")

# Data paths
SFT_DATA_PATH = os.path.join(DATA_BASE_PATH, "sft", "travel-QA")
DPO_DATA_PATH = os.path.join(DATA_BASE_PATH, "dpo", "Human-Like-DPO-Dataset")
RAG_DATA_PATH = os.path.join(DATA_BASE_PATH, "rag", "crosswoz-sft")
KNOWLEDGE_BASE_PATH = os.path.join(DATA_BASE_PATH, "rag", "travel_knowledge")

# Checkpoint paths
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints")
SFT_CHECKPOINT_PATH = os.path.join(CHECKPOINT_PATH, "sft")
DPO_CHECKPOINT_PATH = os.path.join(CHECKPOINT_PATH, "dpo")
PPO_CHECKPOINT_PATH = os.path.join(CHECKPOINT_PATH, "ppo")

# Model configurations
MODEL_CONFIG = {
    "model_name": MODEL_PATH,
    "max_length": 2048,
    "temperature": 0.7,
    "top_p": 0.9,
    "device_map": "auto",
    "torch_dtype": "float16",
}

# Training configurations
TRAINING_CONFIG = {
    "batch_size": 4,
    "learning_rate": 5e-5,
    "num_epochs": 3,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 100,
    "save_steps": 500,
    "logging_steps": 50,
}
```

### Environment Variables

Create `.env` file:

```bash
# Model paths
MODEL_BASE_PATH=/path/to/models
DATA_BASE_PATH=/path/to/data

# API keys (optional)
HUGGINGFACE_TOKEN=your_token_here
OPENAI_API_KEY=your_openai_key
ZHIPU_API_KEY=your_zhipu_key

# Hardware settings
CUDA_VISIBLE_DEVICES=0,1
```

---

## Verification

### Verify Model Downloads

```python
#!/usr/bin/env python3
# scripts/verify_models.py

import os
import sys
from pathlib import Path

def check_directory(path, name, min_size_mb=10):
    """Check if directory exists and has content"""
    if not os.path.exists(path):
        print(f"❌ {name}: Directory not found at {path}")
        return False

    # Check directory size
    total_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, dirnames, filenames in os.walk(path)
        for filename in filenames
    ) / (1024 * 1024)  # Convert to MB

    if total_size < min_size_mb:
        print(f"⚠️ {name}: Directory exists but seems empty ({total_size:.1f}MB)")
        return False

    print(f"✅ {name}: Found ({total_size:.1f}MB)")
    return True

def main():
    models_to_check = [
        ("models/base/Qwen3-1.5B", "Qwen3-1.5B Base Model", 1000),
        ("models/reward/reward-model-deberta-v3-large-v2", "Reward Model", 500),
        ("models/embeddings/all-MiniLM-L6-v2", "Embedding Model", 50),
    ]

    datasets_to_check = [
        ("data/sft/travel-QA", "SFT Dataset", 1),
        ("data/dpo/Human-Like-DPO-Dataset", "DPO Dataset", 1),
        ("data/rag/crosswoz-sft", "RAG Dataset", 1),
    ]

    print("=== Verifying Models ===")
    models_ok = all(check_directory(path, name, size)
                   for path, name, size in models_to_check)

    print("\n=== Verifying Datasets ===")
    data_ok = all(check_directory(path, name, size)
                 for path, name, size in datasets_to_check)

    if models_ok and data_ok:
        print("\n✅ All models and datasets are properly downloaded!")
        return 0
    else:
        print("\n❌ Some components are missing. Please run download scripts.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

### Test Model Loading

```python
# scripts/test_models.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_model_loading():
    """Test if models can be loaded properly"""

    model_path = "models/base/Qwen3-1.5B"

    try:
        print(f"Loading model from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✅ Model loaded successfully!")

        # Test inference
        inputs = tokenizer("Hello, how can I help you plan your trip?", return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Test response: {response}")

        return True

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

if __name__ == "__main__":
    test_model_loading()
```

---

## Troubleshooting

### Common Issues

#### Insufficient Disk Space

```bash
# Check available space
df -h .

# Clean Hugging Face cache
huggingface-cli delete-cache
```

#### Download Interrupted

```bash
# Resume download with --resume-download flag
huggingface-cli download \
    --resume-download \
    --local-dir model_name \
    repo/model
```

#### Authentication Required

```bash
# Login to Hugging Face
huggingface-cli login

# Set token via environment variable
export HF_TOKEN=your_token_here
```

#### CUDA Out of Memory

```python
# Use smaller model or enable quantization
MODEL_CONFIG = {
    "load_in_8bit": True,  # or load_in_4bit
    "device_map": "auto",
}
```

#### Slow Download Speed

```bash
# Use mirror sites (for users in certain regions)
export HF_ENDPOINT=https://hf-mirror.com

# Or use wget with proxy
wget -c https://huggingface.co/model/resolve/main/pytorch_model.bin
```

### Model Size Reference

| Model | Parameters | Disk Space | RAM (FP16) | VRAM (FP16) |
|-------|------------|------------|------------|-------------|
| Qwen3-0.6B | 0.6B | ~1.5GB | ~3GB | ~1.5GB |
| Qwen3-1.5B | 1.5B | ~3GB | ~6GB | ~3GB |
| Qwen3-3B | 3B | ~6GB | ~12GB | ~6GB |

### Directory Markers

Create README files to preserve directory structure in git:

```bash
# models/README.md
echo "# Model Directory

This directory contains downloaded models.
Models are not tracked by git due to their size.

Required models:
- base/Qwen3-1.5B/
- reward/reward-model-deberta-v3-large-v2/
- embeddings/all-MiniLM-L6-v2/
" > models/README.md

# data/README.md
echo "# Data Directory

This directory contains training and evaluation datasets.
Data files are not tracked by git.

Required datasets:
- sft/travel-QA/
- dpo/Human-Like-DPO-Dataset/
- rag/crosswoz-sft/
" > data/README.md
```

---

## Summary

After completing this setup:

1. ✅ All required models downloaded (~5-10GB total)
2. ✅ Training datasets prepared
3. ✅ Configuration files updated
4. ✅ Directory structure organized
5. ✅ Installation verified

The system is now ready for:

- Model training (SFT, DPO, PPO)
- RAG functionality
- Web interface deployment
- Production inference

For additional help, refer to the main [README](README.md) or create an issue on the [GitHub repository](https://github.com/1998frankchen/TravelMind).
