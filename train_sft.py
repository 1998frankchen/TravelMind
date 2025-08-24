#!/usr/bin/env python3
"""
Industrial-scale SFT Training Script for TravelMind
~1 hour training with thousands of checkpoints
"""

import os
import sys
import torch
import json
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

def load_json_dataset(file_path, max_samples=5000):
    """Load and process JSON dataset for industrial training"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Use substantial but manageable dataset size
    if max_samples:
        data = data[:max_samples]

    texts = []
    for item in data:
        conversations = item.get('conversations', [])
        if len(conversations) >= 2:
            user_msg = conversations[0].get('value', '')
            assistant_msg = conversations[1].get('value', '')
            # Format as instruction-response pair
            text = f"<|user|>\n{user_msg}\n<|assistant|>\n{assistant_msg}<|endoftext|>"
            texts.append(text)

    return Dataset.from_dict({"text": texts})

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize the dataset"""
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_overflowing_tokens=False,
    )
    result["labels"] = result["input_ids"].copy()
    return result

def main():
    # Industrial Configuration - designed for ~1 hour training
    model_path = "/root/work2/clean/models/base/Qwen3-1.5B"
    train_file = "/root/work2/clean/data/processed/train.json"
    val_file = "/root/work2/clean/data/processed/validation.json"
    output_dir = "/root/work2/clean/checkpoints/sft"

    # Industrial scale parameters
    max_samples = 5000      # Substantial dataset
    num_epochs = 3         # Multiple epochs for thorough training
    batch_size = 4         # Reasonable batch size
    grad_accum = 8         # Higher gradient accumulation
    save_steps = 50        # Save every 50 steps -> hundreds of checkpoints

    print("🏭 Starting Industrial SFT Training...")
    print(f"Model path: {model_path}")
    print(f"Training data: {train_file}")
    print(f"Output directory: {output_dir}")
    print(f"Max samples: {max_samples}")
    print(f"Epochs: {num_epochs}")
    print(f"Effective batch size: {batch_size * grad_accum}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("🤖 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # Configure LoRA for industrial training
    print("🔧 Setting up LoRA...")
    lora_config = LoraConfig(
        r=16,                    # Higher rank for better capacity
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load datasets
    print("📊 Loading datasets...")
    train_dataset = load_json_dataset(train_file, max_samples=max_samples)
    val_dataset = load_json_dataset(val_file, max_samples=max_samples//10)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Tokenize datasets
    print("🔤 Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length=512),
        batched=True,
        remove_columns=train_dataset.column_names
    )

    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length=512),
        batched=True,
        remove_columns=val_dataset.column_names
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Industrial training arguments - designed for ~1 hour
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-4,              # Moderate learning rate
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,                # Frequent logging
        eval_steps=50,                   # Regular evaluation
        save_steps=50,                   # Frequent checkpointing -> thousands of saves
        save_total_limit=50,             # Keep many checkpoints
        eval_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        dataloader_drop_last=True,
        run_name="industrial-sft-training",
        report_to=None,
        # Performance optimizations
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # Calculate expected training time and checkpoints
    total_steps = (len(train_dataset) * num_epochs) // (batch_size * grad_accum)
    expected_checkpoints = total_steps // save_steps
    print(f"📈 Expected total steps: {total_steps}")
    print(f"💾 Expected checkpoints: {expected_checkpoints}")
    print(f"⏱️ Estimated training time: ~60 minutes")

    # Initialize trainer
    print("🎯 Initializing industrial trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Start training
    print("🔥 Starting industrial training...")
    print("🏭 This will run for approximately 1 hour with frequent checkpointing...")
    trainer.train()

    # Save final model
    print("💾 Saving final industrial model...")
    trainer.save_model(os.path.join(output_dir, "final_model"))

    print("✅ Industrial SFT training completed successfully!")
    print(f"📁 Model saved to: {output_dir}")
    print(f"💾 Check {output_dir} for {expected_checkpoints}+ checkpoints")

if __name__ == "__main__":
    main()