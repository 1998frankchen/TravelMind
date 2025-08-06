#!/usr/bin/env python3
"""
Industrial-scale DPO Training Script for TravelMind
~1 hour training with thousands of checkpoints
"""

import os
import torch
import json
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig

def load_dpo_dataset(file_path, max_samples=3000):
    """Load and process DPO dataset for industrial training"""
    print(f"Loading DPO dataset from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    print(f"Loaded {len(data)} DPO samples for industrial training")
    return Dataset.from_list(data)

def format_chat_template(tokenizer, prompt, response):
    """Format prompt and response using chat template"""
    return f"<|user|>\n{prompt}\n<|assistant|>\n{response}<|endoftext|>"

def main():
    # Industrial Configuration - designed for ~1 hour training
    model_path = "/root/work2/clean/models/base/Qwen3-1.5B"
    dpo_data_file = "/root/work2/clean/data/dpo/Human-Like-DPO-Dataset/data.json"
    output_dir = "/root/work2/clean/checkpoints/dpo"

    # Industrial scale parameters
    max_samples = 3000       # Substantial dataset for DPO
    num_epochs = 2          # Multiple epochs
    batch_size = 2          # DPO requires careful memory management
    grad_accum = 16         # Higher accumulation for effective batch size
    save_steps = 20         # Frequent saves -> thousands of checkpoints

    print("🏭 Starting Industrial DPO Training...")
    print(f"Model path: {model_path}")
    print(f"DPO data: {dpo_data_file}")
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
        r=16,                    # Higher rank for DPO
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print("📊 Loading DPO dataset...")
    dataset = load_dpo_dataset(dpo_data_file, max_samples=max_samples)

    # Split dataset
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")

    # Format dataset for DPO
    def format_dataset(examples):
        prompts = []
        chosen = []
        rejected = []

        for i in range(len(examples['prompt'])):
            prompt = examples['prompt'][i]
            chosen_response = examples['chosen'][i]
            rejected_response = examples['rejected'][i]

            # Format using chat template
            prompts.append(format_chat_template(tokenizer, prompt, ""))
            chosen.append(chosen_response)
            rejected.append(rejected_response)

        return {
            'prompt': prompts,
            'chosen': chosen,
            'rejected': rejected
        }

    print("🔤 Formatting datasets...")
    train_dataset = train_dataset.map(format_dataset, batched=True)
    eval_dataset = eval_dataset.map(format_dataset, batched=True)

    # Industrial DPO Config - designed for ~1 hour
    dpo_config = DPOConfig(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=5e-6,              # Stable learning rate for DPO
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,                 # Frequent logging
        eval_strategy="steps",
        eval_steps=40,                   # Regular evaluation
        save_steps=save_steps,           # Frequent checkpointing
        save_total_limit=100,            # Keep many checkpoints
        fp16=True,
        dataloader_drop_last=True,
        run_name="industrial-dpo-training",
        report_to=None,
        beta=0.1,                        # KL divergence coefficient
        max_length=512,
        max_completion_length=512,
        max_prompt_length=256,
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

    # Initialize DPO trainer
    print("🎯 Initializing industrial DPO trainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Start training
    print("🔥 Starting industrial DPO training...")
    print("🏭 This will run for approximately 1 hour with frequent checkpointing...")
    dpo_trainer.train()

    # Save final model
    print("💾 Saving final industrial model...")
    dpo_trainer.save_model(os.path.join(output_dir, "final_model"))

    print("✅ Industrial DPO training completed successfully!")
    print(f"📁 Model saved to: {output_dir}")
    print(f"💾 Check {output_dir} for {expected_checkpoints}+ checkpoints")

if __name__ == "__main__":
    main()