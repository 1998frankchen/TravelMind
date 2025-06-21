"""
Supervised Fine-Tuning (SFT) Trainer

This module implements supervised fine-tuning for language models using
instruction-following datasets. SFT trains the model to follow instructions
and generate appropriate responses based on supervised learning.

Key features:
- Qwen3/Qwen2.5 model support with AutoModelForCausalLM
- Instruction-following dataset processing
- SwanLab integration for experiment tracking
- DeepSpeed support for distributed training
- Comprehensive evaluation metrics
- LoRA support for efficient fine-tuning
- Memory monitoring and optimization

Classes:
    SFTTrainer: Main trainer class implementing supervised fine-tuning
"""

from typing import Dict, Optional
import os
import torch
import swanlab
from swanlab.integration.huggingface import SwanLabCallback
import numpy as np
import evaluate
import deepspeed
import transformers
from transformers import (
    Trainer,
    TrainingArguments,
    default_data_collator,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from transformers import TrainerCallback
from peft import PeftModel


import sys

sys.path.append("../../")  # Add parent directory's parent directory to sys.path
sys.path.append("../")
from src.configs.config import (
    MODEL_CONFIG,
    BATCH_SIZE,
    DEEPSPEED_CONFIG_PATH,
    SFT_MODEL_PATH,
    SFT_DPO_MODEL_PATH,
)

# from src.models.model import TravelMind  # Disabled - custom class not available
from src.utils.utils import (
    parse_args,
    get_max_length_from_model,
    check_deepspeed_env,
    check_deepspeed_config,
    load_qwen_in_4bit,
    SFTArguments,
    monitor_memory,
)

# from src.data.data_processor import DataProcessor, CrossWOZProcessor  # Temporarily disabled due to syntax issues
from contextlib import contextmanager


# MODEL_PATH = "/root/autodl-tmp/models/Qwen3-1.5B"


"""
python sft_trainer.py \
--model_name "/root/autodl-tmp/models/Qwen3-1.5B" \
--output_dir "output" \
--device "cuda" \
--device_map "auto"


deepspeed --num_gpus=2 sft_trainer.py \
--deepspeed ds_config.json \
--model_name "/root/autodl-tmp/models/Qwen3-1.5B" \
--output_dir "output" \
--device "cuda" \
--device_map "auto"

deepspeed --num_gpus=2 sft_trainer.py \
--deepspeed ds_config.json


deepspeed --num_gpus 2 sft_trainer.py \
    --deepspeed ds_config.json

"""


class MemoryCallback(TrainerCallback):
    """Callback to monitor GPU memory usage during training"""

    def on_step_end(self, args, state, control, **kwargs):
        """Monitor memory usage at the end of each training step"""
        monitor_memory()


class CustomTrainer(Trainer):
    """Custom trainer with optimized gradient synchronization for distributed training"""

    @contextmanager
    def compute_loss_context_manager(self):
        """
        Override compute loss context manager to optimize gradient synchronization

        This method controls when gradients are synchronized across devices during
        distributed training, improving training efficiency with gradient accumulation.

        For DeepSpeed: Uses empty context manager as DeepSpeed handles synchronization internally
        For regular DDP: Uses no_sync() to prevent synchronization until accumulation is complete
        """
        if self.args.gradient_accumulation_steps > 1:
            if self.deepspeed:
                # DeepSpeed handles gradient synchronization internally
                yield
            else:
                # For non-DeepSpeed distributed training
                if self.model.is_gradient_checkpointing:
                    # Gradient checkpointing requires careful synchronization handling
                    yield
                else:
                    # Use no_sync to prevent premature gradient synchronization
                    with self.model.no_sync():
                        yield
        else:
            # No gradient accumulation, proceed normally
            yield


class SFTTrainer:
    """
    Supervised Fine-Tuning (SFT) Trainer for Language Models

    This class implements supervised fine-tuning for language models using instruction-following
    datasets. SFT is typically the first stage in training aligned language models, where the
    model learns to follow instructions and generate appropriate responses.

    Key Features:
    - Support for instruction-following dataset formats
    - Integration with DeepSpeed for distributed training
    - SwanLab experiment tracking and visualization
    - LoRA support for parameter-efficient fine-tuning
    - Comprehensive evaluation metrics (ROUGE, BLEU)
    - Memory monitoring and optimization
    - Custom data collation for sequence-to-sequence tasks

    Training Process:
    1. Load and tokenize instruction-following datasets
    2. Apply LoRA adapters if specified for efficient training
    3. Train using teacher forcing with cross-entropy loss
    4. Evaluate using generation quality metrics
    5. Save fine-tuned model for downstream use (DPO, PPO, etc.)

    Args:
        travel_agent: Legacy parameter for TravelMind integration (deprecated)
        output_dir (str): Directory for saving trained models and checkpoints
        training_args (TrainingArguments, optional): HuggingFace training configuration
        lora_config (dict, optional): LoRA configuration parameters
        use_lora (bool): Whether to use LoRA for parameter-efficient fine-tuning
        max_length (int): Maximum sequence length for input processing
        local_rank (int): Local rank for distributed training (-1 for single GPU)
        args (SFTArguments): Comprehensive SFT training arguments
    """

    def __init__(
        self,
        travel_agent=None,  # TravelMind class disabled
        output_dir: str = SFT_MODEL_PATH,
        training_args: Optional[TrainingArguments] = None,
        lora_config: Optional[Dict] = None,
        use_lora=False,
        max_length=50,
        local_rank=-1,
        args: SFTArguments = None,
    ):
        """
        Initialize SFT trainer with comprehensive configuration

        Performs environment validation, model initialization, and training setup.
        Ensures DeepSpeed compatibility and configures distributed training settings.
        """
        # Validate DeepSpeed environment for distributed training
        if not check_deepspeed_env():
            raise ValueError(
                "DeepSpeed is not installed or not configured correctly. "
                "Please install DeepSpeed for distributed training support."
            )

        # Configure trainer parameters based on input source
        if travel_agent is None:
            # Use provided arguments for standalone initialization
            self.model_name = args.model_name
            self.output_dir = args.output_dir
            self.device = args.device
            self.device_map = args.device_map
            self.local_rank = args.local_rank
            self.use_lora = use_lora
            self.lora_config = lora_config
        else:
            # Legacy TravelMind integration (deprecated but maintained for compatibility)
            self.model_name = travel_agent.model_name
            self.output_dir = output_dir
            self.device = travel_agent.device
            self.device_map = travel_agent.device_map
            self.local_rank = local_rank
            self.use_lora = travel_agent.use_lora
            self.lora_config = travel_agent.lora_config

        # Initialize model and tokenizer (TravelMind integration disabled)
        self.agent = travel_agent if travel_agent is not None else None
        self.max_length = max_length

        # Configure device for distributed training
        if self.local_rank != -1:
            # Distributed training: use specific GPU for current process
            self.device = torch.device("cuda", self.local_rank)
        else:
            # Single GPU or CPU training
            self.device = self.agent.model.device if self.agent else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model and tokenizer from agent (legacy support)
        if self.agent:
            self.model = self.agent.model
            self.tokenizer = self.agent.tokenizer
        else:
            # Direct model loading would go here if TravelMind is not available
            raise ValueError("TravelMind agent is required for model initialization")

        """
        Regardless of which solution is selected, ensure:
            DeepSpeed's train_batch_size equals the actual total batch size
            DeepSpeed's train_micro_batch_size_per_gpu equals TrainingArguments' per_device_train_batch_size
            All values satisfy: total_batch = micro_batch * num_gpus * grad_accum
        """

        # Set default training parameters
        default_training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=5,
            per_device_train_batch_size=1,  # Batch size per GPU
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            weight_decay=0.01,
            warmup_steps=100,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            # Use bf16 instead of fp16, because bf16 has better numerical stability
            bf16=True,  # Modified here
            fp16=False,  # Close fp16
            # fp16=True,
            logging_dir="./logs",  # Specify log directory
            logging_strategy="steps",
            logging_steps=100,
            logging_first_step=True,
            report_to="none",  # SwanLab already configured in callback functions
            save_steps=100,
            eval_steps=100,
            save_total_limit=3,
            eval_strategy="steps",
            load_best_model_at_end=True,
            # report_to="tensorboard",
            # DeepSpeed configuration
            deepspeed=DEEPSPEED_CONFIG_PATH,
            # Distributed training configuration
            local_rank=int(os.getenv("LOCAL_RANK", -1)),
            ddp_find_unused_parameters=False,
            # Add the following parameters to enable 8-bit optimizer
            optim="paged_adamw_8bit",
        )

        # Update training parameters
        if training_args:
            self.training_args = training_args
        else:
            self.training_args = default_training_args

        check_deepspeed_config(self.training_args)

    def train(
        self,
        train_dataset,
        eval_dataset=None,
        resume_from_checkpoint: Optional[str] = None,
    ):
        """
        Start training

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            resume_from_checkpoint: Checkpoint path to resume training
        """
        swanlab_callback = SwanLabCallback(
            project="qwen3-sft",
            log_dir="./swanlab_logs",
            experiment_name="Qwen3-0.5B",
            description="Fine-tune Qwen3-0.5B model on travel_qa dataset using supervised fine-tuning.",
            config={
                "model": "Qwen/Qwen3-0.5B",
                "dataset": "travel_qa",
            },
        )

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            max_length=self.max_length,  # self.max_length parameter refers to the total length of input + output
            padding="max_length",
            return_tensors="pt",
            pad_to_multiple_of=8,  # Improve computational efficiency
            # padding="longest",      # Updates/NewsPadding
            # mlm=False
        )

        sample = next(iter(train_dataset))
        print("Input[0] :", sample["input_ids"])  # Should be similar to (seq_len,)
        # print("Label type:", type(sample["labels"]))      # Should be torch.Tensor
        # print("Label shape:", sample["labels"].shape)     # Should be consistent with input_ids

        # Create trainer
        trainer = CustomTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[
                transformers.EarlyStoppingCallback(
                    early_stopping_patience=3, early_stopping_threshold=0.01
                ),
                MemoryCallback(),
                swanlab_callback,
            ],
        )

        monitor_memory()
        torch.cuda.empty_cache()
        # Start/BeginTrain/Training
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save model only in main process
        if self.local_rank in [-1, 0]:
            # Check if output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            trainer.save_model(self.output_dir)
            # self.tokenizer.save_pretrained(self.output_dir)

            # Scheme/Solution2
            # self.model.save_pretrained(
            #     self.output_dir,
            #     safe_serialization=True  # Use safe serialization
            # )

        return trainer

    @staticmethod
    def load_trained_model(
        base_model_name: str, adapter_path: str = None, device_map: str = "auto"
    ) -> tuple:
        """
        Load trained model

        Args:
            base_model_name: BasicModelName
            adapter_path: LoRAWeightPath
            device_map: Device mapping strategy

        Returns:
            tuple: (model, tokenizer)
        """
        # Load base model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True, padding_side="right"
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=device_map,
        )

        if adapter_path is not None:
            # LoadLoRAWeight
            model = PeftModel.from_pretrained(
                model, adapter_path, device_map=device_map
            )

        return model, tokenizer

    # def compute_metrics(self, eval_pred):
    #     # Calculate evaluation metrics
    #     metric = evaluate.load("perplexity")

    #     predictions, labels = eval_pred
    #     # Remove padding effects
    #     mask = labels != -100
    #     predictions = predictions[mask]
    #     labels = labels[mask]

    #     return metric.compute(predictions=predictions, references=labels)

    def compute_metrics(self, eval_pred):
        # Calculate evaluation metrics
        # Ensure we get tokenizer instance
        tokenizer = self.tokenizer

        tokenizer.pad_token_id = tokenizer.eos_token_id

        # Separate predictions and labels
        predictions, labels = eval_pred  # labels.shape = (batch_size, max_length)

        # Process prediction results
        pred_ids = np.argmax(predictions, axis=-1)

        print("pred_ids = ")

        unique_tokens, counts = np.unique(pred_ids, return_counts=True)
        print(
            "Prediction token distribution:", list(zip(unique_tokens[:5], counts[:5]))
        )  # Display top 5 frequent tokens

        # Check if all predictions are pad tokens
        if (pred_ids == tokenizer.pad_token_id).all():
            print("Warning: All predictions are pad tokens!")

        decoded_preds = tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        print("labels.shape = ", labels.shape)  # (50, 512)

        label_ids = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
        print("Label example label_ids:", label_ids)  # Should display complete answer

        # Process labels (filter padding values -100)
        decoded_labels = []
        for label_seq in labels:
            # Replace -100 with pad_token_id
            valid_label_ids = np.where(
                label_seq != -100, label_seq, tokenizer.pad_token_id
            )
            decoded_label = tokenizer.decode(
                valid_label_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            decoded_labels.append(decoded_label)

        # Print input/output for first 3 samples
        print("\n===== Debug Examples =====")
        for i in range(3):
            print(f"Sample {i+1}:")
            print(f"Prediction: {decoded_preds[i]}")
            print(f"Reference: {decoded_labels[i]}")
            print("-------------------")

        # Calculate ROUGE-L scores (example)
        rouge = evaluate.load("rouge")
        results = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
            use_aggregator=False,  # Get scores for each sample
        )

        # Add BLEU metric
        bleu = evaluate.load("bleu")
        bleu_results = bleu.compute(
            predictions=decoded_preds,
            references=[[l] for l in decoded_labels],
            # max_order=4,
            # smooth=True  # Enable smooth processing
        )

        # Return average metrics
        return {
            "rouge1": np.mean(results["rouge1"]),
            "rouge2": np.mean(results["rouge2"]),
            "rougeL": np.mean(results["rougeL"]),
            "bleu": bleu_results["bleu"],
        }


if __name__ == "__main__":
    args = SFTArguments()  # Usageparse_argsFetchParameter
    trainer = SFTTrainer(args=args)

    processor = CrossWOZProcessor(
        tokenizer=trainer.tokenizer, max_length=trainer.max_length, system_prompt=None
    )

    data_path = (
        "/root/autodl-tmp/Travel-Agent-based-on-LLM-and-SFT/data/processed/crosswoz_sft"
    )
    processed_data = processor.process_conversation_data_huggingface(data_path)

    trainer.train(
        train_dataset=processed_data["train"], eval_dataset=processed_data["test"]
    )
