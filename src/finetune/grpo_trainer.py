"""
Group Relative Policy Optimization (GRPO) Trainer

This module implements GRPO training for language models, which uses group-based
sampling to estimate advantages without requiring a separate critic model.
GRPO generates multiple responses for each prompt and uses relative scoring
within groups to calculate advantages.

Key features:
- Qwen3/Qwen2.5 model support with AutoModelForCausalLM
- Group-based advantage estimation (no critic model needed)
- Relative scoring within response groups
- KL divergence penalty for policy regularization
- Advantage normalization for stable training
- Support for both training and evaluation datasets

Classes:
    QADataset: Dataset class for question-answering evaluation
    GRPOTrainer: Main trainer implementing GRPO algorithm
    GRPODataset: Dataset class for GRPO training data
"""

import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, DatasetDict
from typing import Optional, Dict, List, Union, Tuple
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np
import copy
from torch.nn.utils.rnn import pad_sequence

from src.configs.config import (
    MODEL_PATH,
    DATA_PATH,
    DPO_DATA_PATH,
    CACHED_DPO_DATA_PATH,
    GRPO_MODEL_PATH,
    CACHED_GRPO_DATA_PATH,
)


from src.evaluation.qa_evaluate import QAEvaluator


"""
GRPO (Group Relative Policy Optimization) Algorithm

Core advantages of GRPO:
    - No critic model needed, uses group sampling to estimate advantages
    - Generate multiple outputs for each prompt, calculate advantages using relative scores within groups
    - Use KL penalty term to maintain closeness to reference model

Core idea of GRPO:
    Unlike PPO, GRPO does not use a critic network to estimate value function.
    Instead, it uses multiple responses generated from the same prompt (a group) to calculate relative advantages.
    Each response in the group is compared with the group's average score to obtain relative advantage.

Advantage normalization:
    Advantages are normalized by group mean and standard deviation (optional),
    which helps stabilize training and prevent high variance.
    Formula: (r_i - mean(r)) / std(r)

KL divergence penalty:
    To prevent policy from deviating too far, KL divergence penalty term is added,
    similar to trust region constraint in PPO.

Data processing:
    Code improves data processing by separating prompt and response,
    which helps more accurately calculate response generation probabilities.
"""


class QADataset(Dataset):
    """
    Dataset class for question-answering evaluation

    This dataset wrapper handles tokenized QA data for evaluation purposes.
    It stores input_ids, attention_mask, and labels for each sample.

    Args:
        dataset (dict): Dictionary containing tokenized data with keys:
            - "input_ids": Token IDs for input sequences
            - "attention_mask": Attention mask for input sequences
            - "labels": Ground truth labels for evaluation
    """

    def __init__(self, dataset):
        """Initialize QADataset with tokenized evaluation data"""
        self.dataset = dataset
        self.input_ids = dataset["input_ids"]
        self.attention_mask = dataset["attention_mask"]
        self.labels = dataset["labels"]

    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """Return a single data sample at the given index

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            dict: Dictionary containing input_ids, attention_mask, and labels
        """
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }
        return item


class GRPODataset(Dataset):
    """
    Dataset class for GRPO (Group Relative Policy Optimization) training data

    This dataset handles preference pairs containing both chosen and rejected responses
    for the same prompt. Each sample contains input_ids and attention_mask for both
    the chosen (preferred) and rejected (non-preferred) responses.

    Args:
        tokenized_data (dict): Dictionary containing tokenized preference pairs with keys:
            - "input_ids": Token IDs for chosen responses
            - "attention_mask": Attention mask for chosen responses
            - "rejected_input_ids": Token IDs for rejected responses
            - "rejected_attention_mask": Attention mask for rejected responses
    """

    def __init__(self, tokenized_data):
        """Initialize GRPODataset with tokenized preference pair data"""
        self.data = tokenized_data
        self.input_ids = tokenized_data["input_ids"]
        self.attention_mask = tokenized_data["attention_mask"]
        self.rejected_input_ids = tokenized_data["rejected_input_ids"]
        self.rejected_attention_mask = tokenized_data["rejected_attention_mask"]

    def __len__(self):
        """Return the total number of preference pairs in the dataset"""
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        """Return a single preference pair at the given index

        Args:
            idx (int): Index of the preference pair to retrieve

        Returns:
            dict: Dictionary containing chosen and rejected response data
        """
        return {
            "input_ids": self.data["input_ids"][idx],
            "attention_mask": self.data["attention_mask"][idx],
            "rejected_input_ids": self.data["rejected_input_ids"][idx],
            "rejected_attention_mask": self.data["rejected_attention_mask"][idx],
        }


@dataclass
class GRPOConfig:
    """Configuration class for GRPO (Group Relative Policy Optimization) training

    This configuration class contains all hyperparameters needed for GRPO training,
    including KL divergence penalties, group sampling parameters, and advantage scaling.

    Attributes:
        beta (float): Weight for KL divergence penalty term in loss calculation
        group_size (int): Number of response samples generated per prompt for group-based advantage estimation
        mu (float): Clipping coefficient for policy ratio (1.0 disables clipping)
        kl_coef (float): Coefficient for KL divergence regularization
        scale_rewards (bool): Whether to normalize advantages by standard deviation for stability
    """

    beta: float = 0.1  # KL divergence weight for policy regularization
    group_size: int = 4  # Number of response samples per prompt for group comparison
    mu: float = 1.0  # Policy ratio clipping coefficient (1.0 means no clipping)
    kl_coef: float = 0.1  # KL divergence penalty coefficient
    scale_rewards: bool = True  # Whether to normalize advantages by standard deviation


class GRPOTrainer(Trainer):
    """
    GRPO (Group Relative Policy Optimization) Trainer

    This trainer implements the GRPO algorithm which uses group-based sampling
    to estimate advantages without requiring a separate critic model. It generates
    multiple responses for each prompt and uses relative scoring within groups
    to calculate advantages.

    Args:
        ref_model: Reference model used for KL divergence calculation
        tokenizer: Tokenizer for text processing
        max_seq_length (int): Maximum sequence length for input processing
        **kwargs: Additional arguments passed to the base Trainer class
    """

    def __init__(self, ref_model, tokenizer, max_seq_length, **kwargs):
        """Initialize GRPO trainer with reference model and tokenizer"""
        super().__init__(**kwargs)
        self.ref_model = ref_model  # Reference model for KL divergence calculation
        self.tokenizer = tokenizer  # Tokenizer for text processing
        self.max_seq_length = max_seq_length  # Maximum sequence length
        self.device = self.ref_model.device  # Device for computation

    def compute_metrics(self, eval_preds):
        """Calculate evaluation metrics for GRPO training

        This method creates a QA evaluator and computes various metrics
        to assess model performance during evaluation.

        Args:
            eval_preds: Evaluation prediction results containing predictions and label_ids
                - predictions: Model output logits or probabilities
                - label_ids: Ground truth labels for comparison

        Returns:
            dict: Dictionary containing computed evaluation metrics
        """
        # Create QA evaluator instance with current model and tokenizer
        evaluator = QAEvaluator(self.model, self.tokenizer, self.max_seq_length)

        # Compute and return evaluation metrics
        return evaluator.compute_metrics(eval_preds)

    def compute_loss(self, model, inputs, num_items_in_batch=None):
        """Calculate GRPO (Group Relative Policy Optimization) loss function

        The core of GRPO is to calculate relative advantages of samples within groups,
        then use these advantages to calculate the final loss. This method implements
        the GRPO algorithm which eliminates the need for a separate critic model by
        using group-based relative scoring.

        Algorithm Overview:
        1. Generate multiple responses for each prompt (group-based sampling)
        2. Calculate log probabilities for current policy and reference policy
        3. Compute KL divergence between policies within each group
        4. Calculate relative advantages using group mean and optional scaling
        5. Apply policy ratio clipping and compute surrogate loss
        6. Add KL divergence penalty for regularization

        Args:
            model: The current policy model being trained
            inputs (dict): Batch of input data containing:
                - input_ids: Token IDs for input sequences [batch_size, seq_len]
                - attention_mask: Attention mask for input sequences [batch_size, seq_len]
            num_items_in_batch (int, optional): Number of items in current batch

        Returns:
            torch.Tensor: Computed GRPO loss value (scalar)
        """
        input_ids = inputs["input_ids"]  # Input token sequences
        attention_mask = inputs["attention_mask"]  # Attention masks
        batch_size = input_ids.shape[0]  # Current batch size

        # Calculate number of groups (if batch not evenly divisible, last group may be smaller)
        num_groups = (batch_size + self.group_size - 1) // self.group_size

        # Forward propagation through current policy model to get output logits
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]

        # Create labels for autoregressive language modeling (shift input by one position)
        labels = input_ids.clone()  # Use input_ids as labels for causal LM

        # Create mask to identify non-padding tokens for accurate probability calculation
        non_padding_mask = labels != self.tokenizer.pad_token_id

        # Calculate log probabilities for current policy
        log_probs = self._get_batch_logprobs(logits, labels, non_padding_mask)

        # Calculate reference policy log probabilities (no gradient computation needed)
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            ref_logits = ref_outputs.logits  # Reference model logits
            ref_log_probs = self._get_batch_logprobs(
                ref_logits, labels, non_padding_mask
            )

        # Organize log probabilities into groups for relative advantage calculation
        group_log_probs = []  # Current policy log probs for each group
        group_ref_log_probs = []  # Reference policy log probs for each group

        for i in range(num_groups):
            start_idx = i * self.group_size  # Starting index for current group
            end_idx = min((i + 1) * self.group_size, batch_size)  # End index (handle last group)

            # Extract log probabilities for current group
            group_log_probs.append(log_probs[start_idx:end_idx])
            group_ref_log_probs.append(ref_log_probs[start_idx:end_idx])

        # Calculate relative advantages within each group (core GRPO innovation)
        advantages = []  # List to store advantages for each group

        for g_log_probs, g_ref_log_probs in zip(group_log_probs, group_ref_log_probs):
            # Compute KL divergence for current group: KL = log(π/π_ref) = log(π) - log(π_ref)
            kl_divs = g_log_probs - g_ref_log_probs  # Shape: [group_size]

            # Calculate group mean for relative advantage computation
            group_mean = kl_divs.mean()  # Mean KL divergence within group

            # Calculate relative advantage for each sample in the group
            if self.scale_rewards and len(kl_divs) > 1:
                # Normalize advantages by group standard deviation for stability
                group_std = (
                    kl_divs.std() + 1e-8
                )  # Add epsilon to prevent division by zero
                sample_advantages = (kl_divs - group_mean) / group_std
            else:
                # Use raw relative advantages without normalization
                sample_advantages = kl_divs - group_mean

            advantages.append(sample_advantages)

        # Concatenate all group advantages back into a single tensor
        all_advantages = torch.cat(advantages)  # Shape: [batch_size]

        # Calculate policy importance ratio: π(a|s) / π_ref(a|s)
        policy_ratio = torch.exp(log_probs - ref_log_probs.detach())

        # Apply policy ratio clipping if enabled (similar to PPO clipping)
        if self.mu < 1.0:
            # Clamp policy ratio to prevent large policy updates
            clipped_ratio = torch.clamp(policy_ratio, 1 - self.mu, 1 + self.mu)
            # Calculate both clipped and unclipped surrogate objectives
            surrogate1 = policy_ratio * all_advantages  # Unclipped objective
            surrogate2 = clipped_ratio * all_advantages  # Clipped objective
            # Use minimum for conservative policy update (similar to PPO)
            surrogate_loss = -torch.min(surrogate1, surrogate2).mean()
        else:
            # No clipping applied - use standard policy gradient objective
            surrogate_loss = -(policy_ratio * all_advantages).mean()

        # Add KL divergence penalty to prevent policy from deviating too far from reference
        kl_loss = (log_probs - ref_log_probs.detach()).mean()

        # Combine surrogate loss with KL penalty
        total_loss = surrogate_loss + self.kl_coef * kl_loss

        return total_loss

    def _get_batch_logprobs(self, logits, labels, non_padding_mask):
        """Calculate log probabilities for a batch of sequences

        This method computes the sum of conditional log probabilities for each sequence
        in the batch, considering only non-padding tokens. It implements the standard
        autoregressive language modeling probability calculation.

        Mathematical formulation:
        For sequence s = [s₁, s₂, ..., sₙ], compute:
        log P(s) = Σᵢ log P(sᵢ₊₁ | s₁, s₂, ..., sᵢ)

        Args:
            logits (torch.Tensor): Model output logits [batch_size, seq_len, vocab_size]
            labels (torch.Tensor): Target token IDs [batch_size, seq_len]
            non_padding_mask (torch.Tensor): Boolean mask for non-padding tokens [batch_size, seq_len]

        Returns:
            torch.Tensor: Sum of log probabilities for each sequence [batch_size]
        """
        # Convert logits to log probabilities using log softmax
        log_probs = F.log_softmax(logits, dim=-1)  # Shape: [batch_size, seq_len, vocab_size]

        # Ensure labels are on the same device as log probabilities
        labels = labels.to(log_probs.device)

        # Extract log probabilities for target tokens using teacher forcing
        # Shift by one position: use tokens [0:seq_len-1] to predict [1:seq_len]
        token_log_probs = torch.gather(
            log_probs[:, :-1], -1, labels[:, 1:].unsqueeze(-1)
        ).squeeze(-1)  # Shape: [batch_size, seq_len-1]

        # Apply non-padding mask and sum log probabilities for each sequence
        # Only consider non-padding tokens for accurate probability calculation
        seq_log_probs = (token_log_probs * non_padding_mask[:, 1:]).sum(dim=1)

        return seq_log_probs  # Shape: [batch_size]


class GRPOTrainerWrapper:
    def __init__(
        self,
        output_dir: str = GRPO_MODEL_PATH,
        dataset_name_or_path: str = DPO_DATA_PATH,
        sft_dataset_name_or_path: str = DATA_PATH,
        cached_dataset_name_or_path: str = CACHED_GRPO_DATA_PATH,
        model_name: str = MODEL_PATH,
        is_ds: bool = True,
        ds_config_path: Optional[str] = None,
        is_peft: bool = False,
        peft_config: Optional[LoraConfig] = None,
        is_quantized: bool = False,
        bnb_config: Optional[BitsAndBytesConfig] = None,
        max_seq_length: int = 1024,
        grpo_config: Optional[GRPOConfig] = None,
    ):
        self.output_dir = output_dir
        self.dataset_name_or_path = dataset_name_or_path
        self.sft_dataset_name_or_path = sft_dataset_name_or_path
        self.cached_dataset_name_or_path = cached_dataset_name_or_path
        self.max_seq_length = max_seq_length
        self.is_quantized = is_quantized

        # GRPOConfigure
        self.grpo_config = grpo_config or GRPOConfig()
        self.beta = self.grpo_config.beta
        self.group_size = self.grpo_config.group_size
        self.mu = self.grpo_config.mu
        self.kl_coef = self.grpo_config.kl_coef
        self.scale_rewards = self.grpo_config.scale_rewards

        # Initialize model and tokenizer
        self.model, self.tokenizer = self._init_model_and_tokenizer(
            model_name, is_quantized, bnb_config
        )

        # Create reference model (use deep copy to ensure parameter independence)
        self.ref_model = self._clone_model(self.model)
        self.ref_model.eval()  # Set to evaluation mode
        self.ref_model.requires_grad_(False)  # Freeze parameters

        # Application/AppLoRA
        if is_peft:
            self.peft_config = peft_config or self._default_lora_config()
            self.model = get_peft_model(self.model, self.peft_config)

        # Prepare datasets
        self.dataset = self._prepare_dataset()

        # ConfigureTrain/TrainingParameter
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            deepspeed=ds_config_path if is_ds else None,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=2e-5,
            bf16=True,
            logging_steps=10,
            save_steps=500,
            remove_unused_columns=False,
            optim="adamw_torch",
            max_grad_norm=0.3,
            num_train_epochs=3,
        )

        self.trainer = GRPOTrainer(
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.grpo_collator,
            # compute_metrics=self._compute_metrics,
        )

    def _init_model_and_tokenizer(self, model_name, is_quantized, bnb_config):
        """Initialize model and tokenizer with optional quantization

        This method sets up the base model and tokenizer, applying quantization
        if specified. Quantization helps reduce memory usage for large models.

        Args:
            model_name (str): Path or name of the model to load
            is_quantized (bool): Whether to apply model quantization
            bnb_config (BitsAndBytesConfig, optional): Custom quantization configuration

        Returns:
            tuple: (model, tokenizer) - Initialized model and tokenizer instances
        """
        # Configure quantization settings if enabled
        bnb_config = (
            bnb_config
            or BitsAndBytesConfig(
                load_in_4bit=True,  # Use 4-bit quantization for memory efficiency
                bnb_4bit_quant_type="nf4",  # NF4 quantization type
                bnb_4bit_compute_dtype=torch.bfloat16,  # Computation dtype
                bnb_4bit_use_double_quant=True,  # Double quantization for better accuracy
            )
            if is_quantized
            else None
        )

        # Initialize tokenizer and configure padding token
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token to EOS token

        # Load model with optional quantization configuration
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,  # Apply quantization if specified
            device_map="auto",  # Automatically distribute model across available devices
            trust_remote_code=True,  # Allow custom model code execution
        )
        return model, tokenizer

    def _clone_model(self, model):
        """Create a deep copy of the model for use as reference model

        This method creates an independent copy of the model that will serve
        as the reference policy for KL divergence calculation in GRPO training.
        The cloned model is frozen and used only for inference.

        Args:
            model: The base model to clone

        Returns:
            model_copy: Deep copy of the input model with identical parameters
        """
        model_copy = copy.deepcopy(model)  # Create complete independent copy
        return model_copy

    def _default_lora_config(self):
        """Create default LoRA (Low-Rank Adaptation) configuration

        LoRA enables efficient fine-tuning by adding low-rank matrices to specific
        layers, significantly reducing the number of trainable parameters while
        maintaining performance.

        Returns:
            LoraConfig: Configuration object with optimized LoRA parameters for GRPO
        """
        return LoraConfig(
            r=64,  # Rank of the low-rank matrices (higher = more parameters)
            lora_alpha=16,  # Scaling factor for LoRA weights
            lora_dropout=0.05,  # Dropout rate for LoRA layers
            target_modules=["q_proj", "v_proj"],  # Which linear layers to apply LoRA to
            bias="none",  # How to handle bias terms
            task_type="CAUSAL_LM",  # Task type for language modeling
        )

    def _prepare_dataset(self, train_size=1000, eval_size=500):
        """
        Process data from scratch

        Training data uses DPO dataset, each sample contains (prompt, chosen, rejected) 3 fields
        Evaluation data uses SFT dataset (travel-qa), each sample contains (Question, Answer) 2 fields
        """
        train_dataset = load_dataset(self.dataset_name_or_path, split="train").select(
            range(train_size)
        )
        eval_dataset = load_dataset(self.sft_dataset_name_or_path)

        if "validation" in eval_dataset:
            eval_dataset = load_dataset(
                self.sft_dataset_name_or_path, split="validation"
            ).select(range(eval_size))
        elif "test" in eval_dataset:
            eval_dataset = load_dataset(
                self.sft_dataset_name_or_path, split="test"
            ).select(range(eval_size))
        else:
            eval_dataset = load_dataset(
                self.sft_dataset_name_or_path, split="train"
            ).select(range(eval_size))

        train_dataset = train_dataset.filter(self._data_filter)
        eval_dataset = eval_dataset.filter(self._data_filter)

        train_data = train_dataset.map(
            self._tokenize_train_function,
            batched=True,
            num_proc=1,
            remove_columns=train_dataset.column_names,
        )

        val_data = eval_dataset.map(
            self._tokenize_eval_function,
            batched=True,
            num_proc=1,
            remove_columns=eval_dataset.column_names,
        )

        tokenized_data = DatasetDict({"train": train_data, "validation": val_data})

        # Save preprocessed dataset to local storage
        if not os.path.exists(CACHED_GRPO_DATA_PATH):
            os.makedirs(CACHED_GRPO_DATA_PATH, exist_ok=True)
        tokenized_data.save_to_disk(CACHED_GRPO_DATA_PATH)

        train_data.set_format(type="torch")
        val_data.set_format(type="torch")

        return GRPODataset(train_data), QADataset(val_data)

    def _data_filter(self, sample):
        return (
            all([sample["prompt"], sample["answer"]])
            and len(sample["prompt"]) <= 512
            and len(sample["answer"]) <= 1024
        )

    def _tokenize_train_function(self, samples):
        """
        Process training dataset (DPO format, contains prompt, chosen, rejected)
        Returns format suitable for GRPO training
        """

        # chosen_input_ids have already been added to batch["input_ids"], this logic is correct

        batch = {
            "input_ids": [],  # Complete input for chosen answer
            "attention_mask": [],  # Attention mask for chosen answer
            "rejected_input_ids": [],  # Complete input for rejected answer
            "rejected_attention_mask": [],  # Attention mask for rejected answer
        }
        # batch = {"input_ids": [], "attention_mask": [], "response_ids": [], "group_labels": []}

        for prompt, chosen, rejected in zip(
            samples["prompt"], samples["chosen"], samples["rejected"]
        ):
            # Process chosen answer
            chosen_prompt_tokens = self.tokenizer(
                f"Question: {prompt}\nAnswer:",
                max_length=self.max_seq_length // 2,
                truncation=True,
                return_tensors="pt",
            )

            chosen_response_tokens = self.tokenizer(
                chosen,
                max_length=self.max_seq_length // 2,
                truncation=True,
                return_tensors="pt",
            )

            # Merge input_ids and attention mask (chosen)
            chosen_input_ids = torch.cat(
                [
                    chosen_prompt_tokens["input_ids"][0],
                    chosen_response_tokens["input_ids"][0][
                        1:
                    ],  # Remove BOS token from response
                ]
            )
            chosen_attention_mask = torch.cat(
                [
                    chosen_prompt_tokens["attention_mask"][0],
                    chosen_response_tokens["attention_mask"][0][1:],
                ]
            )

            # Truncate to maximum length
            if len(chosen_input_ids) > self.max_seq_length:
                chosen_input_ids = chosen_input_ids[: self.max_seq_length]
                chosen_attention_mask = chosen_attention_mask[: self.max_seq_length]

            # Process rejected answer
            rejected_prompt_tokens = self.tokenizer(
                f"Question: {prompt}\nAnswer:",
                max_length=self.max_seq_length // 2,
                truncation=True,
                return_tensors="pt",
            )

            rejected_response_tokens = self.tokenizer(
                rejected,
                max_length=self.max_seq_length // 2,
                truncation=True,
                return_tensors="pt",
            )

            # Merge input_ids and attention mask (rejected)
            rejected_input_ids = torch.cat(
                [
                    rejected_prompt_tokens["input_ids"][0],
                    rejected_response_tokens["input_ids"][0][
                        1:
                    ],  # Remove BOS token from response
                ]
            )
            rejected_attention_mask = torch.cat(
                [
                    rejected_prompt_tokens["attention_mask"][0],
                    rejected_response_tokens["attention_mask"][0][1:],
                ]
            )

            # Truncate to maximum length
            if len(rejected_input_ids) > self.max_seq_length:
                rejected_input_ids = rejected_input_ids[: self.max_seq_length]
                rejected_attention_mask = rejected_attention_mask[: self.max_seq_length]

            # Add to batch
            batch["input_ids"].append(chosen_input_ids)
            batch["attention_mask"].append(chosen_attention_mask)
            batch["rejected_input_ids"].append(rejected_input_ids)
            batch["rejected_attention_mask"].append(rejected_attention_mask)

        # Use PyTorch built-in pad_sequence or existing padding method
        if hasattr(self, "_pad_sequences"):
            batch["input_ids"] = self._pad_sequences(
                batch["input_ids"], max_seq_length=self.max_seq_length
            )
            batch["attention_mask"] = self._pad_sequences(
                batch["attention_mask"], max_seq_length=self.max_seq_length
            )
            batch["rejected_input_ids"] = self._pad_sequences(
                batch["rejected_input_ids"], max_seq_length=self.max_seq_length
            )
            batch["rejected_attention_mask"] = self._pad_sequences(
                batch["rejected_attention_mask"], max_seq_length=self.max_seq_length
            )
        else:
            # Use PyTorch built-in padding
            from torch.nn.utils.rnn import pad_sequence

            batch["input_ids"] = pad_sequence(
                batch["input_ids"],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            batch["attention_mask"] = pad_sequence(
                batch["attention_mask"], batch_first=True, padding_value=0
            )
            batch["rejected_input_ids"] = pad_sequence(
                batch["rejected_input_ids"],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            batch["rejected_attention_mask"] = pad_sequence(
                batch["rejected_attention_mask"], batch_first=True, padding_value=0
            )

        return batch

    def _tokenize_eval_function(self, samples):
        """
        Process evaluation dataset (travel_qa format, contains Question and Answer)
        Returns format suitable for evaluation
        """
        batch = {"input_ids": [], "attention_mask": [], "labels": []}

        # Determine Question and Answer field names
        question_field = "Question" if "Question" in samples else "question"
        answer_field = (
            "Response"
            if "Response" in samples
            else "Answer" if "Answer" in samples else "answer"
        )

        for question, answer in zip(samples[question_field], samples[answer_field]):
            # Encode question
            question_tokens = self.tokenizer(
                f"Question: {question}\nAnswer:",
                max_length=self.max_seq_length // 2,  # Reserve half length for answer
                truncation=True,
                return_tensors="pt",
            )

            # Encode answer
            answer_tokens = self.tokenizer(
                answer,
                max_length=self.max_seq_length // 2,
                truncation=True,
                return_tensors="pt",
            )

            # Merge input_ids and attention mask
            input_ids = torch.cat(
                [
                    question_tokens["input_ids"][0],
                    answer_tokens["input_ids"][0][1:],  # Remove BOS token from answer
                ]
            )
            attention_mask = torch.cat(
                [
                    question_tokens["attention_mask"][0],
                    answer_tokens["attention_mask"][0][1:],
                ]
            )

            # Create labels (-100 means tokens for which loss is not calculated, i.e., question part)
            labels = torch.cat(
                [
                    torch.full_like(question_tokens["input_ids"][0], -100),
                    answer_tokens["input_ids"][0][1:],  # Remove BOS token from answer
                ]
            )

            # Truncate to maximum length
            if len(input_ids) > self.max_seq_length:
                input_ids = input_ids[: self.max_seq_length]
                attention_mask = attention_mask[: self.max_seq_length]
                labels = labels[: self.max_seq_length]

            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)

        # Pad to tensors of same length
        batch["input_ids"] = self._pad_sequences(
            batch["input_ids"], max_seq_length=self.max_seq_length
        )
        batch["attention_mask"] = self._pad_sequences(
            batch["attention_mask"], max_seq_length=self.max_seq_length
        )
        batch["labels"] = self._pad_sequences(
            batch["labels"], padding_value=-100, max_seq_length=self.max_seq_length
        )

        return batch

    def _pad_sequences(self, sequences, padding_value=0, max_seq_length=1024):
        """
        Pad sequences of different lengths to same length
        """
        # max_length = max(len(seq) for seq in sequences)
        max_length = max_seq_length
        padded_sequences = []

        for seq in sequences:
            padding_length = max_length - len(seq)
            padded_seq = torch.cat(
                [seq, torch.full((padding_length,), padding_value, dtype=seq.dtype)]
            )
            padded_sequences.append(padded_seq)

        return torch.stack(padded_sequences)

    def grpo_collator(self, features):
        """
        Compose multiple samples into a batch for GRPO training

        Args:
            features: List of samples, each sample contains input_ids, attention_mask, etc.

        Returns:
            Dictionary containing batch data
        """
        # InitializeResultDictionary/Dict
        batch = {}

        # Check if it's training data (contains rejected_input_ids)
        is_train = "rejected_input_ids" in features[0]

        if is_train:
            # Process training batch (contains chosen and rejected)
            batch = {
                "input_ids": torch.stack([f["input_ids"] for f in features]),
                "attention_mask": torch.stack([f["attention_mask"] for f in features]),
                "rejected_input_ids": torch.stack(
                    [f["rejected_input_ids"] for f in features]
                ),
                "rejected_attention_mask": torch.stack(
                    [f["rejected_attention_mask"] for f in features]
                ),
            }

            # Generate group_ids - one group per sample, used in GRPO to distinguish answer pairs for different questions
            batch_size = len(features)
            batch["group_ids"] = torch.arange(0, batch_size, dtype=torch.long)

            # Ensure tensor format is correct
            batch["input_ids"] = batch["input_ids"].to(torch.long)
            batch["attention_mask"] = batch["attention_mask"].to(torch.long)
            batch["rejected_input_ids"] = batch["rejected_input_ids"].to(torch.long)
            batch["rejected_attention_mask"] = batch["rejected_attention_mask"].to(
                torch.long
            )

        else:
            # ProcessEvaluate/EvaluationBatch
            batch = {
                "input_ids": torch.stack([f["input_ids"] for f in features]),
                "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            }

            # If labels field exists, also add to batch
            if "labels" in features[0]:
                batch["labels"] = torch.stack([f["labels"] for f in features])
                batch["labels"] = batch["labels"].to(torch.long)

            # Ensure tensor format is correct
            batch["input_ids"] = batch["input_ids"].to(torch.long)
            batch["attention_mask"] = batch["attention_mask"].to(torch.long)

        return batch

    def train(self):
        self.trainer.train()

    def save_model(self):
        save_path = os.path.join(self.output_dir, "qwen3_grpo")
        self.trainer.save_model(save_path)
        self.tokenizer.save_pretrained(save_path)
