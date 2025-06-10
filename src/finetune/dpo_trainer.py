"""
Direct Preference Optimization (DPO) Trainer

This module implements DPO training for language models, which optimizes models
based on preference data without requiring a separate reward model. DPO directly
optimizes the policy using preference pairs (chosen vs rejected responses).

Key features:
- Qwen3/Qwen2.5 model support with AutoModelForCausalLM
- Preference-based training using chosen/rejected response pairs
- KL divergence penalty to maintain closeness to reference model
- LoRA (Low-Rank Adaptation) support for efficient fine-tuning
- Comprehensive evaluation metrics and logging

Classes:
    DPODataset: Dataset class for handling DPO preference pairs
    DPOTrainer: Main trainer class implementing DPO algorithm
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Callable
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    # AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
)

from transformers import AutoModelForCausalLM
from src.configs.config import (
    REWARD_MODEL_PATH,
    MODEL_PATH,
    SFT_MODEL_PATH,
    PPO_MODEL_PATH,
    DPO_DATA_PATH,
    CACHED_DPO_DATA_PATH,
    DPO_MODEL_PATH,
)


from peft import LoraConfig, get_peft_model
from peft.peft_model import PeftModel, PeftModelForCausalLM
from datasets import load_dataset, load_from_disk, DatasetDict

# from trl import DPOTrainer
# import deepspeed
from deepspeed import DeepSpeedEngine


from dataclasses import dataclass


@dataclass
class DPOTrainingConfig:
    """
    Configuration class for DPO (Direct Preference Optimization) training parameters

    This dataclass encapsulates the key hyperparameters used in DPO training,
    providing default values optimized for language model fine-tuning.

    Attributes:
        batch_size (int): Number of preference pairs per training batch
        learning_rate (float): Learning rate for the optimizer
        max_grad_norm (float): Maximum gradient norm for gradient clipping
        num_train_epochs (int): Number of complete passes through the training dataset
    """

    batch_size: int = 4  # Training batch size (preference pairs per batch)
    learning_rate: float = 2e-5  # AdamW learning rate
    max_grad_norm: float = 0.3  # Gradient clipping threshold
    num_train_epochs: int = 3  # Number of training epochs


class DPODataset(Dataset):
    """
    Dataset class for DPO (Direct Preference Optimization) training data

    This dataset handles preference pairs where each sample contains a prompt with
    both a chosen (preferred) and rejected (non-preferred) response. The dataset
    stores tokenized versions of these preference pairs for efficient training.

    Args:
        tokenized_data (dict): Dictionary containing tokenized preference data with keys:
            - "input_ids": Token IDs for chosen responses [num_samples, seq_len]
            - "attention_mask": Attention masks for chosen responses [num_samples, seq_len]
            - "chosen_labels": Token IDs for chosen responses (labels) [num_samples, seq_len]
            - "rejected_labels": Token IDs for rejected responses (labels) [num_samples, seq_len]
    """

    def __init__(self, tokenized_data):
        """Initialize DPODataset with tokenized preference pairs"""
        self.data = tokenized_data

    def __len__(self):
        """Return the total number of preference pairs in the dataset"""
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        """Return a single preference pair at the given index

        Args:
            idx (int): Index of the preference pair to retrieve

        Returns:
            dict: Dictionary containing tokenized chosen and rejected responses
        """
        return {
            "input_ids": self.data["input_ids"][idx],  # Input tokens for chosen response
            "attention_mask": self.data["attention_mask"][idx],  # Attention mask
            "chosen_labels": self.data["chosen_labels"][idx],  # Chosen response labels
            "rejected_labels": self.data["rejected_labels"][idx],  # Rejected response labels
        }


class DPOTrainer(Trainer):
    """
    DPO (Direct Preference Optimization) Trainer

    This trainer implements the DPO algorithm, which optimizes language models directly
    on preference data without requiring a separate reward model. DPO reformulates the
    preference learning problem as a classification task over pairs of responses.

    The key insight of DPO is that the optimal policy satisfying a Bradley-Terry preference
    model can be expressed in closed form, leading to a simple and stable training objective.

    Mathematical Foundation:
    DPO loss: L = -log(σ(β * [log(π(y_w|x)/π_ref(y_w|x)) - log(π(y_l|x)/π_ref(y_l|x))]))
    where:
    - π is the policy being optimized
    - π_ref is the reference policy (frozen)
    - y_w is the preferred (chosen) response
    - y_l is the rejected response
    - β is the temperature parameter
    - σ is the sigmoid function

    Args:
        ref_model: Reference (frozen) model for KL divergence calculation
        beta (float): Temperature parameter controlling the strength of KL constraint
        **kwargs: Additional arguments passed to the base Trainer class
    """

    def __init__(self, ref_model, beta=0.1, **kwargs):
        """Initialize DPO trainer with reference model and temperature parameter"""
        super().__init__(**kwargs)
        self.ref_model = ref_model  # Frozen reference model
        self.beta = beta  # Temperature parameter for preference strength
        self.device = self.ref_model.device  # Device for computation

    def compute_loss(self, model, inputs, num_items_in_batch=None):
        """Compute the DPO loss for a batch of preference pairs

        This method implements the core DPO loss computation, which directly optimizes
        the policy to prefer chosen responses over rejected ones without requiring
        an explicit reward model.

        Mathematical Formulation:
        1. Compute log probabilities for chosen and rejected responses under current policy
        2. Compute log probabilities under reference policy (frozen)
        3. Calculate implicit rewards: r(x,y) = β * log(π(y|x) / π_ref(y|x))
        4. Apply DPO loss: L = -log(σ(β * (r_chosen - r_rejected)))

        Args:
            model: Current policy model being optimized
            inputs (dict): Batch of preference pairs containing:
                - input_ids: Token IDs for the input sequences
                - attention_mask: Attention mask for valid tokens
                - chosen_labels: Token IDs for preferred responses
                - rejected_labels: Token IDs for rejected responses
            num_items_in_batch (int, optional): Number of items in the current batch

        Returns:
            torch.Tensor: Computed DPO loss (scalar value)
        """
        # Move all inputs to the appropriate device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass through current policy model
        outputs = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )

        # Compute log probabilities for chosen and rejected responses under current policy
        chosen_log_probs = self._get_log_probs(outputs.logits, inputs["chosen_labels"])
        rejected_log_probs = self._get_log_probs(
            outputs.logits, inputs["rejected_labels"]
        )

        # Compute reference model log probabilities (no gradient computation needed)
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            # Calculate log(π_ref(y|x)) for both chosen and rejected responses
            ref_chosen_log_probs = self._get_log_probs(
                ref_outputs.logits, inputs["chosen_labels"]
            )
            ref_rejected_log_probs = self._get_log_probs(
                ref_outputs.logits, inputs["rejected_labels"]
            )

        # Compute DPO loss using Bradley-Terry preference model
        # Implicit rewards: r(x,y) = β * log(π(y|x) / π_ref(y|x))
        chosen_rewards = self.beta * (chosen_log_probs - ref_chosen_log_probs)
        rejected_rewards = self.beta * (rejected_log_probs - ref_rejected_log_probs)

        # DPO loss: -log(σ(r_chosen - r_rejected))
        losses = -F.logsigmoid(chosen_rewards - rejected_rewards)

        loss = losses.mean()  # Average over batch
        return loss

    def _get_log_probs(self, logits, labels):
        """Calculate sequence-level log probabilities for given token sequences

        This method computes the sum of log probabilities for all tokens in each sequence,
        which represents the total log probability of generating that sequence under the model.

        Mathematical formulation:
        For sequence y = [y_1, y_2, ..., y_T]:
        log P(y|x) = Σ_{t=1}^T log P(y_t | x, y_1, ..., y_{t-1})

        Args:
            logits (torch.Tensor): Model output logits [batch_size, seq_len, vocab_size]
            labels (torch.Tensor): Target token sequences [batch_size, seq_len]

        Returns:
            torch.Tensor: Sum of log probabilities for each sequence [batch_size]
                Each value represents log P(sequence | input) for one sample

        Note:
            This method is crucial for DPO as it computes log π(y|x) which is used
            to calculate the implicit reward r(x,y) = β * log(π(y|x)/π_ref(y|x))
        """
        # Convert logits to log probabilities using log softmax
        log_probs = F.log_softmax(logits, dim=-1)  # Shape: [batch_size, seq_len, vocab_size]

        # Gather log probabilities for the target tokens using advanced indexing
        # This selects the log probability of each target token at each position
        token_log_probs = torch.gather(
            log_probs, -1, labels.unsqueeze(-1)
        ).squeeze(-1)  # Shape: [batch_size, seq_len]

        # Sum log probabilities across sequence length to get sequence-level probability
        return token_log_probs.sum(dim=-1)  # Shape: [batch_size]


class DPOTrainerWrapper:
    def __init__(
        self,
        output_dir: str = DPO_MODEL_PATH,
        dataset_name_or_path: str = DPO_DATA_PATH,
        cached_dataset_name_or_path: str = CACHED_DPO_DATA_PATH,
        model_name: str = MODEL_PATH,
        is_ds: bool = True,
        ds_config_path: Optional[str] = None,
        is_peft: bool = False,
        peft_config: Optional[LoraConfig] = None,
        is_quantized: bool = False,
        bnb_config: Optional[BitsAndBytesConfig] = None,
        max_seq_length: int = 1024,
        beta: float = 0.1,
        dpo_training_config=None,
    ):
        self.output_dir = output_dir
        self.dataset_name_or_path = dataset_name_or_path
        self.cached_dataset_name_or_path = cached_dataset_name_or_path
        self.beta = beta
        self.max_seq_length = max_seq_length
        self.is_quantized = is_quantized

        self.dpo_training_config = dpo_training_config or DPOTrainingConfig()

        # Initialize model and tokenizer
        self.model, self.tokenizer = self._init_model_and_tokenizer(
            model_name, is_quantized, bnb_config
        )

        self.device = self.model.device

        self.ref_model = self._get_ref_model()

        # Application/AppLoRA
        if is_peft:
            self.peft_config = peft_config or self._default_lora_config()
            self.model = get_peft_model(self.model, self.peft_config)

        # Prepare dataset
        self.dataset, self.eval_dataset = self._load_cached_dataset(
            self.cached_dataset_name_or_path
        )
        # Configure training parameters
        self.training_args = TrainingArguments(
            label_names=["chosen_labels", "rejected_labels"],  # Prevent PEFT errors
            output_dir=output_dir,
            deepspeed=ds_config_path if is_ds else None,
            per_device_train_batch_size=self.dpo_training_config.batch_size,
            gradient_accumulation_steps=2,
            learning_rate=2e-5,
            bf16=True,
            logging_steps=10,
            save_steps=500,
            remove_unused_columns=False,
            optim="adamw_torch",
            max_grad_norm=0.3,
            num_train_epochs=3,
            report_to="none",
        )

        # InitializeCustomTrainer
        self.trainer = DPOTrainer(
            ref_model=self.ref_model,
            beta=self.beta,
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset,
            data_collator=self.dpo_collator,
            compute_metrics=self._compute_metrics,  # Calculate the proportion of correct model preference selection
        )

    def _get_ref_model(self):
        """Create and return reference model"""
        ref_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        ref_model.load_state_dict(self.model.state_dict())
        ref_model = ref_model.to(self.device)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        return ref_model

    def _init_model_and_tokenizer(self, model_name, is_quantized, bnb_config):
        """Initialize model and tokenizer"""

        bnb_config = bnb_config or BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config if self.is_quantized else None,
            device_map="auto",
            trust_remote_code=True,
        )
        return model, tokenizer

    def _default_lora_config(self):
        """DefaultLoRAConfigure"""
        return LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

    def _load_cached_dataset(self, dataset_path=CACHED_DPO_DATA_PATH):
        if not os.path.exists(dataset_path):
            os.makedirs(CACHED_DPO_DATA_PATH, exist_ok=True)
        try:
            tokenized_data = load_from_disk(dataset_path)
            print("Successfully loaded DPO dataset from cache~~~")
            tokenized_data.set_format(
                type="torch"
            )  # Ensure data format is converted to torch tensor before executing dpo_collator
            train_data = tokenized_data["train"]
            eval_data = tokenized_data["validation"]
            return DPODataset(train_data), DPODataset(eval_data)
        except Exception as e:
            print(
                f"Failed to load cached DPO dataset (tokenized): {e}, will reprocess data"
            )
            ppo_train_data, ppo_eval_data = self._prepare_dataset(
                self.dataset_name_or_path
            )
            return ppo_train_data, ppo_eval_data

    def _prepare_dataset(self, dataset_path):
        """Data preprocessing"""
        dataset = load_dataset(dataset_path)
        train_dataset = load_dataset(dataset_path, split="train").select(range(500))

        if "validation" in dataset:
            eval_dataset = load_dataset(dataset_path, split="validation").select(
                range(500)
            )
        else:
            eval_dataset = load_dataset(dataset_path, split="train").select(
                range(500, 1000)
            )

        train_dataset = train_dataset.filter(self._data_filter)
        eval_dataset = eval_dataset.filter(self._data_filter)

        train_data = train_dataset.map(
            self._tokenize_function,
            batched=True,
            num_proc=1,
            remove_columns=train_dataset.column_names,
        )

        val_data = eval_dataset.map(
            self._tokenize_function,
            batched=True,
            num_proc=1,
            remove_columns=eval_dataset.column_names,
        )

        tokenized_data = DatasetDict({"train": train_data, "validation": val_data})

        # Save preprocessed dataset to local storage
        if not os.path.exists(CACHED_DPO_DATA_PATH):
            os.makedirs(CACHED_DPO_DATA_PATH, exist_ok=True)
        tokenized_data.save_to_disk(CACHED_DPO_DATA_PATH)

        train_data.set_format(type="torch")
        val_data.set_format(type="torch")

        return DPODataset(train_data), DPODataset(val_data)

    def _data_filter(self, sample):
        """Data filter logic"""
        return (
            all([sample["prompt"], sample["chosen"], sample["rejected"]])
            and len(sample["prompt"]) <= 512
            and len(sample["chosen"]) <= 1024
            and len(sample["rejected"]) <= 1024
        )

    def _tokenize_function(self, samples):
        """DPO-specific tokenization process"""
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "chosen_labels": [],
            "rejected_labels": [],
        }

        for prompt, chosen, rejected in zip(
            samples["prompt"], samples["chosen"], samples["rejected"]
        ):
            # GeneratepromptTemplate
            full_prompt = f"Instruction: {prompt}\nResponse: "

            # Tokenize chosenResponse
            chosen_tokens = self.tokenizer(
                full_prompt + chosen,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # Tokenize rejectedResponse
            rejected_tokens = self.tokenizer(
                full_prompt + rejected,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            batch["input_ids"].append(chosen_tokens["input_ids"][0])
            batch["attention_mask"].append(chosen_tokens["attention_mask"][0])
            batch["chosen_labels"].append(chosen_tokens["input_ids"][0])
            batch["rejected_labels"].append(rejected_tokens["input_ids"][0])

        return batch

    def dpo_collator(self, features):
        """Custom data collation function"""
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "chosen_labels": torch.stack([f["chosen_labels"] for f in features]),
            "rejected_labels": torch.stack([f["rejected_labels"] for f in features]),
        }
        return batch

    def _compute_metrics(self, eval_pred):
        """Custom evaluation metric

        eval_pred.predictions  == (logits_chosen, logits_rejected)

        eval_pred.predictions contains two parts:
            Model's prediction logits for chosen responses (logits_chosen) and prediction logits for rejected responses (logits_rejected)

        ### Return
        accuracy = (logits_chosen > logits_rejected).mean()
            Compare the size relationship of these two logits, calculate the proportion of correct model preference (i.e., chosen response score higher than rejected response)

        ### Note:
            1. The logits here are actually the model's prediction values for complete sequences (prompt + response)
            2. What we compare is the comprehensive score of the entire sequence, not just individual tokens
            3. Metric value is between 0-1, 1 indicates perfect preference learning

        """
        logits_chosen, logits_rejected = eval_pred.predictions
        accuracy = (logits_chosen > logits_rejected).mean()
        return {"dpo_accuracy": accuracy}

    def train(self):
        """Start training process"""
        self.trainer.train()

    def save_model(self):
        """Model save logic"""
        if not os.path.exists(DPO_MODEL_PATH):
            os.makedirs(DPO_MODEL_PATH, exist_ok=True)
        self.model.save_pretrained(DPO_MODEL_PATH)


class DPOCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        """Initialization before training starts"""
        self.model = kwargs.pop("model")
        if isinstance(self.model, DeepSpeedEngine):
            self.model = self.model.module

        if isinstance(self.model, PeftModel) or isinstance(
            self.model, PeftModelForCausalLM
        ):
            self.model = self.model.base_model

    def on_step_begin(self, args, state, control, **kwargs):
        # """Freeze reference model during gradient accumulation"""
        # if state.global_step == 0:
        #     # InitializeReferenceModelParameter
        #     self.ref_model = self._clone_model(self.model)
        #     # self.ref_model.requires_grad_(False)
        #     self._frozen_model(self.ref_model)

        """No longer need to initialize ref_model here"""
        pass

    def _frozen_model(self, model: nn.Module):
        """Freeze model parameters"""
        if hasattr(model, "named_parameters"):
            for param in model.parameters():
                param.requires_grad = False
        else:
            raise ValueError("The passed object is not a valid PyTorch model")

    def _clone_model(self, model):
        """
        Create a deep copy of reference model


        type(model): Get the class type of the model (e.g., AutoModelForCausalLM)
        **model.config.to_dict(): Convert model's configuration to dictionary and unpack as keyword parameters
        type(model)(**model.config.to_dict()): Use original model's configuration to create a new model instance
        """
        cloned_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        cloned_model.load_state_dict(model.state_dict())
        return cloned_model

        """
        log_probs = torch.tensor([
            [[-0.5, -1.0, -2.0],  # First sample, first token's log probability
            [-0.8, -1.2, -1.5]], # First sample, second token's log probability
            [[-0.6, -1.1, -2.1],  # Second sample, first token's log probability
            [-0.9, -1.3, -1.6]]  # Second sample, second token's log probability  
        ])  # shape: (2, 2, 3)  (batch_size=2, seq_len=2, vocab_size=3)

        labels = torch.tensor([
            [0, 2],  # First sample's labels
            [1, 0]   # Second sample's labels  
        ])  # shape: (2, 2)  (batch_size=2, seq_len=2)
        
        
        Execution process:  
        labels.unsqueeze(-1)：

        Add a dimension to the last dimension of labels  
        Result:

        python
        Apply
        tensor([
            [[0], [2]],  # First sample
            [[1], [0]]   # Second sample  
        ])  # shape: (2, 2, 1)
        torch.gather(log_probs, -1, labels.unsqueeze(-1))：

        On the last dimension (vocab_size dimension) of log_probs, collect corresponding log probabilities based on labels indices  
        Result:

        python
        Apply
        tensor([
            [[-0.5], [-1.5]],  # First sample
            [[-1.1], [-0.9]]   # Second sample  
        ])  # shape: (2, 2, 1)
        .squeeze(-1)：

        Remove last dimension
        Final result:  

        python
        Apply
        tensor([
            [-0.5, -1.5],  # First sample
            [-1.1, -0.9]   # Second sample  
        ])  # shape: (2, 2)
        Explanation:
        For the first sample's first token, labels[0,0]=0, so collect log_probs[0,0,0]=-0.5
        For the first sample's second token, labels[0,1]=2, so collect log_probs[0,1,2]=-1.5
        For the second sample's first token, labels[1,0]=1, so collect log_probs[1,0,1]=-1.1
        For the second sample's second token, labels[1,1]=0, so collect log_probs[1,1,0]=-0.9

        The final matrix represents the log probability values corresponding to each token in each sample.  
        """


if __name__ == "__main__":
    trainer = DPOTrainerWrapper()

    trainer.train()
