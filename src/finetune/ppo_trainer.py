"""
Proximal Policy Optimization (PPO) Trainer

This module implements PPO training for language models using the Actor-Critic architecture.
PPO optimizes policies through multiple epochs on the same batch of data with clipped
objective function to ensure stable training.

Key features:
- Qwen3/Qwen2.5 model support with AutoModelForCausalLM
- Actor-Critic architecture with separate value function
- Generalized Advantage Estimation (GAE) for advantage computation
- KL divergence penalty against reference policy
- Reward model integration for external reward signals
- Comprehensive PPO loss with policy, value, and entropy components

Classes:
    PPODataset: Dataset class for PPO training data
    Critic: Value network for estimating state values
    PPOTrainer: Main trainer implementing PPO algorithm
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    AutoModelForSequenceClassification,
    DebertaV2Model,
    DebertaV2Config,
)

from datasets import DatasetDict


from tqdm import tqdm


from transformers import AutoModelForCausalLM

from src.configs.config import (
    REWARD_MODEL_PATH,
    MODEL_PATH,
    SFT_MODEL_PATH,
    PPO_MODEL_PATH,
    DPO_DATA_PATH,
    CACHED_DPO_DATA_PATH,
    CACHED_PPO_DATA_PATH,
)


from peft import LoraConfig, get_peft_model
import os
import numpy as np
from datasets import load_dataset, load_from_disk


class PPODataset(Dataset):
    """
    Dataset class for PPO (Proximal Policy Optimization) training data

    This dataset handles tokenized data for PPO training, including input sequences,
    response sequences, and pre-computed log probabilities from the initial policy.
    Unlike other RL datasets, advantages and returns are computed dynamically during training.

    Args:
        tokenized_data (dict): Dictionary containing tokenized training data with keys:
            - "input_ids": Input token sequences [num_samples, seq_len]
            - "attention_mask": Attention masks for input sequences [num_samples, seq_len]
            - "response_ids": Generated response token sequences [num_samples, response_len]
            - "old_log_probs": Log probabilities from initial policy [num_samples, response_len]
    """

    def __init__(self, tokenized_data):
        """Initialize PPODataset with tokenized training data"""
        self.data = tokenized_data

    def __len__(self):
        """Return the total number of training samples"""
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        """Return a single training sample at the given index

        Note: Advantages and returns are computed dynamically during training
        rather than being pre-stored, as they depend on the current value function.

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            dict: Training sample containing input_ids, attention_mask, response_ids, and old_log_probs
        """
        return {
            "input_ids": self.data["input_ids"][idx],  # Input token sequence
            "attention_mask": self.data["attention_mask"][idx],  # Attention mask
            "response_ids": self.data["response_ids"][idx],  # Generated response tokens
            "old_log_probs": self.data["old_log_probs"][idx],  # Initial policy log probabilities
        }


class Critic(nn.Module):
    """
    Value Network (Critic) for PPO Actor-Critic Architecture

    The critic model estimates the value function V(s) for each state, which represents
    the expected future rewards from that state under the current policy. This is essential
    for computing advantages in PPO training.

    Architecture:
    - Base model (frozen): Shared encoder with the policy model for feature extraction
    - Value head: Linear layer that maps hidden states to scalar value predictions

    The critic is initialized from the policy model to leverage pre-trained representations
    while adding a regression head for value prediction.

    Args:
        base_model: Pre-trained language model used as feature encoder (frozen during training)
    """

    def __init__(self, base_model):
        """Initialize critic with a frozen base model and trainable value head"""
        super().__init__()
        self.base_model = base_model  # Shared encoder (frozen backbone)
        self.base_model.eval()  # Set to evaluation mode (frozen)
        # Linear layer to map hidden states to scalar values
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)

    def forward(
        self, input_ids=None, inputs_embeds=None, attention_mask=None, num_actions=None
    ):
        """Forward pass to compute state values

        Processes input sequences through the frozen base model and predicts
        scalar value estimates for each token position.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs [batch_size, seq_len]
            inputs_embeds (torch.Tensor, optional): Input embeddings [batch_size, seq_len, hidden_size]
            attention_mask (torch.Tensor, optional): Attention mask [batch_size, seq_len]
            num_actions (int, optional): Number of generated tokens (unused in current implementation)

        Returns:
            torch.Tensor: Value predictions for each token position [batch_size, seq_len]
        """
        # Ensure at least one input type is provided
        assert (
            input_ids is not None or inputs_embeds is not None
        ), "Either input_ids or inputs_embeds must be provided"

        # Forward pass through frozen base model to get hidden representations
        if input_ids is not None:
            hidden_states = self.base_model.forward(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state  # Shape: [batch_size, seq_len, hidden_size]
        else:
            hidden_states = self.base_model.forward(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask
            ).last_hidden_state

        # Apply value head to predict scalar values for each position
        value_model_output = self.value_head.forward(
            hidden_states
        )  # Shape: [batch_size, seq_len, 1]

        # Remove last dimension and return value predictions
        values = value_model_output.squeeze(-1)  # Shape: [batch_size, seq_len]

        return values


class PPOTrainer:
    """
    PPO (Proximal Policy Optimization) Trainer for Language Models

    This class implements the PPO algorithm for fine-tuning language models using reinforcement learning.
    PPO is an on-policy algorithm that uses a clipped surrogate objective to prevent large policy updates,
    ensuring stable training while maximizing rewards from a reward model.

    Key Features:
    - Actor-Critic architecture with shared base model
    - Generalized Advantage Estimation (GAE) for variance reduction
    - Policy ratio clipping for stable updates
    - KL divergence penalty to prevent policy drift
    - Integration with external reward models
    - Support for LoRA efficient fine-tuning

    Algorithm Overview:
    1. Generate responses using current policy
    2. Evaluate responses with reward model and compute advantages using GAE
    3. Update policy using clipped PPO objective
    4. Update value function using MSE loss
    5. Repeat for multiple epochs on the same batch
    """

    def __init__(
        self,
        output_dir: str = PPO_MODEL_PATH,
        dataset_path: str = DPO_DATA_PATH,
        cached_data_path: str = CACHED_PPO_DATA_PATH,
        model_name: str = MODEL_PATH,
        reward_model_name: str = REWARD_MODEL_PATH,
        clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        ppo_epochs: int = 3,
        lr: float = 3e-5,
        max_seq_length: int = 1024,
        is_peft: bool = True,
        peft_config: LoraConfig = None,
    ):
        """
        Initialize PPO trainer with comprehensive configuration

        Args:
            output_dir (str): Directory for saving trained models and checkpoints
            dataset_path (str): Path to training dataset (preference pairs)
            cached_data_path (str): Path to cached preprocessed dataset
            model_name (str): Path or name of base policy model to fine-tune
            reward_model_name (str): Path or name of reward model for scoring responses
            clip_epsilon (float): PPO clipping parameter εₒ, restricts policy ratio to [1-ε, 1+ε]
            gamma (float): Discount factor γ for future rewards (0 ≤ γ ≤ 1)
            gae_lambda (float): GAE λ parameter for advantage estimation (0 ≤ λ ≤ 1)
            ppo_epochs (int): Number of optimization epochs per batch of experience
            lr (float): Learning rate for policy and value function optimization
            max_seq_length (int): Maximum sequence length for input processing
            is_peft (bool): Whether to use LoRA for parameter-efficient fine-tuning
            peft_config (LoraConfig, optional): LoRA configuration parameters
        """
        # Store configuration parameters
        self.output_dir = output_dir  # Model output directory
        self.dataset_path = dataset_path  # Training dataset path
        self.cached_data_path = cached_data_path  # Cached data path
        self.max_seq_length = max_seq_length  # Maximum sequence length

        # PPO-specific hyperparameters
        self.clip_epsilon = clip_epsilon  # Policy ratio clipping parameter εₒ
        self.gamma = gamma  # Discount factor γ for reward calculation
        self.gae_lambda = gae_lambda  # GAE λ parameter for advantage estimation
        self.ppo_epochs = ppo_epochs  # Number of optimization epochs per experience batch

        # Initialize tokenizers for policy and reward models
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set padding token
        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)

        # Initialize main policy model (Actor) - the model we want to optimize through PPO
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True
        )
        self.device = self.model.device  # Get device for consistency

        # Create reference policy model for KL divergence calculation
        # This frozen model prevents the policy from deviating too far from the initial behavior
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True
        ).to(self.device)

        # Initialize reward model for scoring generated responses
        # This model provides the reward signal for reinforcement learning
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_name, trust_remote_code=True
        ).to(self.device)

        # Create critic model (Value Network) for state value estimation
        # Uses the same base model as policy but with a separate value head
        self.critic_model = Critic(self.model.base_model).to(self.device)

        # Apply LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning if enabled
        if is_peft:
            self.peft_config = peft_config or self._default_lora_config()
            self.model = get_peft_model(self.model, self.peft_config)

        # Load and prepare training datasets
        self.dataset, self.eval_dataset = self._load_cached_dataset(
            self.cached_data_path
        )

        # Configure separate optimizers for Actor and Critic
        # Separate optimizers allow different learning rates and update schedules
        self.optimizer_actor = torch.optim.Adam(
            self.model.parameters(), lr=0.00005  # Policy model optimizer
        )
        self.optimizer_critic = torch.optim.Adam(
            self.critic_model.parameters(), lr=0.00005  # Value function optimizer
        )

    def _default_lora_config(self):
        return LoraConfig(
            r=32,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

    def _load_cached_dataset(self, dataset_path=CACHED_PPO_DATA_PATH):
        if not os.path.exists(dataset_path):
            os.makedirs(CACHED_PPO_DATA_PATH, exist_ok=True)
        try:
            tokenized_data = load_from_disk(dataset_path)
            print("Successfully loaded DPO dataset from cache~~~")
            # tokenized_data.set_format(type="torch")
            train_data = tokenized_data["train"]
            eval_data = tokenized_data["validation"]
            return PPODataset(train_data), PPODataset(eval_data)
        except Exception as e:
            print(
                f"Failed to load cached DPO dataset (tokenized): {e}, Will reprocess the data"
            )
            ppo_train_data, ppo_eval_data = self._prepare_dataset(
                self.dataset_path, self.max_seq_length
            )
            return ppo_train_data, ppo_eval_data

    def _prepare_dataset(self, dataset_path, max_length):
        """Data preprocessing pipeline"""
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

        def process_fn(samples):
            batch = {
                "input_ids": [],
                "attention_mask": [],
                "response_ids": [],
                "old_log_probs": [],
            }

            for prompt, chosen in zip(samples["prompt"], samples["chosen"]):
                # GenerateComplete/Intactprompt
                full_prompt = f"Instruction: {prompt}\nResponse: {chosen}"

                # TokenizeInput
                tokens = self.tokenizer(
                    full_prompt,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)

                # Calculate old policy probabilities  (Use multinomial sampling instead of argmax)
                with torch.no_grad():
                    logits = self.model.forward(**tokens).logits.to(self.device)
                    log_probs = F.log_softmax(logits, dim=-1)

                    # Multinomial sampling
                    probs = torch.softmax(logits, dim=-1)
                    response_ids = (
                        []
                    )  # Greedy sampling, can be changed to multinomial sampling or top-k
                    for i in range(
                        probs.size(1)
                    ):  # Iterate through each token position
                        token_probs = probs[
                            :, i, :
                        ]  # Get probability distribution for current token position (batch_size, vocab_size)
                        sampled_ids = torch.multinomial(
                            token_probs, num_samples=1
                        )  # (batch_size, 1)
                        response_ids.append(sampled_ids)
                    response_ids = torch.cat(
                        response_ids, dim=-1
                    )  # (batch_size, seq_len)

                    # Only collect log_prob for generated part
                    old_log_probs = torch.gather(
                        log_probs,
                        -1,
                        response_ids.unsqueeze(
                            -1
                        ),  # Collected logprobs for entire sequence, not just the action part
                    ).squeeze(
                        -1
                    )  # shape = (batch_size, seq_len)

                batch["input_ids"].append(tokens["input_ids"][0])
                batch["attention_mask"].append(tokens["attention_mask"][0])
                batch["response_ids"].append(response_ids[0])
                batch["old_log_probs"].append(old_log_probs[0])

            return batch

        train_data = train_dataset.map(
            process_fn,
            batched=True,
            num_proc=1,
            remove_columns=train_dataset.column_names,
        )

        val_data = eval_dataset.map(
            process_fn,
            batched=True,
            num_proc=1,
            remove_columns=eval_dataset.column_names,
        )

        tokenized_data = DatasetDict({"train": train_data, "validation": val_data})

        # Save preprocessed dataset to local storage
        if not os.path.exists(CACHED_PPO_DATA_PATH):
            os.makedirs(CACHED_PPO_DATA_PATH, exist_ok=True)
        tokenized_data.save_to_disk(CACHED_PPO_DATA_PATH)

        train_data.set_format(type="torch")
        val_data.set_format(type="torch")

        return PPODataset(train_data), PPODataset(val_data)

    def ppo_collator(self, features):
        return {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "response_ids": torch.stack([f["response_ids"] for f in features]),
            "old_log_probs": torch.stack([f["old_log_probs"] for f in features]),
        }

    def compute_rewards(self, model_outputs, response_ids, attention_mask, kl_coef=0.1):
        """Calculate comprehensive rewards including external rewards and KL divergence penalty

        This method computes the reward signal used in PPO training by combining:
        1. External rewards from a trained reward model (task-specific quality scores)
        2. KL divergence penalty to prevent the policy from deviating too far from reference

        The reward structure follows the standard RLHF (Reinforcement Learning from Human Feedback)
        approach where most tokens receive only KL penalty, while the final token gets both
        KL penalty and the external reward from the reward model.

        Args:
            model_outputs: Output from the policy model containing logits
            response_ids (torch.Tensor): Generated response token IDs [batch_size, seq_len]
            attention_mask (torch.Tensor): Attention mask for valid tokens [batch_size, seq_len]
            kl_coef (float): Coefficient for KL divergence penalty (default: 0.1)

        Returns:
            tuple: (rewards, new_log_probs, kl) where:
                - rewards: Per-token reward values [batch_size, seq_len]
                - new_log_probs: Log probabilities from current policy [batch_size, seq_len]
                - kl: KL divergence per token [batch_size, seq_len]
        """
        # Convert generated response IDs to human-readable text for reward model evaluation
        response_text = self.tokenizer.batch_decode(
            response_ids, skip_special_tokens=True
        )

        # Tokenize responses for reward model input (may use different tokenizer)
        reward_model_inputs = self.reward_tokenizer(
            response_text, return_tensors="pt", padding=True
        )

        # Move reward model inputs to appropriate device
        reward_model_inputs = {
            k: v.to(self.reward_model.device) for k, v in reward_model_inputs.items()
        }

        # Compute external rewards using the trained reward model
        with torch.no_grad():  # No gradients needed for reward computation
            reward_outputs = self.reward_model.forward(
                input_ids=reward_model_inputs["input_ids"],
                attention_mask=reward_model_inputs["attention_mask"],
            )  # Shape: [batch_size, 1]
            # Extract scalar reward scores for each sequence
            external_rewards = reward_outputs.logits.squeeze(-1)  # Shape: [batch_size]

        # Compute KL divergence penalty between current policy and reference policy
        logits = model_outputs.logits  # Current policy logits
        log_probs = F.log_softmax(logits, dim=-1)  # Convert to log probabilities
        # Extract log probabilities for the actually generated tokens
        new_log_probs = torch.gather(
            log_probs, -1, response_ids.unsqueeze(-1)
        ).squeeze(-1)  # Shape: [batch_size, seq_len]

        # Calculate reference model probabilities for KL divergence computation
        with torch.no_grad():  # Reference model is frozen
            ref_outputs = self.ref_model(
                input_ids=response_ids, attention_mask=attention_mask
            )
            ref_logits = ref_outputs.logits
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            # Extract reference log probabilities for generated tokens
            ref_log_probs = torch.gather(
                ref_log_probs, -1, response_ids.unsqueeze(-1)
            ).squeeze(-1)  # Shape: [batch_size, seq_len]

        # Compute KL divergence: KL(π || π_ref) = log(π) - log(π_ref)
        kl = (new_log_probs - ref_log_probs) * attention_mask  # Shape: [batch_size, seq_len]

        # Construct final reward signal following RLHF standard practice:
        # - Most tokens receive only KL penalty (negative reward for deviation)
        # - Final token receives both KL penalty AND external reward from reward model
        rewards = -kl_coef * kl  # KL penalty for all tokens [batch_size, seq_len]
        rewards[:, -1] += external_rewards  # Add external reward to final token of each sequence

        return rewards, new_log_probs, kl

    def compute_gae(self, rewards, values, masks, gamma=0.99, lam=0.95):
        """Calculate Generalized Advantage Estimation (GAE) for PPO training

        GAE is a method for estimating advantage functions that provides a good
        trade-off between bias and variance. It uses exponentially-weighted averages
        of temporal differences to compute advantages.

        Mathematical formulation:
        δ_t = r_t + γ * V(s_{t+1}) - V(s_t)  (temporal difference)
        A_t^{GAE(γ,λ)} = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}  (GAE advantage)

        The λ parameter controls the bias-variance trade-off:
        - λ = 0: Low variance, high bias (uses only immediate TD error)
        - λ = 1: High variance, low bias (uses Monte Carlo returns)

        Args:
            rewards (torch.Tensor): Per-token rewards [batch_size, seq_len]
            values (torch.Tensor): State value predictions from critic [batch_size, seq_len]
            masks (torch.Tensor): Attention masks for valid tokens [batch_size, seq_len]
            gamma (float): Discount factor for future rewards (γ)
            lam (float): GAE parameter for bias-variance trade-off (λ)

        Returns:
            tuple: (advantages, returns) where:
                - advantages: Normalized advantage estimates [batch_size, seq_len]
                - returns: Target values for critic training [batch_size, seq_len]
        """
        advantages = torch.zeros_like(rewards)  # Initialize advantage tensor
        last_gae_lam = 0  # Initialize A_{T+1} = 0 (no future advantage)

        # Compute GAE advantages using backward iteration through time steps
        for t in reversed(range(len(rewards))):
            # Handle bootstrap value for next state
            if t == len(rewards) - 1:
                next_value = 0  # Terminal state has no future value
            else:
                next_value = values[t + 1]  # Use critic's value prediction

            # Compute temporal difference error: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
            delta = rewards[t] + gamma * next_value * masks[t] - values[t]

            # Update GAE advantage: A_t = δ_t + γ*λ*A_{t+1}
            last_gae_lam = delta + gamma * lam * masks[t] * last_gae_lam
            advantages[t] = last_gae_lam

        # Calculate returns for critic training: R_t = A_t + V(s_t)
        returns = advantages + values

        # Normalize advantages to have zero mean and unit variance for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def compute_loss(
        self, actor_model: AutoModelForCausalLM, critic_model: Critic, inputs
    ):
        # ForwardPropagationFetchlogits
        outputs = actor_model.forward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )

        # Calculate rewards and KL divergence
        rewards, new_log_probs, kl = self.compute_rewards(
            outputs, inputs["response_ids"], inputs["attention_mask"]
        )

        # rewards.shape = (bsz, )

        # Get hidden states for value estimation
        # hidden_states = outputs.hidden_states[-1]  # Get hidden states from the last layer

        # Use value head to calculate value for each token
        values = critic_model.forward(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )  # shape = (bsz, seq_len)

        # Calculate mask (to ignore padding)
        masks = inputs["attention_mask"]

        # Calculate GAE advantages and returns
        advantages, returns = self.compute_gae(
            rewards, values, masks, gamma=self.gamma, lam=self.gae_lambda
        )

        # Calculate new policy probabilities
        log_probs = F.log_softmax(outputs.logits, dim=-1)
        new_log_probs = torch.gather(
            log_probs, -1, inputs["response_ids"].unsqueeze(-1)
        ).squeeze(-1)

        # Calculate probability ratio
        ratio = torch.exp(new_log_probs - inputs["old_log_probs"])

        # Calculate policy loss (clipped PPO objective)
        policy_loss = -torch.min(
            ratio * advantages,
            torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            * advantages,
        ).mean()

        # Calculate value function loss
        value_loss = F.mse_loss(values, returns)

        # Calculate entropy bonus (increase exploration)
        entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=-1).mean()

        # Total loss
        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        return total_loss

    def train(self):
        """Custom training loop"""
        train_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=3e-5,
            logging_steps=10,
            save_steps=500,
            remove_unused_columns=False,
            optim="adamw_torch",
            max_grad_norm=0.5,
        )

        train_dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=train_args.per_device_train_batch_size,
            collate_fn=self.ppo_collator,
            shuffle=True,
        )

        # Custom PPO algorithm cannot use HuggingFace's official Trainer

        # trainer = Trainer(
        #     model=self.model,
        #     args=train_args,
        #     train_dataset=self.dataset,
        #     data_collator=self.ppo_collator,
        #     compute_metrics=None,
        #     compute_loss=self.compute_loss
        # )

        best_eval_loss = float("inf")

        # PPO multi-round optimization
        for epoch in range(self.ppo_epochs):
            progress_bar = tqdm(train_dataloader, desc=f"PPO Epoch {epoch}")

            for batch in progress_bar:

                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                self.optimizer_actor.zero_grad()

                self.optimizer_critic.zero_grad()

                # ForwardPropagation
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True,  # Need to get hidden states for value estimation
                )

                # Calculate loss
                loss = self.compute_loss(self.model, self.critic_model, batch)

                # BackwardPropagation
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), train_args.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.critic_model.parameters(), train_args.max_grad_norm
                )

                self.optimizer_actor.step()
                self.optimizer_critic.step()

                # UpdateProgress bar
                progress_bar.set_postfix(loss=loss.item())

            # Copy updated policy model parameters to reference model
            self.ref_model.load_state_dict(self.model.state_dict(), strict=False)

            eval_loss = self.evaluate()

            # Save best model
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                # self.save_model()
                print(f"New best eval loss: {best_eval_loss:.4f}")

        self.save_model()

    def evaluate(self):
        """Evaluate/EvaluationModelPerformance"""
        self.model.eval()
        self.critic_model.eval()

        eval_dataloader = torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=4,  # Evaluation batch_size can be smaller
            collate_fn=self.ppo_collator,
            shuffle=False,
        )

        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Calculate loss
                loss = self.compute_loss(self.model, self.critic_model, batch)

                total_loss += loss.item() * len(batch["input_ids"])
                total_samples += len(batch["input_ids"])

        avg_loss = total_loss / total_samples
        print(f"\nEvaluation - Average Loss: {avg_loss:.4f}")

        self.model.train()
        self.critic_model.train()
        return avg_loss

    def save_model(self):
        # save_path = os.path.join(self.output_dir, "qwen3_ppo")
        self.model.save_pretrained(self.output_dir)
        # self.tokenizer.save_pretrained(save_path)


if __name__ == "__main__":

    trainer = PPOTrainer()

    trainer.train()
