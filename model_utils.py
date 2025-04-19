from typing import Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

class ModelUtils:
    @staticmethod
    def load_base_model(model_name: str = "Qwen/Qwen2-7B", 
                       device_map: str = "auto") -> tuple:
        """
        Load base model and tokenizer

        Args:
            model_name: Model name or path
            device_map: Device mapping strategy

        Returns:
            tuple: (model, tokenizer)
        """
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Ensure tokenizer has pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # LoadModel
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch.float16  # Use float16 to save memory
        )
        
        return model, tokenizer
    
    @staticmethod
    def prepare_model_for_lora(
        model: AutoModelForCausalLM,
        lora_config: Optional[Dict] = None
    ) -> AutoModelForCausalLM:
        """
        Configure model with LoRA settings
        
        Args:
            model: BasicModel
            lora_config: LoRAConfigureParameter
        
        Returns:
            AutoModelForCausalLM: Model with LoRA configuration applied
        """
        default_config = {
            "r": 8,  # LoRA rank
            "lora_alpha": 32,  # LoRA alphaParameter
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": TaskType.CAUSAL_LM,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]  # Target modules for training
        }
        
        # UsageUserConfigureUpdateDefaultConfigure
        if lora_config:
            default_config.update(lora_config)
        
        # CreateLoRAConfigure
        peft_config = LoraConfig(**default_config)
        
        # FetchPEFTModel
        model = get_peft_model(model, peft_config)
        
        return model
    
    @staticmethod
    def generate_response(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        GenerateModelResponse
        
        Args:
            model: Model
            tokenizer: AutoTokenizer object
            prompt: InputTooltip
            max_length: Maximum generation length
            temperature: TemperatureParameter
            top_p: top-pSamplingParameter
        
        Returns:
            str: Generated response text
        """
        # EncodeInput
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # GenerateReply
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # DecodeOutput
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt portion from response
        response = response[len(prompt):]
        
        return response.strip()