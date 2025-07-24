import os
import torch
import logging
import evaluate
import random
import argparse
import numpy as np
from itertools import product


import datetime


from typing import List, Union, Optional  
import torch.nn as nn
import torch.distributed as dist  
from torch.utils.checkpoint import checkpoint
import torch.utils.checkpoint 
import torch.multiprocessing as mp  
from torch.nn.parallel import DistributedDataParallel as DDP 

from torch.utils.data import DataLoader, DistributedSampler

# Using standard transformers AutoModelForCausalLM instead of custom Qwen2 implementation
from transformers import AutoModelForCausalLM

from datasets import (
    Dataset,
    load_dataset
)

from transformers import (
    AutoModel,
    AutoTokenizer,
    RobertaTokenizerFast,
    GPT2TokenizerFast,
    BertTokenizerFast,
    T5TokenizerFast,
    Qwen2TokenizerFast,
    AutoConfig,
    BertTokenizerFast,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    
    BertForSequenceClassification,
    # Using AutoModelForCausalLM for all models including Qwen3
    Qwen2ForSequenceClassification,
    RobertaForSequenceClassification,
    GPT2ForSequenceClassification,
    
    BitsAndBytesConfig,
)


from peft import (
    PeftModel,
    
)

from accelerate import (
    init_empty_weights, 
    load_checkpoint_and_dispatch  
)

from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

import os  
import sys

import sys
sys.path.append("../../")  # Add parent's parent directory to sys.path  
sys.path.append("../")
from src.configs.config import MODEL_CONFIG, MODEL_PATH


# Set environment variables to enable memory optimization    
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True" 

from dataclasses import dataclass
import psutil  
import pynvml

@dataclass
class SFTArguments:
    def __init__(self):
        self.model_name = MODEL_PATH
        self.output_dir = "output"
        self.device = "cuda"
        self.device_map = "auto"
        self.local_rank = -1


def parse_args():
    parser = argparse.ArgumentParser(description="SFT Trainer Arguments")
    parser.add_argument("--model_name", type=str, required=True, help="Base model name", default = MODEL_PATH )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory", default = "output" )
    parser.add_argument("--device", type=str, default="cuda", help="device")  
    parser.add_argument("--device_map", type=str, default="auto", help="device mapping strategy")  
    
    # Add parameters required by DeepSpeed    
    parser.add_argument("--local_rank", type=int, default=-1)  
    parser.add_argument("--deepspeed", type=str, default=None) 
    
    return parser.parse_args()




def check_deepspeed_env():
    """Check DeepSpeed environment"""  
    import pkg_resources  
    import torch  
    
    print("\n=== Environment Check ===")  
    print(f"PyTorch version: {torch.__version__}")  
    print(f"CUDA available: {torch.cuda.is_available()}")  
    if torch.cuda.is_available():  
        print(f"CUDA version: {torch.version.cuda}")  
        print(f"GPU count: {torch.cuda.device_count()}")  
    
    try:  
        ds_version = pkg_resources.get_distribution('deepspeed').version  
        print(f"DeepSpeed version: {ds_version}")  
    except pkg_resources.DistributionNotFound:  
        print("DeepSpeed not found!")  
        
    return True




def check_deepspeed_config(training_args):
    # 1. CheckEnvironmentVariable  
    print("Environment DEEPSPEED_CONFIG:", os.environ.get('DEEPSPEED_CONFIG'))  
    
    # 2. Check configuration in TrainingArguments    
    print("TrainingArguments deepspeed config:", training_args.deepspeed)


    # If you want to see specific content    
    if training_args.deepspeed:  
        import json  
        try:  
            with open(training_args.deepspeed, 'r') as f:  
                ds_config = json.load(f)  
            print("DeepSpeed config content:", json.dumps(ds_config, indent=2))  
        except Exception as e:  
            print(f"Error reading deepspeed config: {e}") 





def setup_cuda_debug_environment():
    """Set up debug environment"""  
    import torch  
    
    torch.backends.cuda.matmul.allow_tf32 = False  # Disable TF32 for more accurate error information    
    torch.backends.cudnn.deterministic = True      # Use deterministic algorithms    
    torch.backends.cudnn.benchmark = False         # Disable benchmark optimization    
    
    import os  
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  
    os.environ['TORCH_USE_CUDA_DSA'] = '1' 
    
    print("=== Debug Environment Setup ===")  
    print(f"CUDA available: {torch.cuda.is_available()}")  
    print(f"CUDA version: {torch.version.cuda}")  
    print(f"PyTorch version: {torch.__version__}")  
    print(f"TORCH_USE_CUDA_DSA: {os.getenv('TORCH_USE_CUDA_DSA')}")  
    print(f"Current device: {torch.cuda.current_device()}")  
    print(f"Device name: {torch.cuda.get_device_name()}")  
    print("===========================")  
    
    



def load_split_model(model_name_or_path):
    """
    Split model across multiple GPUs before loading
    """  
    # 1. First get model configuration    
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)  
    
    # 2. Calculate maximum memory allocation for each GPU    
    max_memory = {  
        0: "20GiB",  # GPU 0 max usage 20GB
        1: "20GiB",  # GPU 1 max usage 20GB
        "cpu": "15GB"  # CPU memory reserve 15GB    
    }  
    
    # 3. Use device_map="auto" to let Accelerate automatically decide optimal allocation    
    try:  
        # Method 1: Direct load and automatic allocation    
        model = AutoModelForCausalLM.from_pretrained(  
            model_name_or_path,  
            device_map="auto",  
            max_memory=max_memory,  
            torch_dtype=torch.bfloat16,  
            trust_remote_code=True,  
            use_flash_attention_2=True  
        )  
        
    except Exception as e:  
        print(f"Direct loading failed, trying alternative method: {str(e)}")  
        
        # Method 2: Use empty weight initialization then load    
        with init_empty_weights():  
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)  
            
        model = load_checkpoint_and_dispatch(  
            model,  
            model_name_or_path,  
            device_map="auto",  
            max_memory=max_memory,  
            no_split_module_classes=["GPTBlock"],  # 适当Set/Configure不可Split的Module    
            dtype=torch.bfloat16,  
            offload_folder="offload"  # Set/ConfigureWeightUninstallDirectory  
        )  
    
    return model  






def load_qwen_in_4bit(  
    model_name,  
    load_in_4bit=True,  
    use_flash_attention=False  
):  
    """  
    Usage更激进的OptimizeScheme/SolutionLoadQwenModel    
    
    Args:  
        model_name: ModelName或Path    
        load_in_4bit: Yes/IsNo/NotUsage4-bit量化    
        use_flash_attention: Yes/IsNo/NotUsageFlash Attention 2  
    """  
    # Initializetokenizer  
    tokenizer = AutoTokenizer.from_pretrained(  
        model_name,  
        trust_remote_code=True  
    )  

    torch.cuda.empty_cache()  
    
    # Configure4-bit量化Parameter    
    quantization_config = BitsAndBytesConfig(  
        load_in_4bit=True,  
        bnb_4bit_compute_dtype=torch.bfloat16,    # torch.float16,  
        bnb_4bit_use_double_quant=True,  
        bnb_4bit_quant_type="nf4",  # Usagenested float 4 量化    
        bnb_4bit_quant_storage=torch.bfloat16,  # Storage时也Usage4-bit   
    )  

    max_memory = {}  
    for i in range(torch.cuda.device_count()):  
        total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3  
        # 预留2GB给System    
        max_memory[i] = f"{int(total_mem - 2)}GiB"  
    max_memory["cpu"] = "15GB"  # CPUMemory预留    

    print("Max memory configuration:", max_memory)
    
    # Set/ConfigureModelLoadConfigure  
    model_kwargs = {  
        "torch_dtype": torch.bfloat16,  
        "trust_remote_code": True,  
        # "device_map": "auto",  
        "quantization_config": quantization_config,  
        "max_memory": max_memory,  # Restriction/LimitationGPU显存Usage    
        "offload_folder": "offload",  # Set/ConfigureModelWeightUninstallDirectory  
        # EnableGradientCheck点以节省显存    
        # "use_gradient_checkpointing": True,  
    }  
    
    if use_flash_attention:  
        model_kwargs["use_flash_attention_2"] = True  
    
    # LoadModel  
    model = AutoModelForCausalLM.from_pretrained(  
        model_name,  
        **model_kwargs,  
        low_cpu_mem_usage=True,  
    )  

    torch.cuda.empty_cache()  

    # 在ModelLoad后Set/Configuregradient checkpointing    
    if hasattr(model, 'gradient_checkpointing_enable'):  
        model.gradient_checkpointing_enable()  
    elif hasattr(model, 'enable_gradient_checkpointing'):  
        model.enable_gradient_checkpointing() 

    # DisableCache  
    model.config.use_cache = False  

     # Note/Attention：这种Method更细粒度，可以ControlSpecific/Concrete哪些LayerUsagecheckpoint    
    for module in model.modules():  
        if isinstance(module, torch.nn.TransformerEncoderLayer):
            # 给forward加了一LayerPackage装，Prohibit/Forbid计算Middle/CenterLayerActivateValue    
            module.forward = torch.utils.checkpoint.checkpoint(module.forward)  
        elif isinstance(module, torch.nn.TransformerDecoderLayer):
            module.forward = torch.utils.checkpoint.checkpoint(module.forward)
    
    
    # 强制进行垃圾回收    
    import gc  
    gc.collect()    
    torch.cuda.empty_cache()  
    
    return model



def load_qwen(
    model_name,  
    use_flash_attention=False  
):
    
     # Initializetokenizer  
    tokenizer = AutoTokenizer.from_pretrained(  
        model_name,  
        trust_remote_code=True  
    )  

    torch.cuda.empty_cache()  
    

    max_memory = {}  
    for i in range(torch.cuda.device_count()):  
        total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3  
        # 预留2GB给System    
        max_memory[i] = f"{int(total_mem - 2)}GiB"  
    max_memory["cpu"] = "15GB"  # CPUMemory预留    

    print("Max memory configuration:", max_memory)
    
    # Set/ConfigureModelLoadConfigure  
    model_kwargs = {  
        "torch_dtype": torch.float16,  # 防止和 deepspeed ConfigureConflict  
        "trust_remote_code": True,  
        # "device_map": "auto",  
        "max_memory": max_memory,  # Restriction/LimitationGPU显存Usage    
        # "offload_folder": "offload",  # Set/ConfigureModelWeightUninstallDirectory  
        # EnableGradientCheck点以节省显存    
        # "use_gradient_checkpointing": True,  
    }  
    
    if use_flash_attention:  
        model_kwargs["use_flash_attention_2"] = True  
    
    # LoadModel  
    model = AutoModelForCausalLM.from_pretrained(  
        model_name,  
        **model_kwargs,  
        low_cpu_mem_usage=True,  
    )  

    torch.cuda.empty_cache()  

    # 在ModelLoad后Set/Configuregradient checkpointing    
    if hasattr(model, 'gradient_checkpointing_enable'):  
        model.gradient_checkpointing_enable()  
    elif hasattr(model, 'enable_gradient_checkpointing'):  
        model.enable_gradient_checkpointing() 

    # DisableCache  
    model.config.use_cache = False  

     # Note/Attention：这种Method更细粒度，可以ControlSpecific/Concrete哪些LayerUsagecheckpoint    
    for module in model.modules():  
        if isinstance(module, torch.nn.TransformerEncoderLayer):
            # 给forward加了一LayerPackage装，Prohibit/Forbid计算Middle/CenterLayerActivateValue    
            module.forward = torch.utils.checkpoint.checkpoint(module.forward)  
        elif isinstance(module, torch.nn.TransformerDecoderLayer):
            module.forward = torch.utils.checkpoint.checkpoint(module.forward)
    
    
    # 强制进行垃圾回收    
    import gc  
    gc.collect()    
    torch.cuda.empty_cache()  
    
    return model
    
    
    
    
    
def monitor_memory():
    """Monitor GPU and CPU memory usage"""    

    try:  
        # Initialize NVML  
        pynvml.nvmlInit()  
        
        print("\nGPU Memory Usage:")  
        # Get information for all GPUs    
        for i in range(torch.cuda.device_count()):  
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)  
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)  
            
            print(f"\nGPU {i}:")  
            print(f"Total memory: {info.total / 1024**3:.2f} GB")  
            print(f"Used memory: {info.used / 1024**3:.2f} GB")  
            print(f"Free memory: {info.free / 1024**3:.2f} GB")  
            
            # Get GPU utilization    
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)  
            print(f"GPU Utilization: {utilization.gpu}%")  
            print(f"Memory Utilization: {utilization.memory}%")  
            
        pynvml.nvmlShutdown()  
        
    except Exception as e:  
        print(f"Error monitoring GPU memory: {str(e)}")  



# 更Simple的Version，只Usage torch    
def print_gpu_memory():  
    """Usage torch Print GPU MemoryUsage情况"""    
    for i in range(torch.cuda.device_count()):  
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")  
        print(f"Memory Allocated: {torch.cuda.memory_allocated(i)/1024**3:.2f} GB")  
        print(f"Memory Reserved: {torch.cuda.memory_reserved(i)/1024**3:.2f} GB")  
        print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated(i)/1024**3:.2f} GB")  

# Complete/Intact的MonitorFunction（Package括CPUMemory）    
def monitor_system_resources():  
    """MonitorSystemResourceUsage情况"""    
    import psutil  
    
    # CPU Usage情况    
    print("\nCPU Usage:")  
    print(f"CPU Usage: {psutil.cpu_percent()}%")  
    
    # MemoryUsage情况    
    memory = psutil.virtual_memory()  
    print("\nSystem Memory:")  
    print(f"Total: {memory.total/1024**3:.2f} GB")  
    print(f"Available: {memory.available/1024**3:.2f} GB")  
    print(f"Used: {memory.used/1024**3:.2f} GB")  
    print(f"Percentage: {memory.percent}%")  
    
    # GPU Usage情况    
    print_gpu_memory()  
    
    
    
def get_model_name_using_model(model):
    '''
    
    use the model object's config file to retrieve the model name, e.g. bert-base-uncased
    '''
    
    if hasattr(model, "module"):
        print("This model is wrapped by Accelerator(DDP), we use model.module")
        model = model.module
        
    config = model.config  
    # 尝试直接FetchModel的Name    
    if hasattr(config, 'name_or_path') and config.name_or_path is not None:  
        # Usage os.path.basename ExtractPathMedium的ModelName    
        model_name = os.path.basename(config.name_or_path)  
        return model_name  
    # 根据ModelClass型和HideLayerSize推断ModelName    
    if config.model_type == "bert":  
        if config.hidden_size == 768:  
            return "bert-base-uncased"  
        elif config.hidden_size == 1024:  
            return "bert-large-uncased"  
    elif config.model_type == "roberta":  
        if config.hidden_size == 768:  
            return "roberta-base"  
        elif config.hidden_size == 1024:  
            return "roberta-large"  
    elif config.model_type == "llama":  
        if config.hidden_size == 4096:  
            return "meta-llama/Llama-2-13b-hf"  
        elif config.hidden_size == 5120:  
            return "meta-llama/Llama-2-70b-hf"  
    elif config.model_type == "qwen2":  
        if config.hidden_size == 896:
            return "Qwen3-0.6B"
        elif config.hidden_size == 1536:
            return "Qwen3-1.5B"
        elif config.hidden_size == 2048:
            return "Qwen3-3B"
        elif config.hidden_size == 3584:
            return "Qwen3-7B"
    elif config.model_type == "gpt2":
        if config.n_embd == 768:
            return "gpt2"
        elif config.n_embd == 1024:
            return "gpt2-medium"
        elif config.n_embd == 1280:
            return "gpt2-large"
        elif config.n_embd== 1600:
            return "gpt2-xl"
    else:  
        # 无法Match已知Model，Return/Back未知ModelTooltip    
        raise ValueError("unknown model, please check your config, it should be [bert | llama | qwen2]") 

def get_base_model_using_model(model):
    """
    FetchModelPackage装器的底Layer的基座ModelObject  

    """
    # Process被Accelerator(DDP)Package装的Model  
    if hasattr(model, "module"):
        print("This model is wrapped by Accelerator(DDP), we use model.module")
        model = model.module
    
        # FetchModelClass型    
    model_type = type(model)

    if hasattr(model, "config"):
        config = model.config
    else:
        raise RuntimeError("This model object does not have a config file, check again~~~")

    try:
        if isinstance(model, AutoModel):
            model = model
        elif isinstance(model, PeftModel):  
            print("Info: Model is a PeftModel, getting the base model")  
            model = model.get_base_model() 
        elif isinstance(model, AutoModelForSequenceClassification):
            model = model.base_model
        elif isinstance(model, BertForSequenceClassification):
            model = model.bert
        elif isinstance(model, RobertaForSequenceClassification):
            model = model.roberta
        elif isinstance(model, Qwen2ForSequenceClassification):
            model = model.model
        elif isinstance(model, GPT2ForSequenceClassification):
            model = model.transformer
         
        else:
            raise ValueError(f"the passed model object is not either SequenceClassification model or AutoModel \
                The current model type = {model_type}")

    except:
        raise ValueError(f"Extracting base model failed, your current model type is {model_type}")

    return model

def get_hidden_size_using_config():
    pass

def get_hidden_size_by_model_name(model_name:str):
    pass

def get_hidden_size_using_model(model):
    # Process被Accelerator(DDP)Package装的Model  
    if hasattr(model, "module"):
        print("This model is wrapped by Accelerator(DDP), we use model.module")
        model = model.module
    
        # FetchModelClass型    
    model_type = type(model)
    
    model_name = get_model_name_using_model(model)

    if hasattr(model, "config"):
        config = model.config
    else:
        raise RuntimeError("This model object does not have a config file, check again~~~")
    
    if hasattr(config,'hidden_size'):
        hidden_size = config.hidden_size
    elif hasattr(config, 'd_model'): # t5
        hidden_size = config.d_model
    elif hasattr(config, 'n_embd'): # gpt2
        hidden_size = config.n_embd
    else:
        raise ValueError(f"the passed model object does not have the attribute `hidden_size` \
            The current model type = {model_type}")
    print(f"model:{model_name}'s hidden_size = {hidden_size}")
    return hidden_size

def get_classifier_from_model(model)-> nn.Module:  
    """  
    Fetch预Train/TrainingModel的分Class器    
    
    Args:  
        model : AutoModelForSequenceClassification or BertForSequenceClassification
        num_labels (int): 分ClassTag数量    
    
    Returns:  
        nn.Module: 分Class器Module    
    """  
    # Process被Accelerator(DDP)Package装的Model  
    if hasattr(model, "module"):
        print("This model is wrapped by Accelerator(DDP), we use model.module")
        model = model.module

    # Fetch分Class器    
    if hasattr(model, 'classifier'):  
        # BERT、RoBERTa 等Model的分Class器    
        classifier = model.classifier  
        print(f"分Class器Class型: {type(classifier).__name__}")  
        
    elif hasattr(model, 'score'):   # qwen2, gpt2
        # 某些Model可能Usage score 作为分Class器Name    
        classifier = model.score  
    else:  
        raise AttributeError("无法找到Model的分Class器Layer")    
    
    # Print分Class器Information    
    print("分Class器结构：")    
    print(classifier)  
    
    in_features=None
    out_features=None
    if hasattr(classifier, 'dense'):
        in_features = classifier.dense.in_features
        print("这Yes/Is一个RobertaClassificationHead，需要PassdenseLayerFetchInputDimension")  
    else:
        in_features = classifier.in_features
        
    if hasattr(classifier, 'out_proj'):
        out_features = classifier.out_proj.out_features
        print("这Yes/Is一个RobertaClassificationHead，需要Passout_projLayerFetchOutputDimension")  
    else:
        out_features = classifier.out_features
        
        
    print(f"\n分Class器InputDimension: {in_features}")    
    print(f"分Class器OutputDimension: {out_features}")   
    
    # Example：直接Usage分Class器进行ForwardPropagation    
    # batch_size = 4  
    # hidden_size = classifier.in_features  
    
    # Mock来自BERT的FeatureOutput    
    # dummy_features = torch.randn(batch_size, hidden_size)  
    
    # # 直接Usage分Class器进行Prediction/Forecast    
    # with torch.no_grad():  
    #     outputs = classifier(dummy_features)  
        
    # print(f"\n分Class器OutputShape: {outputs.shape}")    
    # print("分Class器OutputExample：")    
    # print(outputs)   
    
    
    print("\n分Class器的可Train/TrainingParameter：")    
    for name, param in classifier.named_parameters():  
        print(f"{name}: {param.shape}")  
        
    return classifier 

def get_max_length_from_model(model):  
    """  
    FetchModel的最大SequenceLength    
    model: 既可以base model， 也可以Yes/Is特定Taskmodel  
    
    """  
    if isinstance(model,str):
        model = AutoModel.from_pretrained(model)
    
    # Process被Accelerator(DDP)Package装的Model    
    if hasattr(model, "module"):  
        print("This model is wrapped by Accelerator(DDP), we use model.module")  
        model = model.module  
        
    if hasattr(model, "config"):
        config = model.config  
    else:
        raise ValueError('your model object is not properly defined ... since we can not find a `config` attribute')
    
    # 首先尝试从configMedium直接Fetchmax_position_embeddings    
    if hasattr(config, 'max_position_embeddings'):  
        return config.max_position_embeddings  
    
    # 如果没有max_position_embeddings，尝试Fetchmax_sequence_length    
    elif hasattr(config, 'max_sequence_length'):  
        return config.max_sequence_length  
    
    elif hasattr(config, 'n_positions'):  
        return config.n_positions
    
    elif hasattr(config, 'n_ctx'):  
        return config.n_ctx
    
    else:
        raise ValueError("Error model object, please check your config, it should have either [max_position_embeddings | max_sequence_length]") 

def get_classifier(model:AutoModelForSequenceClassification):
    """
    Fetch预Train/TrainingModel的分Class器  
    """
    # Process被Accelerator(DDP)Package装的Model  
    if hasattr(model, "module"):
        print("This model is wrapped by Accelerator(DDP), we use model.module")
        model = model.module

    classifier = None
    # Fetch分Class器  
    if hasattr(model, 'classifier'):
        # BERT、RoBERTa 等Model的分Class器  
        classifier = model.classifier
        print(f"分Class器Class型: {type(classifier).__name__}")  
    elif hasattr(model, 'score'):
        # 某些Model可能Usage score 作为分Class器Name  
        classifier = model.score
        
    else:
        raise AttributeError("无法找到Model的分Class器Layer")  
    
    return classifier

def print_model_info(model:AutoModelForSequenceClassification):  
    """PrintModel的DetailedInformation"""    
    
    
    print("\n=== Model Classification Head Information ===")  
    
    # 1. Print分Class器的结构    
    print("\nClassifier Architecture:")  
    if hasattr(model,'classifier'):
        print(model.classifier)  
    elif hasattr(model,'score'):
        print(model.score)
    
    # 2. Print分Class器MediumdenseLayer的WeightShape   
    if hasattr(model,'classifier') and hasattr(model.classifier, 'dense'): 
        dense_weight = model.classifier.dense.weight  
        print("\nDense Layer Weight Shape:", dense_weight.shape)  
    
    # 3. Print分Class器Mediumout_projLayer的WeightShape    
    if hasattr(model,'classifier') and hasattr(model.classifier, 'out_proj'):
        out_proj_weight = model.classifier.out_proj.weight  
        print("Output Projection Weight Shape:", out_proj_weight.shape)  
    
    # 4. Print整个Model的Parameter数量    
    total_params = sum(p.numel() for p in model.parameters())  
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  
    print(f"\nTotal Parameters: {total_params:,}")  
    print(f"Trainable Parameters: {trainable_params:,}")  
    print(f"Percentage of Trainable Parameters: {100 * trainable_params / total_params:.2f}%") 

