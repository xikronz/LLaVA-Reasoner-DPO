from dataclasses import dataclass
from trl import SFTConfig
import torch

@dataclass
class TrainingConfig: 
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    small_model_id = "Qwen/Qwen2-VL-2B-Instruct"
    eval_vl_model_id = "Qwen/Qwen2.5-VL-32B-Instruct"
    eval_model_id = "Qwen/Qwen2.5-32B-Instruct"
    
    json_path = "data/stf_data.json"
    image_dir = "data/images"
    output_dir = "./outputs/Qwen2.5-VL-3B-Instruct/newest"
    test_dir = "./output"
    adapter_path = "outputs/Qwen2.5-VL-3B-Instruct/recent"

    wandb_project = "R2Dtuning"
    wandb_run_name = "lr tuning 2"

    num_train_epochs = 10
    per_device_train_batch_size = 2
    lr = 1e-4

    use_qlora = True
    lora_alpha = 16
    lora_dropout = 0.05
    lora_r = 8 

    train = 0.8
    val = 0.15
    test = 0.05
    quant_storage_dtype = "float16"
    max_grad_norm = 0.3
    warmup_ratio = 0.03

@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    model_id: str = "Qwen/Qwen3-VL-4B-Thinking"
    image_path: str = "data/images"  
    dataset_name: str = "Share4oReasoning/dpo_data"
    num_samples: int = 10000
    batch_size: int = 1 
    max_new_tokens: int = 1024
    max_image_size: int = 1024  
    output_dir: str = "outputs/Qwen3-VL-4B-Thinking/results"
    inference_output_dir: str = "outputs/Qwen3-VL-4B-Thinking/Inference" 
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class EvalDPOConfig:
    model_id: str = "Qwen/Qwen3-VL-4B-Thinking"
    image_path: str = "data/images"  
    dataset_name: str = "Share4oReasoning/dpo_data"
    num_samples: int = 10000
    batch_size: int = 1 
    max_new_tokens: int = 1024
    max_image_size: int = 1024 
    output_dir: str = "outputs/Qwen3-VL-4B-Thinking/results" #stats 
    inference_output_dir: str = "outputs/Qwen3-VL-4B-Thinking/Inference"  #incorrect
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"