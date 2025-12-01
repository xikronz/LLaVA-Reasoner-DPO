from dataclasses import dataclass
from trl import SFTConfig

@dataclass
class TrainingConfig: 
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    small_model_id = "Qwen/Qwen2-VL-2B-Instruct"
    eval_vl_model_id = "Qwen/Qwen2.5-VL-32B-Instruct"
    eval_model_id = "Qwen/Qwen2.5-32B-Instruct"
    
    json_path = "data/stf_data.json"
    image_dir = "data"
    output_dir = "./output/Qwen2.5-VL-3B-Instruct/newest"
    test_dir = "./output"
    adapter_path = "output/Qwen2.5-VL-3B-Instruct/recent"

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
    image_path: str = "r2d_reasoner/data/images"  # Base path for images
    dataset_name: str = "Share4oReasoning/sft_data"
    num_samples: int = 10000
    batch_size: int = 1  # VLMs typically need batch_size=1 due to variable image sizes
    max_new_tokens: int = 1024
    max_image_size: int = 1024  # Max dimension for image resizing
    output_dir: str = "output/evaluation"
    inference_output_dir: str = "r2d_reasoner/data/inference"  # Directory for incorrect responses
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
