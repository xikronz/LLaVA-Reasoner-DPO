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

