import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from peft import get_peft_model, LoraConfig, PeftConfig, PeftModel
from transformers import BitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, Qwen2_5_VLProcessor, AutoModelForCausalLM, AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from configs.config import TrainingConfig
from huggingface_hub import login
import os
import json

def load_gen_model_and_processor(config, sharding=False, eval=False, processor = True):
    """ BitsAndBytesConfig is only for non-quantized models! 
        AWQ models are already quantized lol
        requires config.model_id
                 config.use_qlora
    """

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, 
                                    bnb_4bit_use_double_quant=True, 
                                    bnb_4bit_quant_type="nf4", 
                                    bnb_4bit_compute_dtype=torch.float16, 
                                    bnb_4bit_quant_storage=config.quant_storage_dtype,
                                    )

    # figure out which GPU this process should load onto 
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_str = f"cuda:{local_rank}"

    # change the device_map accordingly

    if sharding: 
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config.model_id,
                                                                quantization_config= bnb_config if config.use_qlora else None, 
                                                                device_map="balanced",  # Automatically balance across GPUs
                                                                torch_dtype=torch.float16
                                                            )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config.model_id,
                                                                quantization_config= bnb_config if config.use_qlora else None, 
                                                                torch_dtype=config.quant_storage_dtype or torch.float16,)
        
    # quantisizing weights (lmao what does this even mean :cry:)
    # extracts from config, default is false

    if getattr(config, "use_qlora", False): 
        peft_config = LoraConfig(
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            r=config.lora_r,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)

    if eval:
        model.eval()

    if processor:
        processor = Qwen2_5_VLProcessor.from_pretrained(config.model_id, use_fast=True)
        processor.model_max_length = 4096
        return model, processor

    else:
        return model

def load_sft_model(config, adapter_path, sharding=False):
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, 
                                    bnb_4bit_use_double_quant=True, 
                                    bnb_4bit_quant_type="nf4", 
                                    bnb_4bit_compute_dtype=torch.float16, 
                                    bnb_4bit_quant_storage=config.quant_storage_dtype,
                                    )
    
    
    # Load base model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_id,
        quantization_config=bnb_config if config.use_qlora else None, 
        torch_dtype=config.quant_storage_dtype or torch.float16,
    )
    
    if sharding: 
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config.model_id,
                                                                quantization_config= bnb_config if config.use_qlora else None, 
                                                                device_map="balanced",  # Automatically balance across GPUs
                                                                torch_dtype=torch.float16
                                                            )
        
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model

def load_qwen3b_processor(config):
    return Qwen2_5_VLProcessor.from_pretrained(config.model_id, use_fast=True)

def load_stf_gen_model_and_processor(config, sharding=False): 
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, 
                                    bnb_4bit_use_double_quant=True, 
                                    bnb_4bit_quant_type="nf4", 
                                    bnb_4bit_compute_dtype=torch.float16, 
                                    bnb_4bit_quant_storage=config.quant_storage_dtype,
                                    )
    
    
    # Load base model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_id,
        quantization_config=bnb_config if config.use_qlora else None, 
        torch_dtype=config.quant_storage_dtype or torch.float16,
    )
    
    if sharding: 
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config.model_id,
                                                                quantization_config= bnb_config if config.use_qlora else None, 
                                                                device_map="balanced",  # Automatically balance across GPUs
                                                                torch_dtype=torch.float16
                                                            )
        
    model = PeftModel.from_pretrained(model, config.adapter_path)

    # Load processor
    processor = Qwen2_5_VLProcessor.from_pretrained(config.model_id, use_fast=True)
    
    return model, processor

def load_mini_gen_model_and_processor(config):
    """ BitsAndBytesConfig is only for non-quantized models! 
        AWQ models are already quantized lol
        requires config.model_id
                 config.use_qlora
    """

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, 
                                    bnb_4bit_use_double_quant=True, 
                                    bnb_4bit_quant_type="nf4", 
                                    bnb_4bit_compute_dtype=torch.float16, 
                                    bnb_4bit_quant_storage=config.quant_storage_dtype,
                                    )

    # figure out which GPU this process should load onto 
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_str = f"cuda:{local_rank}"

    # change the device_map accordingly
    model = Qwen2VLForConditionalGeneration.from_pretrained(config.small_model_id,
                                                               quantization_config= bnb_config if config.use_qlora else None, 
                                                               torch_dtype=config.quant_storage_dtype or torch.float16,)
    
    if getattr(config, "use_qlora", False): 
        peft_config = LoraConfig(
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            r=config.lora_r,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)
    
    processor = AutoProcessor.from_pretrained(config.model_id, use_fast=True)

    return model, processor

def load_eval_vl_model_and_processor (config): 
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, 
                                    bnb_4bit_use_double_quant=True, 
                                    bnb_4bit_quant_type="nf4", 
                                    bnb_4bit_compute_dtype=torch.float16, 
                                    bnb_4bit_quant_storage=config.quant_storage_dtype,
                                    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_str = f"cuda:{local_rank}"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config.eval_vl_model_id,
                                                               quantization_config= bnb_config if config.use_qlora else None, 
                                                               device_map = {"": device_str}, 
                                                               torch_dtype=config.quant_storage_dtype or torch.float16,)
     
    if getattr(config, "use_qlora", False): 
        peft_config = LoraConfig(
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            r=config.lora_r,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)
    
    processor = Qwen2_5_VLProcessor.from_pretrained(config.eval_vl_model_id, use_fast=True)

    return model, processor

def load_eval_model_and_tokenizer (config): 
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, 
                                bnb_4bit_use_double_quant=True, 
                                bnb_4bit_quant_type="nf4", 
                                bnb_4bit_compute_dtype=torch.float16, 
                                bnb_4bit_quant_storage=config.quant_storage_dtype,
                                )
 
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_str = f"cuda:{local_rank}"

    model = AutoModelForCausalLM.from_pretrained(config.eval_model_id,
                                                quantization_config= bnb_config if config.use_qlora else None, 
                                                device_map = {"": device_str}, 
                                                torch_dtype=config.quant_storage_dtype or torch.float16,)
    
    if getattr(config, "use_qlora", False): 
        peft_config = LoraConfig(
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            r=config.lora_r,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)
    
    tokenizer = AutoTokenizer.from_pretrained(config.eval_model_id, 
                                            quantization_config= bnb_config if config.use_qlora else None, 
                                            device_map = {"": device_str}, 
                                            torch_dtype=config.quant_storage_dtype or torch.float16,)

    return model, tokenizer

