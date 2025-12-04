import wandb 
import os
import json
import torch
from accelerate import Accelerator
from data.dataset import create_splits
from model.model_loader import load_gen_model_and_processor
from configs.config import TrainingConfig
from trl import SFTTrainer
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLProcessor, TrainerCallback
from trl import SFTConfig
from peft import LoraConfig
from utils import clear_memory
from PIL import Image

config = TrainingConfig()

class ClearCudaCacheCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        return control

def main():

    clear_memory()

    accelerator = Accelerator()

    training_args = SFTConfig(output_dir=config.output_dir,
                               run_name=config.wandb_run_name,
                               num_train_epochs=config.num_train_epochs,
                               per_device_train_batch_size=1,  
                               per_device_eval_batch_size=1,   
                               gradient_accumulation_steps=8, 
                               gradient_checkpointing=True,
                               learning_rate=config.lr,
                               lr_scheduler_type="cosine",
                               logging_steps=5,
                               eval_steps=10,
                               eval_strategy="steps",
                               save_strategy="steps",
                               save_steps=40,
                               metric_for_best_model="eval_loss",
                               greater_is_better=False,
                               load_best_model_at_end=True,
                               fp16=True,
                               bf16 = False,                       
                               max_grad_norm=config.max_grad_norm,
                               warmup_ratio=config.warmup_ratio,
                               push_to_hub=False,
                               report_to="wandb",
                               gradient_checkpointing_kwargs={"use_reentrant": False},
                               dataset_kwargs={"skip_prepare_dataset": True}, 
                               deepspeed="configs/ds_config.json", 
                               max_seq_length=4096)  

    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=config
    )
    model, processor = load_gen_model_and_processor(config)
    model.config.use_cache = False

    #collects data from the dataset and prepares labels (predictors) for the model to 
    #compute loss over the assistant's response only

    def collate_fn(samples):
        """each example is a dictionary of system, user, labels and image inputs like
        [
        {'role': 'system', 'content': [...]},
        {'role': 'user',   'content': [...]},
        {'role': 'assistant','content': [...]}
            ]"""
        prompts = [
            processor.apply_chat_template(sample, tokenize=False) for sample in samples
        ]
        #process vision inputs (returns tuple, so get the image tensor)
        debug_target_size = 524 

        image_inputs = []
        for sample in samples:
            image = process_vision_info(sample)[0]
            if isinstance(image, list):
                if len(image) == 1:
                    image = image[0]
                else:
                    raise ValueError(
                        f"Expected a single image, got a list of length {len(image)}"
                    )
            #Resize image to model's expected input size
            if hasattr(image, "resize"):
                #Get the original dimensions
                width, height = image.size

                if width>debug_target_size or height>debug_target_size:
                    if width > height:
                        new_height = debug_target_size
                        new_width = int((width / height) * debug_target_size)
                    else:
                        new_width = debug_target_size
                        new_height = int((height / width) * debug_target_size)
                    #Use Image.LANCZOS instead of Image.ANTIALIAS
                    image = image.resize((new_width, new_height), Image.LANCZOS)
            else:
                raise TypeError(f"Unsupported image type: {type(image)}")
            image_inputs.append(image)
            
        batch = processor(
            text=prompts, images=image_inputs, return_tensors="pt", padding="max_length", max_length=4096, truncation=True
        )
        
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        #qwen-specific image tokens
        if isinstance(processor, Qwen2_5_VLProcessor):
            image_tokens = [151652, 151653, 151655]
        else:
            image_tokens = [
                processor.tokenizer.convert_tokens_to_ids(processor.image_token)
            ]
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100
        batch["labels"] = labels
        return batch

    train_dataset, eval_dataset, test_dataset = create_splits(config.json_path, config.image_dir, config.train, config.val, config.test)
    
    output_test_dir = os.path.join(config.test_dir, "test")
    os.makedirs(output_test_dir, exist_ok=True)

    test_data = list(test_dataset)

    test_file_path = os.path.join(output_test_dir, "test_data.json")
    with open(test_file_path, "w") as f:
        json.dump(test_data, f, indent=2)

    peft_config = LoraConfig(
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            r=config.lora_r,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,   
        peft_config=peft_config,
        processing_class=processor.tokenizer,
        callbacks=[ClearCudaCacheCallback]
    )

    trainer.train()
 
    model.save_pretrained(config.output_dir)
    processor.save_pretrained(config.output_dir)

if __name__ == "__main__":
    main()
