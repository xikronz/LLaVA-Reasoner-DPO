from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from trl import DPOTrainer, DPOConfig
import torch 
import json
import gc
from data.dataset import JSONLDataset, format_data_rectifying_r1
from model.model_loader import load_stf_gen_model_and_processor
from configs.config import TrainingConfig

def main():
    config = TrainingConfig()
    model, processor = load_stf_gen_model_and_processor(config, sharding=True)

    # {
    #   "prompt": "<instruction text>",
    #   "chosen": "<preferred model output>",
    #   "rejected": "<less preferred output>"
    # }
    raw = load_dataset("data/outputs/preference_data.json")

    def tokenize_pref(sample):
        chosen = processor(sample["prompt"], sample["chosen"], truncation=True, max_length=1024)
        rejected = processor(sample["prompt"], sample["rejected"], truncation=True, max_length=1024)
        return {
            "input_ids_chosen": chosen.input_ids,
            "attention_mask_chosen": chosen.attention_mask,
            "input_ids_rejected": rejected.input_ids,
            "attention_mask_rejected": rejected.attention_mask,
        }

    ds = raw["train"].map(tokenize_pref, remove_columns=raw["train"].column_names)

    dpo_config = DPOConfig(
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        weight_decay=0.01,
        lr_scheduler_type="cosine",  
        warmup_ratio=0.1,
        beta=0.1,                   
        logging_steps=50,
        save_steps=200,
    )

    trainer = DPOTrainer(
        model=model,
        processor=processor,
        args=TrainingArguments(
            output_dir="./qwen-dpo-finetuned",
            **dpo_config.to_dict(),
        ),
        train_dataset=ds,
    )

    trainer.train()

    trainer.save_model("./qwen-dpo-finetuned-final")


if __name__ == "__main__":
    main()
