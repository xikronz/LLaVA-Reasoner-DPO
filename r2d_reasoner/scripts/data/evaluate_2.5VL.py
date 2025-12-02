import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch 
import json
import gc
import random
import numpy as np
from PIL import Image
from datasets import load_dataset
from model.model_loader import load_gen_model_and_processor, load_mini_gen_model_and_processor, load_eval_model_and_tokenizer, load_eval_vl_model_and_processor, load_stf_gen_model_and_processor
from utils import generate_text_from_sample, extract_correct_reasoning, extract_prompt, extract_img, extract_correct_answer, clear_memory
from configs.config import TrainingConfig
from scripts.data.data_fortmat import format_data

class Evaluate_Model(): 
    def __init__(self, model, processor, max:int, model_name:str, cj: str, ij: str):
        self.correct = []
        self.incorrect = []
        self.errors = []
        self.correct_json = cj
        self.incorrect_json = ij
        self.model = model
        self.processor = processor
        self.max = max 
        self.model_name = model_name

    def process_sample(self, sample, i): 
        try: 
            model_response = generate_text_from_sample(self.model, self.processor, sample, self.max, resize=True)
            self.process_response(sample, model_response)
        except RuntimeError as e: 
            error_msg = str(e)
            if "probability tensor contains" in error_msg: 
                print(f"sample {i} triggered probability tensor error, skipping problem and logging error...")
                self.errors.append(self.create_error_log(i, "Invdalid token response distribution", error_msg))
            elif "CUDA out of memory" in error_msg: 
                print(f"Sample {i} triggered CUDA OOM, trying smaller resolution")
                gc.collect()
                torch.cuda.empty_cache()
                try: 
                    model_response = generate_text_from_sample(self.model, self.processor, sample, 524, resize=True)
                    self.process_response(sample, model_response)
                    print("Success")
                except Exception as retry_error: 
                    print("Resizing at 524 also failed. Moving on...")
                    self.errors.append(self.create_error_log(i, "CUDA OOM", error_msg))
            else: 
                print(f"Unexpected RuntimeError: {error_msg}")
                self.errors.append(self.create_error_log(i, "Unexpected RuntimeError", error_msg))
        except Exception as e: 
            print(f"Other Exception: {str(e)}")
            self.errors.append(self.create_error_log(i, e.__class__.__name__, str(e)))

    def process_response(self, sample, model_response):
        correct_rational = extract_correct_reasoning(sample)
        answer = extract_correct_answer(correct_rational)
        model_answer = extract_correct_answer(model_response)
        sample.append(self.create_new_entry(self.model_name, "text", model_response))

        if self.safe_contains(model_answer, answer) or self.safe_contains(answer, model_answer):
            self.correct.append(sample)
        else:
            self.correct.append(sample)

    def dump_info(self): 
        output_dir = "outputs/Qwen2.5-VL-r2d/Inference"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"{output_dir}/{self.incorrect_json}", 'w') as file:
            json.dump(self.incorrect, file, indent=4)
        with open(f"{output_dir}/{self.correct_json}", "w") as file: 
            json.dump(self.correct, file, indent=4)

    def create_error_log(self, idx, error_type, e_msg): 
        log = {"sample_id": idx, 
               "error_type": error_type, 
               "message": e_msg}
        return log  

    def safe_contains(self, text, substring): 
        """Safely check if substring is in text, handling None values."""
        if text is None or substring is None:
            return False
        return substring in text

    def create_new_entry(self, role, entry_type, entry): 
        s = {"role": role, 
             "content": [
                 {"type": entry_type}, 
                 {entry_type : entry}
             ]}
        
        return s
    
def main(): 
    clear_memory()
    config = TrainingConfig()

    print("Loading dataset: Share4oReasoning/sft_data")
    ds = load_dataset("Share4oReasoning/sft_data")
    train_data = ds['train']
    
    random.seed(42)
    num_samples = min(10000, len(train_data))
    sample_indices = random.sample(range(len(train_data)), num_samples)
    print(f"Randomly sampled {num_samples} samples for evaluation")

    sft_model, sft_processor = load_stf_gen_model_and_processor(config, sharding=True)
    base_model, base_processor = load_gen_model_and_processor(config, sharding=True)
    sft_model.eval()
    base_model.eval()

    sft_evaluator = Evaluate_Model(sft_model, sft_processor, 1024, "Qwen2.5-VL-3B-Sft", "r1_correct.json", "r1_incorrect.json")
    base_evaluator = Evaluate_Model(base_model, base_processor, 1024, "Qwen2.5-Vl-3B-Instruct", "base_correct.json", "base_incorrect.json")

    skipped = 0
    for i, idx in enumerate(sample_indices):
        try:
            raw_sample = train_data[idx]
            keys = raw_sample.keys()
            if 'image' not in keys or 'conversations' not in keys:
                print(raw_sample)
                print("unable to find image path from the sample taken from the hf dataset")
                continue 

            sample = format_data(raw_sample, config.image_dir)
            
            sft_evaluator.process_sample(sample, i)
            base_evaluator.process_sample(sample, i)
        except TypeError as t:
            error_msg = str(t)
            print(f"Sample {i} (idx {idx}) failed: {error_msg[:100]}...")
            print(raw_sample)
            skipped += 1
            # Clear CUDA state after errors
            gc.collect()
            torch.cuda.empty_cache()
            continue

        except Exception as e:
            error_msg = str(e)
            print(f"Sample {i} (idx {idx}) failed: {error_msg[:100]}...")
            skipped += 1
            # Clear CUDA state after errors
            gc.collect()
            torch.cuda.empty_cache()
            continue

        if (i % 20 == 0): 
            sft_evaluator.dump_info()
            base_evaluator.dump_info()

            print(f"completed {i}th test data sample {num_samples - i} remaining \nr1 correct: {len(sft_evaluator.correct)} \nbase correct: {len(base_evaluator.correct)} \nskipped: {skipped}")

    # Final dump
    sft_evaluator.dump_info()
    base_evaluator.dump_info()
    print(f"finished evaluating {num_samples} samples from Share4oReasoning/sft_data (skipped {skipped})")

if __name__ == "__main__":
    main()
