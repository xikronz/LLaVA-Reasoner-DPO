import torch 
import json
import gc
from model.model_loader import load_gen_model_and_processor, load_mini_gen_model_and_processor, load_eval_model_and_tokenizer, load_eval_vl_model_and_processor, load_stf_gen_model_and_processor
from utils import generate_text_from_sample, extract_correct_reasoning, extract_prompt, extract_img, extract_correct_answer, clear_memory
from configs.config import TrainingConfig
import numpy as np
from PIL import Image
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
                self.errors.append(self.correct_error_log(i, "Unexpected RuntimeError", error_msg))
        except Exception as e: 
            print(f"Other Exception: {str(e)}")
            self.errors.append(self.create_error_log(i, e.__class__.__name__, error_msg))

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
        with open(f"output/inference/{self.incorrect_json}", 'w') as file:
            json.dump(self.incorrect, file, indent=4)
        with open(f"output/inference/{self.correct_json}", "w") as file: 
            json.dump(self.correct, file, indent=4)

    def create_error_log(idx, error_type, e_msg): 
        log = {"sample_id": idx, 
               "error_type": error_type, 
               "message": e_msg}
        return log  

    def safe_contains (text, substring): 
        """Safely check if substring is in text, handling None values."""
        if text is None or substring is None:
            return False
        return substring in text

    def create_new_entry (role, entry_type, entry): 
        s = {"role": role, 
             "content": [
                 {"type": entry_type}, 
                 {entry_type : entry}
             ]}
        
        return s
    
def main(): 
    clear_memory()
    config = TrainingConfig()

    with open("output/test/test_data2.json", 'r') as f:
        file = json.load(f)

    sft_model, sft_processor = load_stf_gen_model_and_processor(config, sharding=True)
    base_model, base_processor = load_gen_model_and_processor(config, sharding=True)
    sft_model.eval()
    base_model.eval()

    sft_evaluator = Evaluate_Model(sft_model, sft_processor, 1024, "Qwen2.5-VL-3B-Sft", "r1_correct.json", "r1_incorrect.json")
    base_evaluator = Evaluate_Model(base_model, base_processor, 1024, "Qwen2.5-Vl-3B-Instruct", "base_correct.json", "base_incorrect.json")

    for i in range(len(file)):
        sample = file[i]
        sft_evaluator.process_sample(sample, i)
        base_evaluator.process_sample(sample, i)

        if (i%20 ==0): 
            sft_evaluator.dump_info()
            base_evaluator.dump_info()

            print(f"completed {i}th test data sample {len(file)-i} remaining \nr1 correct: {len(sft_evaluator.correct)} \nbase correct: {len(base_evaluator.correct)} \n")

    print(f"finished evaluating test_data2")

if __name__ == "__main__":
    main()
