import torch
import json
import gc
import os
import sys
import random
import regex as re
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForVision2Seq
from torch.utils.data import DataLoader, Dataset
from qwen_vl_utils import process_vision_info
from data_fortmat import format_messages, format_incorrect_for_training

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils import clear_memory, extract_answer, safe_contains, resize_image
from configs.config import EvalConfig, EvalDPOConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
SYSTEM_MESSAGE = "You are a Vision Language Model specialized in interpreting and analyzing visual information from image data. Given an image, provide a detailed explanation based on visual evidence present in the image. Reference specific, visible elements (e.g., signs, people, objects, colors, or positions) to support your reasoning and number your thoughts sequentially. Conclude with the final answer, clearly wrapped in the format: \n\n### Answer: {your answer here}"

class Share4oReasoningDataset(Dataset):    
    def __init__(self, samples: List[Dict], image_path: str, max_image_size: int = 1024):
        self.samples = samples
        self.image_path = image_path
        self.max_image_size = max_image_size
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        
        full_image_path = os.path.join(self.image_path, sample['image'])
        
        question = sample['conversations'][0]['value']
        gpt_response = sample['conversations'][1]['value']
        
        return {
            'id': sample.get('id', idx),
            'image_path': full_image_path, 
            'image_rel_path': sample['image'],  
            'question': question,
            'gpt_response': gpt_response,
            'sample': sample
        }


def collate_fn(batch: List[Dict], processor: AutoProcessor):
    all_messages = []
    metadata = []
    
    for item in batch:
        messages = format_messages(item['image_path'], item['question'])
        
        all_messages.append(messages)
        metadata.append({
            'id': item['id'],
            'gpt_response': item['gpt_response'],
            'question': item['question'],
            'image_path': item['image_path']
        })
    
    if not all_messages:
        return None
    
    processed_inputs = []
    for messages in all_messages:
        text_input = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        inputs = processor(
            text=[text_input],
            return_tensors="pt",
            padding=True
        )
        processed_inputs.append(inputs)
    
    if len(processed_inputs) == 1:
        return {
            'inputs': processed_inputs[0],
            'metadata': metadata
        }
    
    return {
        'inputs': processed_inputs,
        'metadata': metadata
    }


class ModelEvaluator:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.correct = []
        self.incorrect = []
        self.incorrect_conversations = []  
        self.errors = []
        self.model = None
        self.processor = None
        
        self.benchmark_stats = {}
        
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.inference_output_dir, exist_ok=True)
    
    def extract_benchmark(self, image_path: str):
        #aokvqa/33DPuC3HsYxY85pCTfcoxv.jpg" -> "aokvqa"
        path_parts = image_path.split('/')
        if len(path_parts) > 0:
            return path_parts[0]
        return "unknown"
    
    def update_benchmark_stats(self, benchmark: str, is_correct: bool):
        if benchmark not in self.benchmark_stats:
            self.benchmark_stats[benchmark] = {'correct': 0, 'total': 0}
        
        self.benchmark_stats[benchmark]['total'] += 1
        if is_correct:
            self.benchmark_stats[benchmark]['correct'] += 1
    
    def load_model(self):
        print(f"Loading model: {self.config.model_id}")
        
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            trust_remote_code=True
        )
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
            device_map="auto" if self.config.device == "cuda" else None,
            trust_remote_code=True
        )
        
        self.model.eval()
        print("BANGGGGG Model loaded successfully!")
    
    def evaluate_sample(self, item: Dict, idx: int):
        try:
            messages = format_messages(item['image_path'], item['question'])

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt", 
                return_dict=True
            ).to(self.model.device)

            generated_ids = self.model.generate(
                **inputs,
                )

            generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]

            model_response = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            gpt_answer = extract_answer(item['gpt_response'])
            
            model_answer = extract_answer(model_response)
            
            is_correct = (
                safe_contains(model_answer, gpt_answer) or 
                safe_contains(gpt_answer, model_answer)
            )
            
            benchmark = self.extract_benchmark(item.get('image_rel_path', item['image_path']))
            self.update_benchmark_stats(benchmark, is_correct)
            
            result = {
                'id': item['id'],
                'image_path': item['image_rel_path'],
                'benchmark': benchmark,
                'question': item['question'],
                'gpt_response': item['gpt_response'],
                'model_response': model_response,
                'gpt_answer': gpt_answer,
                'model_answer': model_answer,
                'correct': is_correct
            }
            
            if is_correct:
                self.correct.append(result)
            else:
                self.incorrect.append(result)
                self.incorrect_conversations.append(
                    format_incorrect_for_training(item, model_response)
                )
            
            return is_correct
            
        except RuntimeError as e:
            error_msg = str(e)
            if "CUDA out of memory" in error_msg:
                print(f"Sample {idx}: CUDA OOM, trying smaller resolution...")
                gc.collect()
                torch.cuda.empty_cache()
                
                try:
                    image = Image.open(item['image_path']).convert('RGB')
                    image = resize_image(image, 512)  
                    
                    messages = format_messages(image, item['question'])
                    
                    messages = format_messages(item['image_path'], item['question'])

                    inputs = self.processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt", 
                        return_dict=True
                    ).to(self.model.device)

                    generated_ids = self.model.generate(
                        **inputs,
                    )

                    generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]

                    model_response = self.processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True
                    )[0]
                    
                    gpt_answer = extract_answer(item['gpt_response'])
                    model_answer = extract_answer(model_response)
                    
                    is_correct = (
                        safe_contains(model_answer, gpt_answer) or 
                        safe_contains(gpt_answer, model_answer)
                    )
                    
                    benchmark = self.extract_benchmark(item.get('image_rel_path', item['image_path']))
                    self.update_benchmark_stats(benchmark, is_correct)
                    
                    result = {
                        'id': item['id'],
                        'image_path': item['image_rel_path'],
                        'benchmark': benchmark,
                        'question': item['question'],
                        'gpt_response': item['gpt_response'],
                        'model_response': model_response,
                        'gpt_answer': gpt_answer,
                        'model_answer': model_answer,
                        'correct': is_correct
                    }
                    
                    if is_correct:
                        self.correct.append(result)
                    else:
                        self.incorrect.append(result)
                        self.incorrect_conversations.append(
                            format_incorrect_for_training(item, model_response)
                        )
                    
                    return is_correct
                    
                except Exception as retry_error:
                    self.errors.append({
                        'sample_id': idx,
                        'error_type': 'CUDA OOM (retry failed)',
                        'message': str(retry_error)
                    })
            else:
                self.errors.append({
                    'sample_id': idx,
                    'error_type': 'RuntimeError',
                    'message': error_msg
                })
                
        except Exception as e:
            self.errors.append({
                'sample_id': idx,
                'error_type': type(e).__name__,
                'message': str(e)
            })
        
        return False
    
    def log_incorrect(self, prefix: str=""):
        path = os.path.join(self.config.output_dir, f"{prefix}_incorrect.json") 
        with open(path, 'w') as f:
            json.dump(self.incorrect, f, indent=2)
    
    def save_results(self, prefix: str = ""):
        timestamp = prefix if prefix else "eval"
        
        with open(os.path.join(self.config.output_dir, f"{timestamp}_correct.json"), 'w') as f:
            json.dump(self.correct, f, indent=2)
        
        with open(os.path.join(self.config.output_dir, f"{timestamp}_incorrect.json"), 'w') as f:
            json.dump(self.incorrect, f, indent=2)
        
        with open(os.path.join(self.config.output_dir, f"{timestamp}_errors.json"), 'w') as f:
            json.dump(self.errors, f, indent=2)
        
        benchmark_summary = {}
        for benchmark, stats in self.benchmark_stats.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            benchmark_summary[benchmark] = {
                'correct': stats['correct'],
                'total': stats['total'],
                'accuracy': accuracy
            }
        
        sorted_benchmarks = sorted(
            benchmark_summary.items(), 
            key=lambda x: x[1]['accuracy']
        )
        
        summary = {
            'total_samples': len(self.correct) + len(self.incorrect) + len(self.errors),
            'correct': len(self.correct),
            'incorrect': len(self.incorrect),
            'errors': len(self.errors),
            'accuracy': len(self.correct) / (len(self.correct) + len(self.incorrect)) if (len(self.correct) + len(self.incorrect)) > 0 else 0,
            'benchmark_stats': dict(sorted_benchmarks) 
        }
        
        with open(os.path.join(self.config.output_dir, f"{timestamp}_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.save_incorrect_conversations(prefix)
        
        print(f"Results saved to {self.config.output_dir}")
    
    def save_incorrect_conversations(self, prefix: str = ""):
        timestamp = prefix if prefix else "eval"
        output_path = os.path.join(
            self.config.inference_output_dir, 
            f"{timestamp}_incorrect_conversations.json"
        )
        
        with open(output_path, 'w') as f:
            json.dump(self.incorrect_conversations, f, indent=2)
        
        print(f"Incorrect conversations saved to {output_path} ({len(self.incorrect_conversations)} samples)")
    
    def print_stats(self):
        total = len(self.correct) + len(self.incorrect)
        if total > 0:
            accuracy = len(self.correct) / total * 100
            print(f"Correct: {len(self.correct)}, Incorrect: {len(self.incorrect)}, "
                  f"Errors: {len(self.errors)}, Accuracy: {accuracy:.2f}%")
    
    def print_benchmark_stats(self):
        if not self.benchmark_stats:
            print("No benchmark statistics available yet.")
            return
        
        print("\n" + "="*60)
        print("PER-BENCHMARK STATISTICS (sorted by accuracy, lowest first)")
        print("="*60)
        
        #sort by accuracy
        sorted_stats = sorted(
            self.benchmark_stats.items(),
            key=lambda x: x[1]['correct'] / x[1]['total'] if x[1]['total'] > 0 else 0
        )
        
        for benchmark, stats in sorted_stats:
            accuracy = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"  {benchmark:30s}: {stats['correct']:4d}/{stats['total']:4d} ({accuracy:5.2f}%)")


def main(start_idx: int = 0, end_idx: int = 10000, gpu_id: int = 0):
    #set GPU for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    config = EvalConfig()
    clear_memory()
    
    print(f"[GPU {gpu_id}] Loading dataset: {config.dataset_name}")
    ds = load_dataset(config.dataset_name)
    
    train_data = ds['train']
    print(f"[GPU {gpu_id}] Total samples in training set: {len(train_data)}")
    
    random.seed(config.seed)
    total_samples = len(train_data)
    sample_indices = random.sample(range(total_samples), min(config.num_samples, total_samples))
    
    print(f"[GPU {gpu_id}] Randomly sampled {len(sample_indices)} samples for evaluation")
    print(f"[GPU {gpu_id}] Processing range: {start_idx} to {end_idx}")
    
    sampled_data = [train_data[i] for i in sample_indices]
    
    eval_dataset = Share4oReasoningDataset(
        samples=sampled_data,
        image_path=config.image_path,
        max_image_size=config.max_image_size
    )
    
    evaluator = ModelEvaluator(config)
    evaluator.load_model()
    
    print(f"\n[GPU {gpu_id}] Starting evaluation on samples {start_idx} to {end_idx}...")
    
    for idx in tqdm(range(start_idx, min(end_idx, len(eval_dataset))), desc=f"GPU {gpu_id} Evaluating"):
        item = eval_dataset[idx]
        evaluator.evaluate_sample(item, idx)
        
        if (idx + 1) % 500 == 0:
            evaluator.print_stats()
            evaluator.log_incorrect(prefix=f"gpu{gpu_id}")
            clear_memory()
    
    evaluator.save_results(prefix=f"gpu{gpu_id}_final_{start_idx}_{end_idx}")
    
    print("\n" + "="*50)
    print(f"[GPU {gpu_id}] FINAL EVALUATION RESULTS")
    print("="*50)
    evaluator.print_stats()
    
    total = len(evaluator.correct) + len(evaluator.incorrect)
    if total > 0:
        print(f"\n[GPU {gpu_id}] Final Accuracy: {len(evaluator.correct) / total * 100:.2f}%")
    print(f"[GPU {gpu_id}] Total Errors: {len(evaluator.errors)}")
    
    evaluator.print_benchmark_stats()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate VLM on Share4oReasoning dataset")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=10000, help="End index")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    args = parser.parse_args()
    
    main(start_idx=args.start, end_idx=args.end, gpu_id=args.gpu)
