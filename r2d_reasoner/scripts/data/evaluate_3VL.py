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
from data_fortmat import format_messages, format_incorrect_for_training

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils import clear_memory
from configs.config import EvalConfig, EvalDPOConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
SYSTEM_MESSAGE = "You are a Vision Language Model specialized in interpreting and analyzing visual information from image data. Given an image, provide a detailed explanation based on visual evidence present in the image. Reference specific, visible elements (e.g., signs, people, objects, colors, or positions) to support your reasoning and number your thoughts sequentially. Conclude with the final answer, clearly wrapped in the format: \n\n### Answer: {your answer here}"

def extract_answer(text: str) -> Optional[str]:
    if text is None:
        return None
    
    #matching "### Answer: X" pattern
    match = re.search(r"\n\n###\s*Answer:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        return answer
    
    letter_match = re.search(r'\b([A-D])\.\s', text)
    if letter_match:
        return letter_match.group(1)
    
    return None

def safe_contains(text: str, substring: str) -> bool:
    if text is None or substring is None:
        return False
    return substring.lower() in text.lower()


def resize_image(image: Image.Image, max_size: int) -> Image.Image:
    width, height = image.size
    
    if width <= max_size and height <= max_size:
        return image
    
    if width > height:
        new_width = max_size
        new_height = int((height / width) * max_size)
    else:
        new_height = max_size
        new_width = int((width / height) * max_size)
    
    return image.resize((new_width, new_height), Image.LANCZOS)


class Share4oReasoningDataset(Dataset):    
    def __init__(self, samples: List[Dict], image_path: str, max_image_size: int = 1024):
        self.samples = samples
        self.image_path = image_path
        self.max_image_size = max_image_size
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        full_image_path = os.path.join(self.image_path, sample['image'])
        
        question = sample['question']
        ground_truth = sample['chosen']
        answer = sample['answer']
        
        return {
            'id': sample.get('id', idx),
            'image_path': sample['image'],
            'question': question,
            'ground_truth': ground_truth,
            'answer': answer, 
            'sample': sample
        }


def collate_fn(batch: List[Dict], processor: AutoProcessor) -> Dict[str, Any]:
    """
    Collate function for DataLoader that processes samples for Qwen3-VL.
    The processor automatically handles image loading from paths via qwen-vl-utils.
    
    Args:
        batch: List of sample dictionaries from the dataset
        processor: The Qwen3-VL processor
        
    Returns:
        Dictionary containing processed inputs and metadata
    """
    all_messages = []
    metadata = []
    
    for item in batch:
        messages = format_messages(item['image_path'], item['question'])
        
        all_messages.append(messages)
        metadata.append({
            'id': item['id'],
            'ground_truth': item['ground_truth'],
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
    
    def extract_benchmark(self, image_path: str) -> str:
        #aokvqa/33DPuC3HsYxY85pCTfcoxv.jpg" -> "aokvqa"
        parts = image_path.split('/')
        if len(parts) > 0:
            return parts[0]
        return "unknown"
    
    def update_benchmark_stats(self, benchmark: str, is_correct: bool):
        """Update statistics for a specific benchmark."""
        if benchmark not in self.benchmark_stats:
            self.benchmark_stats[benchmark] = {'correct': 0, 'total': 0}
        
        self.benchmark_stats[benchmark]['total'] += 1
        if is_correct:
            self.benchmark_stats[benchmark]['correct'] += 1
    
    def load_model(self):
        """Load the model and processor."""
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
    
    def generate_response(self, inputs: Dict) -> str:
        """Generate response from the model."""
        inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v 
                  for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,  #greedy decoding for evaluation
                pad_token_id=self.processor.tokenizer.pad_token_id if hasattr(self.processor, 'tokenizer') else None
            )
        
        input_len = inputs['input_ids'].shape[-1]
        generated_ids = outputs[0][input_len:]
        response = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        return response
    
    def evaluate_sample(self, item: Dict, idx: int) -> bool:
        """
        Evaluate a single sample.
        
        Returns:
            True if answer is correct, False otherwise
        """
        try:
            messages = format_messages(item['image_path'], item['question'])
            
            text_input = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            
            inputs = self.processor(
                text=[text_input],
                return_tensors="pt"
            )
            
            model_response = self.generate_response(inputs)
            ground_truth_answer = item['answer']
            
            model_answer = extract_answer(model_response)
            
            is_correct = (
                safe_contains(model_answer, ground_truth_answer) or 
                safe_contains(ground_truth_answer, model_answer)
            )
            
            #extract benchmark from image path
            benchmark = self.extract_benchmark(item['image_path'])
            self.update_benchmark_stats(benchmark, is_correct)
            
            result = {
                'id': item['id'],
                'image_path': item['image_path'],
                'benchmark': benchmark,
                'question': item['question'],
                'ground_truth': item['ground_truth'],
                'model_response': model_response,
                'ground_truth_answer': ground_truth_answer,
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
                    
                    text_input = self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False
                    )
                    
                    inputs = self.processor(
                        text=[text_input],
                        return_tensors="pt"
                    )
                    
                    model_response = self.generate_response(inputs)
                    ground_truth_answer = item['answer']
                    model_answer = extract_answer(model_response)
                    
                    is_correct = (
                        safe_contains(model_answer, ground_truth_answer) or 
                        safe_contains(ground_truth_answer, model_answer)
                    )
                    
                    benchmark = self.extract_benchmark(item['image_path'])
                    self.update_benchmark_stats(benchmark, is_correct)
                    
                    result = {
                        'id': item['id'],
                        'image_path': item['image_path'],
                        'benchmark': benchmark,
                        'question': item['question'],
                        'ground_truth': item['ground_truth'],
                        'model_response': model_response,
                        'ground_truth_answer': ground_truth_answer,
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
    
    def save_results(self, prefix: str = ""):
        """Save evaluation results to JSON files."""
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
        """Print current evaluation statistics."""
        total = len(self.correct) + len(self.incorrect)
        if total > 0:
            accuracy = len(self.correct) / total * 100
            print(f"Correct: {len(self.correct)}, Incorrect: {len(self.incorrect)}, "
                  f"Errors: {len(self.errors)}, Accuracy: {accuracy:.2f}%")
    
    def print_benchmark_stats(self):
        """Print per-benchmark statistics, sorted by accuracy (lowest first)."""
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
    
    config = EvalDPOConfig()
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
            evaluator.save_results(prefix=f"gpu{gpu_id}_checkpoint_{idx+1}")
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
