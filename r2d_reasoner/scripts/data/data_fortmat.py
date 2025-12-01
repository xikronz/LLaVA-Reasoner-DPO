import torch
import json 
import os 
import random 
from PIL import Image 
from torch.utils.data import Dataset, random_split
import numpy as np

SYSTEM_MESSAGE ="You are a Vision Language Model specialized in interpreting and analyzing visual information from image data. Given an image, provide a detailed explanation based on visual evidence present in the image. Reference specific, visible elements (e.g., signs, people, objects, colors, or positions) to support your reasoning and number your thoughts sequentially. Conclude with the final answer, clearly wrapped in the format: \n\n### Answer: {your answer here}"
RECTIFIED_INSTRUCT = "You are a Vision Language Model specialized in to help students by identifying and correcting mistakes in their solutions. Given a problem, the student's solution, the correct answer key, and the final correct answer, your task is to find the first mistake in the student's response, starting from the point where their solution deviates from the correct approach. Then, provide a step-by-step correction as if you were the student to go from the mistake to the correct answer while referencing the answer key. Make sure you output in this format: \n\n### Mistake: [the student’s first mistake here], \n\n Correction: Wait, actually [Why was the mistake wrong? How can the student correct their mistake and reach the correct answer?]"

# formats the data for initial fine tuning 
def format_data(sample, image_dir): 
    return [
        { 
            "role": "system", 
            "content": [{"type": "text", "text": SYSTEM_MESSAGE}]
        }, 
        {
            "role": "user", 
            "content": [{"type": "image", "image": image_dir + "/" + sample['image']}, 
                        {"type": "text", "text": sample['conversations'][0]['value']}]
        }, 
        {
            "role": "assistant", 
            "content": [{"type": "text", "text": sample['conversations'][1]['value']}]
        }
    ]

def format_incorrect_for_training(self, item: Dict, model_response: str) -> Dict:
        """
        Format an incorrect sample in conversation format for training.
        
        Args:
            item: The original sample item with question, ground_truth, image_path
            model_response: The FULL response from the model
            
        Returns:
            Dictionary with conversation format, image path, question, and correct answer
        """
        return {
            "image_path": item['image_path'],
            "question": item['question'],
            "correct_answer": item['ground_truth'],  # Full GPT response with reasoning
            "conversations": [
                {
                    "from": "system",
                    "value": SYSTEM_MESSAGE
                },
                {
                    "from": "human",
                    "value": item['question']
                },
                {
                    "from": "model",
                    "value": model_response  # Full model response
                },
                {
                    "from": "gpt",
                    "value": item['ground_truth']  # Full correct answer from GPT
                }
            ]
        }


def format_messages(image: Any, question: str) -> List[Dict]:
    """
    Creates the chat message format for Qwen3-VL.
    The processor's qwen-vl-utils supports paths, URLs, base64, and PIL.Image objects.
    
    Args:
        image: Image path (str) or PIL.Image object
        question: The question text
        
    Returns:
        List of message dictionaries for the chat template
    """
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_MESSAGE}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }
    ]
    return messages

    
def format_query(system_message, image, prompt):
    user_content_list = []

    if image is not None:
        if isinstance(image, list):
            for img_data in image:
                if isinstance(img_data, np.ndarray):
                    pil_image = Image.fromarray(img_data)
                elif isinstance(img_data, Image.Image): 
                    pil_image = img_data
                else:
                    raise TypeError(f"Unsupported image data type in list: {type(img_data)}")
                user_content_list.append({"image": pil_image})
        elif isinstance(image, np.ndarray): 
            pil_image = Image.fromarray(image)
            user_content_list.append({"image": pil_image})
        elif isinstance(image, Image.Image): 
            user_content_list.append({"image": image})
        else:
            raise TypeError(f"Unsupported image data type: {type(image)}")

    user_content_list.append({"text": prompt})

    formatted_query = [
        {"role": "system", "content": [{"text": system_message}]},
        {"role": "user", "content": user_content_list}
    ]
    return formatted_query


def construct_prompt(cr, r1, p, sol):
    instruct =  "Identify the first mistake in the student's solution by comparing it to the answer key. Then, starting from the point of error, provide a corrected version of the student's solution in the perspective of the student that leads to the correct final answer. Start your correction with 'Wait actually,' \n\n"
    prompt = "QUESTION: {"+ p + "}\n\nSTUDENT_ATTEMPT: {" + r1 + "}\n\nANSWER KEY: {" + cr + "}\n\nCORRECT ANSWER: {" + sol+"}"
    return instruct + prompt

# formats the data to create a rectified response 
def format_data_rectifying_r1(correct_rational, base_rational, final_answer, prompts, image_path):
    return[
        {
            "role": "system", 
            "content": [{"type": "text", "text": RECTIFIED_INSTRUCT}]
        }, 
        {
            "role": "user", 
            "content": [{"type": "image", "image": image_path},
                        {"type": "text", "text": construct_prompt(correct_rational, base_rational, prompts, final_answer)}]
        }
    ]

# custom dataset object from json file 
class JSONLDataset (Dataset):
    def __init__(self, json_file_path, image_folder_path):
        super().__init__()
        self.json_fp = json_file_path
        self.image_dir = image_folder_path
        self.dataset = self.load_file()

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        if index <0 or index >= len(self.dataset):
            raise IndexError(f"Index: {index} is out of bounds for {self.json_fp}")
        sample = self.dataset[index]
        #img = Image.open(sample['image_pth']).convert('RGB')
        return format_data(sample, self.image_dir)

    def load_file(self): 
        with open(self.json_fp, 'r') as f: 
            return json.load(f)
    

def create_splits (json_fp, image_dir, train, val, test):
    """ splits train%, val%, test% of the jason file path 
        into train, val, and test Dataset objects respectively
    """
    dataset = JSONLDataset(json_fp, image_dir)
    generator = torch.Generator().manual_seed(42)
    td, vd, testd = random_split(dataset, [train, val, test], generator=generator)
    return td, vd, testd

