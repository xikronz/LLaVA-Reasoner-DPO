import torch
import gc
import time
import regex as re
from qwen_vl_utils import process_vision_info

def generate_text_from_sample(model, processor, sample, max_len, max_new_tokens=1024, device="cuda", resize=False):
    text_input = processor.apply_chat_template(
        sample[0:2], tokenize=False, add_generation_prompt=True
    )

    image_inputs, _ = process_vision_info(sample)
    
    if resize:
        processed_images = []
        for image in image_inputs:
            if isinstance(image, list) and len(image) == 1:
                image = image[0]
            if hasattr(image, "resize"):
                # Get the original dimensions
                width, height = image.size

                if width>max_len or height>max_len:
                    if width > height:
                        new_height = max_len
                        new_width = int((width / height) * max_len)
                    else:
                        new_width = max_len
                        new_height = int((height / width) * max_len)
                    # Use Image.LANCZOS instead of Image.ANTIALIAS
                    image = image.resize((new_width, new_height), Image.LANCZOS)
            else:
                raise TypeError(f"Unsupported image type: {type(image)}")
            processed_images.append(image)

        image_inputs = processed_images

    model_inputs = processor(
        text=[text_input],
        images=image_inputs,  
        return_tensors="pt",
    ).to(device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample = False)

    trimmed_generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]

    output_text = processor.batch_decode(
        trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]

def extract_correct_reasoning(sample):
    return sample[2]['content'][0]['text']

def extract_correct_answer(sample):
    match = re.search(r"\n\n### Answer:\s*(.*)", sample, re.DOTALL)
    if match:
        return match.group(1)
    
def extract_img (sample):
    return sample[1]['content'][0]['image']

def extract_img_raw (sample):
    return sample.get('image_pth', sample.get('image'))

def extract_prompt (sample):
    return sample[1]['content'][1]['text']

def clear_memory():
    # Delete variables if they exist in the current global scope
    if "inputs" in globals():
        del globals()["inputs"]
    if "model" in globals():
        del globals()["model"]
    if "processor" in globals():
        del globals()["processor"]
    if "trainer" in globals():
        del globals()["trainer"]
    if "peft_model" in globals():
        del globals()["peft_model"]
    if "bnb_config" in globals():
        del globals()["bnb_config"]
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

from PIL import Image
import torchvision.transforms as transforms

# Define a transformation to resize images to a fixed size.
# Adjust the (height, width) tuple as needed.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
