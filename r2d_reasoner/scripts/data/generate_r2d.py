from datasets import load_dataset
from PIL import Image
ds = load_dataset("Share4oReasoning/sft_data")
image_path = "/share/cuvl/cc2864/LLaVA-Reasoner-DPO/r2d_reasoner/data/images"
from IPython.display import display


print(ds['train'][0].keys())

sample = ds['train'][0]

print(sample.keys())

image_sample = f"{image_path}/{sample['image']}"

image = Image.open(image_sample).convert("RGB")  # convert to RGB if needed
display(image)
print(image_sample)

