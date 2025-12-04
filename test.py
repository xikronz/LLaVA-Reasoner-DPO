from datasets import load_dataset

ds = load_dataset("Share4oReasoning/sft_data")

print(ds['train'][0])
print(ds['train'][0]['conversations'][0]['value'])