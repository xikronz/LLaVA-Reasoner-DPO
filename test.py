from datasets import load_dataset

ds = load_dataset("Share4oReasoning/dpo_data")

print(ds['train'][0])
print(ds['train'][0].keys())