import os
import json
from datasets import load_dataset

# Define the base paths
base_dir = "/tmp/grpo_test/rl/grpo/data"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Create directories (equivalent to the 'mkdir -p' shell commands)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

print("Downloading GSM8K dataset...")

# --- Process Train Split ---
dataset_train = load_dataset('openai/gsm8k', 'main', split='train')
train_data = [{'question': item['question'], 'answer': item['answer']} for item in dataset_train]

# Save directly to final destination (equivalent to the 'mv' command)
train_path = os.path.join(train_dir, "dataset.json")
with open(train_path, 'w') as f:
    json.dump(train_data, f)
print(f"Train data saved to: {train_path}")

# --- Process Test Split ---
dataset_test = load_dataset('openai/gsm8k', 'main', split='test')
test_data = [{'question': item['question'], 'answer': item['answer']} for item in dataset_test]

# Save directly to final destination (equivalent to the 'mv' command)
test_path = os.path.join(test_dir, "dataset.json")
with open(test_path, 'w') as f:
    json.dump(test_data, f)
print(f"Test data saved to: {test_path}")

print('GSM8K dataset downloaded and organized successfully.')
