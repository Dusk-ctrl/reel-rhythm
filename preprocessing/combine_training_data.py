import json
import random

# Training files
training_files = [
    "movie_train.jsonl",
    "music_train.jsonl",
    "combined_train.jsonl"
]

all_examples = []

# Load all training examples
for filename in training_files:
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            all_examples.append(json.loads(line))

# Shuffle the examples so the model sees mixed domains
random.seed(42)
random.shuffle(all_examples)

# Save the combined training dataset
with open("train_all.jsonl", "w", encoding="utf-8") as file:
    for example in all_examples:
        file.write(json.dumps(example, ensure_ascii=False) + "\n")

print("Combined training dataset created!")
print("Total examples:", len(all_examples))
print("Output: train_all.jsonl")
