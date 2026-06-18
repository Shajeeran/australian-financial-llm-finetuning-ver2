import json
from pathlib import Path


DATASET_FILE = Path("data/instruction_data/asx_llama_chat_dataset.jsonl")


with open(DATASET_FILE, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

print("Total rows:", len(rows))

sample = rows[0]

print("\nCompany:", sample["company"])
print("Task:", sample["task"])

print("\nSYSTEM:")
print(sample["messages"][0]["content"])

print("\nUSER:")
print(sample["messages"][1]["content"][:1000])

print("\nASSISTANT:")
print(sample["messages"][2]["content"])