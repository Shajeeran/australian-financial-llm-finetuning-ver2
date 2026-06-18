import json
from pathlib import Path
from collections import Counter


INPUT_FILE = Path("data/instruction_data/asx_final_dataset.jsonl")
OUTPUT_FILE = Path("data/instruction_data/asx_llama_chat_dataset.jsonl")

SYSTEM_PROMPT = (
    "You are an Australian financial analyst. "
    "You analyse ASX annual report sections and provide clear, concise, investor-focused answers."
)


def build_user_prompt(row):
    return f"""
Task:
{row["instruction"]}

ASX annual report section:
{row["input"][:1800]}
""".strip()


def main():
    rows = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            rows.append({
                "company": row.get("company", ""),
                "task": row.get("task", ""),
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": build_user_prompt(row)
                    },
                    {
                        "role": "assistant",
                        "content": row["output"]
                    }
                ]
            })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Chat dataset created")
    print("=" * 50)
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Total rows: {len(rows)}")

    counts = Counter(row["task"] for row in rows)

    print("\nTask distribution:")
    for task, count in counts.items():
        print(f"{task}: {count}")


if __name__ == "__main__":
    main()