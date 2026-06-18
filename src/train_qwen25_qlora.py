import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATASET_FILE = "data/instruction_data/asx_llama_chat_dataset.jsonl"
OUTPUT_DIR = "models/qwen25_3b_asx_qlora"


def main():
    print("GPU available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Loading dataset...")

    dataset = load_dataset(
        "json",
        data_files=DATASET_FILE
    )["train"]

    split = dataset.train_test_split(
        test_size=0.1,
        seed=42
    )

    train_dataset = split["train"]
    eval_dataset = split["test"]

    print("Train:", len(train_dataset))
    print("Eval:", len(eval_dataset))

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    model.config.use_cache = False
    model.config.torch_dtype = torch.float16

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,

        fp16=False,
        bf16=False,

        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to="none",
        save_total_limit=1,

        optim="paged_adamw_8bit",
        max_grad_norm=0.0,
        gradient_checkpointing=True
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        formatting_func=formatting_func,
        args=training_args
    )

    print("Starting training...")

    trainer.train()

    print("Saving model...")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training completed!")
    print(f"Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()