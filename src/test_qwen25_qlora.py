import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "models/qwen25_3b_asx_qlora"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_PATH,
        trust_remote_code=True
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH
    )

    model.eval()

    return tokenizer, model


def generate_answer(tokenizer, model, task, input_text):
    messages = [
        {
            "role": "system",
            "content": "You are an Australian financial analyst. Give clear, concise investor-focused answers."
        },
        {
            "role": "user",
            "content": f"{task}\n\nASX report section:\n{input_text}"
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1
    )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    if "assistant" in response:
        return response.split("assistant")[-1].strip()

    return response


def main():
    print("Loading fine-tuned Qwen model...")
    tokenizer, model = load_model()

    task = "Identify the key business risks mentioned in the following ASX annual report section."

    input_text = """
The company continues to face inflationary pressure, higher interest rates,
supply chain disruptions, cyber security threats and increased competition
across key markets. Management also noted uncertainty in consumer demand
and potential regulatory changes that may affect future profitability.
"""

    print("\nTASK:")
    print(task)

    print("\nINPUT:")
    print(input_text)

    print("\nMODEL OUTPUT:")
    answer = generate_answer(tokenizer, model, task, input_text)
    print(answer)


if __name__ == "__main__":
    main()