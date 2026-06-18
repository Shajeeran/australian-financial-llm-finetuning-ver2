import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "models/qwen25_3b_asx_qlora"


@st.cache_resource
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

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    return tokenizer, model


st.title("FinetuneASX v2: Qwen 2.5 QLoRA")
st.write("Fine-tuned Qwen 2.5 model for Australian financial report analysis.")

task = st.selectbox(
    "Choose task",
    [
        "Summarise the following ASX annual report section in simple business language.",
        "Identify the key business risks mentioned in the following ASX annual report section.",
        "Explain the financial information in the following annual report section for a non-technical investor.",
        "Extract the main strategic insights from the following ASX annual report section."
    ]
)

input_text = st.text_area("Paste ASX report text here", height=250)

if st.button("Generate Answer"):
    if not input_text.strip():
        st.warning("Please paste text first.")
    else:
        tokenizer, model = load_model()

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

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "assistant" in response:
            answer = response.split("assistant")[-1].strip()
        else:
            answer = response

        st.subheader("Generated Answer")
        st.write(answer)