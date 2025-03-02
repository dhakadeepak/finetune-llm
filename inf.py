from transformers import AutoTokenizer, AutoModelForCausalLM


model_path = "./phi4_mini_finetuned"

# Load tokenizer with fixed special tokens
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # Ensure pad_token is correctly set
tokenizer.sep_token = "<|end|>"

# Load model
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")

def generate_answer(question):
    formatted_prompt = f"""<|system|>\nYou are an AI assistant answering questions about Deepak Dhaka.\n<|end|>\n
    <|user|>\n{question}\n<|end|>\n
    <|assistant|>\n"""

    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to("cuda")

    # Proper attention mask handling
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=200,
        temperature=0.5,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Test inference
print(generate_answer("is deepak gay?"))
