from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from transformers import Trainer, DataCollatorForSeq2Seq


model_name = "microsoft/Phi-4-mini-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure special tokens are set properly.
tokenizer.pad_token = tokenizer.eos_token  # Avoids PAD/EOS confusion
tokenizer.sep_token = "<|end|>"  # End of message separator

 
# Configure 8-bit quantization.
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  
    bnb_8bit_compute_dtype="float16",  # Reduce precision for faster inference
    bnb_8bit_use_double_quant=True  # Enable double quantization
)


# Load the model.
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-4-mini-instruct",
    quantization_config=bnb_config,
    device_map="auto"
)

# LOAD DATASET
dataset = load_dataset("json", data_files="cleaned_qa.json")

# prompt format for Phi-4-mini-instruct.
def format_dataset(example):
    
    formatted_prompt = f"""<|system|>\nYou are an AI assistant answering questions about Deepak Dhaka.\n<|end|>\n
    <|user|>\n{example["question"]}\n<|end|>\n
    <|assistant|>\n{example["answer"]}\n<|end|>"""

    inputs = tokenizer(
        formatted_prompt,
        truncation=True,
        padding="max_length",
        max_length=256,  
        return_tensors="pt"
    )

    labels = inputs["input_ids"].clone()  # Ensure labels match input

    return {
        "input_ids": inputs["input_ids"].squeeze(),
        "attention_mask": inputs["attention_mask"].squeeze(),
        "labels": labels.squeeze()
    }

formatted_dataset = dataset.map(format_dataset, remove_columns=["question", "answer"])


### LORA Configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

### Apply LoRA
model = get_peft_model(model, lora_config)

# set Hypterparameters
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    num_train_epochs=30,
    save_steps=500,
    save_total_limit=1,
    learning_rate=2e-4,
    output_dir="./phi4_mini_finetuned",
    optim="adamw_bnb_8bit",  # Optimizer compatible with 8-bit precision
    fp16=True                # Enables 16-bit precision training for efficiency
)

data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=formatted_dataset["train"],
    data_collator=data_collator
)

# start training
trainer.train()


# Save model and tokenizer
trainer.save_model("./phi4_mini_finetuned")  
tokenizer.save_pretrained("./phi4_mini_finetuned")