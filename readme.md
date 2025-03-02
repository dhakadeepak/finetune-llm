# Fine-tuning Phi-4-Mini-Instruct with LoRA on Resume Question Answers.

## Overview
This script fine-tunes the `microsoft/Phi-4-mini-instruct` model using LoRA (Low-Rank Adaptation) on a custom QA dataset (`cleaned_qa.json`). The model is quantized to 8-bit for efficient training.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place the QA dataset (`cleaned_qa.json`) in the working directory.

## Training Steps
- Loads the Phi-4-mini-instruct model with 8-bit quantization.
- Formats the dataset to match the model's prompt style.
- Applies LoRA for parameter-efficient fine-tuning.
- Trains the model for 30 epochs with a batch size of 2.
- Saves the fine-tuned model and tokenizer to `./phi4_mini_finetuned`.

## Running the Script
Execute the script to start training:
```bash
python train.py
```

## Output
- Fine-tuned model and tokenizer saved in `./phi4_mini_finetuned`.
- Training logs and checkpoints stored in the output directory.

## Notes
- Adjust hyperparameters (`learning_rate`, `num_train_epochs`, etc.) in `TrainingArguments` as needed.
- Ensure the dataset follows the expected JSON format with `question` and `answer` fields.