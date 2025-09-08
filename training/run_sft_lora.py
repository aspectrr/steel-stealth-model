# run_sft_lora.py (pseudocode)
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", torch_dtype="auto")
model = prepare_model_for_kbit_training(model)  # optional for QLoRA
lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj","v_proj"])
model = get_peft_model(model, lora_config)

dataset = load_dataset("json", data_files={"train":"sft_train.jsonl","validation":"sft_val.jsonl"})
# tokenize, collate...
training_args = TrainingArguments(output_dir="sft_lora_out", per_device_train_batch_size=1, num_train_epochs=3, fp16=True)
trainer = Trainer(model=model, args=training_args, train_dataset=dataset["train"], eval_dataset=dataset["validation"])
trainer.train()
model.save_pretrained("sft_lora_out")
