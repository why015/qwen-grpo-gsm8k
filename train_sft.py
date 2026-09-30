import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

BASE_MODEL = str(Path("models/Qwen2.5-0.5B-Instruct").resolve())
TRAIN_FILE = Path("deep_project/data/processed/train_clean.jsonl")
OUTPUT_DIR = "models/sft_adapter_v2"

rows = [
    json.loads(line)
    for line in TRAIN_FILE.read_text(encoding="utf-8").splitlines()
]
dataset = Dataset.from_list([
    {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "completion": [
            {"role": "assistant", "content": row["sft_response"]}
        ],
    }
    for row in rows
])
print(f"训练题数：{len(dataset)}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
)
model.config.use_cache = False

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.0,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

training_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    max_length=512,
    completion_only_loss=True,
    packing=False,
    gradient_checkpointing=True,
    fp16=True,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    args=training_config,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"SFT 适配器已保存：{OUTPUT_DIR}")