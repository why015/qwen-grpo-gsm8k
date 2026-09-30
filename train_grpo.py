import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from verifier import grade

import os
from deep_project.rewards import REWARD_FUNCS, set_audit_path, flush_audit

REWARD_VERSION = os.environ.get("REWARD_VERSION", "r1")
AUDIT_PATH = os.environ.get(
    "AUDIT_PATH",
    f"deep_project/outputs/audit_{REWARD_VERSION}_groups.jsonl",
)
set_audit_path(AUDIT_PATH)
reward_func = REWARD_FUNCS[REWARD_VERSION]
print(f"[reward] using {REWARD_VERSION}, audit -> {AUDIT_PATH}")


SFT_ADAPTER = "models/sft_adapter_v2"
OUTPUT_DIR = f"models/grpo_seed_{os.environ.get('GRPO_SEED', '42')}"

rows = [
    json.loads(line)
    for line in Path("deep_project/data/processed/train_clean.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
]

# 标准答案单独保留给奖励函数；模型生成时只看到 prompt。
dataset = Dataset.from_list([
    {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "answer": row["answer"],
    }
    for row in rows
])
print(f"GRPO 训练题数：{len(dataset)}")

tokenizer = AutoTokenizer.from_pretrained(SFT_ADAPTER)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoPeftModelForCausalLM.from_pretrained(
    SFT_ADAPTER,
    is_trainable=True,
    dtype=torch.float16,
)
model.config.use_cache = False

def final_answer_reward(completions, answer, **kwargs):
    rewards = []
    for completion, expected in zip(completions, answer):
        response = completion[0]["content"]
        rewards.append(float(grade(response, str(expected))["reward"]))
    return rewards

config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    max_steps=1000,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_generations=8,
    max_completion_length=256,
    temperature=0.8,
    top_p=0.95,
    learning_rate=5e-6,
    beta=0.04,
    use_vllm=False,
    gradient_checkpointing=True,
    fp16=True,
    logging_steps=5,
    save_strategy="no",
    report_to="none",
    seed=int(__import__("os").environ.get("GRPO_SEED", "42")),
)

from transformers import TrainerCallback


class StopAtStepCallback(TrainerCallback):
    """在指定 step 提前停止训练，保持 max_steps/scheduler 不变。"""
    def __init__(self, stop_step=200):
        self.stop_step = stop_step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_step:
            print(f"[callback] 到达 step {self.stop_step}，提前停止")
            control.should_training_stop = True
        return control


# 可选：通过环境变量控制提前停止步数
STOP_AT_STEP = os.environ.get("STOP_AT_STEP")
if STOP_AT_STEP:
    callbacks = [StopAtStepCallback(int(STOP_AT_STEP))]
    print(f"[callback] 将在 step {STOP_AT_STEP} 停止")
else:
    callbacks = []

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_func,
    callbacks=callbacks,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()
flush_audit()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"GRPO 适配器已保存：{OUTPUT_DIR}")