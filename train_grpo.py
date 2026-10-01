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
REWARD_MODE = os.environ.get("REWARD_MODE", "lenient")
GRPO_SEED = int(os.environ.get("GRPO_SEED", "42"))

# 显式控制 GRPO / DAPO / Dr.GRPO 关键参数
LOSS_TYPE = os.environ.get("LOSS_TYPE", "dapo")              # grpo | dapo | dr_grpo
SCALE_REWARDS = os.environ.get("SCALE_REWARDS", "group")      # group | none
EPSILON_HIGH_STR = os.environ.get("EPSILON_HIGH", "")
EPSILON_HIGH = float(EPSILON_HIGH_STR) if EPSILON_HIGH_STR else None

# 训练超参（提前定义，用于生成 run_id）
BETA = 0.04
NUM_GEN = 8
MAX_STEPS = 1000
LEARNING_RATE = 5e-6

# === 生成唯一 run_id ===
from deep_project.run_manifest import make_run_id, init_manifest, finalize_manifest

RUN_ID = make_run_id(
    algorithm="grpo",
    loss_type=LOSS_TYPE,
    reward=REWARD_VERSION,
    verifier=REWARD_MODE,
    beta=BETA,
    num_gen=NUM_GEN,
    seed=GRPO_SEED,
)
OUTPUT_DIR = f"runs/{RUN_ID}"

# audit 文件默认进 run 目录（除非显式覆盖）
AUDIT_PATH = os.environ.get("AUDIT_PATH", f"{OUTPUT_DIR}/audit_groups.jsonl")
set_audit_path(AUDIT_PATH)
reward_func = REWARD_FUNCS[REWARD_VERSION]

print(f"[run] run_id     = {RUN_ID}")
print(f"[run] output_dir = {OUTPUT_DIR}")
print(f"[reward] using {REWARD_VERSION} / {REWARD_MODE}, audit -> {AUDIT_PATH}")
print(f"[config] loss_type={LOSS_TYPE}, scale_rewards={SCALE_REWARDS}, epsilon_high={EPSILON_HIGH}")


SFT_ADAPTER = "models/sft_adapter_v2"

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
    max_steps=MAX_STEPS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_generations=NUM_GEN,
    max_completion_length=256,
    temperature=0.8,
    top_p=0.95,
    learning_rate=LEARNING_RATE,
    beta=BETA,
    # === GRPO / DAPO / Dr.GRPO 关键参数（显式指定） ===
    loss_type=LOSS_TYPE,
    scale_rewards=SCALE_REWARDS,
    epsilon=0.2,
    epsilon_high=EPSILON_HIGH,
    mask_truncated_completions=False,
    importance_sampling_level="token",
    # ================================================
    use_vllm=False,
    gradient_checkpointing=True,
    fp16=True,
    logging_steps=5,
    save_strategy="no",
    report_to="none",
    seed=int(__import__("os").environ.get("GRPO_SEED", "42")),
)

print(f"[config] loss_type={LOSS_TYPE}, scale_rewards={SCALE_REWARDS}, "
      f"epsilon=0.2, epsilon_high={EPSILON_HIGH}, "
      f"mask_truncated={config.mask_truncated_completions}")

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

# === 训练前：init manifest (status=running) ===
init_manifest(
    run_dir=OUTPUT_DIR,
    config=config,
    extra={
        "seed": GRPO_SEED,
        "reward_version": REWARD_VERSION,
        "reward_mode": REWARD_MODE,
        "model": "Qwen2.5-0.5B-Instruct (LoRA r=8 alpha=16)",
        "sft_adapter": SFT_ADAPTER,
    },
    notes=f"loss_type={LOSS_TYPE}, scale_rewards={SCALE_REWARDS}, "
          f"epsilon_high={EPSILON_HIGH}, beta={BETA}",
)

try:
    trainer.train()
    flush_audit()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"GRPO 适配器已保存：{OUTPUT_DIR}")

    # 从 trainer state 里提取训练时长
    train_runtime = None
    for log in reversed(trainer.state.log_history):
        if "train_runtime" in log:
            train_runtime = log["train_runtime"]
            break

    finalize_manifest(
        run_dir=OUTPUT_DIR,
        metrics={"train_runtime_sec": train_runtime},
        status="completed",
    )
except Exception as e:
    finalize_manifest(
        run_dir=OUTPUT_DIR,
        status="failed",
        error=f"{type(e).__name__}: {e}",
    )
    raise