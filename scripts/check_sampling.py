import json
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from verifier import grade

ADAPTER = "models/sft_adapter"

rows = [
    json.loads(line)
    for line in Path("data/multistep/train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[:20]
]

tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
model = AutoPeftModelForCausalLM.from_pretrained(
    ADAPTER,
    dtype=torch.float16,
    device_map="cuda",
)
model.eval()

mixed_groups = 0

for index, row in enumerate(rows, start=1):
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": row["prompt"]}],
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            num_return_sequences=4,
            pad_token_id=tokenizer.eos_token_id,
        )

    rewards = []
    for sequence in generated:
        answer_tokens = sequence[inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(
            answer_tokens, skip_special_tokens=True
        ).strip()
        rewards.append(grade(response, row["answer"])["reward"])

    mixed_groups += len(set(rewards)) > 1
    print(f"{index:02d} {row['id']}: {rewards}")

print(f"有奖励差异的组：{mixed_groups}/20")