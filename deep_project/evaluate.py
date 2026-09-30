import argparse
import json
import os
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from verifier import grade

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data",
    default=str(ROOT / "data/processed/validation.jsonl"),
)
parser.add_argument(
    "--model",
    default=str(PROJECT / "models/Qwen2.5-0.5B-Instruct"),
)
parser.add_argument("--output", required=True)
parser.add_argument("--limit", type=int, default=None)
args = parser.parse_args()

with Path(args.data).open(encoding="utf-8") as file:
    questions = [json.loads(line) for line in file if line.strip()]
if args.limit is not None:
    questions = questions[:args.limit]
if not questions:
    raise ValueError("评测数据为空")

tokenizer = AutoTokenizer.from_pretrained(args.model)


def load_model(model_path: str):
    """支持完整模型和 LoRA adapter 两种路径。"""
    adapter_cfg = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        with open(adapter_cfg, encoding="utf-8") as f:
            cfg = json.load(f)
        base_path = cfg["base_model_name_or_path"]
        if not os.path.exists(base_path):
            local_base = str(PROJECT / "models" / "Qwen2.5-0.5B-Instruct")
            if os.path.exists(local_base):
                base_path = local_base
            else:
                raise FileNotFoundError(f"找不到 base model: {base_path}")
        print(f"[LoRA] adapter = {model_path}")
        print(f"[LoRA] base    = {base_path}")
        base = AutoModelForCausalLM.from_pretrained(
            base_path, dtype=torch.float16, device_map="cuda",
        )
        m = PeftModel.from_pretrained(base, model_path)
        m = m.merge_and_unload()
        return m
    print(f"[Full] {model_path}")
    return AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map="cuda",
    )


model = load_model(args.model)
model.eval()

output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)

correct = 0
parsed = 0
total_tokens = 0
hit_limit = 0
start = time.perf_counter()

with output_path.open("w", encoding="utf-8") as file:
    for index, question in enumerate(questions, start=1):
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": question["prompt"]}],
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        answer_tokens = generated[0, inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(
            answer_tokens, skip_special_tokens=True
        ).strip()
        result = grade(response, question["answer"])

        correct += result["reward"]
        parsed += result["parsed"] is not None
        total_tokens += len(answer_tokens)
        hit_limit += len(answer_tokens) == 1024

        record = {
            "id": question["id"],
            "expected": question["answer"],
            "response": response,
            "generated_tokens": len(answer_tokens),
            **result,
        }
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

        if index % 10 == 0 or index == len(questions):
            print(f"已评测 {index}/{len(questions)}")

elapsed = time.perf_counter() - start
print(f"正确率：{correct}/{len(questions)}")
print(f"解析成功率：{parsed}/{len(questions)}")
print(f"平均输出 token：{total_tokens / len(questions):.1f}")
print(f"达到 1024 token 上限：{hit_limit}/{len(questions)}")
print(f"总耗时：{elapsed:.1f} 秒")
print(f"逐题结果：{output_path}")