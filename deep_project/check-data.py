import json
from pathlib import Path

from transformers import AutoTokenizer

from verifier import grade

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent / "models/Qwen2.5-0.5B-Instruct"

def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]

train = read_jsonl(ROOT / "data/processed/train.jsonl")
validation = read_jsonl(ROOT / "data/processed/validation.jsonl")

assert len(train) == 6973
assert len(validation) == 500
assert not ({r["id"] for r in train} & {r["id"] for r in validation})

tokenizer = AutoTokenizer.from_pretrained(str(MODEL))
lengths = []
bad_answers = []

for row in train + validation:
    result = grade(row["sft_response"], row["answer"])
    if result["reward"] != 1:
        bad_answers.append((row["id"], result))

    messages = [
        {"role": "user", "content": row["prompt"]},
        {"role": "assistant", "content": row["sft_response"]},
    ]
    encoded = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    return_dict=True,
    return_tensors="pt",
    )
    lengths.append(encoded["input_ids"].shape[-1])

lengths.sort()
print(f"训练题：{len(train)}，验证题：{len(validation)}")
print(f"标准回答判分失败：{len(bad_answers)}")
print(f"token 长度中位数：{lengths[len(lengths) // 2]}")
print(f"token 长度 P95：{lengths[int(len(lengths) * 0.95)]}")
print(f"token 长度 P99：{lengths[int(len(lengths) * 0.99)]}")
print(f"最长：{lengths[-1]}")
print(f"超过 512 token：{sum(n > 512 for n in lengths)}")
if bad_answers:
    print("失败样例：", bad_answers[:5])
    raise SystemExit(1)

print("数据检查通过")