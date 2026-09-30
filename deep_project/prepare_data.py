import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_TRAIN = ROOT / "data/raw/gsm8k_train.jsonl"
OUTPUT = ROOT / "data/processed"

rows = [
    json.loads(line)
    for line in RAW_TRAIN.read_text(encoding="utf-8").splitlines()
]
assert len(rows) == 7473, f"训练文件行数异常：{len(rows)}"

processed = []
for original_id, row in enumerate(rows):
    question = row["question"].strip()
    full_answer = row["answer"].strip()

    match = re.search(
        r"####\s*([+-]?\d[\d,]*(?:\.\d+)?)\s*$",
        full_answer,
    )
    if match is None:
        raise ValueError(f"题目 {original_id} 的标准答案无法解析")

    number = match.group(1).replace(",", "")
    reasoning = full_answer.rsplit("####", 1)[0].strip()
    reasoning = re.sub(r"<<.*?>>", "", reasoning).strip()

    processed.append({
        "id": f"gsm8k-train-{original_id}",
        "prompt": (
            "请解答下面的数学应用题，给出简短计算过程。"
        "最后一行以“答案：”开头，冒号后只写结果数字，不写单位。\n\n"
        + question
        ),
        "answer": number,
        "sft_response": reasoning + f"\n答案：{number}",
    })

random.Random(20260929).shuffle(processed)
splits = {
    "validation": processed[:500],
    "train": processed[500:],
}

OUTPUT.mkdir(parents=True, exist_ok=True)
for name, subset in splits.items():
    path = OUTPUT / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as file:
        for row in subset:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{name}: {len(subset)} 题 → {path}")

assert not (
    {row["id"] for row in splits["train"]}
    & {row["id"] for row in splits["validation"]}
)
print("训练集与验证集 ID 无交集")