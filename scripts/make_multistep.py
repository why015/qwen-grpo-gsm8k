import json
import random
from pathlib import Path

rng = random.Random(20260929)
rows = []
seen = set()

templates = [
    "仓库有{a}箱铅笔，每箱{b}支，又收到{c}支，送出{d}支。现在有多少支？",
    "商店有{a}盒零件，每盒{b}个，后来增加{c}个，再卖出{d}个。还剩多少个？",
    "工厂每天生产{b}件，连续生产{a}天后，又收到{c}件，发走{d}件。现在有多少件？",
]

while len(rows) < 300:
    a = rng.randint(3, 25)
    b = rng.randint(4, 25)
    c = rng.randint(2, 40)
    d = rng.randint(2, 40)
    template_id = rng.randrange(len(templates))
    key = (template_id, a, b, c, d)
    if key in seen:
        continue
    seen.add(key)

    answer = a * b + c - d
    question = templates[template_id].format(a=a, b=b, c=c, d=d)
    rows.append({
        "id": f"multi-{template_id}-{a}-{b}-{c}-{d}",
        "prompt": (
            question
            + " 请列式计算。最后一行以“答案：”开头，冒号后写出具体数值。"
        ),
        "answer": str(answer),
        "sft_response": (
            f"先算 {a} × {b} = {a * b}。"
            f"再算 {a * b} + {c} − {d} = {answer}。\n"
            f"答案：{answer}"
        ),
    })

rng.shuffle(rows)
output_dir = Path("data/multistep")
output_dir.mkdir(parents=True, exist_ok=True)

for name, subset in {
    "train": rows[:210],
    "validation": rows[210:255],
    "test": rows[255:300],
}.items():
    path = output_dir / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as file:
        for row in subset:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{path}: {len(subset)} 题")