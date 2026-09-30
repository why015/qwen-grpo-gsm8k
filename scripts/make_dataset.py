import json
import random
from pathlib import Path

rng = random.Random(42)
questions = []

for a in range(1, 31):
    for b in range(a, 31):
        for symbol, answer in [
            ("+", a + b),
            ("×", a * b),
            ("−", b - a),
        ]:
            question_id = f"{symbol}-{a}-{b}"
            expression = f"{b} − {a}" if symbol == "−" else f"{a} {symbol} {b}"
            questions.append({
                "id": question_id,
                "prompt": f"请计算 {expression}。最后一行以“答案：”开头，冒号后写出计算所得的具体数值。",
                "answer": str(answer),
                "sft_response": f"{expression} = {answer}\n答案：{answer}",
            })

rng.shuffle(questions)
selected = questions[:300]
splits = {
    "train": selected[:210],
    "validation": selected[210:255],
    "test": selected[255:300],
}

output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

for name, rows in splits.items():
    path = output_dir / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{path}: {len(rows)} 题")

assert len({row["id"] for row in selected}) == 300