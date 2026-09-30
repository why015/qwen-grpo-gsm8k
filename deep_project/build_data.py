"""
build_data.py — 一键构建所有训练/验证/测试文件

输入:
  deep_project/data/raw/gsm8k_train.jsonl       (7473 条原始 GSM8K train)
  deep_project/data/splits/train_ids.json       (6973 个 id，按切分顺序)
  deep_project/data/splits/val_ids.json         (500 个 id，按切分顺序)

输出:
  deep_project/data/processed/train_clean.jsonl      (6973)
  deep_project/data/processed/validation.jsonl       (500)
  deep_project/data/processed/development500.jsonl   (500)
  deep_project/data/processed/dev200.jsonl           (200)
  deep_project/data/processed/heldout300.jsonl       (300)
"""
import json
import re
from pathlib import Path

RAW_PATH = Path("deep_project/data/raw/gsm8k_train.jsonl")
SPLITS_DIR = Path("deep_project/data/splits")
OUT_DIR = Path("deep_project/data/processed")

PROMPT_TEMPLATE = (
    "请解答下面的数学应用题，给出简短计算过程。"
    "最后一行以“答案：”开头，冒号后只写结果数字，不写单位。\n\n{question}"
)

_CALC_RE = re.compile(r"<<[^>]*>>")
_FINAL_RE = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)


def process_raw(raw_item: dict, item_id: str) -> dict:
    question = raw_item["question"].replace("\u2028", "")
    raw_answer = raw_item["answer"].replace("\u2028", "")

    m = _FINAL_RE.search(raw_answer)
    if not m:
        raise ValueError(f"{item_id}: missing '####' separator")
    final_answer = m.group(1).strip().replace(",", "")

    sft_response = _CALC_RE.sub("", raw_answer)
    sft_response = _FINAL_RE.sub(f"答案：{final_answer}", sft_response)
    sft_response = sft_response.strip()

    return {
        "id": item_id,
        "prompt": PROMPT_TEMPLATE.format(question=question),
        "answer": final_answer,
        "sft_response": sft_response,
    }


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    print(f"[1/4] 读入 raw: {RAW_PATH}")
    raw_rows = load_jsonl(RAW_PATH)
    raw_by_idx = {i: r for i, r in enumerate(raw_rows)}
    print(f"      → {len(raw_rows)} 条")

    print(f"[2/4] 读入 splits")
    train_ids = json.loads((SPLITS_DIR / "train_ids.json").read_text(encoding="utf-8"))
    val_ids = json.loads((SPLITS_DIR / "val_ids.json").read_text(encoding="utf-8"))
    print(f"      train={len(train_ids)}  val={len(val_ids)}")

    print(f"[3/4] 处理")
    train_rows = [process_raw(raw_by_idx[int(i.replace("gsm8k-train-", ""))], i) for i in train_ids]
    val_rows   = [process_raw(raw_by_idx[int(i.replace("gsm8k-train-", ""))], i) for i in val_ids]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[4/4] 写出")
    write_jsonl(OUT_DIR / "train_clean.jsonl", train_rows)
    print(f"      → train_clean.jsonl ({len(train_rows)})")
    write_jsonl(OUT_DIR / "validation.jsonl", val_rows)
    print(f"      → validation.jsonl ({len(val_rows)})")
    write_jsonl(OUT_DIR / "development500.jsonl", val_rows)
    print(f"      → development500.jsonl ({len(val_rows)})")
    write_jsonl(OUT_DIR / "dev200.jsonl", val_rows[:200])
    print(f"      → dev200.jsonl (200)")
    write_jsonl(OUT_DIR / "heldout300.jsonl", val_rows[200:])
    print(f"      → heldout300.jsonl ({len(val_rows)-200})")

    print("\n✅ 完成")


if __name__ == "__main__":
    main()
