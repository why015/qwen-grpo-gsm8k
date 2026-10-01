"""从 HuggingFace 下载 GSM8K train split 并导出到 data/raw/。

用法：
    python prepare_gsm8k.py

国内网络：
    export HF_ENDPOINT=https://hf-mirror.com
    python prepare_gsm8k.py
"""
import json
import os
from pathlib import Path

OUT = Path("deep_project/data/raw/gsm8k_train.jsonl")


def main():
    from datasets import load_dataset

    print(f"HF_ENDPOINT = {os.environ.get('HF_ENDPOINT', 'https://huggingface.co (default)')}")
    print("下载 GSM8K train split ...")
    ds = load_dataset("gsm8k", "main", split="train")
    print(f"  → {len(ds)} 条")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in ds:
            f.write(json.dumps({
                "question": r["question"],
                "answer": r["answer"],
            }, ensure_ascii=False) + "\n")
    print(f"✅ 写出 {OUT} ({len(ds)} 条)")


if __name__ == "__main__":
    main()
