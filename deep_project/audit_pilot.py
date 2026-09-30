import json
import re
from decimal import Decimal

PATH = "deep_project/outputs/base_validation_pilot20.jsonl"
NUM = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"

def extract(text):
    tail = text.strip()[-500:]

    patterns = [
        rf"\\boxed\{{\s*({NUM})\s*\}}",
        rf"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*\$?\s*({NUM})\b",
        rf"答案\s*[:：]\s*({NUM})\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, tail, flags=re.IGNORECASE)
        if matches:
            return matches[-1].replace(",", "")
    return None

rows = [json.loads(line) for line in open(PATH, encoding="utf-8")]
for row in rows:
    if row["generated_tokens"] >= 256:
        status = "达到长度上限"
        value = None
    else:
        value = extract(row["response"])
        if value is None:
            status = "需人工查看"
        else:
            status = (
                "末尾数值正确"
                if Decimal(value) == Decimal(row["expected"])
                else "末尾数值错误"
            )
    print(
        f'{row["id"]}: {status}; '
        f'提取={value!r}; 标准={row["expected"]}; '
        f'token={row["generated_tokens"]}'
    )
    if status == "需人工查看":
        print("  结尾：", repr(row["response"][-250:]))
