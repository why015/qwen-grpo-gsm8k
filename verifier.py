"""判分模块：从模型输出中提取答案并与标准答案比对。"""
import re

# 提取候选答案的正则，按优先级从高到低尝试
_PATTERNS = [
    # 1) GSM8K 官方格式：#### 123
    re.compile(r"####\s*([\-\d\.,]+)"),
    # 2) 中英文“答案：X” / “The answer is X” / “final answer is X”
    re.compile(
        r"(?:答案|answer|final\s+answer)\s*(?:is|为)?\s*[:：]?\s*\$?([\-\d\.,]+)",
        re.I,
    ),
    # 3) \boxed{123}
    re.compile(r"\\boxed\{([^}]*)\}"),
]

# 兜底：最后一个数字
_FALLBACK_NUMBER = re.compile(r"([\-\d]+(?:\.\d+)?)")


def extract_answer(text: str):
    """从模型输出中提取答案；失败返回 None。"""
    if not text:
        return None
    text = text.strip()

    for pat in _PATTERNS:
        matches = pat.findall(text)
        if matches:
            return matches[-1]

    # 兜底：只在最后一行找数字，避免从中间推理步骤误抓
    last_line = text.split("\n")[-1].strip()
    matches = _FALLBACK_NUMBER.findall(last_line)
    if matches:
        return matches[-1]

    return None


def normalize(x):
    """归一化答案：去 $ 和逗号，转数值。"""
    if x is None:
        return None
    x = str(x).replace("$", "").replace(",", "").strip().rstrip(".")
    try:
        f = float(x)
        return int(f) if f.is_integer() else f
    except ValueError:
        return x


def grade(response: str, expected: str) -> dict:
    """判分主函数。返回 {parsed, reward, reason}。"""
    pred = normalize(extract_answer(response))
    gold = normalize(expected)

    if pred is None:
        return {"parsed": None, "reward": 0.0, "reason": "PARSE_FAIL"}
    if gold is None:
        return {"parsed": pred, "reward": 0.0, "reason": "GOLD_INVALID"}
    if pred == gold:
        return {"parsed": pred, "reward": 1.0, "reason": "CORRECT"}
    return {
        "parsed": pred,
        "reward": 0.0,
        "reason": f"WRONG pred={pred} gold={gold}",
    }