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
    if not text:
        return None
    text = text.strip()

    # 第一优先级：模式匹配（####、答案：、\boxed{} 等）
    pattern_candidates = []
    for pat in _PATTERNS:
        for m in pat.finditer(text):
            pattern_candidates.append((m.end(), m.group(1)))

    if pattern_candidates:
        pattern_candidates.sort(key=lambda x: -x[0])
        return pattern_candidates[0][1]

    # 第二优先级（兜底）：只在没有任何模式匹配时启用
    last_line_start = text.rfind("\n") + 1
    last_line = text[last_line_start:].strip()
    fallback_candidates = []
    for m in _FALLBACK_NUMBER.finditer(last_line):
        fallback_candidates.append((last_line_start + m.end(), m.group(0)))

    if not fallback_candidates:
        return None

    fallback_candidates.sort(key=lambda x: -x[0])
    return fallback_candidates[0][1]


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