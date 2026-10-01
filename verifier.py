"""判分模块：从模型输出中提取答案并与标准答案比对。

提供三种接口：
- lenient_parse / grade:       宽松解析，用于评测与分析
- strict_parse / grade_strict: 严格解析，用于训练 reward（防 reward hacking）
- extract_answer:              兼容别名，等价于 lenient_parse
"""
import re

# === 通用：多格式候选提取 ===
_PATTERNS = [
    # 1) GSM8K 官方格式：#### 123
    re.compile(r"####\s*([\-\d\.,]+)"),
    # 2) 中英文"答案：X" / "The answer is X" / "final answer is X"
    re.compile(
        r"(?:答案|answer|final\s+answer)\s*(?:is|为)?\s*[:：]?\s*\$?([\-\d\.,]+)",
        re.I,
    ),
    # 3) \boxed{123}
    re.compile(r"\\boxed\{([^}]*)\}"),
]

# 兜底：最后一个数字
_FALLBACK_NUMBER = re.compile(r"([\-\d]+(?:\.\d+)?)")

# 严格格式：最后一行完整为 "答案：<number>"（允许末尾中文句号 / 英文句号）
_STRICT_FORMAT_RE = re.compile(r"^\s*答案\s*[:：]\s*([+-]?[\d.,]+)\s*[。.]?\s*$")


# === 通用：归一化 ===
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


# === 宽松解析（分析用） ===
def lenient_parse(text: str):
    """按文本位置取最后一个候选，允许多种格式。"""
    if not text:
        return None
    text = text.strip()

    # 模式匹配优先
    pattern_candidates = []
    for pat in _PATTERNS:
        for m in pat.finditer(text):
            pattern_candidates.append((m.end(), m.group(1)))

    if pattern_candidates:
        pattern_candidates.sort(key=lambda x: -x[0])
        return pattern_candidates[0][1]

    # 兜底：最后一行里的数字
    last_line_start = text.rfind("\n") + 1
    last_line = text[last_line_start:].strip()
    fallback_candidates = []
    for m in _FALLBACK_NUMBER.finditer(last_line):
        fallback_candidates.append((last_line_start + m.end(), m.group(0)))

    if not fallback_candidates:
        return None

    fallback_candidates.sort(key=lambda x: -x[0])
    return fallback_candidates[0][1]


# === 严格解析（训练 reward 用） ===
def strict_parse(text: str):
    """只接受最后一行完整为 '答案：<number>'。"""
    if not text:
        return None
    last_line = text.strip().split("\n")[-1].strip()
    m = _STRICT_FORMAT_RE.match(last_line)
    if not m:
        return None
    return m.group(1)


# === 兼容别名 ===
def extract_answer(text: str):
    """【兼容旧代码】等价于 lenient_parse。"""
    return lenient_parse(text)


# === 高层判分接口 ===
def grade(response: str, expected: str) -> dict:
    """【评测 / 分析用】宽松判分。"""
    pred = normalize(lenient_parse(response))
    gold = normalize(expected)

    if pred is None:
        return {"parsed": None, "reward": 0.0, "reason": "PARSE_FAIL"}
    if gold is None:
        return {"parsed": pred, "reward": 0.0, "reason": "GOLD_INVALID"}
    if pred == gold:
        return {"parsed": pred, "reward": 1.0, "reason": "CORRECT"}
    return {"parsed": pred, "reward": 0.0, "reason": f"WRONG pred={pred} gold={gold}"}


def grade_strict(response: str, expected: str) -> dict:
    """【训练 reward 用】严格判分：只接受最后一行 '答案：X'。"""
    pred = normalize(strict_parse(response))
    gold = normalize(expected)

    if pred is None:
        return {"parsed": None, "reward": 0.0, "reason": "STRICT_PARSE_FAIL"}
    if gold is None:
        return {"parsed": pred, "reward": 0.0, "reason": "GOLD_INVALID"}
    if pred == gold:
        return {"parsed": pred, "reward": 1.0, "reason": "CORRECT_STRICT"}
    return {"parsed": pred, "reward": 0.0, "reason": f"WRONG_STRICT pred={pred} gold={gold}"}
