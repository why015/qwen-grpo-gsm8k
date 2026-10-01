"""Reward functions for GRPO training with per-group audit logging."""
import json
import os
import re
from pathlib import Path

# === 根据环境变量选择 verifier 严格度 ===
# strict (默认):   只接受最后一行"答案：X"，防 reward hacking
# lenient:          多格式解析，复现历史 v3 训练需显式传 REWARD_MODE=lenient
REWARD_MODE = os.environ.get("REWARD_MODE", "strict")  # 默认 strict；复现历史 v3 需显式传 REWARD_MODE=lenient

if REWARD_MODE == "strict":
    from deep_project.verifier import grade_strict as _grade_fn
    print("[reward] using STRICT verifier (last-line only)")
elif REWARD_MODE == "lenient":
    from deep_project.verifier import grade as _grade_fn
    print("[reward] using LENIENT verifier (multi-format)")
else:
    raise ValueError(f"REWARD_MODE must be 'lenient' or 'strict', got {REWARD_MODE!r}")

# R2/R3 的格式判定始终使用 strict_parse，与训练 reward 严格度一致
from deep_project.verifier import strict_parse

_AUDIT_LOG = []
_AUDIT_PATH = None
_STEP = {"n": 0}


def set_audit_path(path: str):
    global _AUDIT_PATH
    _AUDIT_PATH = Path(path)
    _AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[audit] 记录到 {_AUDIT_PATH}")


def _flush():
    if _AUDIT_PATH is None:
        return
    with _AUDIT_PATH.open("w", encoding="utf-8") as f:
        for rec in _AUDIT_LOG:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _record(step, completions, answers, rewards):
    batch = []
    for comp, ans, r in zip(completions, answers, rewards):
        content = comp[0]["content"] if isinstance(comp, list) else str(comp)
        batch.append({
            "answer": str(ans),
            "response": content,
            "reward": float(r),
        })
    _AUDIT_LOG.append({"step": step, "batch": batch})
    if len(_AUDIT_LOG) % 10 == 0:
        _flush()
        print(f"[audit] flushed {len(_AUDIT_LOG)} steps")


def flush_audit():
    _flush()
    print(f"[audit] 最终写入 {len(_AUDIT_LOG)} steps")


def reward_r1(completions, answer, **kwargs):
    rewards = [float(_grade_fn(c[0]["content"], str(a))["reward"])
               for c, a in zip(completions, answer)]
    _STEP["n"] += 1
    _record(_STEP["n"], completions, answer, rewards)
    return rewards


def reward_r2(completions, answer, **kwargs):
    rewards = []
    for c, a in zip(completions, answer):
        resp = c[0]["content"]
        correct = float(_grade_fn(resp, str(a))["reward"])
        last_line = resp.strip().split("\n")[-1].strip()
        format_ok = 1.0 if strict_parse(last_line) is not None else 0.0
        rewards.append(correct + 0.1 * format_ok)
    _STEP["n"] += 1
    _record(_STEP["n"], completions, answer, rewards)
    return rewards


def reward_r3(completions, answer, **kwargs):
    rewards = []
    for c, a in zip(completions, answer):
        resp = c[0]["content"]
        correct = float(_grade_fn(resp, str(a))["reward"])
        last_line = resp.strip().split("\n")[-1].strip()
        format_ok = 1.0 if strict_parse(last_line) is not None else 0.0
        lines = [l.strip() for l in resp.strip().split("\n") if l.strip()]
        if len(lines) > 1:
            repeats = sum(1 for x, y in zip(lines, lines[1:]) if x == y)
            rep_ratio = repeats / (len(lines) - 1)
        else:
            rep_ratio = 0.0
        invalid = 0.0 if format_ok else 1.0
        rewards.append(correct + 0.1 * format_ok - 0.1 * rep_ratio - 0.1 * invalid)
    _STEP["n"] += 1
    _record(_STEP["n"], completions, answer, rewards)
    return rewards


REWARD_FUNCS = {"r1": reward_r1, "r2": reward_r2, "r3": reward_r3}
