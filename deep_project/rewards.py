"""Reward functions for GRPO training with per-group audit logging."""
import json
import re
from pathlib import Path

from deep_project.verifier import grade

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
    rewards = [float(grade(c[0]["content"], str(a))["reward"])
               for c, a in zip(completions, answer)]
    _STEP["n"] += 1
    _record(_STEP["n"], completions, answer, rewards)
    return rewards


_FORMAT_RE = re.compile(r"^答案\s*[:：]\s*[+-]?[\d.,]+")


def reward_r2(completions, answer, **kwargs):
    rewards = []
    for c, a in zip(completions, answer):
        resp = c[0]["content"]
        correct = float(grade(resp, str(a))["reward"])
        last_line = resp.strip().split("\n")[-1].strip()
        format_ok = 1.0 if _FORMAT_RE.match(last_line) else 0.0
        rewards.append(correct + 0.1 * format_ok)
    _STEP["n"] += 1
    _record(_STEP["n"], completions, answer, rewards)
    return rewards


def reward_r3(completions, answer, **kwargs):
    rewards = []
    for c, a in zip(completions, answer):
        resp = c[0]["content"]
        correct = float(grade(resp, str(a))["reward"])
        last_line = resp.strip().split("\n")[-1].strip()
        format_ok = 1.0 if _FORMAT_RE.match(last_line) else 0.0
        lines = [l.strip() for l in resp.strip().split("\n") if l.strip()]
        if len(lines) > 1:
            repeats = sum(1 for x, y in zip(lines, lines[1:]) if x == y)
            rep_ratio = repeats / (len(lines) - 1)
        else:
            rep_ratio = 0.0
        invalid = 1.0 if len(resp.strip()) < 10 else 0.0
        rewards.append(correct + 0.1 * format_ok - 0.1 * rep_ratio - 0.1 * invalid)
    _STEP["n"] += 1
    _record(_STEP["n"], completions, answer, rewards)
    return rewards


REWARD_FUNCS = {"r1": reward_r1, "r2": reward_r2, "r3": reward_r3}
