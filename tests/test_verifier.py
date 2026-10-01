"""Verifier 单元测试 —— 防止未来修改引入 bug。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deep_project.verifier import grade, grade_strict, extract_answer, normalize


# ============ 基础测试（lenient 主路径）============

def test_exact_format():
    assert grade("答案：42", "42")["reward"] == 1.0
    assert grade("答案：42", "43")["reward"] == 0.0


def test_gsm8k_hash_format():
    assert grade("some reasoning\n#### 42", "42")["reward"] == 1.0


def test_boxed_format():
    assert grade(r"Therefore \boxed{42}", "42")["reward"] == 1.0


def test_english_format():
    assert grade("The answer is 42.", "42")["reward"] == 1.0
    assert grade("Final answer: 42", "42")["reward"] == 1.0


def test_priority_over_fallback():
    """模式匹配优先于兜底 —— 防止 7191 类 bug"""
    resp = r"45 \times 1.20 = 54\]\n\nTherefore, the answer is: $\boxed{54}$\n\nNote: used $1.20 instead."
    assert grade(resp, "54")["reward"] == 1.0


def test_last_pattern_wins():
    """多个模式匹配时取位置最靠后的"""
    resp = "#### 42\n重新计算后\n答案：43"
    assert grade(resp, "43")["reward"] == 1.0
    assert grade(resp, "42")["reward"] == 0.0


def test_normalize():
    assert normalize("1,000") == 1000
    assert normalize("$3.14") == 3.14
    assert normalize("42.0") == 42
    assert normalize("42.") == 42


def test_parse_fail():
    assert grade("I don't know", "42")["reward"] == 0.0
    assert grade("", "42")["reward"] == 0.0


# ============ strict vs lenient 对比测试 ============

def test_strict_accepts_strict_format():
    """strict 模式：只接受最后一行 '答案：X'"""
    assert grade_strict("答案：42", "42")["reward"] == 1.0
    assert grade_strict("答案：42。", "42")["reward"] == 1.0
    assert grade_strict("先算一下\n答案：42", "42")["reward"] == 1.0


def test_strict_rejects_loose_format():
    """strict 模式：拒绝宽松格式"""
    assert grade_strict("#### 42", "42")["reward"] == 0.0
    assert grade_strict(r"\boxed{42}", "42")["reward"] == 0.0
    assert grade_strict("The answer is 42.", "42")["reward"] == 0.0
    assert grade_strict("答案：42 units", "42")["reward"] == 0.0


def test_strict_rejects_answer_not_in_last_line():
    """strict 模式：答案必须出现在最后一行"""
    resp = "答案：42\n后面还有别的话"
    assert grade_strict(resp, "42")["reward"] == 0.0


def test_lenient_accepts_loose_format():
    """lenient 模式：接受宽松格式"""
    assert grade("#### 42", "42")["reward"] == 1.0
    assert grade(r"\boxed{42}", "42")["reward"] == 1.0
    assert grade("The answer is 42.", "42")["reward"] == 1.0


# ============ 运行器 ============

if __name__ == "__main__":
    import traceback
    passed = failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"✅ {name}")
                passed += 1
            except AssertionError:
                print(f"❌ {name}")
                traceback.print_exc()
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
