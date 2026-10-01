"""Run manifest: 为每次训练/评测自动生成可追溯的配置文件。

生成:
  results/<run_id>/manifest.json   - 完整配置 + 环境 + 指标
  results/runs.csv                 - 所有 run 一行一条，方便汇总
"""
import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("results")
RUNS_CSV = RESULTS_DIR / "runs.csv"

CSV_HEADER = [
    "run_id", "timestamp", "git_sha", "git_dirty",
    "model", "loss_type", "scale_rewards", "epsilon_high",
    "beta", "num_generations", "learning_rate", "max_steps",
    "seed", "reward_mode",
    "dev500_acc", "heldout_acc",
    "train_runtime_sec", "notes",
]


def _get_git_info():
    """返回 (short_sha, is_dirty)。如果不在 git repo 中，返回 (None, None)。"""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip())
        return sha, dirty
    except Exception:
        return None, None


def _get_env_info():
    """收集环境信息。"""
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        pass
    try:
        import transformers, peft, trl
        env["transformers"] = transformers.__version__
        env["peft"] = peft.__version__
        env["trl"] = trl.__version__
    except ImportError:
        pass
    return env


def _config_to_dict(config):
    """把 GRPOConfig / SFTConfig 转成可序列化的 dict。"""
    if config is None:
        return {}
    try:
        from dataclasses import asdict
        d = asdict(config)
        # 只保留关键字段，避免 manifest 太大
        keys = [
            "loss_type", "scale_rewards", "epsilon", "epsilon_high",
            "beta", "num_generations", "learning_rate", "max_steps",
            "per_device_train_batch_size", "gradient_accumulation_steps",
            "max_completion_length", "temperature", "top_p",
            "mask_truncated_completions", "importance_sampling_level",
            "num_train_epochs",
        ]
        return {k: d.get(k) for k in keys if k in d}
    except Exception:
        return {}


def dump_manifest(
    output_dir,
    config=None,
    run_id=None,
    extra=None,
    metrics=None,
    notes="",
):
    """生成 manifest.json + 追加一行 runs.csv。

    Args:
        output_dir: 训练输出目录，如 "models/grpo_seed_42"
        config: GRPOConfig 或 SFTConfig 对象
        run_id: 自定义 run_id，默认从 output_dir 推导
        extra: 额外信息 dict，比如 {"seed": 42, "reward_mode": "lenient"}
        metrics: 评测指标 dict，比如 {"dev500_acc": 0.468, "heldout_acc": 0.457}
        notes: 备注
    """
    output_dir = Path(output_dir)
    if run_id is None:
        run_id = output_dir.name

    timestamp = datetime.now().isoformat(timespec="seconds")
    git_sha, git_dirty = _get_git_info()
    env_info = _get_env_info()
    config_dict = _config_to_dict(config)
    extra = extra or {}
    metrics = metrics or {}

    manifest = {
        "run_id": run_id,
        "timestamp": timestamp,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "output_dir": str(output_dir),
        "env": env_info,
        "config": config_dict,
        "extra": extra,
        "metrics": metrics,
        "notes": notes,
    }

    # 写入 manifest.json（双写：output_dir 便于本地查看；results/ 用于 git 提交）
    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    print(f"[manifest] 写入 {manifest_path}")

    results_manifest = RESULTS_DIR / run_id / "manifest.json"
    results_manifest.parent.mkdir(parents=True, exist_ok=True)
    with results_manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    print(f"[manifest] 写入 {results_manifest}")

    # 追加到 runs.csv
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = RUNS_CSV.exists()
    row = {
        "run_id": run_id,
        "timestamp": timestamp,
        "git_sha": git_sha,
        "git_dirty": "yes" if git_dirty else "no",
        "model": extra.get("model", ""),
        "loss_type": config_dict.get("loss_type", ""),
        "scale_rewards": config_dict.get("scale_rewards", ""),
        "epsilon_high": config_dict.get("epsilon_high", ""),
        "beta": config_dict.get("beta", ""),
        "num_generations": config_dict.get("num_generations", ""),
        "learning_rate": config_dict.get("learning_rate", ""),
        "max_steps": config_dict.get("max_steps", ""),
        "seed": extra.get("seed", ""),
        "reward_mode": extra.get("reward_mode", ""),
        "dev500_acc": metrics.get("dev500_acc", ""),
        "heldout_acc": metrics.get("heldout_acc", ""),
        "train_runtime_sec": metrics.get("train_runtime_sec", ""),
        "notes": notes,
    }
    with RUNS_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"[manifest] 追加到 {RUNS_CSV}")


def update_metrics(run_id, metrics):
    """训练完成后，回填评测指标到 manifest.json 和 runs.csv。

    Args:
        run_id: manifest 里的 run_id
        metrics: {"dev500_acc": 0.468, "heldout_acc": 0.457}
    """
    # 更新 manifest.json
    for mpath in Path("models").glob("*/manifest.json"):
        with mpath.open(encoding="utf-8") as f:
            m = json.load(f)
        if m.get("run_id") == run_id:
            m["metrics"].update(metrics)
            with mpath.open("w", encoding="utf-8") as f:
                json.dump(m, f, indent=2, ensure_ascii=False, default=str)
            print(f"[manifest] 更新 {mpath}")

    # 更新 runs.csv
    if not RUNS_CSV.exists():
        return
    rows = []
    with RUNS_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["run_id"] == run_id:
                for k, v in metrics.items():
                    if k in row:
                        row[k] = v
            rows.append(row)
    with RUNS_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[manifest] 更新 {RUNS_CSV}")


if __name__ == "__main__":
    # 自测
    dump_manifest(
        output_dir="/tmp/test_manifest",
        config=None,
        extra={"seed": 42, "reward_mode": "lenient", "model": "Qwen2.5-0.5B"},
        metrics={"dev500_acc": 0.468},
        notes="self-test",
    )
    print("\n✅ 自测完成，检查 /tmp/test_manifest/manifest.json 和 results/runs.csv")
