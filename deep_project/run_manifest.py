"""Run manifest: 为每次训练/评测生成唯一、可追溯的实验记录。

设计原则:
  - 每次运行生成唯一 run_id
  - 所有产物放在 runs/<run_id>/ 下
  - manifest 是唯一权威记录（不再双写）
  - status 字段跟踪生命周期: running / completed / failed
"""
import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

RUNS_DIR = Path("runs")
RUNS_CSV = Path("results/runs.csv")

CSV_HEADER = [
    "run_id", "timestamp", "git_sha", "git_dirty", "status",
    "model", "loss_type", "scale_rewards", "epsilon_high",
    "beta", "num_generations", "learning_rate", "max_steps",
    "seed", "reward_version", "reward_mode",
    "dev500_acc", "heldout_acc", "train_runtime_sec",
    "notes",
]


def make_run_id(
    algorithm="grpo",
    loss_type="dapo",
    reward="r1",
    verifier="lenient",
    beta=0.04,
    num_gen=8,
    seed=42,
    timestamp=None,
):
    """生成唯一的 run_id。"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (
        f"{algorithm}__loss-{loss_type}__reward-{reward}__verifier-{verifier}"
        f"__beta-{beta}__g{num_gen}__seed-{seed}__{timestamp}"
    )


def _get_git_info():
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
    if config is None:
        return {}
    try:
        from dataclasses import asdict
        d = asdict(config)
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


def _append_to_csv(run_id, timestamp, git_sha, git_dirty, status,
                   config_dict, extra, metrics, notes):
    RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    file_exists = RUNS_CSV.exists()
    row = {
        "run_id": run_id,
        "timestamp": timestamp,
        "git_sha": git_sha,
        "git_dirty": "yes" if git_dirty else "no",
        "status": status,
        "model": extra.get("model", ""),
        "loss_type": config_dict.get("loss_type", ""),
        "scale_rewards": config_dict.get("scale_rewards", ""),
        "epsilon_high": config_dict.get("epsilon_high", ""),
        "beta": config_dict.get("beta", ""),
        "num_generations": config_dict.get("num_generations", ""),
        "learning_rate": config_dict.get("learning_rate", ""),
        "max_steps": config_dict.get("max_steps", ""),
        "seed": extra.get("seed", ""),
        "reward_version": extra.get("reward_version", ""),
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


def _rewrite_csv_row(run_id, updates):
    """更新 runs.csv 里某一行的字段。"""
    if not RUNS_CSV.exists():
        return
    rows = []
    with RUNS_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["run_id"] == run_id:
                for k, v in updates.items():
                    if k in row:
                        row[k] = v
            rows.append(row)
    with RUNS_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def init_manifest(run_dir, config, extra=None, notes=""):
    """训练开始时调用。创建 manifest.json，status=running。

    Returns:
        Path: manifest.json 路径
    """
    run_dir = Path(run_dir)
    run_id = run_dir.name
    timestamp = datetime.now().isoformat(timespec="seconds")
    git_sha, git_dirty = _get_git_info()
    env_info = _get_env_info()
    config_dict = _config_to_dict(config)
    extra = extra or {}

    manifest = {
        "run_id": run_id,
        "timestamp_start": timestamp,
        "timestamp_end": None,
        "status": "running",
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "output_dir": str(run_dir),
        "env": env_info,
        "config": config_dict,
        "extra": extra,
        "metrics": {},
        "notes": notes,
        "error": None,
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    print(f"[manifest] init → {manifest_path} (status=running)")

    _append_to_csv(
        run_id=run_id, timestamp=timestamp, git_sha=git_sha, git_dirty=git_dirty,
        status="running", config_dict=config_dict, extra=extra, metrics={}, notes=notes,
    )
    print(f"[manifest] appended to {RUNS_CSV}")

    return manifest_path


def finalize_manifest(run_dir, metrics=None, status="completed", error=None, notes=None):
    """训练结束时调用。更新 manifest.json：status + metrics + timestamp_end。"""
    run_dir = Path(run_dir)
    run_id = run_dir.name
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[manifest] ⚠️ 未找到 {manifest_path}，跳过 finalize")
        return

    with manifest_path.open(encoding="utf-8") as f:
        m = json.load(f)

    m["timestamp_end"] = datetime.now().isoformat(timespec="seconds")
    m["status"] = status
    if error:
        m["error"] = error
    if metrics:
        m["metrics"].update(metrics)
    if notes is not None:
        m["notes"] = notes

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False, default=str)
    print(f"[manifest] finalize → {manifest_path} (status={status})")

    updates = {"status": status}
    if metrics:
        for k, v in metrics.items():
            if k in CSV_HEADER:
                updates[k] = v
    _rewrite_csv_row(run_id, updates)
    print(f"[manifest] updated {RUNS_CSV}")


def update_metrics(run_dir, metrics):
    """评测后调用。更新 manifest.json 和 runs.csv 里的评测指标。

    Args:
        run_dir: runs/<run_id> 目录（或 Path）
        metrics: {"dev500_acc": 0.468, "heldout_acc": 0.457}
    """
    run_dir = Path(run_dir)
    run_id = run_dir.name
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[manifest] ⚠️ 未找到 {manifest_path}")
        return

    with manifest_path.open(encoding="utf-8") as f:
        m = json.load(f)
    m["metrics"].update(metrics)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False, default=str)
    print(f"[manifest] updated {manifest_path}")

    _rewrite_csv_row(run_id, {k: v for k, v in metrics.items() if k in CSV_HEADER})
    print(f"[manifest] updated {RUNS_CSV}")


if __name__ == "__main__":
    # 自测
    run_id = make_run_id()
    print(f"sample run_id: {run_id}")
    init_manifest(
        run_dir=f"/tmp/{run_id}",
        config=None,
        extra={"seed": 42, "reward_version": "r1", "reward_mode": "lenient"},
        notes="self-test",
    )
    finalize_manifest(f"/tmp/{run_id}", metrics={"train_runtime_sec": 123.45})
    print("\n✅ 自测完成")
