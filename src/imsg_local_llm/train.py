"""
LoRA fine-tune a local model on your iMessage style with Apple MLX.

A wrapper around `mlx_lm lora` with defaults for an Apple silicon laptop
(M-series, 16GB or more). It trains a small LoRA adapter over a 4-bit base model.

Usage:
    python -m imsg_local_llm.train              # full run from config.yaml
    python -m imsg_local_llm.train --smoke      # short sanity run
    python -m imsg_local_llm.train --model mlx-community/Qwen2.5-7B-Instruct-4bit
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULTS = dict(
    model="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    data="data",
    adapter_path="adapters",
    iters=600,
    batch_size=2,
    num_layers=16,
    learning_rate=1e-4,
    max_seq_length=2048,
    steps_per_report=20,
    steps_per_eval=100,
    save_every=100,
    grad_checkpoint=True,
)


def load_config(path: Path) -> dict:
    cfg = dict(DEFAULTS)
    if path.exists():
        try:
            import yaml
            loaded = (yaml.safe_load(path.read_text()) or {}).get("train", {})
            cfg.update({k: v for k, v in loaded.items() if k in DEFAULTS})
        except ModuleNotFoundError:
            pass
    return cfg


def build_command(cfg: dict) -> list[str]:
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", str(cfg["model"]),
        "--train",
        "--data", str(cfg["data"]),
        "--adapter-path", str(cfg["adapter_path"]),
        "--iters", str(cfg["iters"]),
        "--batch-size", str(cfg["batch_size"]),
        "--num-layers", str(cfg["num_layers"]),
        "--learning-rate", str(cfg["learning_rate"]),
        "--max-seq-length", str(cfg["max_seq_length"]),
        "--steps-per-report", str(cfg["steps_per_report"]),
        "--steps-per-eval", str(cfg["steps_per_eval"]),
        "--save-every", str(cfg["save_every"]),
    ]
    if cfg.get("grad_checkpoint"):
        cmd.append("--grad-checkpoint")
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA fine-tune on your iMessages (MLX).")
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--model")
    ap.add_argument("--iters", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--adapter-path")
    ap.add_argument("--smoke", action="store_true",
                    help="Quick end-to-end sanity run (60 iters).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    for key in ("model", "iters", "batch_size", "adapter_path"):
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    if args.smoke:
        cfg.update(iters=60, steps_per_eval=60, save_every=60)

    data_dir = Path(cfg["data"])
    if not (data_dir / "train.jsonl").exists():
        sys.exit(f"No training data at {data_dir}/train.jsonl. Run prepare first "
                 "(make prepare).")

    cmd = build_command(cfg)
    print("Fine-tuning locally with MLX. Nothing leaves your machine.")
    print("  model:", cfg["model"])
    print("  iters:", cfg["iters"], "| batch:", cfg["batch_size"],
          "| lora layers:", cfg["num_layers"])
    print("  adapters ->", cfg["adapter_path"])
    print("$", " ".join(cmd), "\n")
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
