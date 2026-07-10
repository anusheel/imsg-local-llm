"""
Chat with your fine-tuned iMessage clone.

Loads a 4-bit base model plus your LoRA adapter and holds a streaming
conversation, applying the same persona prompt you trained with.

Usage:
    python -m imsg_local_llm.chat
    python -m imsg_local_llm.chat --temp 0.8 --my-name Sam
    python -m imsg_local_llm.chat --no-adapter        # talk to the base model

Slash commands inside the REPL:
    /reset     start a fresh conversation
    /system    print the active system prompt
    /exit      quit
"""
from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_SYSTEM = (
    "You are {my_name} texting on iMessage. Reply exactly the way {my_name} really "
    "texts: same tone, length, slang, capitalization, punctuation, and emoji habits. "
    "Keep it casual and in-character. Do not explain yourself."
)


def load_cfg(path: Path) -> dict:
    cfg = {"my_name": "Me", "model": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
           "adapter_path": "adapters", "system_prompt": DEFAULT_SYSTEM}
    if path.exists():
        try:
            import yaml
            data = yaml.safe_load(path.read_text()) or {}
            cfg["my_name"] = data.get("my_name", cfg["my_name"])
            cfg["system_prompt"] = data.get("system_prompt", cfg["system_prompt"])
            cfg["model"] = data.get("train", {}).get("model", cfg["model"])
            cfg["adapter_path"] = data.get("train", {}).get("adapter_path", cfg["adapter_path"])
        except ModuleNotFoundError:
            pass
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Chat with your fine-tuned iMessage clone.")
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--model")
    ap.add_argument("--adapter-path")
    ap.add_argument("--no-adapter", action="store_true", help="Use the base model only.")
    ap.add_argument("--my-name")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    try:
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler
    except ModuleNotFoundError:
        raise SystemExit("mlx-lm is not installed. Run: make setup   (or: uv pip install mlx-lm)")

    cfg = load_cfg(args.config)
    if args.my_name:
        cfg["my_name"] = args.my_name
    model_id = args.model or cfg["model"]
    adapter = None if args.no_adapter else (args.adapter_path or cfg["adapter_path"])
    if adapter and not Path(adapter).exists():
        print(f"(no adapter at '{adapter}', using base model; run `make train` first)")
        adapter = None

    system = cfg["system_prompt"].format(my_name=cfg["my_name"])
    print(f"Loading {model_id}" + (f" + adapter {adapter}" if adapter else " (base model)") + " ...")
    model, tokenizer = load(model_id, adapter_path=adapter)
    sampler = make_sampler(temp=args.temp, top_p=args.top_p)

    print(f"\nChatting as a clone of '{cfg['my_name']}'. Type a message; /exit to quit.\n")
    history = [{"role": "system", "content": system}]
    while True:
        try:
            user = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            history = [{"role": "system", "content": system}]
            print("(conversation reset)\n")
            continue
        if user == "/system":
            print(system + "\n")
            continue

        history.append({"role": "user", "content": user})
        prompt = tokenizer.apply_chat_template(history, add_generation_prompt=True)
        print(f"{cfg['my_name'].lower()} › ", end="", flush=True)
        reply = ""
        for chunk in stream_generate(model, tokenizer, prompt,
                                     max_tokens=args.max_tokens, sampler=sampler):
            print(chunk.text, end="", flush=True)
            reply += chunk.text
        print("\n")
        history.append({"role": "assistant", "content": reply.strip()})


if __name__ == "__main__":
    main()
