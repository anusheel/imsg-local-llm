"""
Build an Ollama persona from your iMessage style, without training.

Computes a style profile from your messages and pulls a few dozen short
(their text -> your reply) pairs as few-shot examples, then bakes them into an
Ollama Modelfile over a base model (default llama3.1). The Modelfile contains
real snippets of your messages, so it is git-ignored.

Usage:
    python -m imsg_local_llm.quickstart_ollama --my-name Sam
    python -m imsg_local_llm.quickstart_ollama --base llama3.1 --chat
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

EMOJI_RANGES = [
    (0x1F300, 0x1FAFF), (0x2600, 0x27BF), (0x1F1E6, 0x1F1FF), (0x2190, 0x21FF),
    (0xFE00, 0xFE0F),
]


def is_emoji(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in EMOJI_RANGES)


def style_profile(train_path: Path) -> dict:
    my_msgs: list[str] = []
    with train_path.open(encoding="utf-8") as fh:
        for line in fh:
            for m in json.loads(line)["messages"]:
                if m["role"] == "assistant":
                    my_msgs.append(m["content"])
    if not my_msgs:
        return {}
    lengths = [len(m) for m in my_msgs]
    emoji_msgs = sum(1 for m in my_msgs if any(is_emoji(c) for c in m))
    lower_msgs = sum(1 for m in my_msgs if m and m == m.lower())
    top_emoji: dict[str, int] = {}
    for m in my_msgs:
        for c in m:
            if is_emoji(c):
                top_emoji[c] = top_emoji.get(c, 0) + 1
    emojis = sorted(top_emoji, key=top_emoji.get, reverse=True)[:8]
    return {
        "count": len(my_msgs),
        "avg_len": round(sum(lengths) / len(lengths), 1),
        "emoji_pct": round(100 * emoji_msgs / len(my_msgs), 1),
        "lowercase_pct": round(100 * lower_msgs / len(my_msgs), 1),
        "top_emojis": "".join(emojis),
    }


def sample_pairs(train_path: Path, n: int, max_len: int = 120) -> list[tuple[str, str]]:
    """Short (their message -> your reply) pairs for few-shot priming."""
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    with train_path.open(encoding="utf-8") as fh:
        for line in fh:
            msgs = json.loads(line)["messages"]
            for a, b in zip(msgs, msgs[1:]):
                if a["role"] == "user" and b["role"] == "assistant":
                    u, r = a["content"].strip(), b["content"].strip()
                    if not u or not r or len(u) > max_len or len(r) > max_len:
                        continue
                    if "\n" in u or "\n" in r or r in seen:
                        continue
                    seen.add(r)
                    pairs.append((u, r))
            if len(pairs) >= n * 4:
                break
    # spread picks across the pool for variety, deterministically
    step = max(1, len(pairs) // n)
    return pairs[::step][:n]


def build_modelfile(base: str, system: str, pairs: list[tuple[str, str]], temp: float) -> str:
    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')
    lines = [f"FROM {base}", "", f'SYSTEM """{system}"""', "",
             f"PARAMETER temperature {temp}", "PARAMETER top_p 0.9", ""]
    for u, r in pairs:
        lines.append(f'MESSAGE user "{esc(u)}"')
        lines.append(f'MESSAGE assistant "{esc(r)}"')
    return "\n".join(lines) + "\n"


def ollama_up() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


def chat_repl(model: str, temp: float) -> None:
    print(f"\nChatting with '{model}' (Ollama). Type a message; /exit to quit.\n")
    history: list[dict] = []
    while True:
        try:
            user = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not user:
            continue
        if user == "/exit":
            break
        history.append({"role": "user", "content": user})
        payload = json.dumps({"model": model, "messages": history, "stream": True,
                              "options": {"temperature": temp}}).encode()
        req = urllib.request.Request("http://localhost:11434/api/chat", data=payload,
                                     headers={"Content-Type": "application/json"})
        print("clone › ", end="", flush=True)
        reply = ""
        try:
            with urllib.request.urlopen(req) as resp:
                for raw in resp:
                    if not raw.strip():
                        continue
                    obj = json.loads(raw)
                    tok = obj.get("message", {}).get("content", "")
                    print(tok, end="", flush=True)
                    reply += tok
        except Exception as e:
            print(f"[error talking to ollama: {e}]")
        print("\n")
        history.append({"role": "assistant", "content": reply.strip()})


def main() -> None:
    ap = argparse.ArgumentParser(description="Instant iMessage clone via Ollama (no training).")
    ap.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    ap.add_argument("--base", default="llama3.1", help="Ollama base model (e.g. llama3.1, qwen2.5)")
    ap.add_argument("--my-name", default="Me")
    ap.add_argument("--name", help="Ollama model name to create (default: imsg-<my-name>)")
    ap.add_argument("--pairs", type=int, default=40, help="few-shot examples to embed")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--chat", action="store_true", help="Open a chat REPL after creating.")
    args = ap.parse_args()

    if not args.train.exists():
        sys.exit(f"No {args.train}. Run `make extract && make prepare` first.")

    prof = style_profile(args.train)
    pairs = sample_pairs(args.train, args.pairs)
    model_name = (args.name or f"imsg-{args.my_name}").lower().replace(" ", "-")

    system = (
        f"You are {args.my_name}, texting friends on iMessage. Mimic {args.my_name}'s "
        f"real texting style precisely.\n"
        f"Style profile from real messages: average reply ~{prof.get('avg_len', 0):.0f} "
        f"characters (keep replies SHORT); uses emoji in ~{prof.get('emoji_pct', 0):.0f}% "
        f"of messages (favorites: {prof.get('top_emojis') or 'none'}); "
        f"all-lowercase ~{prof.get('lowercase_pct', 0):.0f}% of the time.\n"
        "Match that tone, brevity, slang and punctuation. Never sound like an "
        "assistant, never over-explain, never use corporate phrasing."
    )

    modelfile = build_modelfile(args.base, system, pairs, args.temp)
    mf_path = Path("Modelfile")
    mf_path.write_text(modelfile, encoding="utf-8")

    print("Style profile from your messages:")
    for k, v in prof.items():
        print(f"  {k:14} {v}")
    print(f"\nWrote Modelfile ({len(pairs)} few-shot examples baked in) -> {mf_path}")

    if not ollama_up():
        print("\nOllama server isn't running. Start it, then create the model:")
        print("  ollama serve   # (in another terminal, if the app isn't running)")
        print(f"  ollama pull {args.base}")
        print(f"  ollama create {model_name} -f Modelfile")
        print(f"  ollama run {model_name}")
        return

    print(f"\nCreating Ollama model '{model_name}' from base '{args.base}' ...")
    rc = subprocess.call(["ollama", "create", model_name, "-f", str(mf_path)])
    if rc != 0:
        print(f"\n`ollama create` failed. Make sure the base is pulled: ollama pull {args.base}")
        return
    print(f"\nDone. Talk to your clone:  ollama run {model_name}")
    if args.chat:
        chat_repl(model_name, args.temp)


if __name__ == "__main__":
    main()
