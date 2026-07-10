"""
Turn extracted messages into fine-tuning data: given a conversation, produce the
reply you would send.

  raw messages -> per-chat sessions (split on long idle gaps)
              -> merged turns (consecutive same-side messages combined)
              -> chat examples: other people = "user", you = "assistant"
              -> data/train.jsonl + data/valid.jsonl (MLX chat format)

Each line is {"messages": [{"role": "system", ...}, {"role": "user", ...},
{"role": "assistant", ...}, ...]}, where the assistant turns are your replies.

Usage:
    python -m imsg_local_llm.prepare [--no-groups] [--my-name NAME] [--max-examples N]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DEFAULTS = dict(
    my_name="Me",
    gap_minutes=240,       # new session after a 4h+ silence
    max_turns=40,          # cap turns per training example
    min_turns=3,           # need at least a little back-and-forth
    include_groups=True,
    valid_fraction=0.05,
    seed=7,
    max_examples=0,        # 0 = no cap
    system_prompt=(
        "You are {my_name} texting on iMessage. Reply exactly the way {my_name} "
        "really texts: same tone, length, slang, capitalization, punctuation, and "
        "emoji habits. Keep it casual and in-character. Do not explain yourself."
    ),
)


def load_config(path: Path) -> dict:
    cfg = dict(DEFAULTS)
    if path.exists():
        try:
            import yaml  # optional
            loaded = yaml.safe_load(path.read_text()) or {}
            cfg.update({k: v for k, v in loaded.items() if k in DEFAULTS})
        except ModuleNotFoundError:
            pass
    return cfg


def parse_ts(ts: str | None) -> float:
    if not ts:
        return 0.0
    try:
        return datetime.fromisoformat(ts).timestamp()
    except Exception:
        return 0.0


def sessionize(msgs: list[dict], gap_seconds: float, max_turns: int) -> list[list[dict]]:
    """Split one chat's messages into sessions on large time gaps."""
    sessions: list[list[dict]] = []
    cur: list[dict] = []
    last_t = None
    turns = 0
    last_side = None
    for m in msgs:
        t = parse_ts(m["timestamp"])
        side = m["is_from_me"]
        new_turn = side != last_side            # speaking side flipped
        gap_break = last_t is not None and (t - last_t) > gap_seconds
        turn_break = new_turn and turns >= max_turns
        if cur and (gap_break or turn_break):
            sessions.append(cur)
            cur, turns, last_side = [], 0, None
            new_turn = True                     # first msg of a fresh session
        if new_turn:
            turns += 1
            last_side = side
        cur.append(m)
        last_t = t
    if cur:
        sessions.append(cur)
    return sessions


def build_turns(session: list[dict], is_group: bool) -> list[dict]:
    """Merge consecutive same-side messages into alternating user/assistant turns.

    In group chats, non-you speakers are name-tagged so the model can follow who
    said what; consecutive non-you speakers are merged into one 'user' turn so the
    dialogue strictly alternates (chat templates require this).
    """
    turns: list[dict] = []
    for m in session:
        role = "assistant" if m["is_from_me"] else "user"
        line = m["text"].strip()
        if not line:
            continue
        if is_group and role == "user":
            line = f"{m['sender']}: {line}"
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n" + line
        else:
            turns.append({"role": role, "content": line})
    return turns


def trim(turns: list[dict]) -> list[dict]:
    """Start on a user turn, end on an assistant turn (your reply = the target)."""
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    while turns and turns[-1]["role"] != "assistant":
        turns.pop()
    return turns


def prepare(raw_path: Path, out_dir: Path, cfg: dict) -> dict:
    by_chat: dict[int, list[dict]] = defaultdict(list)
    group_flag: dict[int, bool] = {}
    with raw_path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            by_chat[r["chat_id"]].append(r)
            group_flag[r["chat_id"]] = r["is_group"]

    gap_seconds = cfg["gap_minutes"] * 60
    sys_prompt = cfg["system_prompt"].format(my_name=cfg["my_name"])

    examples: list[dict] = []
    for chat_id, msgs in by_chat.items():
        is_group = group_flag[chat_id]
        if is_group and not cfg["include_groups"]:
            continue
        msgs.sort(key=lambda m: parse_ts(m["timestamp"]))
        for session in sessionize(msgs, gap_seconds, cfg["max_turns"]):
            turns = trim(build_turns(session, is_group))
            if len(turns) < cfg["min_turns"]:
                continue
            if not any(t["role"] == "assistant" for t in turns):
                continue
            examples.append({"messages": [{"role": "system", "content": sys_prompt}] + turns})

    rng = random.Random(cfg["seed"])
    rng.shuffle(examples)
    if cfg["max_examples"]:
        examples = examples[: cfg["max_examples"]]

    n_valid = max(1, int(len(examples) * cfg["valid_fraction"])) if examples else 0
    valid, train = examples[:n_valid], examples[n_valid:]

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("valid", valid)):
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    assistant_turns = sum(
        1 for e in examples for t in e["messages"] if t["role"] == "assistant"
    )
    return {
        "examples": len(examples),
        "train": len(train),
        "valid": len(valid),
        "your_reply_targets": assistant_turns,
        "avg_turns": round(
            sum(len(e["messages"]) - 1 for e in examples) / max(1, len(examples)), 1
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build MLX-LM chat training data from messages.")
    ap.add_argument("--raw", type=Path, default=Path("data/raw/messages.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--my-name", dest="my_name")
    ap.add_argument("--no-groups", dest="include_groups", action="store_false", default=None)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.my_name:
        cfg["my_name"] = args.my_name
    if args.include_groups is False:
        cfg["include_groups"] = False
    if args.max_examples:
        cfg["max_examples"] = args.max_examples

    stats = prepare(args.raw, args.out, cfg)
    print("Training data built (local):")
    print(f"  examples ............... {stats['examples']:,}")
    print(f"    train ................ {stats['train']:,}")
    print(f"    valid ................ {stats['valid']:,}")
    print(f"  your reply targets ..... {stats['your_reply_targets']:,}")
    print(f"  avg turns/example ...... {stats['avg_turns']}")
    print(f"  -> {args.out}/train.jsonl, {args.out}/valid.jsonl")


if __name__ == "__main__":
    main()
