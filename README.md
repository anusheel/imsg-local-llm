# imsg-local-llm

Fine-tune a small language model on your own iMessage history so it texts like you, running entirely on a Mac. No cloud, no API keys. Your `chat.db` is read locally and nothing derived from it leaves the machine.

This follows the group-chat cloning demos from a couple of years back (Izzy Miller's "I turned my group chat into an LLM" being the well-known one), reimplemented to run start to finish on Apple silicon with [MLX](https://github.com/ml-explore/mlx).

## What it does

It reads the Messages SQLite database, rebuilds conversations, and trains a LoRA adapter that reproduces your replies in context. Two ways to try it:

- `make train`: LoRA fine-tune Llama-3.1-8B (4-bit) on your conversations. Roughly 20 to 40 minutes on an M-series Mac, and the adapter is a few MB.
- `make quickstart`: no training. Builds an Ollama persona from a style profile plus a few dozen real example exchanges. Ready in about a minute, lower fidelity.

Then talk to it:

```
you › yo you around this weekend?
me  › yeah should be, whats up
you › thinking of a hike sat
me  › im down. early tho, tryna beat the heat
```

## Requirements

- Apple silicon Mac (M1 or newer), macOS 13+, 16 GB RAM or more.
- [uv](https://github.com/astral-sh/uv) for the Python environment: `brew install uv`
- [Ollama](https://ollama.com), only for the quickstart path: `brew install ollama`

## Full Disk Access

macOS guards `~/Library/Messages/chat.db`, so grant your terminal access once under System Settings > Privacy & Security > Full Disk Access. Add whichever terminal you use (Terminal, iTerm, VS Code) and restart it. Confirm with:

```bash
sqlite3 ~/Library/Messages/chat.db "select count(*) from message;"
```

## Usage

```bash
make setup      # .venv on Python 3.12, installs mlx-lm
make data       # chat.db -> data/train.jsonl + data/valid.jsonl
make train      # LoRA fine-tune per config.yaml
make chat       # talk to the result (make base uses the untuned model)
```

Set your name in `config.yaml` (`my_name`), or override per command: `make chat NAME=Sam`.

## How it works

Four stages, one small module each under `src/imsg_local_llm/`:

1. `extract.py` reads `chat.db` (read-only) into `data/raw/messages.jsonl`, one row per message with sender, text, chat, and timestamp.
2. `prepare.py` groups messages into conversation sessions and writes MLX chat-format training data.
3. `train.py` wraps `mlx_lm lora` to fine-tune a 4-bit base model with QLoRA.
4. `chat.py` loads the base model plus your adapter and runs a streaming REPL.

### The attributedBody problem

`SELECT text FROM message` returns almost nothing on a modern Mac. Since Ventura, Messages keeps the body in `attributedBody`: a binary `NSAttributedString` serialized in Apple's old `typedstream` format, with `text` left null. On a recent install that is around 96% of all messages, so any extractor that reads only the `text` column silently drops most of the history.

`decode_attributed_body` pulls the string out of the blob directly. It finds the `NSString` class marker, then reads the length-prefixed UTF-8 introduced by the `+` (`0x2b`) type code, decoding typedstream's variable-length integer for the length (a single byte, or `0x81`/`0x82`/`0x83` prefixing a 2/4/8-byte little-endian value). That recovers about 98% of messages; the remainder are attachment-only with no text to extract.

### Building the training data

Within each chat, messages are split into sessions on a configurable idle gap (default 4 hours) so unrelated exchanges do not bleed together. Consecutive messages from the same person are merged into one turn, since people fire off several texts in a row. Then the roles are mapped: everyone else becomes `user`, you become `assistant`. Group chats are included with each speaker name-tagged, and consecutive non-you speakers are merged so turns strictly alternate (chat templates require it). Each session is trimmed to start on a `user` turn and end on one of yours.

The output is MLX chat format, one JSON object per line:

```json
{"messages": [
  {"role": "system",    "content": "You are Sam texting on iMessage. Reply the way Sam really texts ..."},
  {"role": "user",      "content": "you around this weekend?"},
  {"role": "assistant", "content": "yeah should be, whats up"}
]}
```

Loss is computed over the whole conversation, so every one of your turns is a training target, not only the last reply in a thread.

## Configuration

Everything lives in `config.yaml`:

| Field | Default | Purpose |
| --- | --- | --- |
| `my_name` | `Me` | Name used in the persona prompt and REPL |
| `system_prompt` | persona text | Instruction the model is trained and chatted with |
| `gap_minutes` | `240` | Idle gap that starts a new conversation |
| `include_groups` | `true` | Include group chats, speakers name-tagged |
| `min_turns` / `max_turns` | `3` / `40` | Skip trivially short sessions, cap long ones |
| `train.model` | Llama-3.1-8B-Instruct-4bit | Base model from the mlx-community hub |
| `train.iters` | `600` | Training iterations |
| `train.num_layers` | `16` | How many layers get a LoRA adapter |
| `train.batch_size` | `2` | Batch size |
| `train.learning_rate` | `1e-4` | Learning rate |

Any 4-bit model on the [mlx-community](https://huggingface.co/mlx-community) hub works as a base. Qwen2.5-7B and Mistral-7B are good alternatives to Llama; edit `train.model`.

To use real names instead of `Friend 1`, `Friend 2`, copy `contacts.example.csv` to `contacts.csv` (git-ignored) and map `handle,name`, where the handle is the phone number or email as stored by iMessage.

## Commands

| Command | Description |
| --- | --- |
| `make setup` | Create `.venv` (Python 3.12) and install mlx-lm |
| `make data` | Extract `chat.db` and build the training set |
| `make quickstart` | Instant Ollama persona, no training |
| `make train` / `make train-smoke` | Full or short LoRA fine-tune |
| `make chat` / `make base` | Chat with the tuned or untuned model |
| `make fuse` | Merge the adapter into a standalone `fused_model/` |
| `make clean` / `make wipe` | Remove generated data, or that plus the venv |

## Privacy

Your messages stay on the machine. `.gitignore` blocks `data/`, `adapters/`, `Modelfile`, `contacts.csv`, and every `*.jsonl` and `*.db`, so neither the raw history nor anything trained from it can be committed. The database is opened read-only, so your live Messages data is never modified.

## Layout

```
imsg-local-llm/
├── config.yaml                    hyperparameters and persona
├── Makefile                       setup / data / train / chat
├── contacts.example.csv           optional handle -> name map
└── src/imsg_local_llm/
    ├── extract.py                 chat.db -> messages.jsonl
    ├── prepare.py                 messages -> train/valid.jsonl
    ├── train.py                   mlx_lm lora wrapper
    ├── chat.py                    chat REPL (base + adapter)
    └── quickstart_ollama.py       no-train Ollama persona
```

## License

MIT.
