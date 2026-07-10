# imsg-local-llm: one-command pipeline. Everything runs locally.
.PHONY: help setup extract prepare data quickstart train train-smoke train-fast train-fast-3b chat base fuse clean wipe

SYS_PY := python3          # stdlib-only steps (extract/prepare/quickstart)
PY     := .venv/bin/python # venv with mlx-lm (train/chat)
NAME   ?= Me
export PYTHONPATH := src

help:
	@echo "imsg-local-llm targets:"
	@echo "  make setup         create .venv (py3.12) and install mlx-lm"
	@echo "  make extract       chat.db  -> data/raw/messages.jsonl   (decodes attributedBody)"
	@echo "  make prepare       messages -> data/train.jsonl + valid.jsonl"
	@echo "  make data          extract + prepare"
	@echo "  make quickstart    INSTANT clone via Ollama (no training)"
	@echo "  make train         LoRA fine-tune Llama-3.1-8B on your texts (MLX)"
	@echo "  make train-smoke   short end-to-end sanity fine-tune"
	@echo "  make train-fast    faster LoRA: grad-checkpoint off, batch 4, seq 1024, 400 iters"
	@echo "  make train-fast-3b same settings on a 3B base (fastest)"
	@echo "  make chat          chat with your fine-tuned clone"
	@echo "  make base          chat with the base model (no adapter) for comparison"
	@echo "  make fuse          merge adapter into a standalone model -> fused_model/"
	@echo "  make clean         delete data/adapters/Modelfile (keeps .venv)"
	@echo "  make wipe          clean + delete .venv"

setup:
	uv venv --python 3.12
	uv pip install mlx-lm pyyaml
	@echo "\n.venv ready. Next: make data"

extract:
	$(SYS_PY) -m imsg_local_llm.extract

prepare:
	$(SYS_PY) -m imsg_local_llm.prepare --my-name $(NAME)

data: extract prepare

quickstart:
	$(SYS_PY) -m imsg_local_llm.quickstart_ollama --my-name $(NAME) --chat

train:
	$(PY) -m imsg_local_llm.train

train-smoke:
	$(PY) -m imsg_local_llm.train --smoke

train-fast:
	$(PY) -m imsg_local_llm.train --config config.fast.yaml

train-fast-3b:
	$(PY) -m imsg_local_llm.train --config config.fast.yaml --model mlx-community/Llama-3.2-3B-Instruct-4bit

chat:
	$(PY) -m imsg_local_llm.chat --my-name $(NAME)

base:
	$(PY) -m imsg_local_llm.chat --my-name $(NAME) --no-adapter

fuse:
	$(PY) -m mlx_lm.fuse \
		--model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
		--adapter-path adapters --save-path fused_model

clean:
	rm -rf data adapters Modelfile fused_model

wipe: clean
	rm -rf .venv
