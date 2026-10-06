.PHONY: train evaluate compare check
CONFIG ?= configs/smoke.json
train:
	llm-train --config $(CONFIG)
evaluate:
	llm-evaluate --config $(CONFIG)
compare:
	llm-compare --root outputs/qwen3-1.7b-base
check:
	ruff check src tests
	ruff format --check src tests
	pytest
